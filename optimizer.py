import pandas as pd
import json
import os
from pypfopt import EfficientFrontier, risk_models, expected_returns, HRPOpt
import traceback
import numpy as np
from scipy.optimize import minimize
import warnings

# 경고 메시지 필터링
warnings.filterwarnings('ignore', category=UserWarning, module='pypfopt')

# --- 경로 설정 ---
DATA_DIR = "data"
PRICE_DATA_DIR = os.path.join(DATA_DIR, "fund_prices")
MASTER_FILE_PATH = os.path.join(DATA_DIR, "etf_master.json")

def clean_price_data(df):
    """가격 데이터를 정리하고 이상값을 제거합니다."""
    # 1. 무한값을 NaN으로 변환
    df = df.replace([np.inf, -np.inf], np.nan)
    
    # 2. 0 또는 음수 가격을 NaN으로 변환
    df = df.where(df > 0, np.nan)
    
    # 3. 극단적인 일일 수익률 제거 (±30% 초과) - 더 보수적으로 설정
    returns = df.pct_change()
    outlier_mask = (returns.abs() > 0.3)
    df = df.where(~outlier_mask, np.nan)
    
    # 4. forward fill로 단기 결측값 보완 (최대 3일) - 새로운 pandas 버전 대응
    df = df.ffill(limit=3)
    
    # 5. 연속된 NaN이 너무 많은 행 제거
    df = df.dropna(thresh=len(df.columns) * 0.8)  # 80% 이상의 컬럼에 데이터가 있는 행만 유지
    
    return df

def safe_annualize_performance(returns_series, weights, risk_free_rate=0.02):
    """안전한 포트폴리오 성과 연율화 계산"""
    try:
        # 일일 수익률 계산
        daily_returns = returns_series.pct_change().dropna()
        
        # 이상값 제거
        daily_returns = daily_returns.replace([np.inf, -np.inf], np.nan)
        daily_returns = daily_returns[daily_returns.abs() <= 0.3]  # ±30% 제한
        daily_returns = daily_returns.dropna()
        
        if daily_returns.empty or len(daily_returns) < 10:
            return 0.0, 0.1, 0.0
        
        # 포트폴리오 일일 수익률
        if isinstance(weights, dict):
            portfolio_daily_returns = sum(weights.get(col, 0) * daily_returns[col] 
                                        for col in daily_returns.columns 
                                        if col in weights and not pd.isna(daily_returns[col]).all())
        else:
            # weights가 numpy array인 경우
            portfolio_daily_returns = np.dot(daily_returns.values, weights)
        
        if isinstance(portfolio_daily_returns, pd.Series):
            portfolio_daily_returns = portfolio_daily_returns.dropna()
        
        if len(portfolio_daily_returns) == 0:
            return 0.0, 0.1, 0.0
        
        # 연율화 계산 (252 거래일 기준)
        mean_daily_return = np.mean(portfolio_daily_returns)
        daily_vol = np.std(portfolio_daily_returns, ddof=1)
        
        # NaN 또는 무한값 체크
        if np.isnan(mean_daily_return) or np.isinf(mean_daily_return):
            mean_daily_return = 0.0
        if np.isnan(daily_vol) or np.isinf(daily_vol) or daily_vol <= 0:
            daily_vol = 0.01  # 최소 변동성 설정
        
        # 연율화
        annual_return = mean_daily_return * 252 + risk_free_rate
        annual_vol = daily_vol * np.sqrt(252)
        
        # 변동성 상한선 설정 (100%)
        annual_vol = min(annual_vol, 1.0)
        
        # 샤프 비율
        sharpe = (annual_return - risk_free_rate) / annual_vol if annual_vol > 0 else 0.0
        
        return float(annual_return), float(annual_vol), float(sharpe)
        
    except Exception as e:
        print(f"성과 연율화 계산 오류: {str(e)}")
        return 0.0, 0.1, 0.0

def get_risk_asset_info(selected_tickers):
    """위험자산 정보를 반환합니다."""
    try:
        with open(MASTER_FILE_PATH, 'r', encoding='utf-8') as f:
            master_df = pd.DataFrame(json.load(f))
            master_df.columns = [col.lower() for col in master_df.columns]
        
        selected_info = master_df[master_df['ticker'].isin(selected_tickers)]
        risk_keywords = ["국내주식", "해외주식", "대체투자"]
        risk_tickers = selected_info[selected_info['saa_class'].str.contains('|'.join(risk_keywords), na=False)]['ticker'].tolist()
        return risk_tickers
    except:
        return []

def safe_optimize_with_constraints(mu, S, selected_tickers, target_return, risk_free_rate, price_data, objective="max_sharpe"):
    """안전한 제약조건 최적화를 수행합니다."""
    # None 값 처리
    if target_return is None:
        target_return = risk_free_rate if risk_free_rate is not None else 0.02
    if risk_free_rate is None:
        risk_free_rate = 0.02
    
    # mu와 S의 컬럼 순서와 selected_tickers 순서를 맞춤
    if hasattr(mu, 'index'):
        # NaN이나 무한값이 있는 티커 제거
        valid_tickers = []
        for ticker in selected_tickers:
            if ticker in mu.index and not (np.isnan(mu[ticker]) or np.isinf(mu[ticker])):
                valid_tickers.append(ticker)
        
        if len(valid_tickers) < 2:
            raise ValueError("유효한 데이터를 가진 자산이 2개 미만입니다.")
        
        mu_array = np.array([mu[ticker] for ticker in valid_tickers])
        available_tickers = valid_tickers
    else:
        mu_array = mu
        available_tickers = selected_tickers
    
    if len(available_tickers) != len(selected_tickers):
        print(f"경고 : 일부 티커의 수익률 데이터가 없습니다. 요청: {selected_tickers}, 사용가능: {available_tickers}")
    
    n = len(available_tickers)
    if n < 2:
        raise ValueError("최적화를 위해 최소 2개의 유효한 자산이 필요합니다.")
    
    # 공분산 행렬도 순서 맞춤
    if hasattr(S, 'index'):
        S_array = S.loc[available_tickers, available_tickers].values
    else:
        S_array = S
    
    # 공분산 행렬의 특이값 확인
    try:
        np.linalg.cholesky(S_array)
    except np.linalg.LinAlgError:
        # 정규화 추가
        S_array += np.eye(n) * 1e-8
    
    # 초기 가중치
    x0 = np.array([1/n] * n)
    
    # 제약조건 실행 가능성 검사
    max_possible_return = np.max(mu_array)
    min_possible_return = np.min(mu_array)
    
    # 제약조건 설정
    constraints = []
    
    # 가중치 합계 = 1
    constraints.append({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    
    # 목표수익률 이상 (조정된 값 사용)
    constraints.append({'type': 'ineq', 'fun': lambda x: np.dot(x, mu_array) - target_return})

    
    # 경계 조건 (0 <= 가중치 <= 1)
    bounds = [(0.05, 0.65) for _ in range(n)]
    
    # 목적함수 정의
    if objective == "max_sharpe":
        def objective_func(x):
            portfolio_return = np.dot(x, mu_array)
            portfolio_vol = np.sqrt(np.dot(x, np.dot(S_array, x)))
            if portfolio_vol == 0 or np.isnan(portfolio_vol):
                return 1e6
            sharpe = (portfolio_return - risk_free_rate) / portfolio_vol
            return -sharpe  # 최소화 문제이므로 음수 반환
    elif objective == "min_vol":
        def objective_func(x):
            vol = np.sqrt(np.dot(x, np.dot(S_array, x)))
            return vol if not np.isnan(vol) else 1e6
    
    # 여러 최적화 방법을 순차적으로 시도
    methods = ['SLSQP', 'trust-constr']
    result = None
    
    for method in methods:
        try:
            result = minimize(objective_func, x0, method=method, bounds=bounds, constraints=constraints)
            if result.success and not np.any(np.isnan(result.x)):
                break
            else:
                print(f"{method} 방법 실패: {result.message}")
        except Exception as e:
            print(f"{method} 방법에서 오류 발생: {str(e)}")
            continue
    
    # 모든 방법이 실패하면 제약조건을 완화하여 재시도
    if result is None or not result.success or np.any(np.isnan(result.x)):
        print("제약조건을 완화하여 재시도합니다...")
        
        # 목표수익률을 평균 수익률로 설정
        adjusted_target_return = np.mean(mu_array)
        print(f"목표수익률을 {target_return:.4f}에서 {adjusted_target_return:.4f}로 조정")
        
        # 완화된 제약조건
        relaxed_constraints = []
        relaxed_constraints.append({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        relaxed_constraints.append({'type': 'ineq', 'fun': lambda x: np.dot(x, mu_array) - adjusted_target_return})
        
        try:
            result = minimize(objective_func, x0, method='SLSQP', bounds=bounds, constraints=relaxed_constraints)
        except Exception as e:
            print(f"완화된 제약조건에서도 실패: {str(e)}")
    
    # 최후의 수단: 동일가중 포트폴리오 반환
    if result is None or not result.success or np.any(np.isnan(result.x)):
        print("최적화 실패. 동일가중 포트폴리오를 반환합니다.")
        result_weights = np.array([1/n] * n)
        weights = dict(zip(available_tickers, result_weights))
        
        # 실제 가격 데이터로 성과 계산
        portfolio_return, portfolio_vol, sharpe = safe_annualize_performance(
            price_data[available_tickers], weights, risk_free_rate
        )
        
        return weights, portfolio_return, portfolio_vol, sharpe
    
    weights = dict(zip(available_tickers, result.x))
    
    # 실제 가격 데이터로 성과 계산
    portfolio_return, portfolio_vol, sharpe = safe_annualize_performance(
        price_data[available_tickers], weights, risk_free_rate
    )
    
    return weights, portfolio_return, portfolio_vol, sharpe

# ★★★ 함수 정의와 내부 로직을 모두 수정합니다 ★★★
def safe_optimize_risk_parity_with_constraints(prices, target_return=None, risk_free_rate=0.02):
    """
    가격 데이터를 기반으로 Risk Parity 포트폴리오를 계산하고 제약조건을 처리합니다.
    """
    # 1. 함수 내부에서 수익률(returns)을 직접 계산합니다.
    returns = prices.pct_change().dropna()
    n = returns.shape[1]
    tickers = returns.columns.tolist()
    cov_matrix = returns.cov().values

    # --- Core Risk Parity Optimization ---
    def risk_budget_objective(weights):
        weights = np.array(weights)
        portfolio_vol = np.sqrt(weights.T @ cov_matrix @ weights)
        if portfolio_vol == 0: return 1e9
        marginal_contrib = cov_matrix @ weights / portfolio_vol
        risk_contrib = weights * marginal_contrib
        target_contrib = portfolio_vol / n
        return np.sum((risk_contrib - target_contrib)**2)

    inv_vol = 1.0 / np.sqrt(np.diag(cov_matrix))
    x0 = inv_vol / np.sum(inv_vol)
    bounds = tuple((0.0, 1.0) for _ in range(n))
    constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}]
    result = minimize(risk_budget_objective, x0, method='SLSQP', bounds=bounds, constraints=constraints)

    rp_weights = result.x if result.success else x0

    # --- Constraint Handling ---
    weights_array = rp_weights
    # 2. 올바른 가격 데이터(prices)로 기대수익률(mu)을 계산합니다.
    mu = expected_returns.mean_historical_return(prices)
    mu = mu.replace([np.inf, -np.inf], np.nan).fillna(0)
    
    current_return = mu.values @ weights_array

    if target_return is not None and current_return < target_return:
        print(f"Initial RP return ({current_return:.2%}) is below target ({target_return:.2%}). Adjusting...")
        def adjustment_objective(weights):
            return np.sum((weights - rp_weights)**2)
        adj_constraints = constraints + [{'type': 'ineq', 'fun': lambda w: w @ mu.values - target_return}]
        adj_result = minimize(adjustment_objective, rp_weights, method='SLSQP', bounds=bounds, constraints=adj_constraints)
        if adj_result.success:
            weights_array = adj_result.x
        else:
            print("Warning: Could not meet target return. Returning original RP portfolio.")
    
    # --- Final Performance Calculation ---
    final_weights = dict(zip(tickers, weights_array))
    final_return = mu.values @ weights_array
    # 3. 일일 변동성을 연간 변동성으로 변환합니다.
    daily_volatility = np.sqrt(weights_array.T @ cov_matrix @ weights_array)
    annual_volatility = daily_volatility * np.sqrt(252)
    
    sharpe = (final_return - risk_free_rate) / annual_volatility if annual_volatility > 0 else 0.0

    return final_weights, final_return, annual_volatility, sharpe

def get_optimized_portfolio(selected_tickers, params):
    """
    선택된 티커 목록과 파라미터를 기반으로 포트폴리오를 최적화합니다.
    """
    try:
        print("\n" + "="*50)
        print(" STEP 1: 데이터 로드 및 정제 ".center(50, "="))
        print("="*50)
        print(f"요청된 티커: {selected_tickers}")

        if len(selected_tickers) < 2:
            raise ValueError("최적화를 위해 2개 이상의 종목이 필요합니다.")
            
        all_prices = []
        for ticker in selected_tickers:
            file_path = os.path.join(PRICE_DATA_DIR, f"{ticker}.csv")
            if os.path.exists(file_path):
                try:
                    df = pd.read_csv(
                        file_path, 
                        skiprows=3, 
                        header=None,
                        names=['Date', 'Close', 'High', 'Low', 'Open', 'Volume'],
                        index_col='Date',
                        parse_dates=True,
                        on_bad_lines='skip'
                    )
                    
                    df = df[df.index.notna()]
                    df = df[~df.index.duplicated(keep='first')]
                    df['Adj Close'] = df['Close']
                    
                    if not df.empty and 'Adj Close' in df.columns:
                        price_series = df[['Adj Close']].rename(columns={'Adj Close': ticker})
                        price_series = clean_price_data(price_series)
                        
                        if not price_series.empty:
                            all_prices.append(price_series)
                        else:
                            print(f"⚠️  정리 후 {ticker}의 데이터가 비어있습니다.")
                    else:
                        print(f"⚠️  {ticker}의 가격 데이터가 없습니다.")
                        
                except Exception as e:
                    print(f"⚠️  {ticker} 파일 읽기 오류: {str(e)}")
            else:
                print(f"⚠️  파일이 존재하지 않습니다: {file_path}")

        if not all_prices:
            raise FileNotFoundError("유효한 가격 데이터를 가진 ETF가 하나도 없습니다.")

        price_df_cleaned = pd.concat(all_prices, axis=1, join='outer', sort=True)
        price_df_cleaned = clean_price_data(price_df_cleaned)
        price_df_cleaned = price_df_cleaned.dropna()

        if price_df_cleaned.empty or price_df_cleaned.shape[0] < 10:
            raise ValueError(f"선택된 ETF들의 공통된 거래 기간 데이터가 부족합니다. (티커: {selected_tickers})")
        
        available_tickers = list(price_df_cleaned.columns)
        print(f"최종 분석 대상 티커: {available_tickers}")
        print(f"공통 분석 기간: {price_df_cleaned.index.min().strftime('%Y-%m-%d')} ~ {price_df_cleaned.index.max().strftime('%Y-%m-%d')} ({len(price_df_cleaned)}일)")
        
        if len(available_tickers) < len(selected_tickers):
            print(f"경고: 일부 티커의 데이터가 누락되었습니다. 요청: {selected_tickers}, 사용가능: {available_tickers}")
        
        if len(available_tickers) < 2:
            raise ValueError("최적화를 위해 유효한 데이터를 가진 2개 이상의 종목이 필요합니다.")
        
        print("\n" + "="*50)
        print(" STEP 2: 개별 자산 성과 분석 ".center(50, "="))
        print("="*50)
        
        # ★★★ 핵심 수정사항: mu를 계산 직후 바로 정제하여 모든 하위 로직에 안정적인 데이터를 전달합니다.
        mu = expected_returns.mean_historical_return(price_df_cleaned)
        mu = mu.replace([np.inf, -np.inf], np.nan).fillna(0)
        
        S = risk_models.sample_cov(price_df_cleaned)
        
        for ticker in available_tickers:
            ticker_return = mu.get(ticker, 0)
            ticker_volatility = np.sqrt(S.loc[ticker, ticker]) 
            print(f"▶ {ticker}")
            print(f"  - 연율화 기대수익률: {ticker_return:.2%}")
            print(f"  - 연율화 변동성: {ticker_volatility:.2%}")

        print("\n" + "="*50)
        print(" STEP 3: 포트폴리오 최적화 실행 ".center(50, "="))
        print("="*50)
        
        mode = params.get("mode", "MVO")
        
        risk_free_rate = params.get("risk_free_rate") if params.get("risk_free_rate") is not None else 0.02
        target_return = params.get("target_return") if params.get("target_return") is not None else risk_free_rate

        print(f"최적화 모드: {mode}, 무위험수익률: {risk_free_rate:.2%}, 목표수익률: {target_return:.2%}")
        
        if mode == "EqualWeight":
            weights, e_ret, ann_vol, sharpe = safe_optimize_with_constraints(
                mu, S, available_tickers, target_return, risk_free_rate, price_df_cleaned, "min_vol"
            )
        elif mode == "RiskParity":
            # ★★★ 수정: 수익률(returns) 대신 가격(prices) 데이터를 전달합니다. ★★★
            weights, e_ret, ann_vol, sharpe = safe_optimize_risk_parity_with_constraints(
                prices=price_df_cleaned,
                target_return=target_return,
                risk_free_rate=risk_free_rate
            )
        elif mode == "MVO":
            mvo_objective = params.get("mvo_objective", "max_sharpe")
            print(f"MVO 목적 함수: {mvo_objective}")
            weights, e_ret, ann_vol, sharpe = safe_optimize_with_constraints(
                mu, S, available_tickers, target_return, risk_free_rate, price_df_cleaned, mvo_objective
            )
        else:
            raise ValueError(f"알 수 없는 모드입니다: {mode}")

        cleaned_weights = {k: max(0, v) for k, v in weights.items()}
        total_weight = sum(cleaned_weights.values())
        if total_weight > 0:
            cleaned_weights = {k: v/total_weight for k, v in cleaned_weights.items()}
        else:
            cleaned_weights = {k: 1/len(available_tickers) for k in available_tickers}

        print("\n" + "="*50)
        print(" STEP 4: 최종 최적화 결과 ".center(50, "="))
        print("="*50)
        print("▶ 최적 비중:")
        for ticker, weight in cleaned_weights.items():
            print(f"  - {ticker}: {weight:.2%}")
        
        print("\n▶ 포트폴리오 성과:")
        print(f"  - 기대수익률: {e_ret:.2%}")
        print(f"  - 변동성: {ann_vol:.2%}")
        print(f"  - 샤프지수: {sharpe:.2f}")
        print("="*50 + "\n")
        
        def safe_float(value, default=0.0):
            try:
                f_val = float(value)
                if np.isnan(f_val) or np.isinf(f_val):
                    return default
                return f_val
            except (ValueError, TypeError):
                return default

        performance = {
            "expected_annual_return": safe_float(e_ret),
            "annual_volatility": safe_float(ann_vol),
            "sharpe_ratio": safe_float(sharpe)
        }
        
        return cleaned_weights, performance

    except np.linalg.LinAlgError:
        traceback.print_exc()
        raise ValueError("선택된 종목들의 상관관계가 너무 높아 계산이 불가능합니다. 다른 자산 조합을 선택해주세요.")
    except Exception as e:
        traceback.print_exc()
        raise e