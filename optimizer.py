import pandas as pd
import json
import os
from pypfopt import EfficientFrontier, risk_models, expected_returns, HRPOpt
import traceback
import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm
import warnings
import math

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

def calculate_trimmed_mean_returns(price_data, trim_ratio=0.3):
    """
    🆕 새로운 함수: 상위/하위 30% 제거 후 평균 계산하는 Trimmed Mean 기대수익률
    
    Args:
        price_data (pd.DataFrame): 가격 데이터
        trim_ratio (float): 제거할 비율 (0.3 = 상위30% + 하위30% 제거)
    
    Returns:
        pd.Series: 연율화된 기대수익률
    """
    try:
        print("🔄 Trimmed Mean 기대수익률 계산 중...")
        
        # 1. 일일 수익률 계산
        returns = price_data.pct_change().dropna()
        print(f"   - 수익률 데이터 기간: {len(returns)}일")
        
        if returns.empty:
            print("   ⚠️ 수익률 데이터가 비어있습니다.")
            return pd.Series(index=price_data.columns, data=0.0)
        
        # 2. 각 자산별로 Trimmed Mean 계산
        trimmed_returns = {}
        
        for ticker in returns.columns:
            # 해당 자산의 일일 수익률 시리즈 (NaN 제거)
            asset_returns = returns[ticker].dropna()
            
            if len(asset_returns) < 10:  # 최소 10일 데이터 필요
                print(f"   ⚠️ {ticker}: 데이터 부족 ({len(asset_returns)}일)")
                trimmed_returns[ticker] = 0.0
                continue
            
            # 3. 상위 30%, 하위 30% 제거
            # scipy.stats.trim_mean을 사용하거나 직접 구현
            sorted_returns = asset_returns.sort_values()
            n = len(sorted_returns)
            
            # 제거할 개수 계산 (양쪽에서 각각 30%)
            trim_count = int(n * trim_ratio)
            
            # 양쪽 극값 제거
            if trim_count > 0:
                trimmed_data = sorted_returns.iloc[trim_count:-trim_count]
            else:
                trimmed_data = sorted_returns
            
            # 4. 중간값들의 평균 계산
            if len(trimmed_data) > 0:
                daily_mean = trimmed_data.mean()
                # 5. 연율화 (252 거래일 기준)
                annual_return = daily_mean * 252
            else:
                annual_return = 0.0
            
            trimmed_returns[ticker] = annual_return
            
            print(f"   - {ticker}: 원본 {n}일 → 트림 후 {len(trimmed_data)}일 → 연수익률 {annual_return:.2%}")
        
        # 6. pd.Series로 변환
        mu_trimmed = pd.Series(trimmed_returns)
        
        # 7. 이상값 처리
        mu_trimmed = mu_trimmed.replace([np.inf, -np.inf], np.nan).fillna(0)
        
        print(f"✅ Trimmed Mean 기대수익률 계산 완료")
        return mu_trimmed
        
    except Exception as e:
        print(f"❌ Trimmed Mean 계산 오류: {str(e)}")
        # 오류 시 기본값 반환
        return pd.Series(index=price_data.columns, data=0.02)  # 2% 기본값

def safe_annualize_performance(price_data, weights, risk_free_rate=0.02, mu=None):
    """
    ✅ 통일된 수익률 계산을 사용한 포트폴리오 성과 연율화 계산
    외부에서 계산된 mu(Trimmed Mean 기대수익률)를 우선 사용하고, 없으면 내부에서 Trimmed Mean으로 계산
    """
    try:
        # ✅ 1. 외부에서 전달받은 mu가 있으면 그것을 사용 (일관성 유지)
        if mu is not None:
            # 이론적 포트폴리오 수익률 계산
            portfolio_return = sum(weights.get(ticker, 0) * mu.get(ticker, 0) 
                                 for ticker in mu.index if ticker in weights)
            
            # 공분산 행렬로 변동성 계산 (변경 없음)
            if isinstance(price_data, pd.DataFrame) and len(price_data.columns) > 1:
                S = risk_models.sample_cov(price_data)
                tickers = list(weights.keys())
                weights_array = np.array([weights[ticker] for ticker in tickers])
                cov_matrix = S.loc[tickers, tickers].values
                daily_vol = np.sqrt(weights_array.T @ cov_matrix @ weights_array)
                annual_vol = daily_vol 
            else:
                # 단일 자산인 경우
                annual_vol = 0.1  # 기본값
            
            sharpe = (portfolio_return - risk_free_rate) / annual_vol if annual_vol > 0 else 0.0
            return float(portfolio_return), float(annual_vol), float(sharpe)
        
        # ✅ 2. mu가 없으면 Trimmed Mean으로 내부 계산 (기존 EMA에서 변경)
        else:
            # 🆕 Trimmed Mean 기대수익률 계산
            mu_trimmed = calculate_trimmed_mean_returns(price_data)
            
            # 포트폴리오 수익률
            portfolio_return = sum(weights.get(ticker, 0) * mu_trimmed.get(ticker, 0) 
                                 for ticker in mu_trimmed.index if ticker in weights)
            
            # 변동성 계산 (공분산 방식은 그대로 유지)
            S = risk_models.sample_cov(price_data)
            tickers = list(weights.keys())
            weights_array = np.array([weights[ticker] for ticker in tickers])
            cov_matrix = S.loc[tickers, tickers].values
            annual_vol = np.sqrt(weights_array.T @ cov_matrix @ weights_array)
            
            # 변동성 상한선 설정 (100%)
            annual_vol = min(annual_vol, 1.0)
            
            # 샤프 비율
            sharpe = (portfolio_return - risk_free_rate) / annual_vol if annual_vol > 0 else 0.0
            
            return float(portfolio_return), float(annual_vol), float(sharpe)
        
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
        
        # ✅ 통일된 mu를 사용하여 성과 계산
        portfolio_return, portfolio_vol, sharpe = safe_annualize_performance(
            price_data[available_tickers], weights, risk_free_rate, mu
        )
        
        return weights, portfolio_return, portfolio_vol, sharpe
    
    weights = dict(zip(available_tickers, result.x))
    
    # ✅ 통일된 mu를 사용하여 성과 계산
    portfolio_return, portfolio_vol, sharpe = safe_annualize_performance(
        price_data[available_tickers], weights, risk_free_rate, mu
    )
    
    return weights, portfolio_return, portfolio_vol, sharpe

def safe_optimize_risk_parity_with_constraints(mu, S, prices, target_return=None, risk_free_rate=0.02):
    """
    ✅ 외부에서 계산된 mu, S를 받아서 사용하는 Risk Parity 최적화
    """
    returns = prices.pct_change().dropna()
    n = returns.shape[1]
    tickers = returns.columns.tolist()
    
    # ✅ 외부에서 전달받은 공분산 행렬 사용
    cov_matrix = S.values

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
    
    # ✅ 외부에서 전달받은 mu 사용 (다시 계산하지 않음!)
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
    
    # ✅ 통일된 mu를 사용하여 성과 계산
    final_return, annual_volatility, sharpe = safe_annualize_performance(
        prices, final_weights, risk_free_rate, mu
    )

    return final_weights, final_return, annual_volatility, sharpe

def get_optimized_portfolio(selected_tickers, params):
    """
    🆕 Trimmed Mean 기대수익률을 사용한 통일된 포트폴리오 최적화
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
        print(" STEP 2: 개별 자산 성과 분석 (Trimmed Mean 방식) ".center(50, "="))
        print("="*50)
        
        # 🆕 핵심 수정사항: Trimmed Mean 기대수익률로 통일
        print("📊 Trimmed Mean 기대수익률 계산 중...")
        mu = calculate_trimmed_mean_returns(price_df_cleaned)
        
        # 공분산 계산은 기존 방식 그대로 유지
        print("📊 공분산 행렬 계산 중 (기존 방식 유지)...")
        S = risk_models.sample_cov(price_df_cleaned)
        
        print("✅ 모든 함수가 동일한 Trimmed Mean 기대수익률 사용")
        for ticker in available_tickers:
            ticker_return = mu.get(ticker, 0)
            ticker_volatility = np.sqrt(S.loc[ticker, ticker]) 
            print(f"▶ {ticker}")
            print(f"  - Trimmed Mean 연율화 기대수익률: {ticker_return:.2%}")
            print(f"  - 연율화 변동성: {ticker_volatility:.2%}")

        print("\n" + "="*50)
        print(" STEP 3: 포트폴리오 최적화 실행 ".center(50, "="))
        print("="*50)
        
        mode = params.get("mode", "MVO")
        
        risk_free_rate = params.get("risk_free_rate") if params.get("risk_free_rate") is not None else 0.02
        target_return = params.get("target_return") if params.get("target_return") is not None else risk_free_rate

        print(f"최적화 모드: {mode}, 무위험수익률: {risk_free_rate:.2%}, 목표수익률: {target_return:.2%}")
        
        # ✅ 모든 최적화 모드에서 동일한 mu, S 사용
        if mode == "EqualWeight":
            weights, e_ret, ann_vol, sharpe = safe_optimize_with_constraints(
                mu, S, available_tickers, target_return, risk_free_rate, price_df_cleaned, "min_vol"
            )
        elif mode == "RiskParity":
            # ✅ 통일된 mu, S 전달
            weights, e_ret, ann_vol, sharpe = safe_optimize_risk_parity_with_constraints(
                mu, S, price_df_cleaned, target_return, risk_free_rate
            )
        elif mode == "MVO":
            mvo_objective = params.get("mvo_objective", "max_sharpe")
            print(f"MVO 목적 함수: {mvo_objective}")
            weights, e_ret, ann_vol, sharpe = safe_optimize_with_constraints(
                mu, S, available_tickers, target_return, risk_free_rate, price_df_cleaned, mvo_objective
            )
        elif mode == "Rebalancing":
            # 🆕 리밸런싱 모드 추가
            rebalancing_objective = params.get("mvo_objective", "max_sharpe")
            weight_change_limit = params.get("risk_asset_limit", 0.5)  # 리밸런싱 강도를 weight_change로 사용
            current_weights = params.get("current_weights", {})  # 현재 보유 비중
            
            print(f"Rebalancing 목적 함수: {rebalancing_objective}")
            print(f"Weight change 한계: {weight_change_limit:.2%}")
            print(f"현재 보유 비중: {current_weights}")
            
            weights, e_ret, ann_vol, sharpe = safe_rebalancing_optimize(
                mu, S, available_tickers, target_return, risk_free_rate, price_df_cleaned, 
                rebalancing_objective, weight_change_limit, current_weights
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
        print(" STEP 4: 최종 최적화 결과 (Trimmed Mean 통일) ".center(50, "="))
        print("="*50)
        print("✅ 모든 함수가 동일한 Trimmed Mean 기대수익률 사용")
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
    

def safe_rebalancing_optimize(mu, S, selected_tickers, target_return, risk_free_rate, price_data, 
                            objective, weight_change_limit, current_weights):
    """
    리밸런싱 제약조건이 포함된 포트폴리오 최적화
    
    Args:
        mu: 기대수익률
        S: 공분산 행렬  
        selected_tickers: 선택된 티커 목록
        target_return: 목표 수익률
        risk_free_rate: 무위험 수익률
        price_data: 가격 데이터
        objective: 목적 함수 ("max_sharpe" or "min_vol")
        weight_change_limit: 비중 변경 한계 (0.0 ~ 1.0)
        current_weights: 현재 보유 비중 딕셔너리
    """
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
        
        if len(valid_tickers) < 1:
            raise ValueError("유효한 데이터를 가진 자산이 1개 미만입니다.")
        
        mu_array = np.array([mu[ticker] for ticker in valid_tickers])
        available_tickers = valid_tickers
    else:
        mu_array = mu
        available_tickers = selected_tickers
    
    n = len(available_tickers)
    if n < 1:
        raise ValueError("리밸런싱을 위해 최소 1개의 유효한 자산이 필요합니다.")
    
    # 공분산 행렬도 순서 맞춤
    if hasattr(S, 'index'):
        if n == 1:
            S_array = np.array([[S.loc[available_tickers[0], available_tickers[0]]]])
        else:
            S_array = S.loc[available_tickers, available_tickers].values
    else:
        S_array = S
    
    # 공분산 행렬의 특이값 확인 (다중 자산인 경우만)
    if n > 1:
        try:
            np.linalg.cholesky(S_array)
        except np.linalg.LinAlgError:
            # 정규화 추가
            S_array += np.eye(n) * 1e-8
    
    # 현재 비중 벡터 생성
    current_weights_array = np.array([current_weights.get(ticker, 1/n) for ticker in available_tickers])
    
    # 현재 비중 정규화 (합계 = 1)
    if np.sum(current_weights_array) > 0:
        current_weights_array = current_weights_array / np.sum(current_weights_array)
    else:
        current_weights_array = np.array([1/n] * n)
    
    # 초기 가중치 (현재 비중에서 시작)
    x0 = current_weights_array.copy()
    
    # 제약조건 설정
    constraints = []
    
    # 가중치 합계 = 1
    constraints.append({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    
    # 목표수익률 이상 (단일 자산이 아닌 경우만)
    if n > 1:
        constraints.append({'type': 'ineq', 'fun': lambda x: np.dot(x, mu_array) - target_return})
    
    # 🆕 핵심: Weight Change 제약조건 추가
    # |w_new - w_current|의 합이 weight_change_limit 이하
    def weight_change_constraint(x):
        weight_changes = np.abs(x - current_weights_array)
        total_change = np.sum(weight_changes)
        return weight_change_limit - total_change  # >= 0이어야 함
    
    constraints.append({'type': 'ineq', 'fun': weight_change_constraint})
    
    print(f"현재 비중: {dict(zip(available_tickers, current_weights_array))}")
    print(f"Weight change 한계: {weight_change_limit:.2%}")

    # 경계 조건 (0 <= 가중치 <= 1)
    bounds = [(0.0, 1.0) for _ in range(n)]
    
    # 목적함수 정의
    if objective == "max_sharpe":
        def objective_func(x):
            portfolio_return = np.dot(x, mu_array)
            if n == 1:
                portfolio_vol = np.sqrt(S_array[0,0])
            else:
                portfolio_vol = np.sqrt(np.dot(x, np.dot(S_array, x)))
            if portfolio_vol == 0 or np.isnan(portfolio_vol):
                return 1e6
            sharpe = (portfolio_return - risk_free_rate) / portfolio_vol
            return -sharpe  # 최소화 문제이므로 음수 반환
    elif objective == "min_vol":
        def objective_func(x):
            if n == 1:
                vol = np.sqrt(S_array[0,0])
            else:
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
    
    # 모든 방법이 실패하면 weight_change_limit을 완화하여 재시도
    if result is None or not result.success or np.any(np.isnan(result.x)):
        print("제약조건을 완화하여 재시도합니다...")
        
        # weight_change_limit을 50% 늘려서 재시도
        relaxed_limit = min(weight_change_limit * 1.5, 2.0)  # 최대 200%까지
        print(f"Weight change 한계를 {weight_change_limit:.2%}에서 {relaxed_limit:.2%}로 완화")
        
        def relaxed_weight_change_constraint(x):
            weight_changes = np.abs(x - current_weights_array)
            total_change = np.sum(weight_changes)
            return relaxed_limit - total_change
        
        # 완화된 제약조건
        relaxed_constraints = [c for c in constraints if c['fun'] != weight_change_constraint]
        relaxed_constraints.append({'type': 'ineq', 'fun': relaxed_weight_change_constraint})
        
        try:
            result = minimize(objective_func, x0, method='SLSQP', bounds=bounds, constraints=relaxed_constraints)
        except Exception as e:
            print(f"완화된 제약조건에서도 실패: {str(e)}")
    
    # 최후의 수단: 현재 비중 유지 (아주 작은 조정만)
    if result is None or not result.success or np.any(np.isnan(result.x)):
        print("최적화 실패. 현재 비중에서 소폭 조정합니다.")
        
        # 현재 비중에서 5% 내에서만 조정
        small_adjustment = 0.05
        result_weights = current_weights_array.copy()
        
        # 가장 성과가 좋은 자산의 비중을 약간 늘리고, 나쁜 자산의 비중을 약간 줄임
        if n > 1:
            best_asset_idx = np.argmax(mu_array)
            worst_asset_idx = np.argmin(mu_array)
            
            # 5% 이내에서 조정
            adjustment = min(small_adjustment, result_weights[worst_asset_idx])
            result_weights[worst_asset_idx] -= adjustment
            result_weights[best_asset_idx] += adjustment
        
        weights = dict(zip(available_tickers, result_weights))
        
        # ✅ 통일된 mu를 사용하여 성과 계산
        portfolio_return, portfolio_vol, sharpe = safe_annualize_performance(
            price_data[available_tickers] if hasattr(price_data, 'columns') else None, 
            weights, risk_free_rate, mu
        )
        
        return weights, portfolio_return, portfolio_vol, sharpe
    
    weights = dict(zip(available_tickers, result.x))
    
    # ✅ 통일된 mu를 사용하여 성과 계산
    portfolio_return, portfolio_vol, sharpe = safe_annualize_performance(
        price_data[available_tickers] if hasattr(price_data, 'columns') else None, 
        weights, risk_free_rate, mu
    )
    
    # 🔍 결과 분석 출력
    print(f"\n리밸런싱 결과:")
    for ticker in available_tickers:
        old_weight = current_weights.get(ticker, 0)
        new_weight = weights.get(ticker, 0)
        change = new_weight - old_weight
        print(f"  {ticker}: {old_weight:.2%} → {new_weight:.2%} ({change:+.2%})")
    
    total_change = sum(abs(weights.get(ticker, 0) - current_weights.get(ticker, 0)) 
                      for ticker in available_tickers)
    print(f"총 비중 변경량: {total_change:.2%} (한계: {weight_change_limit:.2%})")
    
    return weights, portfolio_return, portfolio_vol, sharpe


def ValueAtRisk(annual_return, annual_vol, risk_free_rate=0.02):
    """
    이미 계산된 연간 수익률과 변동성으로 VaR 계산
    정규분포 가정 하에 좌측 1%, 5%, 10% VaR 계산
    """
    try:
        # VaR 계산 (정규분포 가정)
        confidence_levels = [0.99, 0.95, 0.90]  # 1%, 5%, 10% VaR
        var_results = {}
        
        for confidence in confidence_levels:
            # 정규분포의 분위수 계산
            z_score = norm.ppf(1 - confidence)  # 좌측 꼬리 분위수
            
            # VaR = 기대수익률 + (Z-score × 변동성)
            var_1year = annual_return + z_score * annual_vol
            
            confidence_pct = int((1 - confidence) * 100)
            var_results[f"var_{confidence_pct}pct"] = {
                "confidence_level": f"{confidence_pct}%",
                "var_1year": round(var_1year * 100, 2),  # 백분율로 변환
                "var_1year_display": f"{var_1year:.1%}"
            }
        
        return {
            "portfolio_return": round(annual_return * 100, 2),
            "portfolio_volatility": round(annual_vol * 100, 2),
            "var_calculations": var_results
        }
        
    except Exception as e:
        print(f"VaR 계산 오류: {str(e)}")
        return {
            "error": str(e),
            "portfolio_return": 0.0,
            "portfolio_volatility": 0.0,
            "var_calculations": {}
        }


def shortfallrisk(annual_return, annual_vol, risk_free_rate=0.02):
    """
    이미 계산된 연간 수익률과 변동성으로 shortfall risk 계산
    1년~20년 투자 시 손실확률 계산
    n년 투자시: return(mean)*n, vol(std)*sqrt(n)
    """
    try:
        # Shortfall Risk 계산 (1년~20년)
        shortfall_results = []
        
        for years in range(1, 21):
            # n년 투자 시 기대수익률과 변동성
            expected_return_n_years = annual_return * years
            volatility_n_years = annual_vol * math.sqrt(years)
            
            # 손실확률 계산 (정규분포 가정)
            # P(수익률 < 0) = Φ((0 - μ) / σ)
            if volatility_n_years > 0:
                z_score = (0 - expected_return_n_years) / volatility_n_years
                loss_probability = norm.cdf(z_score)
            else:
                loss_probability = 0.0 if expected_return_n_years > 0 else 1.0
            
            shortfall_results.append({
                "years": years,
                "expected_return": round(expected_return_n_years * 100, 2),
                "volatility": round(volatility_n_years * 100, 2),
                "loss_probability": round(loss_probability * 100, 2),
                "loss_probability_display": f"{loss_probability:.1%}"
            })
        
        return {
            "portfolio_annual_return": round(annual_return * 100, 2),
            "portfolio_annual_volatility": round(annual_vol * 100, 2),
            "shortfall_risk_by_years": shortfall_results
        }
        
    except Exception as e:
        print(f"Shortfall Risk 계산 오류: {str(e)}")
        return {
            "error": str(e),
            "portfolio_annual_return": 0.0,
            "portfolio_annual_volatility": 0.0,
            "shortfall_risk_by_years": []
        }