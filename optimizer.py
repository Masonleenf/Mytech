import pandas as pd
import numpy as np
from pymongo import MongoClient
from pypfopt import EfficientFrontier, risk_models, expected_returns
from scipy.optimize import minimize
from scipy.stats import norm
import warnings
import math

warnings.filterwarnings('ignore', category=UserWarning, module='pypfopt')

# MongoDB 설정
MONGO_URI = "mongodb+srv://rator9521_db_user:qwe343434@cluster0.d126rkt.mongodb.net/"
ETF_DATABASE = "etf_database"

client = MongoClient(MONGO_URI)
db = client[ETF_DATABASE]
synthetic_indices_collection = db['synthetic_indices']
etf_master_collection = db['etf_master']

def get_price_data_from_mongodb(ticker):
    """MongoDB에서 개별 ETF 가격 데이터 로드 (리밸런싱용)"""
    try:
        from pymongo import MongoClient
        
        MONGO_URI = "mongodb+srv://rator9521_db_user:qwe343434@cluster0.d126rkt.mongodb.net/"
        client = MongoClient(MONGO_URI)
        db = client["etf_database"]
        fund_prices_collection = db['fund_prices']
        
        doc = fund_prices_collection.find_one({'ticker': ticker})
        
        if not doc or 'prices' not in doc:
            print(f"MongoDB에서 {ticker} 개별 ETF 데이터를 찾을 수 없습니다.")
            return None
        
        df = pd.DataFrame(doc['prices'])
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.set_index('Date').sort_index()
        df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
        df = df.dropna(subset=['Close'])
        df['Adj Close'] = df['Close']
        
        return df[['Adj Close']]
        
    except Exception as e:
        print(f"MongoDB에서 {ticker} 로드 오류: {e}")
        return None

def get_synthetic_index_from_mongodb(code):
    """MongoDB에서 합성 지수 데이터를 메모리로 직접 로드"""
    try:
        doc = synthetic_indices_collection.find_one({'code': code})
        
        if not doc or 'data' not in doc:
            print(f"MongoDB에서 {code} 합성 지수를 찾을 수 없습니다.")
            return None
        
        df = pd.DataFrame(doc['data'])
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.set_index('Date').sort_index()
        df['close'] = pd.to_numeric(df['close'], errors='coerce')
        df = df.dropna(subset=['close'])
        df['Adj Close'] = df['close']
        
        return df[['Adj Close']]
        
    except Exception as e:
        print(f"MongoDB에서 {code} 로드 오류: {e}")
        return None

def get_optimized_portfolio_rebalancing(holding_tickers, selected_codes, code_to_ticker_map, params):
    """
    리밸런싱 전용: 보유종목(ticker) + 추가자산(code) 혼합 최적화
    """
    try:
        print("\n" + "="*50)
        print(" 리밸런싱: 혼합 데이터 로드 ".center(50, "="))
        print("="*50)
        print(f"보유 종목 (ticker): {holding_tickers}")
        print(f"추가 자산 (code): {selected_codes}")
        
        all_prices = []
        final_identifiers = []  # ticker 또는 code
        
        # 1. 보유 종목 데이터 로드 (ticker → fund_prices)
        for ticker in holding_tickers:
            price_series = get_price_data_from_mongodb(ticker)
            
            if price_series is not None and not price_series.empty:
                price_series = clean_price_data(price_series)
                
                if not price_series.empty:
                    price_series = price_series.rename(columns={'Adj Close': ticker})
                    all_prices.append(price_series)
                    final_identifiers.append(ticker)
                    print(f"✅ 보유 종목 로드: {ticker}")
                else:
                    print(f"⚠️ {ticker} 데이터 정리 후 비어있음")
            else:
                print(f"⚠️ {ticker} 데이터 없음")
        
        # 2. 추가 자산 데이터 로드 (code → synthetic_indices)
        for code in selected_codes:
            price_series = get_synthetic_index_from_mongodb(code)
            
            if price_series is not None and not price_series.empty:
                price_series = clean_price_data(price_series)
                
                if not price_series.empty:
                    price_series = price_series.rename(columns={'Adj Close': code})
                    all_prices.append(price_series)
                    final_identifiers.append(code)
                    print(f"✅ 추가 자산 로드: {code}")
                else:
                    print(f"⚠️ {code} 데이터 정리 후 비어있음")
            else:
                print(f"⚠️ {code} 합성 지수 없음")
        
        if len(all_prices) < 2:
            raise ValueError(f"최적화를 위해 2개 이상의 자산이 필요합니다. 현재: {len(all_prices)}개")
        
        # 3. 데이터 병합
        price_df_cleaned = pd.concat(all_prices, axis=1, join='outer', sort=True)
        price_df_cleaned = clean_price_data(price_df_cleaned)
        price_df_cleaned = price_df_cleaned.dropna()
        
        if price_df_cleaned.empty or price_df_cleaned.shape[0] < 10:
            raise ValueError(f"공통 거래 기간 데이터 부족")
        
        available_identifiers = list(price_df_cleaned.columns)
        print(f"\n최종 분석 대상: {available_identifiers}")
        print(f"공통 분석 기간: {price_df_cleaned.index.min().strftime('%Y-%m-%d')} ~ {price_df_cleaned.index.max().strftime('%Y-%m-%d')}")
        
        # 4. 최적화 수행
        print("\n" + "="*50)
        print(" 최적화 수행 ".center(50, "="))
        print("="*50)
        
        mu = calculate_trimmed_mean_returns(price_df_cleaned)
        S = risk_models.sample_cov(price_df_cleaned)
        
        risk_free_rate = params.get("risk_free_rate", 0.02)
        target_return = params.get("target_return", risk_free_rate)
        mvo_objective = params.get("mvo_objective", "max_sharpe")
        
        weights, e_ret, ann_vol, sharpe = safe_optimize_with_constraints(
            mu, S, available_identifiers, target_return, risk_free_rate, price_df_cleaned, mvo_objective
        )
        
        cleaned_weights = {k: max(0, v) for k, v in weights.items()}
        total_weight = sum(cleaned_weights.values())
        if total_weight > 0:
            cleaned_weights = {k: v/total_weight for k, v in cleaned_weights.items()}
        
        # 5. code를 ticker로 변환
        print("\n" + "="*50)
        print(" 결과 변환 ".center(50, "="))
        print("="*50)
        
        ticker_weights = {}
        for identifier, weight in cleaned_weights.items():
            if identifier in holding_tickers:
                # 보유 종목은 그대로
                ticker_weights[identifier] = weight
                print(f"보유 종목: {identifier} → {weight:.2%}")
            else:
                # 추가 자산은 code → ticker 변환
                ticker = code_to_ticker_map.get(identifier, f"{identifier}.KS")
                ticker_weights[ticker] = weight
                print(f"추가 자산: {identifier} → {ticker} → {weight:.2%}")
        
        performance = {
            "expected_annual_return": float(e_ret),
            "annual_volatility": float(ann_vol),
            "sharpe_ratio": float(sharpe)
        }
        
        return ticker_weights, performance
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise e

def clean_price_data(df):
    """가격 데이터 정리"""
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.where(df > 0, np.nan)
    
    returns = df.pct_change()
    outlier_mask = (returns.abs() > 0.3)
    df = df.where(~outlier_mask, np.nan)
    
    df = df.ffill(limit=3)
    df = df.dropna(thresh=len(df.columns) * 0.8)
    
    return df

def calculate_trimmed_mean_returns(price_data, trim_ratio=0.3):
    """Trimmed Mean 기대수익률 계산"""
    try:
        print("📊 Trimmed Mean 기대수익률 계산 중...")
        
        returns = price_data.pct_change().dropna()
        print(f"   - 수익률 데이터 기간: {len(returns)}일")
        
        if returns.empty:
            print("   ⚠️ 수익률 데이터가 비어있습니다.")
            return pd.Series(index=price_data.columns, data=0.0)
        
        trimmed_returns = {}
        
        for code in returns.columns:
            asset_returns = returns[code].dropna()
            
            if len(asset_returns) < 10:
                print(f"   ⚠️ {code}: 데이터 부족 ({len(asset_returns)}일)")
                trimmed_returns[code] = 0.0
                continue
            
            sorted_returns = asset_returns.sort_values()
            n = len(sorted_returns)
            trim_count = int(n * trim_ratio)
            
            if trim_count > 0:
                trimmed_data = sorted_returns.iloc[trim_count:-trim_count]
            else:
                trimmed_data = sorted_returns
            
            if len(trimmed_data) > 0:
                daily_mean = trimmed_data.mean()
                annual_return = daily_mean * 252
            else:
                annual_return = 0.0
            
            trimmed_returns[code] = annual_return
            print(f"   - {code}: 원본 {n}일 → 트림 후 {len(trimmed_data)}일 → 연수익률 {annual_return:.2%}")
        
        mu_trimmed = pd.Series(trimmed_returns)
        mu_trimmed = mu_trimmed.replace([np.inf, -np.inf], np.nan).fillna(0)
        
        print(f"✅ Trimmed Mean 기대수익률 계산 완료")
        return mu_trimmed
        
    except Exception as e:
        print(f"❌ Trimmed Mean 계산 오류: {str(e)}")
        return pd.Series(index=price_data.columns, data=0.02)

def safe_annualize_performance(price_data, weights, risk_free_rate=0.02, mu=None):
    """포트폴리오 성과 연율화 계산"""
    try:
        if mu is not None:
            portfolio_return = sum(weights.get(code, 0) * mu.get(code, 0) 
                                 for code in mu.index if code in weights)
            
            if isinstance(price_data, pd.DataFrame) and len(price_data.columns) > 1:
                S = risk_models.sample_cov(price_data)
                codes = list(weights.keys())
                weights_array = np.array([weights[code] for code in codes])
                cov_matrix = S.loc[codes, codes].values
                daily_vol = np.sqrt(weights_array.T @ cov_matrix @ weights_array)
                annual_vol = daily_vol
            else:
                annual_vol = 0.1
            
            sharpe = (portfolio_return - risk_free_rate) / annual_vol if annual_vol > 0 else 0.0
            return float(portfolio_return), float(annual_vol), float(sharpe)
        
        else:
            mu_trimmed = calculate_trimmed_mean_returns(price_data)
            
            portfolio_return = sum(weights.get(code, 0) * mu_trimmed.get(code, 0) 
                                 for code in mu_trimmed.index if code in weights)
            
            S = risk_models.sample_cov(price_data)
            codes = list(weights.keys())
            weights_array = np.array([weights[code] for code in codes])
            cov_matrix = S.loc[codes, codes].values
            annual_vol = np.sqrt(weights_array.T @ cov_matrix @ weights_array)
            annual_vol = min(annual_vol, 1.0)
            
            sharpe = (portfolio_return - risk_free_rate) / annual_vol if annual_vol > 0 else 0.0
            
            return float(portfolio_return), float(annual_vol), float(sharpe)
        
    except Exception as e:
        print(f"성과 연율화 계산 오류: {str(e)}")
        return 0.0, 0.1, 0.0

def safe_optimize_with_constraints(mu, S, selected_codes, target_return, risk_free_rate, price_data, objective="max_sharpe"):
    """제약조건 최적화"""
    if target_return is None:
        target_return = risk_free_rate if risk_free_rate is not None else 0.02
    if risk_free_rate is None:
        risk_free_rate = 0.02
    
    if hasattr(mu, 'index'):
        valid_codes = []
        for code in selected_codes:
            if code in mu.index and not (np.isnan(mu[code]) or np.isinf(mu[code])):
                valid_codes.append(code)
        
        if len(valid_codes) < 2:
            raise ValueError("유효한 데이터를 가진 자산이 2개 미만입니다.")
        
        mu_array = np.array([mu[code] for code in valid_codes])
        available_codes = valid_codes
    else:
        mu_array = mu
        available_codes = selected_codes
    
    n = len(available_codes)
    if n < 2:
        raise ValueError("최적화를 위해 최소 2개의 유효한 자산이 필요합니다.")
    
    if hasattr(S, 'index'):
        S_array = S.loc[available_codes, available_codes].values
    else:
        S_array = S
    
    try:
        np.linalg.cholesky(S_array)
    except np.linalg.LinAlgError:
        S_array += np.eye(n) * 1e-8
    
    x0 = np.array([1/n] * n)
    
    constraints = []
    constraints.append({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    constraints.append({'type': 'ineq', 'fun': lambda x: np.dot(x, mu_array) - target_return})
    
    bounds = [(0.05, 0.65) for _ in range(n)]
    
    if objective == "max_sharpe":
        def objective_func(x):
            portfolio_return = np.dot(x, mu_array)
            portfolio_vol = np.sqrt(np.dot(x, np.dot(S_array, x)))
            if portfolio_vol == 0 or np.isnan(portfolio_vol):
                return 1e6
            sharpe = (portfolio_return - risk_free_rate) / portfolio_vol
            return -sharpe
    elif objective == "min_vol":
        def objective_func(x):
            vol = np.sqrt(np.dot(x, np.dot(S_array, x)))
            return vol if not np.isnan(vol) else 1e6
    
    methods = ['SLSQP', 'trust-constr']
    result = None
    
    for method in methods:
        try:
            result = minimize(objective_func, x0, method=method, bounds=bounds, constraints=constraints)
            if result.success and not np.any(np.isnan(result.x)):
                break
        except Exception as e:
            continue
    
    if result is None or not result.success or np.any(np.isnan(result.x)):
        print("제약조건을 완화하여 재시도합니다...")
        adjusted_target_return = np.mean(mu_array)
        relaxed_constraints = []
        relaxed_constraints.append({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        relaxed_constraints.append({'type': 'ineq', 'fun': lambda x: np.dot(x, mu_array) - adjusted_target_return})
        
        try:
            result = minimize(objective_func, x0, method='SLSQP', bounds=bounds, constraints=relaxed_constraints)
        except Exception as e:
            pass
    
    if result is None or not result.success or np.any(np.isnan(result.x)):
        print("최적화 실패. 동일가중 포트폴리오를 반환합니다.")
        result_weights = np.array([1/n] * n)
        weights = dict(zip(available_codes, result_weights))
        
        portfolio_return, portfolio_vol, sharpe = safe_annualize_performance(
            price_data[available_codes], weights, risk_free_rate, mu
        )
        
        return weights, portfolio_return, portfolio_vol, sharpe
    
    weights = dict(zip(available_codes, result.x))
    
    portfolio_return, portfolio_vol, sharpe = safe_annualize_performance(
        price_data[available_codes], weights, risk_free_rate, mu
    )
    
    return weights, portfolio_return, portfolio_vol, sharpe

def get_optimized_portfolio(selected_codes, params, code_to_ticker_map):
    """
    CODE 기반으로 최적화 수행 후 ticker로 변환하여 반환
    """
    try:
        print("\n" + "="*50)
        print(" STEP 1: MongoDB에서 합성 지수 로드 ".center(50, "="))
        print("="*50)
        print(f"요청된 코드: {selected_codes}")

        if len(selected_codes) < 2:
            raise ValueError("최적화를 위해 2개 이상의 종목이 필요합니다.")
            
        all_prices = []
        for code in selected_codes:
            # MongoDB에서 합성 지수 로드
            price_series = get_synthetic_index_from_mongodb(code)
            
            if price_series is not None and not price_series.empty:
                price_series = clean_price_data(price_series)
                
                if not price_series.empty:
                    price_series = price_series.rename(columns={'Adj Close': code})
                    all_prices.append(price_series)
                else:
                    print(f"⚠️ 정리 후 {code}의 데이터가 비어있습니다.")
            else:
                print(f"⚠️ MongoDB에서 {code} 합성 지수를 찾을 수 없습니다.")

        if not all_prices:
            raise FileNotFoundError("유효한 가격 데이터를 가진 자산이 하나도 없습니다.")

        price_df_cleaned = pd.concat(all_prices, axis=1, join='outer', sort=True)
        price_df_cleaned = clean_price_data(price_df_cleaned)
        price_df_cleaned = price_df_cleaned.dropna()

        if price_df_cleaned.empty or price_df_cleaned.shape[0] < 10:
            raise ValueError(f"선택된 자산들의 공통된 거래 기간 데이터가 부족합니다.")
        
        available_codes = list(price_df_cleaned.columns)
        print(f"최종 분석 대상 코드: {available_codes}")
        print(f"공통 분석 기간: {price_df_cleaned.index.min().strftime('%Y-%m-%d')} ~ {price_df_cleaned.index.max().strftime('%Y-%m-%d')} ({len(price_df_cleaned)}일)")
        
        if len(available_codes) < 2:
            raise ValueError("최적화를 위해 유효한 데이터를 가진 2개 이상의 종목이 필요합니다.")
        
        print("\n" + "="*50)
        print(" STEP 2: Trimmed Mean 기대수익률 계산 ".center(50, "="))
        print("="*50)
        
        mu = calculate_trimmed_mean_returns(price_df_cleaned)
        S = risk_models.sample_cov(price_df_cleaned)
        
        print("\n개별 자산 성과:")
        for code in available_codes:
            code_return = mu.get(code, 0)
            code_volatility = np.sqrt(S.loc[code, code])
            print(f"▶ {code}")
            print(f"  - 기대수익률: {code_return:.2%}")
            print(f"  - 변동성: {code_volatility:.2%}")

        print("\n" + "="*50)
        print(" STEP 3: 포트폴리오 최적화 ".center(50, "="))
        print("="*50)
        
        mode = params.get("mode", "MVO")
        risk_free_rate = params.get("risk_free_rate") if params.get("risk_free_rate") is not None else 0.02
        target_return = params.get("target_return") if params.get("target_return") is not None else risk_free_rate

        print(f"최적화 모드: {mode}, 무위험수익률: {risk_free_rate:.2%}, 목표수익률: {target_return:.2%}")
        
        if mode == "MVO":
            mvo_objective = params.get("mvo_objective", "max_sharpe")
            weights, e_ret, ann_vol, sharpe = safe_optimize_with_constraints(
                mu, S, available_codes, target_return, risk_free_rate, price_df_cleaned, mvo_objective
            )
        else:
            weights, e_ret, ann_vol, sharpe = safe_optimize_with_constraints(
                mu, S, available_codes, target_return, risk_free_rate, price_df_cleaned, "max_sharpe"
            )

        cleaned_weights = {k: max(0, v) for k, v in weights.items()}
        total_weight = sum(cleaned_weights.values())
        if total_weight > 0:
            cleaned_weights = {k: v/total_weight for k, v in cleaned_weights.items()}
        else:
            cleaned_weights = {k: 1/len(available_codes) for k in available_codes}

        print("\n" + "="*50)
        print(" STEP 4: CODE를 TICKER로 변환 ".center(50, "="))
        print("="*50)
        
        # ✅ code 기반 weights를 ticker 기반으로 변환
        ticker_weights = {}
        for code, weight in cleaned_weights.items():
            ticker = code_to_ticker_map.get(code, f"{code}.KS")  # 매핑이 없으면 기본값
            ticker_weights[ticker] = weight
            print(f"  - {code} → {ticker}: {weight:.2%}")
        
        print("\n" + "="*50)
        print(" STEP 5: 최종 결과 ".center(50, "="))
        print("="*50)
        print("▶ 최적 비중 (ticker 기반):")
        for ticker, weight in ticker_weights.items():
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
        
        # ✅ ticker 기반 weights 반환
        return ticker_weights, performance

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise e

def ValueAtRisk(annual_return, annual_vol, risk_free_rate=0.02):
    """VaR 계산"""
    try:
        confidence_levels = [0.99, 0.95, 0.90]
        var_results = {}
        
        for confidence in confidence_levels:
            z_score = norm.ppf(1 - confidence)
            var_1year = annual_return + z_score * annual_vol
            
            confidence_pct = int((1 - confidence) * 100)
            var_results[f"var_{confidence_pct}pct"] = {
                "confidence_level": f"{confidence_pct}%",
                "var_1year": round(var_1year * 100, 2),
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
    """Shortfall Risk 계산"""
    try:
        shortfall_results = []
        
        for years in range(1, 21):
            expected_return_n_years = annual_return * years
            volatility_n_years = annual_vol * math.sqrt(years)
            
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