# beg_optimize.py
# Beginner 모드 전용 포트폴리오 최적화 - 완전한 독립 모듈

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm
from pymongo import MongoClient
from pypfopt import risk_models

# MongoDB 연결
MONGO_URI = "mongodb+srv://rator9521_db_user:qwe343434@cluster0.d126rkt.mongodb.net/"
client = MongoClient(MONGO_URI)
db = client["etf_database"]
synthetic_indices_collection = db['synthetic_indices']
etf_master_collection = db['etf_master']

# SAA/TAA 매핑 데이터 (sata.xlsx 기반)
BEGINNER_STYLE_MAPPING = {
    0: {  # 초안정형
        'saa': ['단기자금', '국내채권', '국내채권'],
        'taa': ['단기자금', '단기국채', '종합']
    },
    1: {  # 안정추구형
        'saa': ['단기자금','국내채권', '해외주식', '해외채권'],
        'taa': ['단기자금', '종합', '지수/MSCIW', '종합']
    },
    2: {  # 배당주 선호형
        'saa': ['국내주식', '해외주식', '국내주식', '국내주식', '대체투자'],
        'taa': ['전략/배당', '글로벌/전략/배당', '전략/퀄리티', '전략/라지캡', '부동산']
    },
    3: {  # 채권 이자 선호형
        'saa': ['국내채권', '국내채권', '국내채권', '해외채권', '해외채권', '국내채권'],
        'taa': ['특수채', '회사채', '장기국채', '종합', '하이일드', 'MBS']
    },
    4: {  # 글로벌 분산형
        'saa': ['국내주식', '해외주식', '해외주식', '국내채권', '해외채권'],
        'taa': ['지수/코스피', '지수/MSCIW', '지수/미국', '종합', '종합']
    },
    5: {  # 올웨더 추구형
        'saa': ['국내주식', '해외주식', '해외주식', '국내채권', '해외채권', '대체투자', '대체투자', '대체투자', '대체투자'],
        'taa': ['지수/코스피', '지수/MSCIW', '지수/미국', '종합', '종합', '귀금속', '부동산', 'SOC', '에너지']
    },
    6: {  # 미국 우량주 집중형
        'saa': ['해외주식', '해외주식', '해외주식', '해외주식', '해외주식'],
        'taa': ['지수/미국', '글로벌/테마/AI반도체', '글로벌/테마/AI소프트', '글로벌/테마/M7', '글로벌/전략/퀄리티']
    },
    7: {  # IT/기술주 성장형
        'saa': ['국내주식', '국내주식', '해외주식', '해외주식', '해외주식', '해외주식'],
        'taa': ['테마/AI', '테마/반도체', '글로벌/테마/AI', '중국/테마/AI', '글로벌/테마/바이오', '글로벌/테마/전기차']
    },
    8: {  # 신흥국 개척형
        'saa': ['해외주식', '해외주식', '해외주식', '해외주식', '해외주식', '해외채권', '국내채권'],
        'taa': ['지수/MSCIW', '지수/유럽', '지수/일본', '지수/중국', '지수/아시아', '종합', '종합']
    },
    9: {  # ESG/친환경 투자형
        'saa': ['국내주식', '해외주식', '해외주식', '해외주식', '해외주식', '대체투자', '국내채권'],
        'taa': ['테마/ESG', '중국/테마/ESG', '글로벌/테마/전기차', '중국/테마/전기차', '글로벌/테마/SMR', '에너지', '종합']
    }
}

# Risk level별 설정
RISK_LEVEL_CONFIG = {
    0: {'threshold': 0.03, 'step': 0.01},  # 원금은 소중해요
    1: {'threshold': 0.08, 'step': 0.02},  # 잠깐의 추위는 OK
    2: {'threshold': 0.15, 'step': 0.03}   # 위기는 곧 기회!
}

def get_synthetic_index_from_mongodb(code):
    """MongoDB에서 합성 지수 데이터 로드"""
    try:
        doc = synthetic_indices_collection.find_one({'code': code})
        if not doc or 'data' not in doc:
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

def clean_price_data(price_df):
    """가격 데이터 정리"""
    if price_df.empty:
        return price_df
    
    price_df = price_df.replace([np.inf, -np.inf], np.nan)
    price_df = price_df.dropna()
    
    for col in price_df.columns:
        if price_df[col].nunique() < 2:
            price_df = price_df.drop(columns=[col])
    
    return price_df

def calculate_trimmed_mean_returns(price_data, trim_percent=0.1):
    """Trimmed Mean 수익률 계산"""
    try:
        returns = price_data.pct_change().dropna()
        
        def trimmed_mean(series):
            sorted_data = np.sort(series.dropna())
            n = len(sorted_data)
            trim_count = int(n * trim_percent)
            if trim_count > 0:
                trimmed = sorted_data[trim_count:-trim_count]
            else:
                trimmed = sorted_data
            return np.mean(trimmed) if len(trimmed) > 0 else 0.0
        
        mu = returns.apply(trimmed_mean)
        mu = mu * 252  # 연율화
        
        return mu
    except Exception as e:
        print(f"Trimmed Mean 계산 오류: {str(e)}")
        return pd.Series(index=price_data.columns, data=0.02)

def calculate_shortfall_probability(weights, mu, S):
    """
    1년 투자 시 원금손실 확률 계산 (Shortfall Risk)
    
    Parameters:
    - weights: 포트폴리오 가중치
    - mu: 연간 기대수익률
    - S: 연간 공분산 행렬
    
    Returns:
    - shortfall_prob: P(수익률 < 0)
    """
    portfolio_return = np.dot(weights, mu)
    portfolio_variance = np.dot(weights, np.dot(S, weights))
    portfolio_std = np.sqrt(portfolio_variance)
    
    # 정규분포 가정: P(R < 0)
    shortfall_prob = norm.cdf(0, loc=portfolio_return, scale=portfolio_std)
    
    return shortfall_prob

def optimize_with_shortfall_constraint(mu, S, codes, initial_threshold, step, max_threshold=0.5):
    """
    Shortfall 제약조건을 만족하는 최적화
    해가 없으면 제약을 완화하면서 반복
    
    Parameters:
    - mu: 기대수익률 벡터
    - S: 공분산 행렬
    - codes: 자산 코드 리스트
    - initial_threshold: 초기 shortfall 제약값 (예: 0.03 = 3%)
    - step: 제약 완화 단계 (예: 0.01 = 1%p)
    - max_threshold: 최대 허용 shortfall (0.5 = 50%)
    
    Returns:
    - weights: 최적 가중치
    - final_threshold: 실제 사용된 제약값
    - success: 성공 여부
    """
    n = len(codes)
    threshold = initial_threshold
    
    # 랜덤 bounds 설정 (3~5% min, 60~65% max)
    min_weight = np.random.uniform(0.03, 0.05)
    max_weight = np.random.uniform(0.60, 0.65)
    bounds = [(min_weight, max_weight) for _ in range(n)]
    
    # 제약조건: 가중치 합 = 1
    constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}]
    
    # 목적함수: Sharpe ratio 최대화
    risk_free_rate = 0.02
    
    def negative_sharpe(w):
        port_return = np.dot(w, mu)
        port_variance = np.dot(w, np.dot(S, w))
        port_std = np.sqrt(port_variance)
        sharpe = (port_return - risk_free_rate) / port_std if port_std > 0 else 0
        return -sharpe
    
    # Shortfall 제약 완화하면서 반복
    while threshold <= max_threshold:
        try:
            # Shortfall 제약 추가
            def shortfall_constraint(w):
                shortfall_prob = calculate_shortfall_probability(w, mu, S)
                return threshold - shortfall_prob  # >= 0이어야 함
            
            current_constraints = constraints + [
                {'type': 'ineq', 'fun': shortfall_constraint}
            ]
            
            # 초기값: 동일가중
            x0 = np.array([1.0 / n] * n)
            
            # 최적화 실행
            result = minimize(
                negative_sharpe,
                x0,
                method='SLSQP',
                bounds=bounds,
                constraints=current_constraints,
                options={'maxiter': 1000, 'ftol': 1e-9}
            )
            
            if result.success:
                # 추가 검증: 실제 shortfall이 threshold 이하인지
                actual_shortfall = calculate_shortfall_probability(result.x, mu, S)
                if actual_shortfall <= threshold + 0.001:  # 약간의 여유
                    print(f"✅ 최적화 성공 - Shortfall 제약: {threshold*100:.1f}%")
                    return result.x, threshold, True
            
        except Exception as e:
            print(f"⚠️ Threshold {threshold*100:.1f}%에서 오류: {e}")
        
        # 제약 완화
        print(f"⚠️ Threshold {threshold*100:.1f}%에서 해 없음 - 제약 완화 중...")
        threshold += step
    
    # 최대치까지 완화해도 해가 없으면 실패
    print(f"❌ 최대 Shortfall {max_threshold*100:.0f}%까지 시도했으나 해를 찾지 못함")
    return None, max_threshold, False

def get_beginner_portfolio(style_index, risk_index):
    """
    Beginner 모드 포트폴리오 최적화 메인 함수
    main.py에서 호출하는 유일한 진입점
    
    Parameters:
    - style_index: 투자 스타일 인덱스 (0~9)
    - risk_index: 위험 감수 수준 인덱스 (0~2)
    
    Returns:
    - selected_tickers: 선택된 ETF ticker 리스트
    - weights: {ticker: weight} 딕셔너리
    - performance: 성과 지표
    """
    try:
        print("\n" + "="*60)
        print(" Beginner 모드 최적화 시작 ".center(60, "="))
        print("="*60)
        print(f"스타일 인덱스: {style_index}")
        print(f"위험 수준 인덱스: {risk_index}")
        
        # 1. 스타일 매핑 가져오기
        style_mapping = BEGINNER_STYLE_MAPPING.get(style_index)
        if not style_mapping:
            raise ValueError(f"유효하지 않은 style_index: {style_index}")
        
        risk_config = RISK_LEVEL_CONFIG.get(risk_index)
        if not risk_config:
            raise ValueError(f"유효하지 않은 risk_index: {risk_index}")
        
        saa_list = style_mapping['saa']
        taa_list = style_mapping['taa']
        
        print(f"SAA: {saa_list}")
        print(f"TAA: {taa_list}")
        print(f"Shortfall 제약: {risk_config['threshold']*100}% (단계: {risk_config['step']*100}%p)")
        
        # 2. MongoDB에서 ETF 마스터 로드
        etf_data = list(etf_master_collection.find({}, {'_id': 0}))
        if not etf_data:
            raise FileNotFoundError("ETF 마스터 데이터가 없습니다.")
        
        etf_df = pd.DataFrame(etf_data)
        
        # 3. SAA/TAA 조합으로 CODE 찾기
        selected_codes = []
        code_to_ticker_map = {}
        
        for saa, taa in zip(saa_list, taa_list):
            matched_etf = etf_df[
                (etf_df['saa_class'] == saa) & 
                (etf_df['taa_class'] == taa)
            ]
            
            if not matched_etf.empty:
                code = matched_etf['code'].iloc[0]
                ticker = matched_etf['ticker'].iloc[0]
                selected_codes.append(code)
                code_to_ticker_map[code] = ticker
                print(f"✅ [{saa} - {taa}] → code: {code}, ticker: {ticker}")
            else:
                print(f"⚠️ [{saa} - {taa}] 매칭 실패")
        
        if len(selected_codes) < 2:
            raise ValueError("최소 2개 이상의 유효한 자산이 필요합니다.")
        
        # 중복 제거
        selected_codes = list(dict.fromkeys(selected_codes))
        print(f"\n최종 선택된 CODE: {selected_codes}")
        
        # 4. 가격 데이터 로드
        all_prices = []
        for code in selected_codes:
            price_series = get_synthetic_index_from_mongodb(code)
            
            if price_series is not None and not price_series.empty:
                price_series = clean_price_data(price_series)
                
                if not price_series.empty:
                    price_series = price_series.rename(columns={'Adj Close': code})
                    all_prices.append(price_series)
                    print(f"✅ {code} 데이터 로드 성공")
                else:
                    print(f"⚠️ {code} 데이터 정리 후 비어있음")
            else:
                print(f"⚠️ {code} 데이터 없음")
        
        if len(all_prices) < 2:
            raise ValueError("최소 2개 이상의 유효한 자산 데이터가 필요합니다.")
        
        # 5. 데이터 병합
        price_df = pd.concat(all_prices, axis=1, join='outer', sort=True)
        price_df = clean_price_data(price_df)
        price_df = price_df.dropna()
        
        if price_df.empty or price_df.shape[0] < 10:
            raise ValueError("공통 거래 기간 데이터가 부족합니다.")
        
        available_codes = list(price_df.columns)
        print(f"\n최종 분석 대상: {available_codes}")
        print(f"분석 기간: {price_df.index.min().strftime('%Y-%m-%d')} ~ {price_df.index.max().strftime('%Y-%m-%d')} ({len(price_df)}일)")
        
        # 6. 기대수익률 및 공분산 계산
        mu = calculate_trimmed_mean_returns(price_df)
        S = risk_models.sample_cov(price_df)
        
        print(f"\n기대수익률 (연간):")
        for code in available_codes:
            print(f"  {code}: {mu[code]*100:.2f}%")
        
        # 7. 최적화 실행
        print("\n" + "="*60)
        print(" 최적화 실행 ".center(60, "="))
        print("="*60)
        
        weights_array, final_threshold, success = optimize_with_shortfall_constraint(
            mu, S, available_codes, 
            risk_config['threshold'], 
            risk_config['step']
        )
        
        if not success:
            raise ValueError(f"최적화 실패: Shortfall 제약을 만족하는 해를 찾지 못했습니다. (최대 50%까지 시도)")
        
        # 8. CODE를 TICKER로 변환
        code_weights = dict(zip(available_codes, weights_array))
        ticker_weights = {}
        
        for code, weight in code_weights.items():
            ticker = code_to_ticker_map.get(code, f"{code}.KS")
            ticker_weights[ticker] = weight
            print(f"  {code} → {ticker}: {weight*100:.2f}%")
        
        # 9. 성과 지표 계산
        portfolio_return = np.dot(weights_array, mu)
        portfolio_variance = np.dot(weights_array, np.dot(S, weights_array))
        portfolio_std = np.sqrt(portfolio_variance)
        sharpe_ratio = (portfolio_return - 0.02) / portfolio_std if portfolio_std > 0 else 0
        final_shortfall = calculate_shortfall_probability(weights_array, mu, S)
        
        performance = {
            'expected_annual_return': float(portfolio_return),
            'annual_volatility': float(portfolio_std),
            'sharpe_ratio': float(sharpe_ratio),
            'shortfall_probability': float(final_shortfall),
            'final_shortfall_threshold': float(final_threshold)
        }
        
        print("\n" + "="*60)
        print(" 최적화 결과 ".center(60, "="))
        print("="*60)
        print(f"기대수익률: {portfolio_return*100:.2f}%")
        print(f"변동성: {portfolio_std*100:.2f}%")
        print(f"샤프비율: {sharpe_ratio:.3f}")
        print(f"원금손실확률: {final_shortfall*100:.2f}%")
        print(f"사용된 제약: {final_threshold*100:.1f}%")
        
        selected_tickers = list(ticker_weights.keys())
        
        return selected_tickers, ticker_weights, performance
        
    except Exception as e:
        print(f"❌ Beginner 최적화 오류: {str(e)}")
        raise