"""
beg_optimize.py
Beginner 모드 전용 포트폴리오 최적화 - 리팩토링 버전
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm
from pypfopt import risk_models

import config
from db import db_manager
from optimizer import clean_price_data, calculate_trimmed_mean_returns, get_synthetic_index_from_mongodb

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


def calculate_shortfall_probability(weights, mu, S):
    """1년 투자 시 원금손실 확률 계산 (Shortfall Risk)"""
    portfolio_return = np.dot(weights, mu)
    portfolio_variance = np.dot(weights, np.dot(S, weights))
    portfolio_std = np.sqrt(portfolio_variance)
    
    return norm.cdf(0, loc=portfolio_return, scale=portfolio_std)


def optimize_with_shortfall_constraint(mu, S, codes, initial_threshold, step, max_threshold=0.5):
    """Shortfall 제약조건을 만족하는 최적화 수행"""
    n = len(codes)
    threshold = initial_threshold
    
    # 랜덤 bounds 설정
    min_weight = np.random.uniform(0.03, 0.05)
    max_weight = np.random.uniform(0.60, 0.65)
    bounds = [(min_weight, max_weight) for _ in range(n)]
    
    # 제약조건: 가중치 합 = 1
    constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}]
    
    # 목적함수: Sharpe ratio 최대화
    risk_free_rate = config.DEFAULT_RISK_FREE_RATE
    
    def negative_sharpe(w):
        port_return = np.dot(w, mu)
        port_variance = np.dot(w, np.dot(S, w))
        port_std = np.sqrt(port_variance)
        sharpe = (port_return - risk_free_rate) / port_std if port_std > 0 else 0
        return -sharpe
    
    # Shortfall 제약 완화하면서 반복
    while threshold <= max_threshold:
        try:
            def shortfall_constraint(w):
                shortfall_prob = calculate_shortfall_probability(w, mu, S)
                return threshold - shortfall_prob
            
            current_constraints = constraints + [{'type': 'ineq', 'fun': shortfall_constraint}]
            x0 = np.array([1.0 / n] * n)
            
            result = minimize(
                negative_sharpe, x0, method='SLSQP',
                bounds=bounds, constraints=current_constraints,
                options={'maxiter': 1000, 'ftol': 1e-9}
            )
            
            if result.success:
                actual_shortfall = calculate_shortfall_probability(result.x, mu, S)
                if actual_shortfall <= threshold + 0.001:
                    print(f"✅ 최적화 성공 - Shortfall 제약: {threshold*100:.1f}%")
                    return result.x, threshold, True
            
        except Exception as e:
            print(f"⚠️ Threshold {threshold*100:.1f}%에서 오류: {e}")
        
        print(f"⚠️ Threshold {threshold*100:.1f}%에서 해 없음 - 제약 완화 중...")
        threshold += step
    
    print(f"❌ 최대 Shortfall {max_threshold*100:.0f}%까지 시도했으나 해를 찾지 못함")
    return None, max_threshold, False


def get_beginner_portfolio(style_index, risk_index):
    """
    Beginner 모드 포트폴리오 최적화 메인 함수
    
    Parameters:
        style_index: 투자 스타일 인덱스 (0~9)
        risk_index: 위험 감수 수준 인덱스 (0~2)
    
    Returns:
        selected_tickers, weights, performance
    """
    try:
        print("\n" + "="*60)
        print(" Beginner 모드 최적화 시작 ".center(60, "="))
        print(f"스타일 인덱스: {style_index}, 위험 수준 인덱스: {risk_index}")
        
        # 1. 스타일/위험 설정 검증
        style_mapping = BEGINNER_STYLE_MAPPING.get(style_index)
        risk_config = RISK_LEVEL_CONFIG.get(risk_index)
        
        if not style_mapping:
            raise ValueError(f"유효하지 않은 style_index: {style_index}")
        if not risk_config:
            raise ValueError(f"유효하지 않은 risk_index: {risk_index}")
        
        saa_list = style_mapping['saa']
        taa_list = style_mapping['taa']
        
        # 2. SAA/TAA 조합으로 CODE 찾기
        etf_data = list(db_manager.etf_master.find({}, {'_id': 0}))
        if not etf_data:
            raise FileNotFoundError("ETF 마스터 데이터가 없습니다.")
        
        etf_df = pd.DataFrame(etf_data)
        
        selected_codes = []
        code_to_ticker_map = {}
        
        for saa, taa in zip(saa_list, taa_list):
            matched_etf = etf_df[(etf_df['saa_class'] == saa) & (etf_df['taa_class'] == taa)]
            
            if not matched_etf.empty:
                code = matched_etf['code'].iloc[0]
                ticker = matched_etf['ticker'].iloc[0]
                selected_codes.append(code)
                code_to_ticker_map[code] = ticker
                print(f"✅ [{saa} - {taa}] → {code}")
        
        if len(selected_codes) < 2:
            raise ValueError("최소 2개 이상의 유효한 자산이 필요합니다.")
        
        # 중복 제거
        selected_codes = list(dict.fromkeys(selected_codes))
        
        # 3. 가격 데이터 로드
        all_prices = []
        for code in selected_codes:
            price_series = get_synthetic_index_from_mongodb(code)
            if price_series is not None and not price_series.empty:
                price_series = clean_price_data(price_series)
                if not price_series.empty:
                    price_series = price_series.rename(columns={'Adj Close': code})
                    all_prices.append(price_series)
        
        if len(all_prices) < 2:
            raise ValueError("최소 2개 이상의 유효한 자산 데이터가 필요합니다.")
        
        # 4. 데이터 병합 및 정리
        price_df = pd.concat(all_prices, axis=1, join='outer', sort=True)
        price_df = clean_price_data(price_df).dropna()
        
        if price_df.empty or price_df.shape[0] < 10:
            raise ValueError("공통 거래 기간 데이터가 부족합니다.")
        
        available_codes = list(price_df.columns)
        
        # 5. 기대수익률 및 공분산 계산
        mu = calculate_trimmed_mean_returns(price_df)
        S = risk_models.sample_cov(price_df)
        
        # 6. 최적화 실행
        weights_array, final_threshold, success = optimize_with_shortfall_constraint(
            mu, S, available_codes, risk_config['threshold'], risk_config['step']
        )
        
        if not success:
            raise ValueError("최적화 실패: Shortfall 제약을 만족하는 해를 찾지 못했습니다.")
        
        # 7. CODE를 TICKER로 변환
        code_weights = dict(zip(available_codes, weights_array))
        ticker_weights = {code_to_ticker_map.get(c, f"{c}.KS"): w for c, w in code_weights.items()}
        
        # 8. 성과 지표 계산
        portfolio_return = np.dot(weights_array, mu)
        portfolio_std = np.sqrt(np.dot(weights_array, np.dot(S, weights_array)))
        sharpe_ratio = (portfolio_return - config.DEFAULT_RISK_FREE_RATE) / portfolio_std if portfolio_std > 0 else 0
        final_shortfall = calculate_shortfall_probability(weights_array, mu, S)
        
        performance = {
            'expected_annual_return': float(portfolio_return),
            'annual_volatility': float(portfolio_std),
            'sharpe_ratio': float(sharpe_ratio),
            'shortfall_probability': float(final_shortfall),
            'final_shortfall_threshold': float(final_threshold)
        }
        
        print(f"\n기대수익률: {portfolio_return*100:.2f}%, 변동성: {portfolio_std*100:.2f}%")
        print(f"샤프비율: {sharpe_ratio:.3f}, 원금손실확률: {final_shortfall*100:.2f}%")
        
        return list(ticker_weights.keys()), ticker_weights, performance
        
    except Exception as e:
        print(f"❌ Beginner 최적화 오류: {str(e)}")
        raise