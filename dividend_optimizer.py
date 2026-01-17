# dividend_optimizer.py
# 해외 배당 ETF 최적화 모듈 - MVSK 최적화 통합

import json
import os
import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from datetime import datetime
from db import db_manager

# MVSK 최적화 모듈 임포트
try:
    from mvsk_optimizer import MVSKOptimizer, MomentCalculator, print_optimization_report
    MVSK_AVAILABLE = True
except ImportError:
    MVSK_AVAILABLE = False
    print("⚠ MVSK optimizer not available, using simple scoring")


def get_dividend_etf_summary() -> List[Dict]:
    """MongoDB에서 배당 ETF 요약 정보 조회"""
    try:
        cursor = db_manager.dividend_etf_summary.find({}, {'_id': 0})
        return list(cursor)
    except Exception as e:
        print(f"❌ 배당 ETF 요약 조회 실패: {e}")
        return []


def get_market_data() -> Optional[pd.DataFrame]:
    """MongoDB에서 시장 데이터(가격) 조회 - parquet 대신 DB사용"""
    try:
        # dividen_model의 data/market_data.parquet 파일 경로
        parquet_path = "/Volumes/X31/github/Fundplatter/dividen_model/gsheet/data/market_data.parquet"
        if os.path.exists(parquet_path):
            df = pd.read_parquet(parquet_path)
            if isinstance(df.columns, pd.MultiIndex):
                if 'Adj Close' in df.columns.get_level_values(1):
                    return df.xs('Adj Close', axis=1, level=1)
                elif 'Close' in df.columns.get_level_values(1):
                    return df.xs('Close', axis=1, level=1)
            return df
        return None
    except Exception as e:
        print(f"❌ 마켓 데이터 로드 실패: {e}")
        return None


def _extract_dividend_months(schedule: List[Dict]) -> List[int]:
    """배당 스케줄에서 배당 월 추출"""
    months = set()
    for payment in schedule:
        date_str = payment.get('date', '')
        if date_str:
            try:
                month = int(date_str.split('-')[1])
                months.add(month)
            except (IndexError, ValueError):
                pass
    return sorted(list(months))


def _get_etf_data(etf: Dict) -> Dict:
    """ETF 데이터를 통일된 형식으로 변환"""
    metrics = etf.get('Key Metrics', {})
    schedule = etf.get('Dividend Schedule Summary', [])
    
    return {
        'ticker': etf.get('Ticker', ''),
        'name': etf.get('Name', ''),
        'dividend_yield': metrics.get('current_dividend_yield', 0) or 0,
        'annual_return': (metrics.get('cagr_price_5y', 0) or 0) * 100,
        'volatility': metrics.get('volatility', 0) or 0,
        'max_drawdown': metrics.get('max_drawdown', 0) or 0,
        'dividend_months': _extract_dividend_months(schedule),
        'dividend_frequency': len(schedule),
        'dividend_schedule': schedule[:6],  # 최근 6개 배당 기록
    }


def filter_etfs_by_frequency(etf_list: List[Dict], frequency: str) -> List[Dict]:
    """배당 주기로 ETF 필터링"""
    if frequency == 'any':
        return etf_list
    
    filtered = []
    for etf in etf_list:
        div_freq = etf.get('dividend_frequency', 0)
        if frequency == 'monthly' and div_freq >= 11:
            filtered.append(etf)
        elif frequency == 'quarterly' and div_freq >= 3:
            filtered.append(etf)
    
    return filtered


def calculate_portfolio_score(etf: Dict, alpha: float) -> float:
    """ETF 스코어 계산 (alpha 기반) - MVSK 미사용 시 대체"""
    div_yield = etf.get('dividend_yield', 0) or 0
    total_return = etf.get('annual_return', 0) or 0
    
    dividend_score = div_yield * 10
    growth_score = total_return + 10
    
    score = (1 - alpha) * dividend_score + alpha * growth_score
    return score


def optimize_dividend_portfolio_mvsk(
    alpha: float = 0.5,
    frequency: str = 'any',
    top_n: int = 8,
    initial_investment: int = 5000,
    universe_size: int = 50,  # MVSK는 O(N^4)이므로 50개로 제한
    max_weight: float = 0.20,
    lambdas: tuple = (1.0, 2.0, 2.0)
) -> Dict:
    """MVSK 기반 배당 포트폴리오 최적화
    
    Args:
        alpha: 성장/배당 균형 (0=배당, 1=성장/MVSK)
        frequency: 배당 주기 필터
        top_n: 최종 포트폴리오 종목 수
        initial_investment: 초기 투자금 (만원)
        universe_size: 유니버스 크기
        max_weight: 종목당 최대 비중
        lambdas: MVSK 람다 파라미터 (variance, skewness, kurtosis)
    """
    print(f"\n📊 [MVSK 최적화] alpha={alpha}, freq={frequency}, universe={universe_size}")
    
    # 1. ETF 데이터 로드
    raw_etf_list = get_dividend_etf_summary()
    if not raw_etf_list:
        return _get_mock_result(alpha, frequency, initial_investment)
    
    # 2. 마켓 데이터(가격) 로드
    price_df = get_market_data()
    if price_df is None:
        print("⚠ 마켓 데이터 없음, 단순 스코어 방식 사용")
        return optimize_dividend_portfolio_simple(alpha, frequency, top_n, initial_investment, universe_size)
    
    # 3. ETF 데이터 변환
    etf_list = [_get_etf_data(etf) for etf in raw_etf_list]
    dividend_etfs = [e for e in etf_list if e['dividend_yield'] > 0]
    filtered_etfs = filter_etfs_by_frequency(dividend_etfs, frequency)
    
    if len(filtered_etfs) < 5:
        return _get_mock_result(alpha, frequency, initial_investment)
    
    # 4. 유니버스 필터링 (스코어 기반 상위 N개)
    for etf in filtered_etfs:
        etf['_score'] = calculate_portfolio_score(etf, alpha)
    sorted_etfs = sorted(filtered_etfs, key=lambda x: x.get('_score', 0), reverse=True)
    universe_etfs = sorted_etfs[:universe_size]
    
    # 5. 가격 데이터와 매칭되는 티커만 사용
    valid_tickers = []
    ticker_to_etf = {}
    for etf in universe_etfs:
        ticker = etf['ticker']
        if ticker in price_df.columns:
            valid_tickers.append(ticker)
            ticker_to_etf[ticker] = etf
    
    print(f"  유효 티커: {len(valid_tickers)}개 (가격 데이터 매칭)")
    
    if len(valid_tickers) < 5:
        print("⚠ 가격 데이터 부족, 단순 스코어 방식 사용")
        return optimize_dividend_portfolio_simple(alpha, frequency, top_n, initial_investment, universe_size)
    
    # 6. 수익률 계산
    price_subset = price_df[valid_tickers].dropna()
    lookback_days = min(252, len(price_subset))
    price_subset = price_subset.tail(lookback_days)
    returns_df = price_subset.pct_change().dropna()
    returns_df = returns_df.dropna(axis=1, how='any')
    
    # 티커 리스트 업데이트
    valid_tickers = list(returns_df.columns)
    
    print(f"  수익률 데이터: {len(returns_df)}일, {len(valid_tickers)}개 자산")
    
    # 7. 배당 수익률 배열 준비
    dividend_yields = []
    for t in valid_tickers:
        etf = ticker_to_etf.get(t, {})
        dy = etf.get('dividend_yield', 0) or 0
        # 비정상 수익률 캡핑 (30% 최대)
        if dy > 30:
            dy = 30.0
        dividend_yields.append(dy / 100.0)
    
    dividend_yields = np.array(dividend_yields)
    
    # 8. MVSK 최적화 실행
    print(f"  MVSK 최적화 실행 중... (λ={lambdas})")
    
    optimizer = MVSKOptimizer(
        returns=returns_df,
        dividend_yields=dividend_yields,
        lambdas=lambdas,
        risk_free_rate=0.04
    )
    
    # 최적화 실행
    result = optimizer.optimize(
        alpha=alpha,
        max_weight=max_weight,
        verbose=False
    )
    
    if not result.success:
        print(f"⚠ 최적화 실패: {result.message}")
        return optimize_dividend_portfolio_simple(alpha, frequency, top_n, initial_investment, universe_size)
    
    # 9. 결과 변환 (상위 top_n개만)
    portfolio = []
    sorted_weights = sorted(result.weights.items(), key=lambda x: -x[1])
    
    for ticker, weight in sorted_weights[:top_n]:
        if weight < 0.01:
            continue
        etf = ticker_to_etf.get(ticker, {})
        portfolio.append({
            'ticker': ticker,
            'name': etf.get('name', ticker),
            'weight': round(weight, 4),
            'yield': round(etf.get('dividend_yield', 0), 2),
            'annual_return': round(etf.get('annual_return', 0), 2),
            'volatility': round(etf.get('volatility', 0) * 100, 2),
            'max_drawdown': round(etf.get('max_drawdown', 0) * 100, 2),
            'dividend_months': etf.get('dividend_months', []),
            'dividend_schedule': etf.get('dividend_schedule', []),
        })
    
    # 비중 재정규화
    total_weight = sum(p['weight'] for p in portfolio)
    if total_weight > 0:
        for p in portfolio:
            p['weight'] = round(p['weight'] / total_weight, 4)
    
    # 10. 월별 배당금 계산
    monthly_dividends = _calculate_monthly_dividends(portfolio, initial_investment)
    
    # 11. 포트폴리오 메트릭
    portfolio_yield = sum(p['yield'] * p['weight'] for p in portfolio)
    portfolio_return = sum(p.get('annual_return', 0) * p['weight'] for p in portfolio)
    
    print(f"  ✅ MVSK 최적화 완료: {len(portfolio)}개 ETF, 배당률={portfolio_yield:.2f}%")
    
    return {
        'portfolio': portfolio,
        'monthly_dividends': monthly_dividends,
        'portfolio_yield': round(portfolio_yield, 2),
        'portfolio_return': round(portfolio_return, 2),
        'alpha': alpha,
        'frequency': frequency,
        'metrics': {
            'expected_return': result.moments.mean,
            'volatility': result.moments.volatility,
            'skewness': result.moments.skewness,
            'kurtosis': result.moments.kurtosis,
            'sharpe_ratio': result.risk_metrics.sharpe_ratio,
        },
        '_mvsk': True
    }


def optimize_dividend_portfolio_simple(
    alpha: float = 0.5,
    frequency: str = 'any',
    top_n: int = 8,
    initial_investment: int = 5000,
    universe_size: int = 50
) -> Dict:
    """단순 스코어 기반 최적화 (MVSK 대체)"""
    raw_etf_list = get_dividend_etf_summary()
    
    if not raw_etf_list:
        return _get_mock_result(alpha, frequency, initial_investment)
    
    etf_list = [_get_etf_data(etf) for etf in raw_etf_list]
    dividend_etfs = [e for e in etf_list if e['dividend_yield'] > 0]
    
    if not dividend_etfs:
        return _get_mock_result(alpha, frequency, initial_investment)
    
    filtered_etfs = filter_etfs_by_frequency(dividend_etfs, frequency)
    
    if not filtered_etfs:
        return _get_mock_result(alpha, frequency, initial_investment)
    
    for etf in filtered_etfs:
        etf['_score'] = calculate_portfolio_score(etf, alpha)
    
    sorted_etfs = sorted(filtered_etfs, key=lambda x: x.get('_score', 0), reverse=True)
    universe_etfs = sorted_etfs[:universe_size]
    top_etfs = universe_etfs[:top_n]
    
    total_score = sum(etf.get('_score', 1) for etf in top_etfs)
    portfolio = []
    
    for etf in top_etfs:
        weight = etf.get('_score', 1) / total_score if total_score > 0 else 1 / len(top_etfs)
        portfolio.append({
            'ticker': etf.get('ticker'),
            'name': etf.get('name'),
            'weight': round(weight, 4),
            'yield': round(etf.get('dividend_yield', 0), 2),
            'annual_return': round(etf.get('annual_return', 0), 2),
            'volatility': round(etf.get('volatility', 0) * 100, 2),
            'max_drawdown': round(etf.get('max_drawdown', 0) * 100, 2),
            'dividend_months': etf.get('dividend_months', []),
            'dividend_schedule': etf.get('dividend_schedule', []),
        })
    
    monthly_dividends = _calculate_monthly_dividends(portfolio, initial_investment)
    portfolio_yield = sum(p['yield'] * p['weight'] for p in portfolio)
    portfolio_return = sum(p.get('annual_return', 0) * p['weight'] for p in portfolio)
    
    return {
        'portfolio': portfolio,
        'monthly_dividends': monthly_dividends,
        'portfolio_yield': round(portfolio_yield, 2),
        'portfolio_return': round(portfolio_return, 2),
        'alpha': alpha,
        'frequency': frequency,
        '_mvsk': False
    }


def optimize_dividend_portfolio(
    alpha: float = 0.5,
    frequency: str = 'any',
    top_n: int = 8,
    initial_investment: int = 5000,
    universe_size: int = 50  # MVSK O(N^4) 성능 위해 50개 제한
) -> Dict:
    """배당 포트폴리오 최적화 (MVSK 우선, 불가 시 단순 스코어)"""
    if MVSK_AVAILABLE:
        try:
            return optimize_dividend_portfolio_mvsk(
                alpha=alpha,
                frequency=frequency,
                top_n=top_n,
                initial_investment=initial_investment,
                universe_size=universe_size
            )
        except Exception as e:
            print(f"⚠ MVSK 최적화 오류: {e}, 단순 스코어 방식 사용")
    
    return optimize_dividend_portfolio_simple(
        alpha=alpha,
        frequency=frequency,
        top_n=top_n,
        initial_investment=initial_investment,
        universe_size=universe_size
    )


def _calculate_monthly_dividends(portfolio: List[Dict], initial_investment: int) -> List[float]:
    """포트폴리오 월별 배당금 계산 (만원 단위)"""
    monthly = [0.0] * 12
    
    for p in portfolio:
        invest_amt = initial_investment * p['weight']
        annual_div = invest_amt * (p['yield'] / 100)
        
        dividend_months = p.get('dividend_months', [])
        if dividend_months:
            per_payment = annual_div / len(dividend_months)
            for m in dividend_months:
                if 1 <= m <= 12:
                    monthly[m - 1] += per_payment
        elif p['yield'] > 0:
            per_month = annual_div / 12
            for i in range(12):
                monthly[i] += per_month
    
    return [round(m, 1) for m in monthly]


def simulate_30_year_growth(
    initial_investment: int,
    monthly_savings: int,
    portfolio_return: float = 0.08,
    dividend_yield: float = 0.04
) -> Dict:
    """30년 자산 성장 시뮬레이션 (만원 단위)"""
    total_asset = float(initial_investment)
    total_principal = float(initial_investment)
    
    yearly_data = []
    
    for year in range(31):
        yearly_data.append({
            'year': year,
            'total_asset': int(total_asset),
            'principal': int(total_principal)
        })
        
        total_asset = total_asset * (1 + portfolio_return) + monthly_savings * 12
        total_principal += monthly_savings * 12
    
    final_asset = yearly_data[-1]['total_asset']
    monthly_dividend = final_asset * dividend_yield / 12
    
    return {
        'total_asset_30y': int(final_asset),
        'total_principal': int(total_principal),
        'monthly_dividend': int(monthly_dividend),
        'yearly_data': yearly_data
    }


def _get_mock_result(alpha: float, frequency: str, initial_investment: int = 5000) -> Dict:
    """데이터가 없을 때 반환할 목업 결과"""
    mock_portfolio = [
        {'ticker': 'SCHD', 'name': 'Schwab US Dividend Equity ETF', 'weight': 0.20, 'yield': 3.5, 'annual_return': 8.2, 'dividend_months': [3, 6, 9, 12]},
        {'ticker': 'JEPI', 'name': 'JPMorgan Equity Premium Income', 'weight': 0.18, 'yield': 7.2, 'annual_return': 6.5, 'dividend_months': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]},
        {'ticker': 'DIVO', 'name': 'Amplify CWP Enhanced Dividend', 'weight': 0.16, 'yield': 4.8, 'annual_return': 7.8, 'dividend_months': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]},
        {'ticker': 'VYM', 'name': 'Vanguard High Dividend Yield', 'weight': 0.14, 'yield': 3.1, 'annual_return': 9.1, 'dividend_months': [3, 6, 9, 12]},
        {'ticker': 'DGRO', 'name': 'iShares Core Dividend Growth', 'weight': 0.12, 'yield': 2.4, 'annual_return': 10.5, 'dividend_months': [3, 6, 9, 12]},
        {'ticker': 'HDV', 'name': 'iShares Core High Dividend', 'weight': 0.10, 'yield': 3.8, 'annual_return': 7.2, 'dividend_months': [3, 6, 9, 12]},
        {'ticker': 'SPYD', 'name': 'SPDR Portfolio S&P 500 High Dividend', 'weight': 0.05, 'yield': 4.5, 'annual_return': 6.8, 'dividend_months': [3, 6, 9, 12]},
        {'ticker': 'VIG', 'name': 'Vanguard Dividend Appreciation', 'weight': 0.05, 'yield': 1.8, 'annual_return': 11.2, 'dividend_months': [3, 6, 9, 12]},
    ]
    
    monthly_dividends = _calculate_monthly_dividends(mock_portfolio, initial_investment)
    
    return {
        'portfolio': mock_portfolio,
        'monthly_dividends': monthly_dividends,
        'portfolio_yield': round(sum(p['yield'] * p['weight'] for p in mock_portfolio), 2),
        'portfolio_return': round(sum(p['annual_return'] * p['weight'] for p in mock_portfolio), 2),
        'alpha': alpha,
        'frequency': frequency,
        '_mock': True
    }
