import pandas as pd
import os

DATA_DIR = "data"
PRICE_DATA_DIR = os.path.join(DATA_DIR, "fund_prices")

def run_backtest(weights: dict):
    """
    주어진 가중치로 포트폴리오의 백테스팅을 수행합니다.
    """
    tickers = list(weights.keys())
    if not tickers:
        return {"error": "백테스팅할 종목이 없습니다."}

    # 모든 가격 데이터를 읽어와 하나의 데이터프레임으로 합칩니다.
    all_prices = []
    for ticker in tickers:
        # CSV 파일로 변경
        file_path = os.path.join(PRICE_DATA_DIR, f"{ticker}.csv")
        if os.path.exists(file_path):
            try:
                # CSV 파일 읽기
                df = pd.read_csv(
                    file_path, 
                    skiprows=3, 
                    header=None,
                    names=['Date', 'Price', 'Close', 'High', 'Low', 'Open', 'Volume'],
                    index_col='Date',
                    parse_dates=True
                )
                
                # 인덱스 중복 제거
                df = df[~df.index.duplicated(keep='first')]
                
                # Adj Close 컬럼 생성
                df['Adj Close'] = df['Price']
                
                if 'Adj Close' in df.columns:
                    price_series = df[['Adj Close']].rename(columns={'Adj Close': ticker})
                    all_prices.append(price_series)
            except Exception as e:
                print(f"Error reading {ticker}: {e}")
                continue

    if not all_prices:
        return {"error": "백테스팅에 사용할 가격 데이터가 없습니다."}

    # concat 시 join='outer'와 sort=True 명시
    price_df = pd.concat(all_prices, axis=1, join='outer', sort=True)
    
    # NaN 값 처리
    price_df = price_df.ffill().bfill()

    # 백테스팅 기간 설정 (데이터가 있는 마지막 날짜로부터 1년)
    end_date = price_df.index.max()
    start_date = end_date - pd.DateOffset(years=1)
    
    # 데이터가 1년 미만일 경우, 가장 오래된 날짜부터 시작
    if price_df.index.min() > start_date:
        start_date = price_df.index.min()

    price_df = price_df.loc[start_date:end_date].dropna()
    
    if len(price_df) < 2:
        return {"error": "백테스팅 기간의 데이터가 부족합니다."}

    # 월별 수익률 계산
    monthly_prices = price_df.resample('ME').last()
    monthly_returns = monthly_prices.pct_change(fill_method=None).dropna()
    
    # 포트폴리오 월별 수익률 계산
    portfolio_monthly_returns = monthly_returns.dot(pd.Series(weights))

    # 월별 누적 수익률 계산
    cumulative_returns = (1 + portfolio_monthly_returns).cumprod()
    
    # 차트용 데이터 생성 ('YYYY-MM', 수익률)
    chart_data = [
        {"date": date.strftime('%Y-%m'), "return": (value - 1) * 100}
        for date, value in cumulative_returns.items()
    ]

    # 직전 1년 수익률
    last_year_return = (cumulative_returns.iloc[-1] - 1) * 100 if len(cumulative_returns) > 0 else 0

    # MDD (최대 낙폭) 계산
    if len(cumulative_returns) > 0:
        peak = cumulative_returns.cummax()
        drawdown = (cumulative_returns - peak) / peak
        mdd = drawdown.min() * 100
    else:
        mdd = 0

    return {
        "monthly_cumulative_returns": chart_data,
        "last_year_return": f"{last_year_return:.2f}%",
        "mdd": f"{mdd:.2f}%"
    }