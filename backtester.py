import pandas as pd
from pymongo import MongoClient

# MongoDB 설정
MONGO_URI = "mongodb+srv://rator9521_db_user:qwe343434@cluster0.d126rkt.mongodb.net/"
ETF_DATABASE = "etf_database"

client = MongoClient(MONGO_URI)
db = client[ETF_DATABASE]
fund_prices_collection = db['fund_prices']

def get_price_data_from_mongodb(ticker):
    """MongoDB에서 가격 데이터를 메모리로 직접 로드"""
    try:
        doc = fund_prices_collection.find_one({'ticker': ticker})
        
        if not doc or 'prices' not in doc:
            print(f"MongoDB에서 {ticker} 데이터를 찾을 수 없습니다.")
            return None
        
        df = pd.DataFrame(doc['prices'])
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.set_index('Date').sort_index()
        
        # 중복 제거
        df = df[~df.index.duplicated(keep='first')]
        
        # Adj Close 컬럼 생성
        df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
        df['Adj Close'] = df['Close']
        
        return df[['Adj Close']]
        
    except Exception as e:
        print(f"MongoDB에서 {ticker} 로드 오류: {e}")
        return None

def run_backtest(weights: dict):
    """
    주어진 가중치로 포트폴리오의 백테스팅을 수행합니다.
    MongoDB에서 직접 데이터를 읽어 메모리에서만 처리 (파일 저장 없음)
    """
    tickers = list(weights.keys())
    if not tickers:
        return {"error": "백테스팅할 종목이 없습니다."}

    print(f"백테스팅 대상: {tickers}")

    # MongoDB에서 가격 데이터 로드
    all_prices = []
    for ticker in tickers:
        price_series = get_price_data_from_mongodb(ticker)
        
        if price_series is not None and not price_series.empty:
            price_series = price_series.rename(columns={'Adj Close': ticker})
            all_prices.append(price_series)
        else:
            print(f"⚠️ {ticker} 데이터 로드 실패")
            continue

    if not all_prices:
        return {"error": "백테스팅에 사용할 가격 데이터가 없습니다."}

    # 데이터 병합
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

    print(f"백테스팅 기간: {start_date.strftime('%Y-%m-%d')} ~ {end_date.strftime('%Y-%m-%d')}")

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