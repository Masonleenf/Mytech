
import os
import sys
import pandas as pd
from pymongo import MongoClient
import config
from datetime import datetime

def upload_market_data():
    # 1. Parquet 파일 로드
    parquet_path = '/Volumes/X31/github/Fundplatter/dividen_model/gsheet/data/market_data.parquet'
    if not os.path.exists(parquet_path):
        print(f"❌ 파일을 찾을 수 없습니다: {parquet_path}")
        return

    print(f"📂 Parquet 파일 로드 중... {parquet_path}")
    try:
        df = pd.read_parquet(parquet_path)
    except Exception as e:
        print(f"❌ Parquet 로드 실패: {e}")
        return

    print(f"📊 데이터 로드 완료: {df.shape}")
    
    # 2. MongoDB 연결
    try:
        # config.py의 설정을 사용하거나 직접 연결
        mongo_uri = getattr(config, 'MONGO_URI', None)
        if not mongo_uri:
            # Fallback (db.py나 ecos_main_mongo.py에서 본 URI 참조)
            mongo_uri = "mongodb+srv://rator9521_db_user:qwe343434@cluster0.d126rkt.mongodb.net/"
            
        client = MongoClient(mongo_uri)
        db = client[getattr(config, 'ETF_DATABASE', 'etf_database')]
        collection_name = getattr(config, 'COLLECTION_DIVIDEND_ETF_PRICES', 'dividend_etf_prices')
        collection = db[collection_name]
        
        print(f"🔗 MongoDB 연결: {db.name}.{collection_name}")
        
    except Exception as e:
        print(f"❌ MongoDB 연결 실패: {e}")
        return

    # 3. 데이터 변환 및 업로드
    # MultiIndex Columns: (Ticker, Attribute)
    if isinstance(df.columns, pd.MultiIndex):
        tickers = df.columns.levels[0]
        total_tickers = len(tickers)
        print(f"🚀 {total_tickers}개 티커 데이터 업로드 시작...")
        
        count = 0
        for ticker in tickers:
            try:
                # 해당 티커의 Close 데이터 추출
                if 'Close' in df[ticker].columns:
                    ticker_df = df[ticker][['Close']].copy()
                    ticker_df = ticker_df.dropna() # NaN 제거
                    
                    if ticker_df.empty:
                        continue
                        
                    # 날짜 인덱스를 리셋하고 문자열로 변환
                    ticker_df = ticker_df.reset_index()
                    ticker_df.columns = ['Date', 'Close']
                    
                    # Date를 문자열 혹은 datetime 객체로 변환 
                    # (dividend_optimizer는 pd.to_datetime(df['Date'])를 쓰므로 유연하지만, 
                    #  일반적으로 DB에는 datetime이나 ISO string 저장)
                    # 여기서는 원본 포맷 유지 (Timestamp)하되, pymongo가 datetime으로 변환 지원
                    
                    price_list = ticker_df.to_dict('records')
                    
                    # 문서 생성
                    doc = {
                        'ticker': ticker,
                        'prices': price_list,
                        'last_updated': datetime.now()
                    }
                    
                    # Upsert (기존 데이터 있으면 업데이트)
                    collection.update_one(
                        {'ticker': ticker}, 
                        {'$set': doc}, 
                        upsert=True
                    )
                    
                    count += 1
                    if count % 100 == 0:
                        print(f"  ... {count}/{total_tickers} 완료")
                        
            except Exception as e:
                print(f"⚠️ {ticker} 처리 중 오류: {e}")
                continue
                
        print(f"✅ 업로드 완료! 총 {count}개 티커 저장됨.")
        
    else:
        print("❌ 예상치 못한 데이터 구조입니다 (MultiIndex 아님).")

if __name__ == "__main__":
    upload_market_data()
