# market_ranking.py
# 자산군/ETF 등락률 순위 계산

import pandas as pd
from pymongo import MongoClient
from datetime import datetime, timedelta

MONGO_URI = "mongodb+srv://rator9521_db_user:qwe343434@cluster0.d126rkt.mongodb.net/"
ETF_DATABASE = "etf_database"

client = MongoClient(MONGO_URI)
db = client[ETF_DATABASE]

synthetic_indices_collection = db['synthetic_indices']
fund_prices_collection = db['fund_prices']
etf_master_collection = db['etf_master']

def calculate_return_for_period(prices_df, days):
    """특정 기간의 수익률 계산"""
    try:
        if len(prices_df) < days:
            return None
        
        current_price = prices_df.iloc[-1]
        past_price = prices_df.iloc[-days]
        
        return_pct = ((current_price - past_price) / past_price) * 100
        return round(return_pct, 2)
    except:
        return None

def get_timeframe_days(timeframe):
    """타임프레임을 일수로 변환"""
    mapping = {
        '당일': 1,
        '1주일': 5,
        '1달': 21,
        '3개월': 63,
        '6개월': 126,
        '1년': 252,
        '3년': 756
    }
    return mapping.get(timeframe, 21)

def get_asset_class_rankings(timeframe='1달'):
    """
    자산군(SAA/TAA 합성지수) 등락률 순위 반환
    """
    try:
        days = get_timeframe_days(timeframe)
        rankings = []
        
        # 모든 합성 지수 조회
        all_indices = synthetic_indices_collection.find({})
        
        for index_doc in all_indices:
            code = index_doc.get('code')
            data = index_doc.get('data', [])
            
            if not data or len(data) < days:
                continue
            
            # DataFrame으로 변환
            df = pd.DataFrame(data)
            df['Date'] = pd.to_datetime(df['Date'])
            df = df.sort_values('Date')
            df['close'] = pd.to_numeric(df['close'], errors='coerce')
            df = df.dropna(subset=['close'])
            
            if len(df) < days:
                continue
            
            # 수익률 계산
            return_pct = calculate_return_for_period(df['close'], days)
            
            if return_pct is None:
                continue
            
            # etf_master에서 SAA/TAA 정보 가져오기
            master_info = etf_master_collection.find_one({'code': code})
            
            if master_info:
                saa = master_info.get('saa_class', '미분류')
                taa = master_info.get('taa_class', '미분류')
                
                rankings.append({
                    'code': code,
                    'saa': saa,
                    'taa': taa,
                    'name': f"{saa} / {taa}",
                    'return': return_pct
                })
        
        # 수익률 기준 정렬
        rankings_sorted = sorted(rankings, key=lambda x: x['return'], reverse=True)
        
        # 상위 10개, 하위 10개 추출
        top_10 = rankings_sorted[:10]
        bottom_10 = rankings_sorted[-10:]
        
        # 순위 매기기
        for idx, item in enumerate(top_10, 1):
            item['rank'] = idx
        
        for idx, item in enumerate(bottom_10, 1):
            item['rank'] = idx
        
        return {
            'status': 'success',
            'timeframe': timeframe,
            'top': top_10,
            'bottom': bottom_10
        }
        
    except Exception as e:
        print(f"자산군 순위 계산 오류: {e}")
        import traceback
        traceback.print_exc()
        return {
            'status': 'error',
            'message': str(e)
        }

def get_etf_rankings(timeframe='1달'):
    """
    ETF 등락률 순위 반환
    """
    try:
        days = get_timeframe_days(timeframe)
        rankings = []
        
        # 모든 ETF 조회
        all_etfs = etf_master_collection.find({})
        
        for etf in all_etfs:
            ticker = etf.get('ticker')
            name = etf.get('한글종목약명', etf.get('한글종목명', ticker))
            
            # fund_prices에서 가격 데이터 가져오기
            price_doc = fund_prices_collection.find_one({'ticker': ticker})
            
            if not price_doc or 'prices' not in price_doc:
                continue
            
            prices = price_doc['prices']
            if len(prices) < days:
                continue
            
            # DataFrame으로 변환
            df = pd.DataFrame(prices)
            df['Date'] = pd.to_datetime(df['Date'])
            df = df.sort_values('Date')
            df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
            df = df.dropna(subset=['Close'])
            
            if len(df) < days:
                continue
            
            # 수익률 계산
            return_pct = calculate_return_for_period(df['Close'], days)
            
            if return_pct is None:
                continue
            
            # 현재가
            current_price = int(df['Close'].iloc[-1])
            
            rankings.append({
                'ticker': ticker,
                'name': name,
                'price': f"{current_price:,}원",
                'return': return_pct
            })
        
        # 수익률 기준 정렬
        rankings_sorted = sorted(rankings, key=lambda x: x['return'], reverse=True)
        
        # 상위 10개, 하위 10개 추출
        top_10 = rankings_sorted[:10]
        bottom_10 = rankings_sorted[-10:]
        
        # 순위 매기기
        for idx, item in enumerate(top_10, 1):
            item['rank'] = idx
        
        for idx, item in enumerate(bottom_10, 1):
            item['rank'] = idx
        
        return {
            'status': 'success',
            'timeframe': timeframe,
            'top': top_10,
            'bottom': bottom_10
        }
        
    except Exception as e:
        print(f"ETF 순위 계산 오류: {e}")
        import traceback
        traceback.print_exc()
        return {
            'status': 'error',
            'message': str(e)
        }