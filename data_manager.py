import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import time
from pymongo import MongoClient, ASCENDING
from pymongo.errors import DuplicateKeyError

# ============= MongoDB 설정 =============
MONGO_URI = "mongodb+srv://rator9521_db_user:qwe343434@cluster0.d126rkt.mongodb.net/"
DATABASE_NAME = "etf_database"

# 컬렉션 이름
COLLECTION_ETF_MASTER = "etf_master"
COLLECTION_ASSET_PAIRS = "asset_pairs"
COLLECTION_FUND_PRICES = "fund_prices"
COLLECTION_SYNTHETIC_INDICES = "synthetic_indices"

# 기존 로컬 파일 경로 (CSV 읽기용)
USER_CSV_FILE = "data/etf_info.csv"

# MongoDB 연결
client = MongoClient(MONGO_URI)
db = client[DATABASE_NAME]

def convert_csv_to_master_json():
    """CSV를 읽어 MongoDB의 etf_master 컬렉션에 저장"""
    print(f"'{USER_CSV_FILE}' 파일을 읽어 MongoDB에 저장합니다...")
    
    try:
        # CSV 읽기
        try:
            df = pd.read_csv(USER_CSV_FILE, encoding='utf-8-sig')
        except UnicodeDecodeError:
            df = pd.read_csv(USER_CSV_FILE, encoding='cp949')

        df.columns = [col.lower() for col in df.columns]

        # 데이터 전처리
        df.rename(columns={'saaclass': 'saa_class', 'taaclass': 'taa_class'}, inplace=True)
        df['단축코드'] = df['단축코드'].astype(str).str.zfill(6)
        df['ticker'] = df['단축코드'].astype(str).str.zfill(6) + '.KS'
        df['상장일'] = pd.to_datetime(df['상장일'], errors='coerce')
        df = df.dropna(subset=['상장일', 'saa_class', 'taa_class'])
        df = df[~df['saa_class'].isin(['미분류', ''])]

        # Score 생성
        df = df.sort_values(by='상장일')
        df['score'] = df.groupby(['saa_class', 'taa_class']).cumcount() + 1
        print("✅ 'score' 컬럼 생성 완료.")

        # Code 생성
        saa_prefix_map = {
            '국내주식': 'SK', '국내채권': 'BK', '해외주식': 'SG',
            '해외채권': 'BG', '대체투자': 'AI', '단기자금': 'MM'
        }
        
        unique_pairs = df[['saa_class', 'taa_class']].drop_duplicates().sort_values(
            by=['saa_class', 'taa_class']
        ).reset_index(drop=True)
        
        unique_pairs['pair_rank'] = unique_pairs.groupby('saa_class').cumcount() + 1
        unique_pairs['code'] = (
            unique_pairs['saa_class'].map(saa_prefix_map) + 
            unique_pairs['pair_rank'].apply(lambda x: f'{x:02d}')
        )

        df = df.merge(unique_pairs[['saa_class', 'taa_class', 'code']], on=['saa_class', 'taa_class'], how='left')
        print("✅ 'code' 컬럼 생성 완료.")
        
        # 날짜를 문자열로 변환
        df['상장일'] = df['상장일'].dt.strftime('%Y-%m-%d')

        # MongoDB에 저장
        etf_list = df.to_dict('records')
        
        # 기존 데이터 삭제 후 새로 삽입
        db[COLLECTION_ETF_MASTER].delete_many({})
        db[COLLECTION_ETF_MASTER].insert_many(etf_list)
        
        # 인덱스 생성 (ticker로 빠른 검색)
        db[COLLECTION_ETF_MASTER].create_index([("ticker", ASCENDING)])
        
        print(f"✅ MongoDB '{COLLECTION_ETF_MASTER}' 컬렉션에 {len(etf_list)}개 저장 완료!")
        return True
        
    except Exception as e:
        print(f"❌ 오류: {e}")
        import traceback
        traceback.print_exc()
        return False

def create_asset_pairs():
    """etf_master 컬렉션에서 SAA/TAA 조합 추출하여 asset_pairs 컬렉션 생성"""
    print(f"'{COLLECTION_ETF_MASTER}' 컬렉션에서 자산 조합을 생성합니다...")
    
    try:
        # MongoDB에서 데이터 읽기
        etf_list = list(db[COLLECTION_ETF_MASTER].find({}, {'_id': 0}))
        
        df = pd.DataFrame(etf_list)
        asset_pairs_df = df[['saa_class', 'taa_class']].drop_duplicates().dropna()
        result = asset_pairs_df.to_dict('records')
        
        # MongoDB에 저장
        db[COLLECTION_ASSET_PAIRS].delete_many({})
        db[COLLECTION_ASSET_PAIRS].insert_many(result)
        
        print(f"✅ {len(result)}개의 자산 조합을 '{COLLECTION_ASSET_PAIRS}' 컬렉션에 저장 완료!")
        
    except Exception as e:
        print(f"❌ 오류: {e}")

def update_etf_prices_to_mongodb():
    """ETF 가격 데이터를 MongoDB에 저장/업데이트"""
    print(f"\nETF 가격 데이터를 MongoDB에 저장합니다...")
    
    # etf_master에서 ticker 목록 가져오기
    etf_list = list(db[COLLECTION_ETF_MASTER].find({}, {'ticker': 1, '_id': 0}))
    
    print(f"총 {len(etf_list)}개 ETF의 가격 데이터 업데이트 시작...")
    
    for i, etf in enumerate(etf_list):
        ticker = etf.get('ticker')
        if not ticker:
            continue
        
        # MongoDB에서 기존 데이터 확인
        existing_doc = db[COLLECTION_FUND_PRICES].find_one({'ticker': ticker})
        
        start_date_str = (datetime.now() - timedelta(days=10*365)).strftime('%Y-%m-%d')
        
        if existing_doc and 'prices' in existing_doc and existing_doc['prices']:
            # 최신 날짜 찾기
            dates = [pd.to_datetime(p['Date']) for p in existing_doc['prices']]
            if dates:
                latest_date = max(dates)
                start_date_str = (latest_date + timedelta(days=1)).strftime('%Y-%m-%d')
        
        if pd.to_datetime(start_date_str).date() >= datetime.now().date():
            print(f"({i+1}/{len(etf_list)}) {ticker}: 이미 최신 데이터")
            continue
        
        try:
            time.sleep(0.2)
            print(f"({i+1}/{len(etf_list)}) {ticker}: {start_date_str}부터 다운로드 중...")
            df_new = yf.download(ticker, start=start_date_str, progress=False, auto_adjust=False)
            
            if df_new.empty:
                print(f" -> {ticker}: 데이터 없음")
                continue
            
            # DataFrame을 dict 리스트로 변환
            df_new = df_new.reset_index()
            
            # 컬럼명 정리 (멀티인덱스 대응)
            if isinstance(df_new.columns, pd.MultiIndex):
                df_new.columns = df_new.columns.get_level_values(0)
            
            price_data = []
            for idx in range(len(df_new)):
                try:
                    date_val = df_new.iloc[idx]['Date'] if 'Date' in df_new.columns else df_new.index[idx]
                    price_data.append({
                        'Date': pd.to_datetime(date_val).strftime('%Y-%m-%d'),
                        'Close': float(df_new.iloc[idx]['Close']) if 'Close' in df_new.columns and pd.notna(df_new.iloc[idx]['Close']) else None,
                        'High': float(df_new.iloc[idx]['High']) if 'High' in df_new.columns and pd.notna(df_new.iloc[idx]['High']) else None,
                        'Low': float(df_new.iloc[idx]['Low']) if 'Low' in df_new.columns and pd.notna(df_new.iloc[idx]['Low']) else None,
                        'Open': float(df_new.iloc[idx]['Open']) if 'Open' in df_new.columns and pd.notna(df_new.iloc[idx]['Open']) else None,
                        'Volume': int(df_new.iloc[idx]['Volume']) if 'Volume' in df_new.columns and pd.notna(df_new.iloc[idx]['Volume']) else None
                    })
                except Exception as e:
                    print(f"   데이터 변환 오류: {e}")
                    continue
            
            if existing_doc:
                # 기존 데이터에 추가
                db[COLLECTION_FUND_PRICES].update_one(
                    {'ticker': ticker},
                    {'$push': {'prices': {'$each': price_data}}}
                )
            else:
                # 새 문서 생성
                db[COLLECTION_FUND_PRICES].insert_one({
                    'ticker': ticker,
                    'prices': price_data
                })
            
            print(f" -> {ticker}: {len(price_data)}건 저장 완료")
            
        except Exception as e:
            print(f" -> 오류: {ticker} - {e}")
    
    # 인덱스 생성
    db[COLLECTION_FUND_PRICES].create_index([("ticker", ASCENDING)], unique=True)
    
    print("\n✅ ETF 가격 데이터 업데이트 완료!")

def create_synthetic_indices():
    """합성 지수 생성하여 MongoDB에 저장"""
    print("\n합성 지수 생성 시작...")
    
    # etf_master에서 데이터 로드
    master_list = list(db[COLLECTION_ETF_MASTER].find({}, {'_id': 0}))
    master_df = pd.DataFrame(master_list)
    master_df['상장일'] = pd.to_datetime(master_df['상장일'])
    
    # fund_prices에서 데이터 로드하여 수익률 계산
    price_data = {}
    print("ETF 수익률 계산 중...")
    
    for ticker in master_df['ticker'].unique():
        doc = db[COLLECTION_FUND_PRICES].find_one({'ticker': ticker})
        if not doc or 'prices' not in doc:
            continue
        
        try:
            # MongoDB 데이터를 DataFrame으로 변환
            df = pd.DataFrame(doc['prices'])
            df['Date'] = pd.to_datetime(df['Date'])
            df = df.set_index('Date').sort_index()
            df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
            df = df.dropna(subset=['Close'])
            
            if len(df) > 1:
                price_data[ticker] = df['Close'].pct_change()
                
        except Exception as e:
            print(f"경고: {ticker} 처리 오류 - {e}")
    
    unique_codes = master_df['code'].dropna().unique()
    print(f"총 {len(unique_codes)}개의 합성 지수 생성...")
    
    for code in unique_codes:
        try:
            print(f" - '{code}' 생성 중...")
            group_df = master_df[master_df['code'] == code]
            
            # 실제 데이터 시작일 찾기
            actual_start_date = None
            for ticker in group_df['ticker']:
                if ticker in price_data and not price_data[ticker].empty:
                    first_valid_date = price_data[ticker].first_valid_index()
                    if first_valid_date is not None:
                        if actual_start_date is None or first_valid_date < actual_start_date:
                            actual_start_date = first_valid_date
            
            if actual_start_date is None:
                print(f"   -> 경고: '{code}' 데이터 없음, 건너뜀")
                continue
            
            end_date = datetime.now().date()
            date_range = pd.date_range(start=actual_start_date, end=end_date, freq='B')
            
            daily_avg_returns = []
            
            for dt in date_range:
                active_etfs = group_df[group_df['상장일'] <= dt]
                returns_for_day = []
                
                for ticker in active_etfs['ticker']:
                    if ticker in price_data and dt in price_data[ticker].index:
                        ret = price_data[ticker].loc[dt]
                        if pd.notna(ret):
                            returns_for_day.append(ret)
                
                avg_return = sum(returns_for_day) / len(returns_for_day) if returns_for_day else 0.0
                daily_avg_returns.append(avg_return)
            
            # 지수 계산
            index_df = pd.DataFrame({'Date': date_range, 'return': daily_avg_returns})
            index_df.iloc[0, index_df.columns.get_loc('return')] = 0.0
            index_df['close'] = 100 * (1 + index_df['return']).cumprod()
            
            # MongoDB에 저장
            index_data = []
            for _, row in index_df.iterrows():
                index_data.append({
                    'Date': row['Date'].strftime('%Y-%m-%d'),
                    'close': float(row['close']),
                    'return': float(row['return'])
                })
            
            # Upsert (있으면 업데이트, 없으면 삽입)
            db[COLLECTION_SYNTHETIC_INDICES].update_one(
                {'code': code},
                {'$set': {
                    'code': code,
                    'start_date': actual_start_date.strftime('%Y-%m-%d'),
                    'data': index_data
                }},
                upsert=True
            )
            
            print(f"   -> '{code}' 저장 완료 (시작일: {actual_start_date.date()})")
            
        except Exception as e:
            print(f"   -> 오류: '{code}' - {e}")
            import traceback
            traceback.print_exc()
    
    # 인덱스 생성
    db[COLLECTION_SYNTHETIC_INDICES].create_index([("code", ASCENDING)], unique=True)
    
    print("\n✅ 합성 지수 생성 완료!")

def calculate_and_store_market_summary():
    """
    매일 자산군/ETF 순위를 계산하여 MongoDB에 저장
    """
    print("\n" + "="*60)
    print("마켓 서머리 순위 계산 시작...")
    print("="*60)
    
    from datetime import datetime
    import pandas as pd
    
    COLLECTION_MARKET_SUMMARY = "market_summary"
    
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
    
    timeframe_days = {
        '당일': 2,
        '1주일': 5,
        '1달': 21,
        '3개월': 63,
        '6개월': 126,
        '1년': 252,
        '3년': 756
    }
    
    # 기존 데이터 삭제
    db[COLLECTION_MARKET_SUMMARY].delete_many({})
    print("기존 마켓 서머리 데이터 삭제 완료")
    
    total_saved = 0
    
    # 각 타임프레임별로 계산
    for timeframe, days in timeframe_days.items():
        print(f"\n📊 [{timeframe}] 순위 계산 중...")
        
        # ========== 자산군 순위 계산 ==========
        print(f"  - 자산군 순위 계산...")
        asset_rankings = []
        
        all_indices = db[COLLECTION_SYNTHETIC_INDICES].find({})
        
        for index_doc in all_indices:
            code = index_doc.get('code')
            data = index_doc.get('data', [])
            
            if not data or len(data) < days:
                continue
            
            df = pd.DataFrame(data)
            df['Date'] = pd.to_datetime(df['Date'])
            df = df.sort_values('Date')
            df['close'] = pd.to_numeric(df['close'], errors='coerce')
            df = df.dropna(subset=['close'])
            
            if len(df) < days:
                continue
            
            return_pct = calculate_return_for_period(df['close'], days)
            
            if return_pct is None:
                continue
            
            master_info = db[COLLECTION_ETF_MASTER].find_one({'code': code})
            
            if master_info:
                saa = master_info.get('saa_class', '미분류')
                taa = master_info.get('taa_class', '미분류')
                
                asset_rankings.append({
                    'code': code,
                    'saa': saa,
                    'taa': taa,
                    'name': f"{saa} / {taa}",
                    'return': return_pct
                })
        
        # 수익률 기준 정렬
        asset_rankings_sorted = sorted(asset_rankings, key=lambda x: x['return'], reverse=True)
        
        # 상위 10개, 하위 10개
        asset_top_10 = asset_rankings_sorted[:10] if len(asset_rankings_sorted) >= 10 else asset_rankings_sorted
        asset_bottom_10 = asset_rankings_sorted[-10:] if len(asset_rankings_sorted) >= 10 else []
        
        # 순위 매기기
        for idx, item in enumerate(asset_top_10, 1):
            item['rank'] = idx
        
        for idx, item in enumerate(asset_bottom_10, 1):
            item['rank'] = idx
        
        print(f"    ✅ 자산군: 상위 {len(asset_top_10)}개, 하위 {len(asset_bottom_10)}개")
        
        # ========== ETF 순위 계산 ==========
        print(f"  - ETF 순위 계산...")
        etf_rankings = []
        
        all_etfs = db[COLLECTION_ETF_MASTER].find({})
        
        for etf in all_etfs:
            ticker = etf.get('ticker')
            name = etf.get('한글종목약명', etf.get('한글종목명', ticker))
            
            price_doc = db[COLLECTION_FUND_PRICES].find_one({'ticker': ticker})
            
            if not price_doc or 'prices' not in price_doc:
                continue
            
            prices = price_doc['prices']
            if len(prices) < days:
                continue
            
            df = pd.DataFrame(prices)
            df['Date'] = pd.to_datetime(df['Date'])
            df = df.sort_values('Date')
            df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
            df = df.dropna(subset=['Close'])
            
            if len(df) < days:
                continue
            
            return_pct = calculate_return_for_period(df['Close'], days)
            
            if return_pct is None:
                continue
            
            current_price = int(df['Close'].iloc[-1])
            
            etf_rankings.append({
                'ticker': ticker,
                'name': name,
                'price': f"{current_price:,}원",
                'return': return_pct
            })
        
        # 수익률 기준 정렬
        etf_rankings_sorted = sorted(etf_rankings, key=lambda x: x['return'], reverse=True)
        
        # 상위 10개, 하위 10개
        etf_top_10 = etf_rankings_sorted[:10] if len(etf_rankings_sorted) >= 10 else etf_rankings_sorted
        etf_bottom_10 = etf_rankings_sorted[-10:] if len(etf_rankings_sorted) >= 10 else []
        
        # 순위 매기기
        for idx, item in enumerate(etf_top_10, 1):
            item['rank'] = idx
        
        for idx, item in enumerate(etf_bottom_10, 1):
            item['rank'] = idx
        
        print(f"    ✅ ETF: 상위 {len(etf_top_10)}개, 하위 {len(etf_bottom_10)}개")
        
        # ========== MongoDB에 저장 ==========
        summary_data = {
            'timeframe': timeframe,
            'updated_at': datetime.now(),
            'asset': {
                'top': asset_top_10,
                'bottom': asset_bottom_10
            },
            'etf': {
                'top': etf_top_10,
                'bottom': etf_bottom_10
            }
        }
        
        db[COLLECTION_MARKET_SUMMARY].insert_one(summary_data)
        total_saved += 1
        print(f"  ✅ [{timeframe}] 저장 완료")
    
    # 인덱스 생성 (빠른 조회를 위해)
    db[COLLECTION_MARKET_SUMMARY].create_index([("timeframe", 1)])
    
    print("\n" + "="*60)
    print(f"✅ 마켓 서머리 순위 계산 완료! (총 {total_saved}개 타임프레임)")
    print("="*60)

# ========== 메인 실행 부분 수정 ==========
if __name__ == '__main__':
    print("=" * 60)
    print("ETF 데이터 관리 시스템 (MongoDB 버전)")
    print("=" * 60)
    
    if convert_csv_to_master_json():
        create_asset_pairs()
        update_etf_prices_to_mongodb()
        create_synthetic_indices()
        
        # ✨ 마켓 서머리 순위 계산 추가
        calculate_and_store_market_summary()
    
    print("\n" + "=" * 60)
    print("모든 작업 완료!")
    print("=" * 60)
