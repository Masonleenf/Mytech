import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import time
import numpy as np
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

def remove_duplicate_dates_and_add_new_data(existing_prices, new_data):
    """기존 데이터의 중복 제거 + 새 데이터 추가 + 재중복 제거"""
    if not existing_prices and not new_data:
        return []
    
    # 모든 데이터를 합치기
    all_data = existing_prices + new_data if existing_prices else new_data
    
    if not all_data:
        return []
    
    # DataFrame으로 변환
    df = pd.DataFrame(all_data)
    
    if df.empty:
        return []
    
    # 날짜를 datetime으로 변환
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    
    # 잘못된 날짜 데이터 제거
    df = df.dropna(subset=['Date'])
    
    # 날짜 순으로 정렬
    df = df.sort_values('Date')
    
    # 중복된 날짜는 마지막 것만 남기기 (keep='last') + 명시적으로 copy 생성
    df_unique = df.drop_duplicates(subset=['Date'], keep='last').copy()
    
    # 날짜를 다시 문자열로 변환 (이제 경고 없음)
    df_unique['Date'] = df_unique['Date'].dt.strftime('%Y-%m-%d')
    
    # 다시 dict 리스트로 변환
    return df_unique.to_dict('records')

def convert_csv_to_master_json():
    """CSV를 읽어 MongoDB의 etf_master 컬렉션에 저장"""
    print(f"'{USER_CSV_FILE}' 파일을 읽어 MongoDB에 저장합니다...")
    
    try:
        # CSV 읽기 - 단축코드만 문자열로 지정 (과학적 표기법 방지)
        try:
            df = pd.read_csv(USER_CSV_FILE, encoding='utf-8-sig', dtype={'단축코드': str})
        except UnicodeDecodeError:
            df = pd.read_csv(USER_CSV_FILE, encoding='cp949', dtype={'단축코드': str})

        df.columns = [col.lower() for col in df.columns]

        # 데이터 전처리
        df.rename(columns={'saaclass': 'saa_class', 'taaclass': 'taa_class'}, inplace=True)
        
        # 단축코드 처리 (문자열로 읽어와서 6자리로 맞춤)
        df['단축코드'] = df['단축코드'].astype(str).str.zfill(6)
        df['ticker'] = df['단축코드'] + '.KS'
        
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
    """ETF 가격 데이터를 MongoDB에 저장/업데이트 (중복 제거 포함)"""
    print(f"\nETF 가격 데이터를 MongoDB에 저장합니다... (중복 제거 기능 포함)")
    
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
            # ✅ 1단계: 기존 데이터 중복 제거
            print(f"({i+1}/{len(etf_list)}) {ticker}: 기존 데이터 중복 확인 중...", end=" ")
            
            existing_prices = existing_doc['prices']
            cleaned_existing = remove_duplicate_dates_and_add_new_data(existing_prices, [])
            
            if len(cleaned_existing) != len(existing_prices):
                print(f"중복 제거: {len(existing_prices)}건 → {len(cleaned_existing)}건")
            else:
                print("중복 없음")
            
            # 최신 날짜 찾기 (중복 제거된 데이터에서)
            if cleaned_existing:
                dates = [pd.to_datetime(p['Date']) for p in cleaned_existing]
                latest_date = max(dates)
                start_date_str = (latest_date + timedelta(days=1)).strftime('%Y-%m-%d')
        else:
            print(f"({i+1}/{len(etf_list)}) {ticker}: 새 데이터")
            cleaned_existing = []
        
        # ✅ 2단계: 새 데이터 다운로드 필요한지 확인
        if pd.to_datetime(start_date_str).date() >= datetime.now().date():
            # 오늘날짜거나 미래 날짜면 다운로드 불필요
            if existing_doc and len(cleaned_existing) != len(existing_doc['prices']):
                # 중복만 제거하고 업데이트
                db[COLLECTION_FUND_PRICES].update_one(
                    {'ticker': ticker},
                    {'$set': {'prices': cleaned_existing}}
                )
                print(f"   -> 중복 제거 완료 ({len(cleaned_existing)}건)")
            else:
                print(f"   -> 이미 최신 데이터")
            continue
        
        # ✅ 3단계: 새 데이터 다운로드
        try:
            time.sleep(0.2)  # API 호출 간격
            print(f"   -> {start_date_str}부터 다운로드 중...", end=" ")
            
            df_new = yf.download(ticker, start=start_date_str, progress=False, auto_adjust=False)
            
            if df_new.empty:
                # 새 데이터는 없지만 중복 제거된 데이터로 업데이트
                if existing_doc and len(cleaned_existing) != len(existing_doc['prices']):
                    db[COLLECTION_FUND_PRICES].update_one(
                        {'ticker': ticker},
                        {'$set': {'prices': cleaned_existing}}
                    )
                    print("새 데이터 없음, 중복 제거만 완료")
                else:
                    print("새 데이터 없음")
                continue
            
            # DataFrame을 dict 리스트로 변환
            df_new = df_new.reset_index()
            
            # 컬럼명 정리 (멀티인덱스 대응)
            if isinstance(df_new.columns, pd.MultiIndex):
                df_new.columns = df_new.columns.get_level_values(0)
            
            new_price_data = []
            for idx in range(len(df_new)):
                try:
                    date_val = df_new.iloc[idx]['Date'] if 'Date' in df_new.columns else df_new.index[idx]
                    new_price_data.append({
                        'Date': pd.to_datetime(date_val).strftime('%Y-%m-%d'),
                        'Close': float(df_new.iloc[idx]['Close']) if 'Close' in df_new.columns and pd.notna(df_new.iloc[idx]['Close']) else None,
                        'High': float(df_new.iloc[idx]['High']) if 'High' in df_new.columns and pd.notna(df_new.iloc[idx]['High']) else None,
                        'Low': float(df_new.iloc[idx]['Low']) if 'Low' in df_new.columns and pd.notna(df_new.iloc[idx]['Low']) else None,
                        'Open': float(df_new.iloc[idx]['Open']) if 'Open' in df_new.columns and pd.notna(df_new.iloc[idx]['Open']) else None,
                        'Volume': int(df_new.iloc[idx]['Volume']) if 'Volume' in df_new.columns and pd.notna(df_new.iloc[idx]['Volume']) else None
                    })
                except Exception as e:
                    print(f"데이터 변환 오류: {e}")
                    continue
            
            # ✅ 4단계: 기존 데이터 + 새 데이터 합치고 중복 제거
            final_data = remove_duplicate_dates_and_add_new_data(cleaned_existing, new_price_data)
            
            # MongoDB에 저장
            if existing_doc:
                db[COLLECTION_FUND_PRICES].update_one(
                    {'ticker': ticker},
                    {'$set': {'prices': final_data}}
                )
            else:
                db[COLLECTION_FUND_PRICES].insert_one({
                    'ticker': ticker,
                    'prices': final_data
                })
            
            added_count = len(final_data) - len(cleaned_existing)
            print(f"신규 {len(new_price_data)}건 추가, 총 {len(final_data)}건 ({added_count:+d})")
            
        except Exception as e:
            print(f"   -> 오류: {e}")
            # 오류가 발생해도 중복 제거는 수행
            if existing_doc and cleaned_existing and len(cleaned_existing) != len(existing_doc['prices']):
                db[COLLECTION_FUND_PRICES].update_one(
                    {'ticker': ticker},
                    {'$set': {'prices': cleaned_existing}}
                )
                print(f"   -> 다운로드 실패, 중복 제거만 완료")
    
    # 인덱스 생성
    try:
        db[COLLECTION_FUND_PRICES].create_index([("ticker", ASCENDING)], unique=True)
    except Exception as e:
        print(f"인덱스 생성 오류: {e}")
    
    print("\n✅ ETF 가격 데이터 업데이트 완료! (중복 제거 포함)")

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
                        
                        # ★ 핵심 수정: Series인 경우를 처리
                        if isinstance(ret, pd.Series):
                            if len(ret) > 0:
                                ret = ret.iloc[0]  # 첫 번째 값 사용
                            else:
                                continue
                        
                        # ★ 핵심 수정: 스칼라 값에 대해서만 notna 체크
                        if pd.notna(ret) and np.isfinite(ret):
                            returns_for_day.append(float(ret))
                
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
    try:
        db[COLLECTION_SYNTHETIC_INDICES].create_index([("code", ASCENDING)], unique=True)
    except Exception as e:
        print(f"인덱스 생성 오류: {e}")
    
    print("\n✅ 합성 지수 생성 완료!")

if __name__ == '__main__':
    print("=" * 60)
    print("ETF 데이터 관리 시스템 (MongoDB 버전) - 중복 제거 기능 포함")
    print("=" * 60)
    
    # 현재 시간 출력 (새벽 4시 실행 확인용)
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"실행 시간: {current_time}")
    
    if convert_csv_to_master_json():
        create_asset_pairs()
        update_etf_prices_to_mongodb()
        create_synthetic_indices()
    
    print("\n" + "=" * 60)
    print("모든 작업 완료!")
    print("=" * 60)
