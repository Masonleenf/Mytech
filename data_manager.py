import pandas as pd
import yfinance as yf
import json
import os
from datetime import datetime, timedelta
import time

# --- 설정 ---
USER_CSV_FILE = "etf_info.csv"
DATA_DIR = "data"
PRICE_DATA_DIR = os.path.join(DATA_DIR, "fund_prices")
CSV_FILE_PATH = os.path.join(DATA_DIR, USER_CSV_FILE)
MASTER_FILE_PATH = os.path.join(DATA_DIR, "etf_master.json")
ASSET_PAIRS_PATH = os.path.join(DATA_DIR, 'asset_pairs.json')

def convert_csv_to_master_json():
    """
    CSV를 읽어 'score', 'code' 컬럼을 추가하고 최종 etf_master.json을 생성합니다.
    """
    print(f"'{CSV_FILE_PATH}' 파일을 읽어 최종 마스터 파일을 생성합니다...")
    if not os.path.exists(CSV_FILE_PATH):
        print(f"오류: '{CSV_FILE_PATH}' 파일이 없습니다.")
        return False
    try:
        try:
            df = pd.read_csv(CSV_FILE_PATH, encoding='utf-8-sig')
        except UnicodeDecodeError:
            df = pd.read_csv(CSV_FILE_PATH, encoding='cp949')

        df.columns = [col.lower() for col in df.columns]

        # --- 데이터 전처리 ---
        df.rename(columns={'saaclass': 'saa_class', 'taaclass': 'taa_class'}, inplace=True)
        df['단축코드'] = df['단축코드'].astype(str).str.zfill(6)
        df['ticker'] = df['단축코드'].astype(str).str.zfill(6) + '.KS'
        df['상장일'] = pd.to_datetime(df['상장일'], errors='coerce')
        df = df.dropna(subset=['상장일', 'saa_class', 'taa_class'])
        df = df[~df['saa_class'].isin(['미분류', ''])]

        # --- 2-1. Score 생성 ---
        df = df.sort_values(by='상장일')
        df['score'] = df.groupby(['saa_class', 'taa_class']).cumcount() + 1
        print("✅ 'score' 컬럼 생성 완료.")

        # --- 2-2. Code 생성 ---
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
        
        # --- 파일 저장 전, 날짜 타입을 문자열로 변환 ---
        # 👇👇👇 이 한 줄을 추가해주세요! 👇👇👇
        df['상장일'] = df['상장일'].dt.strftime('%Y-%m-%d')

        # --- 파일 저장 ---
        etf_list = df.to_dict('records')
        with open(MASTER_FILE_PATH, 'w', encoding='utf-8') as f:
            json.dump(etf_list, f, ensure_ascii=False, indent=4)
        print(f"성공! '{MASTER_FILE_PATH}' 파일이 생성/업데이트되었습니다.")
        return True
    except Exception as e:
        print(f"오류: CSV 파일을 처리하는 중 문제가 발생했습니다 - {e}")
        import traceback
        traceback.print_exc()
        return False

def create_asset_pairs():
    """etf_master.json에서 SAA/TAA 조합을 추출하여 asset_pairs.json을 생성합니다."""
    print(f"'{MASTER_FILE_PATH}' 파일을 읽어 자산 조합을 생성합니다...")
    try:
        with open(MASTER_FILE_PATH, 'r', encoding='utf-8') as f:
            etf_list = json.load(f)
        
        df = pd.DataFrame(etf_list)
        asset_pairs_df = df[['saa_class', 'taa_class']].drop_duplicates().dropna()
        result = asset_pairs_df.to_dict('records')
        
        with open(ASSET_PAIRS_PATH, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=4)
        print(f"성공! 총 {len(result)}개의 자산 조합을 '{ASSET_PAIRS_PATH}'에 저장했습니다.")
    except Exception as e:
        print(f"오류: {e}")

def update_etf_prices_as_csv():
    """etf_master.json을 기준으로 개별 ETF 가격 데이터를 CSV로 저장/업데이트합니다."""
    # (이 함수는 기존과 동일하게 유지)
    os.makedirs(PRICE_DATA_DIR, exist_ok=True)
    if not os.path.exists(MASTER_FILE_PATH):
        print(f"오류: '{MASTER_FILE_PATH}' 파일이 없습니다.")
        return

    with open(MASTER_FILE_PATH, 'r', encoding='utf-8') as f:
        etf_list = json.load(f)

    print(f"\n총 {len(etf_list)}개 ETF의 가격 데이터 업데이트를 시작합니다 (CSV 형식)...")
    
    for i, etf in enumerate(etf_list):
        ticker = etf.get('ticker')
        if not ticker: continue
        
        file_path = os.path.join(PRICE_DATA_DIR, f"{ticker}.csv")
        start_date_str = (datetime.now() - timedelta(days=10*365)).strftime('%Y-%m-%d')
        
        if os.path.exists(file_path):
            try:
                df_existing = pd.read_csv(file_path, index_col='Date', parse_dates=True)
                if not df_existing.empty:
                    start_date_str = (df_existing.index.max() + timedelta(days=1)).strftime('%Y-%m-%d')
            except Exception: pass
        
        if pd.to_datetime(start_date_str).date() >= datetime.now().date():
            print(f"({i+1}/{len(etf_list)}) {ticker}: 이미 최신 데이터입니다.")
            continue
            
        try:
            time.sleep(0.2)
            print(f"({i+1}/{len(etf_list)}) {ticker}: {start_date_str}부터 데이터 다운로드 중...")
            df_new = yf.download(ticker, start=start_date_str, progress=False)
            
            if df_new.empty:
                print(f" -> {ticker}: 데이터 없음")
                continue

            header = not os.path.exists(file_path)
            df_new.to_csv(file_path, mode='a' if not header else 'w', header=header, index=True)
            print(f" -> {ticker}: 데이터를 성공적으로 업데이트했습니다.")
        except Exception as e:
            print(f" -> 오류: {ticker} 데이터 처리 실패 - {e}")

    print("\n개별 ETF 가격 데이터 업데이트가 완료되었습니다.")


def create_synthetic_indices():
    """
    'code'별로 합성 지수를 생성하여 'code.csv' 파일로 저장합니다. (시작일 최적화 버전)
    """
    print("\n합성 지수(synthetic index) 생성을 시작합니다...")
    if not os.path.exists(MASTER_FILE_PATH):
        print(f"오류: '{MASTER_FILE_PATH}'가 필요합니다.")
        return

    master_df = pd.read_json(MASTER_FILE_PATH)
    master_df['상장일'] = pd.to_datetime(master_df['상장일'])
    
    price_data = {}
    print("모든 ETF의 일일 수익률을 계산 중입니다...")
    for ticker in master_df['ticker'].unique():
        file_path = os.path.join(PRICE_DATA_DIR, f"{ticker}.csv")
        if not os.path.exists(file_path):
            continue
        
        try:
            df = pd.read_csv(file_path, skiprows=2)
            df.columns = ['Date', 'Close', 'High', 'Low', 'Open', 'Volume']
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
            df = df.dropna(subset=['Date'])
            df.set_index('Date', inplace=True)
            df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
            df = df.dropna(subset=['Close'])

            if len(df) > 1:
                df = df[~df.index.duplicated(keep='first')]
                df.sort_index(inplace=True)
                price_data[ticker] = df['Close'].pct_change()

        except Exception as e:
            print(f"경고: {ticker}.csv 파일을 처리하는 중 오류 발생 - {e}")

    unique_codes = master_df['code'].dropna().unique()
    print(f"총 {len(unique_codes)}개의 합성 지수를 생성합니다.")

    for code in unique_codes:
        try:
            print(f" - 지수 '{code}' 생성 중...")
            group_df = master_df[master_df['code'] == code]
            
            # ★★★★★ [수정된 로직 시작] ★★★★★
            # 그룹 내 ETF들의 '공식 상장일'이 아닌 '실제 데이터 시작일' 중 가장 빠른 날짜를 찾습니다.
            actual_start_date = None
            for ticker in group_df['ticker']:
                if ticker in price_data and not price_data[ticker].empty:
                    # 데이터가 있는 첫 날짜
                    first_valid_date = price_data[ticker].first_valid_index()
                    
                    # ✅ None 값 체크 추가
                    if first_valid_date is not None:
                        if actual_start_date is None or first_valid_date < actual_start_date:
                            actual_start_date = first_valid_date
            
            # 실제 데이터가 전혀 없는 그룹이라면 건너뜁니다.
            if actual_start_date is None:
                print(f"   -> 경고: '{code}' 그룹에 유효한 가격 데이터가 없어 건너뜁니다.")
                continue
            
            end_date = datetime.now().date()
            # 찾은 '실제 시작일'부터 날짜 범위를 생성합니다.
            date_range = pd.date_range(start=actual_start_date, end=end_date, freq='B')
            # ★★★★★ [수정된 로직 끝] ★★★★★
            
            daily_avg_returns = []
            
            for dt in date_range:
                # 공식 상장일이 아닌, 실제 날짜(dt)를 기준으로 active ETF를 판단합니다.
                active_etfs = group_df[group_df['상장일'] <= dt]
                
                returns_for_day = []
                for ticker in active_etfs['ticker']:
                    # 해당 날짜에 수익률 데이터가 있는지 확인
                    if ticker in price_data and dt in price_data[ticker].index:
                        ret = price_data[ticker].loc[dt]
                        if pd.notna(ret):
                            returns_for_day.append(ret)
                
                if returns_for_day:
                    avg_return = sum(returns_for_day) / len(returns_for_day)
                    daily_avg_returns.append(avg_return)
                else:
                    # 주말/휴일 등 모든 ETF 데이터가 없는 경우 0으로 처리
                    daily_avg_returns.append(0.0)

            index_df = pd.DataFrame({'Date': date_range, 'return': daily_avg_returns})
            index_df.set_index('Date', inplace=True)
            
            # 첫 날의 수익률은 0으로 설정하여 기준점(100)을 만듭니다.
            if not index_df.empty:
                index_df.iloc[0, index_df.columns.get_loc('return')] = 0.0

            index_df['close'] = 100 * (1 + index_df['return']).cumprod()
            
            output_path = os.path.join(PRICE_DATA_DIR, f"{code}.csv")
            index_df[['close', 'return']].to_csv(output_path)
            print(f"   -> 성공: '{output_path}'에 저장 완료. (시작일: {actual_start_date.date()})")

        except Exception as e:
            print(f"   -> 오류: '{code}' 지수 생성 실패 - {e}")
            # 디버깅을 위한 추가 정보 출력
            import traceback
            traceback.print_exc()
    
    print("\n모든 합성 지수 생성이 완료되었습니다.")
if __name__ == '__main__':
    if convert_csv_to_master_json():
        create_asset_pairs()
        update_etf_prices_as_csv()
        create_synthetic_indices()