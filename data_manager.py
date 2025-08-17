import pandas as pd
import yfinance as yf
import json
import os
from datetime import datetime, timedelta
import time # 시간 지연(휴식)을 위해 time 라이브러리 추가

# --- 설정 ---
USER_CSV_FILE = "etf_info.csv"
DATA_DIR = "data"
PRICE_DATA_DIR = os.path.join(DATA_DIR, "fund_prices")
CSV_FILE_PATH = os.path.join(DATA_DIR, USER_CSV_FILE)
MASTER_FILE_PATH = os.path.join(DATA_DIR, "etf_master.json")
source_file = 'data/etf_master.json'
output_file = 'App/mytechapp/assets/data/asset_pairs.json' # Flutter 프로젝트 assets 폴더로 옮길 최종 파일

def convert_csv_to_master_json():
    """사용자가 수정한 CSV를 읽어 최종 etf_master.json 파일을 생성합니다."""
    print(f"'{CSV_FILE_PATH}' 파일을 읽어 최종 마스터 파일을 생성합니다...")
    if not os.path.exists(CSV_FILE_PATH):
        print(f"오류: '{CSV_FILE_PATH}' 파일이 없습니다. data 폴더에 파일을 올바르게 넣었는지 확인해주세요.")
        return False
    try:
        try:
            df = pd.read_csv(CSV_FILE_PATH, encoding='utf-8-sig')
        except UnicodeDecodeError:
            df = pd.read_csv(CSV_FILE_PATH, encoding='cp949')

        df.columns = [col.lower() for col in df.columns]
        
        if 'ticker' not in df.columns:
            # '단축코드'가 소문자로 바뀌었으므로 '단축코드'로 접근
            df['ticker'] = df['단축코드'].astype(str).str.zfill(6) + '.KS'
        
        etf_list = df.to_dict('records')
        with open(MASTER_FILE_PATH, 'w', encoding='utf-8') as f:
            json.dump(etf_list, f, ensure_ascii=False, indent=4)
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(etf_list, f, ensure_ascii=False, indent=4)
        print(f"성공! '{MASTER_FILE_PATH}' 파일이 생성/업데이트되었습니다.")
        return True
    except KeyError as e:
        print(f"오류: CSV 파일에서 필요한 컬럼({e})을 찾을 수 없습니다. 컬럼 이름을 확인해주세요.")
        return False
    except Exception as e:
        print(f"오류: CSV 파일을 처리하는 중 문제가 발생했습니다 - {e}")
        return False

def create_asset_pairs():
    """
    etf_master.json에서 SAA와 TAA 조합을 추출하고 중복을 제거하여
    asset_pairs.json 파일을 생성합니다.
    """
    print(f"'{source_file}' 파일을 읽어 자산 조합을 생성합니다...")
    try:
        with open(source_file, 'r', encoding='utf-8') as f:
            etf_list = json.load(f)
        
        df = pd.DataFrame(etf_list)
        
        # SAA와 TAA 클래스 컬럼명을 소문자로 통일
        df.rename(columns={'SAAclass': 'saa_class', 'TAAclass': 'taa_class'}, inplace=True)
        
        # 두 컬럼을 선택하고, 중복된 행을 제거
        asset_pairs_df = df[['saa_class', 'taa_class']].drop_duplicates()
        
        # 결측값(NaN)이나 "미분류" 항목은 제외
        asset_pairs_df = asset_pairs_df.dropna()
        asset_pairs_df = asset_pairs_df[asset_pairs_df['saa_class'] != '미분류']
        
        # JSON 형식으로 변환
        result = asset_pairs_df.to_dict('records')
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=4)
            
        print(f"성공! 총 {len(result)}개의 자산 조합을 '{output_file}'에 저장했습니다.")

    except FileNotFoundError:
        print(f"오류: '{source_file}'을 찾을 수 없습니다.")
    except Exception as e:
        print(f"오류: {e}")


def update_etf_prices_as_csv(): # Parquet 대신 CSV로 저장하도록 함수 이름 변경
    """etf_master.json을 기준으로 가격 데이터를 표준 CSV 파일로 저장/업데이트합니다."""
    os.makedirs(PRICE_DATA_DIR, exist_ok=True)
    if not os.path.exists(MASTER_FILE_PATH):
        print(f"오류: '{MASTER_FILE_PATH}' 파일이 없습니다.")
        return

    with open(MASTER_FILE_PATH, 'r', encoding='utf-8') as f:
        etf_list = json.load(f)

    print(f"\n총 {len(etf_list)}개 ETF의 가격 데이터 업데이트를 시작합니다 (CSV 형식)...")
    
    for i, etf in enumerate(etf_list):
        ticker = etf.get('ticker')
        if not ticker:
            continue
        
        file_path = os.path.join(PRICE_DATA_DIR, f"{ticker}.csv")
        start_date_str = (datetime.now() - timedelta(days=5*365)).strftime('%Y-%m-%d')
        
        df_existing = None
        if os.path.exists(file_path):
            try:
                df_existing = pd.read_csv(file_path, index_col='Date', parse_dates=True)
                if not df_existing.empty:
                    last_date = df_existing.index.max()
                    start_date_str = (last_date + timedelta(days=1)).strftime('%Y-%m-%d')
            except Exception:
                pass 
        
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

            # ★★★ 핵심 수정: 항상 표준 형식의 CSV로 저장 ★★★
            if df_existing is not None and not df_existing.empty:
                # 새 데이터만 기존 파일에 추가
                df_new.to_csv(file_path, mode='a', header=False, index=True)
            else:
                # 새 파일로 저장 (헤더 포함)
                df_new.to_csv(file_path, mode='w', header=True, index=True)
            
            print(f" -> {ticker}: 데이터를 성공적으로 업데이트했습니다.")
        except Exception as e:
            print(f" -> 오류: {ticker} 데이터 처리 실패 - {e}")

    print("\n모든 ETF 가격 데이터 업데이트가 완료되었습니다.")

if __name__ == '__main__':
    if convert_csv_to_master_json():
        update_etf_prices_as_csv()