import pandas as pd
import json
import os
from pypfopt import EfficientFrontier, risk_models, expected_returns, HRPOpt
import traceback
import numpy as np
from scipy.optimize import minimize
import warnings

# 경고 메시지 필터링
warnings.filterwarnings('ignore', category=UserWarning, module='pypfopt')

# --- 경로 설정 ---
DATA_DIR = "data"
PRICE_DATA_DIR = os.path.join(DATA_DIR, "360750.KS")

def clean_price_data(df):
    """가격 데이터를 정리하고 이상값을 제거합니다."""
    # 1. 무한값을 NaN으로 변환
    df = df.replace([np.inf, -np.inf], np.nan)
    
    # 2. 0 또는 음수 가격을 NaN으로 변환
    df = df.where(df > 0, np.nan)
    
    # 3. 극단적인 일일 수익률 제거 (±30% 초과) - 더 보수적으로 설정
    returns = df.pct_change()
    outlier_mask = (returns.abs() > 0.3)
    df = df.where(~outlier_mask, np.nan)
    
    # 4. forward fill로 단기 결측값 보완 (최대 3일) - 새로운 pandas 버전 대응
    df = df.ffill(limit=3)
    
    # 5. 연속된 NaN이 너무 많은 행 제거
    df = df.dropna(thresh=len(df.columns) * 0.8)  # 80% 이상의 컬럼에 데이터가 있는 행만 유지
    
    return df

file_path = "C:\Users\mason\Desktop\Project\Mytech\data\fund_prices\data\360750.KS.csv"

if os.path.exists(file_path):
    try:
        df = pd.read_csv(
            file_path, 
            skiprows=3, 
            header=None,
            names=['Date', 'Close', 'High', 'Low', 'Open', 'Volume'],
            index_col='Date',
            parse_dates=True,
            on_bad_lines='skip' # ★★★ 1. 깨진 줄은 건너뛰도록 수정
        )
        
        # ★★★ 2. 인덱스가 날짜가 아닌 행(NaT)을 먼저 제거하여 안정성 확보 ★★★
        df = df[df.index.notna()]
        
        # ★★★ 3. 중복된 인덱스(날짜)가 있을 경우 첫 번째 값만 남김 ★★★
        df = df[~df.index.duplicated(keep='first')]
        
        df['Adj Close'] = df['Close']
        
        if not df.empty and 'Adj Close' in df.columns:
            price_series = df[['Adj Close']].rename(columns={'Adj Close': ticker})
            price_series = clean_price_data(price_series)
            
            if not price_series.empty:
                all_prices.append(price_series)
            else:
                print(f"⚠️  정리 후 {ticker}의 데이터가 비어있습니다.")
        else:
            print(f"⚠️  {ticker}의 가격 데이터가 없습니다.")
            
    except Exception as e:
        print(f"⚠️  {ticker} 파일 읽기 오류: {str(e)}")
else:
    print(f"⚠️  파일이 존재하지 않습니다: {file_path}")