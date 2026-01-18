import os
import re
import io
import json
import time
import numpy as np
import pandas as pd
import yfinance as yf
import requests
import urllib3
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple

# SSL 경고 비활성화
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


# =============================================================================
# Constants
# =============================================================================

# ETF 티커 소스
ETF_SOURCES = [
    {
        "name": "DumbStockAPI",
        "url": "https://dumbstockapi.com/stock?exchanges=NYSE,NASDAQ,AMEX&ticker_type=ETF&format=csv"
    },
    {
        "name": "NASDAQ Traded",
        "url": "http://www.nasdaqtrader.com/dynamic/SymDir/nasdaqtraded.txt",
        "sep": "|"
    },
    {
        "name": "GitHub Backup",
        "url": "https://raw.githubusercontent.com/rreichel3/US-Stock-Symbols/main/etf/etf_list.csv"
    }
]

# 레버리지/인버스 ETF 패턴
LEVERAGED_PATTERNS = [
    r'\b2x\b', r'\b3x\b', r'\b-1x\b', r'\b-2x\b', r'\b-3x\b',
    r'\bUltra\b', r'\bUltraShort\b', r'\bUltraPro\b',
    r'\bBull\s*2x\b', r'\bBull\s*3x\b', r'\bBear\s*1x\b', r'\bBear\s*2x\b', r'\bBear\s*3x\b',
    r'\bDouble\b', r'\bTriple\b',
    r'\bLeveraged\b', r'\bInverse\b',
]

LEVERAGED_TICKERS = {
    'TQQQ', 'SQQQ', 'UPRO', 'SPXU', 'SPXS', 'SOXL', 'SOXS', 'LABU', 'LABD',
    'NUGT', 'DUST', 'JNUG', 'JDST', 'UVXY', 'SVXY', 'VXX', 'VIXY',
    'TNA', 'TZA', 'FAS', 'FAZ', 'ERX', 'ERY', 'GUSH', 'DRIP',
    'TECL', 'TECS', 'UDOW', 'SDOW', 'UMDD', 'SMDD', 'URTY', 'SRTY',
    'TMF', 'TMV', 'TBT', 'BOIL', 'KOLD', 'UCO', 'SCO', 'UNG', 'DGAZ',
    'YINN', 'YANG', 'EDC', 'EDZ', 'INDL', 'RUSL', 'RUSS',
    'CURE', 'PILL', 'WEBL', 'WEBS', 'NAIL', 'CLAW',
    'FNGU', 'FNGD', 'DPST', 'DRN', 'DRV',
}

# 기본 ETF 리스트 (소스 다운로드 실패 시)
DEFAULT_ETFS = [
    'SPY', 'IVV', 'VOO', 'QQQ', 'VTI', 'VEA', 'IEFA', 'VWO', 'AGG', 'BND',
    'IEMG', 'IJH', 'IWF', 'GLD', 'VUG', 'IJR', 'VIG', 'VTV', 'BNDX', 'VXUS',
    'IWM', 'VO', 'IWD', 'XLK', 'VGT', 'VB', 'TLT', 'IVW', 'VNQ', 'LQD',
    'SCHD', 'JEPI', 'JEPQ', 'DGRO', 'VYM', 'XLV', 'XLF', 'XLE', 'XLY', 'XLI'
]


class FinancialDataManager:
    """
    금융 데이터 관리자.
    
    ETF 유니버스 수집, 가격 데이터 다운로드, 배당 정보 처리,
    공분산 계산 등 전체 데이터 파이프라인을 관리합니다.
    """
    
    def __init__(self, output_dir='data'):
        """
        Initialize the FinancialDataManager.
        
        Args:
            output_dir (str): Directory where output files will be saved.
        """
        self.output_dir = output_dir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"Created directory: {output_dir}")
    
    # =========================================================================
    # ETF Universe Collection (from divid.py)
    # =========================================================================
    
    def get_etf_universe(self) -> List[str]:
        """
        여러 소스에서 ETF 티커 목록을 수집합니다.
        
        Returns:
            List of ETF ticker symbols
        """
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
        
        for source in ETF_SOURCES:
            print(f"📥 {source['name']} 다운로드 중...")
            try:
                response = requests.get(
                    source['url'], 
                    headers=headers, 
                    timeout=10, 
                    verify=False
                )
                
                if response.status_code != 200:
                    print(f"   ⚠️ 응답 코드: {response.status_code}")
                    continue
                
                content = response.content.decode('utf-8')
                sep = source.get('sep', ',')
                df = pd.read_csv(io.StringIO(content), sep=sep)
                
                # 소스별 컬럼 처리
                tickers = []
                if 'ticker' in df.columns:
                    tickers = df['ticker'].tolist()
                elif 'Symbol' in df.columns:
                    if 'ETF' in df.columns:
                        df = df[df['ETF'] == 'Y']
                    tickers = df['Symbol'].tolist()
                elif 'symbol' in df.columns:
                    tickers = df['symbol'].tolist()
                
                # 정제
                clean_tickers = [str(t).strip() for t in tickers if str(t).isalpha()]
                
                if clean_tickers:
                    print(f"   ✅ {len(clean_tickers)}개 티커 발견")
                    return clean_tickers
                    
            except Exception as e:
                print(f"   ⚠️ 다운로드 실패: {e}")
        
        print("⚠️ 모든 소스 실패. 기본 리스트 사용.")
        return DEFAULT_ETFS.copy()
    
    def filter_by_market_cap(self, tickers: List[str], threshold: int = 300_000_000) -> List[str]:
        """
        시가총액 기준으로 ETF를 필터링합니다.
        
        Args:
            tickers: List of ticker symbols
            threshold: Minimum market cap in USD (default: 300M)
        
        Returns:
            List of tickers meeting the threshold
        """
        print(f"\n🔍 시가총액 {threshold:,} USD 이상 필터링 ({len(tickers)}개)...")
        
        qualified = []
        chunk_size = 50
        
        for i in range(0, len(tickers), chunk_size):
            chunk = tickers[i:i+chunk_size]
            print(f"   [{i+len(chunk)}/{len(tickers)}] 처리 중... (선정: {len(qualified)})", end='\r')
            
            try:
                tickers_obj = yf.Tickers(" ".join(chunk))
                
                for symbol in chunk:
                    try:
                        ticker = tickers_obj.tickers[symbol]
                        mc = self._get_market_cap(ticker)
                        
                        if mc and mc >= threshold:
                            qualified.append(symbol)
                    except:
                        continue
            except:
                pass
        
        print(f"\n✅ 필터링 완료: {len(qualified)}개 ETF 선정")
        return qualified
    
    def _get_market_cap(self, ticker_obj) -> Optional[float]:
        """yfinance 객체에서 시가총액 추출"""
        # fast_info 시도
        try:
            mc = ticker_obj.fast_info.market_cap
            if mc and mc > 0:
                return mc
        except:
            pass
        
        try:
            mc = ticker_obj.fast_info['market_cap']
            if mc and mc > 0:
                return mc
        except:
            pass
        
        # info 시도
        try:
            mc = ticker_obj.info.get('marketCap')
            if mc and mc > 0:
                return mc
        except:
            pass
        
        return None
    
    # =========================================================================
    # Leveraged/Inverse ETF Filter (from recalculate_yields.py)
    # =========================================================================
    
    def is_leveraged_etf(self, ticker: str, name: str = "") -> bool:
        """
        레버리지/인버스 ETF 여부를 판별합니다.
        
        Args:
            ticker: ETF ticker symbol
            name: ETF name (optional)
        
        Returns:
            True if leveraged/inverse ETF
        """
        if ticker in LEVERAGED_TICKERS:
            return True
        
        if name:
            for pattern in LEVERAGED_PATTERNS:
                if re.search(pattern, name, re.IGNORECASE):
                    return True
            
            # Direxion Daily는 항상 레버리지
            if 'Direxion' in name and ('Daily' in name or 'Ultra' in name or 'Short' in name):
                return True
        
        return False
    
    def filter_leveraged_etfs(self, summary: List[Dict]) -> Tuple[List[Dict], List[str]]:
        """
        레버리지/인버스 ETF를 제거합니다.
        
        Args:
            summary: List of ETF summary dictionaries
        
        Returns:
            Tuple of (filtered_summary, removed_tickers)
        """
        filtered = []
        removed = []
        
        for item in summary:
            ticker = item.get('Ticker', item.get('ticker', ''))
            name = item.get('Name', '')
            
            if self.is_leveraged_etf(ticker, name):
                removed.append(ticker)
            else:
                filtered.append(item)
        
        if removed:
            print(f"🚫 레버리지/인버스 ETF {len(removed)}개 제거: {', '.join(removed[:10])}...")
        
        return filtered, removed
    
    # =========================================================================
    # Dividend Yield Calculation (from recalculate_yields.py)
    # =========================================================================
    
    def infer_dividend_frequency(self, schedule: List[Dict]) -> int:
        """
        배당 스케줄에서 빈도를 추론합니다.
        
        Args:
            schedule: List of dividend payments with 'date' keys
        
        Returns:
            Annualization factor (12=monthly, 4=quarterly, 2=semi, 1=annual, 0=none)
        """
        if not schedule:
            return 0
        
        # 최근 12개월 배당 필터링
        now = datetime.now()
        one_year_ago = now - timedelta(days=365)
        
        recent = []
        for div in schedule:
            try:
                div_date = datetime.strptime(div['date'], '%Y-%m-%d')
                if div_date >= one_year_ago:
                    recent.append(div_date)
            except:
                pass
        
        count = len(recent)
        
        if count >= 11:
            return 12  # Monthly
        elif count >= 4:
            return 4   # Quarterly
        elif count >= 2:
            return 2   # Semi-annual
        elif count >= 1:
            return 1   # Annual
        
        # 날짜 간격으로 추론
        if len(schedule) >= 2:
            try:
                dates = sorted(
                    [datetime.strptime(d['date'], '%Y-%m-%d') for d in schedule],
                    reverse=True
                )
                gaps = [(dates[i] - dates[i+1]).days for i in range(min(4, len(dates)-1))]
                avg_gap = sum(gaps) / len(gaps) if gaps else 365
                
                if avg_gap < 45:
                    return 12
                elif avg_gap < 120:
                    return 4
                elif avg_gap < 250:
                    return 2
            except:
                pass
        
        return 1
    
    def calculate_dividend_yield(
        self, 
        schedule: List[Dict], 
        price_series: pd.Series
    ) -> Tuple[float, str]:
        """
        정확한 배당 수익률을 계산합니다.
        
        Formula: Yield = (Dividend / Price_at_ExDiv) × Frequency × 100
        
        Args:
            schedule: List of dividend payments [{'date': ..., 'amount': ...}]
            price_series: Price series with DatetimeIndex
        
        Returns:
            Tuple of (yield_percent, detail_message)
        """
        if not schedule:
            return 0.0, "No dividends"
        
        if price_series is None or len(price_series) < 10:
            return 0.0, "Insufficient price data"
        
        # 최근 배당 (스케줄은 내림차순 정렬 가정)
        recent_div = schedule[0]
        div_date = recent_div['date']
        div_amount = recent_div.get('amount', 0)
        
        if div_amount <= 0:
            return 0.0, "Invalid dividend amount"
        
        # 배당락일 2영업일 전 가격
        price_at_exdiv = self._get_price_before_date(price_series, div_date, 2)
        
        if price_at_exdiv is None or price_at_exdiv <= 0:
            price_at_exdiv = price_series.iloc[-1]
        
        # 빈도 추론
        freq = self.infer_dividend_frequency(schedule)
        
        if freq == 0:
            return 0.0, "Unknown frequency"
        
        # 연환산 수익률 계산
        single_yield = div_amount / price_at_exdiv
        annual_yield = single_yield * freq * 100
        
        # 50% 상한 (비정상적으로 높은 수익률 제한)
        if annual_yield > 50:
            # 최근 4개 배당 평균 사용
            recent_divs = [d['amount'] for d in schedule[:min(4, len(schedule))]]
            avg_div = sum(recent_divs) / len(recent_divs)
            annual_yield = (avg_div / price_at_exdiv) * freq * 100
            
            if annual_yield > 50:
                detail = f"Capped from {annual_yield:.1f}%"
                annual_yield = 50.0
            else:
                detail = f"Used avg of {len(recent_divs)} divs"
        else:
            detail = f"Freq={freq}, Div=${div_amount:.4f}, Price=${price_at_exdiv:.2f}"
        
        return round(annual_yield, 2), detail
    
    def _get_price_before_date(
        self, 
        price_series: pd.Series, 
        target_date: str, 
        business_days: int = 2
    ) -> Optional[float]:
        """N 영업일 전 가격 조회"""
        try:
            target = pd.to_datetime(target_date)
            valid_dates = price_series.index[price_series.index < target]
            
            if len(valid_dates) < business_days:
                return None
            
            return price_series.loc[valid_dates[-business_days]]
        except:
            return None
    
    # =========================================================================
    # Dividend Schedule Collection (from patch_metadata.py)
    # =========================================================================
    
    def fetch_dividend_schedule(self, ticker: str, limit: int = 12) -> List[Dict]:
        """
        yfinance에서 배당 스케줄을 수집합니다.
        
        Args:
            ticker: ETF ticker symbol
            limit: Maximum number of recent dividends to return
        
        Returns:
            List of dividend payments [{'date': ..., 'amount': ...}]
        """
        try:
            tik = yf.Ticker(ticker)
            divs = tik.dividends
            
            if divs.empty:
                return []
            
            # 최근 N개 배당, 내림차순
            recent = divs.sort_index(ascending=False).head(limit)
            
            schedule = []
            for d_date, d_amt in recent.items():
                schedule.append({
                    "date": d_date.strftime('%Y-%m-%d'),
                    "amount": round(float(d_amt), 4)
                })
            
            return schedule
            
        except Exception as e:
            return []
    
    def get_dividend_data(self, tickers: List[str]) -> List[Dict]:
        """
        여러 ETF의 배당 정보를 수집합니다.
        
        Args:
            tickers: List of ticker symbols
        
        Returns:
            List of dividend data dictionaries
        """
        print(f"\n💰 {len(tickers)}개 ETF 배당 정보 수집 중...")
        results = []
        
        for i, symbol in enumerate(tickers):
            print(f"   [{i+1}/{len(tickers)}] {symbol}...", end='\r')
            
            try:
                ticker = yf.Ticker(symbol)
                
                # Yield
                yield_val = 0
                try:
                    yield_val = ticker.info.get('dividendYield', 0)
                    if yield_val is None:
                        yield_val = 0
                except:
                    pass
                
                # Schedule
                schedule = self.fetch_dividend_schedule(symbol)
                
                results.append({
                    'ticker': symbol,
                    'dividend_yield': yield_val,
                    'dividend_schedule': schedule,
                    'last_updated': datetime.now().strftime('%Y-%m-%d')
                })
                
            except:
                pass
        
        print(f"\n✅ 배당 정보 수집 완료: {len(results)}개")
        return results
    
    # =========================================================================
    # Data Loading
    # =========================================================================
    
    def load_tickers_from_file(self, file_path: str) -> List[Dict]:
        """Load the base metadata from JSON file."""
        if not os.path.exists(file_path):
            print(f"Error: File {file_path} not found.")
            return []
        
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            print(f"Loaded {len(data)} items from {file_path}")
            return data
    
    # =========================================================================
    # Market Data Download
    # =========================================================================
    
    def download_data_batch(self, tickers: List[str], chunk_size=100, period="5y") -> pd.DataFrame:
        """
        Download data in batches to handle large lists of tickers.
        """
        all_dfs = []
        total_tickers = len(tickers)
        
        print(f"📊 {total_tickers}개 티커 다운로드 시작 (청크 크기: {chunk_size})...")
        
        for i in range(0, total_tickers, chunk_size):
            chunk = tickers[i : i + chunk_size]
            print(f"   청크 {i//chunk_size + 1}/{(total_tickers + chunk_size - 1) // chunk_size} ({len(chunk)}개)...")
            
            try:
                df = yf.download(chunk, period=period, group_by='ticker', threads=True, progress=False)
                
                if not df.empty:
                    all_dfs.append(df)
                
                time.sleep(1.0)
                
            except Exception as e:
                print(f"   ⚠️ 청크 {i} 오류: {e}")
        
        if not all_dfs:
            return pd.DataFrame()
        
        print("   데이터 병합 중...")
        full_df = pd.concat(all_dfs, axis=1)
        print(f"✅ 다운로드 완료. Shape: {full_df.shape}")
        
        return full_df
    
    # =========================================================================
    # Market Data Processing
    # =========================================================================
    
    def process_market_data(self, price_df: pd.DataFrame, file_name='market_data.parquet') -> pd.DataFrame:
        """
        Save formatted market data to Parquet.
        Transforms (Date, Ticker-Levels) -> (Date, Ticker-Attributes) wide format.
        """
        print("📁 마켓 데이터 처리 중...")
        
        try:
            if not isinstance(price_df.columns, pd.MultiIndex):
                print("Warning: DataFrame is not MultiIndex.")
                return price_df
            
            def get_level_df(df, col_name):
                if col_name in df.columns.get_level_values(1):
                    return df.xs(col_name, axis=1, level=1, drop_level=True)
                return pd.DataFrame(index=df.index)
            
            adj_close = get_level_df(price_df, 'Adj Close')
            close = get_level_df(price_df, 'Close')
            
            # Adj Close 우선, 없으면 Close 사용
            best_price = adj_close.combine_first(close)
            
            # 수익률 계산
            returns_df = best_price.pct_change()
            
            # MultiIndex 재구성
            if not returns_df.empty:
                tuples = [(ticker, 'Daily_Return') for ticker in returns_df.columns]
                returns_cols = pd.MultiIndex.from_tuples(tuples, names=price_df.columns.names)
                returns_df.columns = returns_cols
            
            # 원본 데이터 필터링 (Close, Adj Close만)
            cols_to_keep = []
            if 'Close' in price_df.columns.get_level_values(1):
                cols_to_keep.append('Close')
            if 'Adj Close' in price_df.columns.get_level_values(1):
                cols_to_keep.append('Adj Close')
            
            idx = pd.IndexSlice
            base_data = price_df.loc[:, idx[:, cols_to_keep]]
            
            # 병합
            final_df = pd.concat([base_data, returns_df], axis=1)
            final_df = final_df.sort_index(axis=1, level=0)
            
            # 저장
            parquet_path = os.path.join(self.output_dir, file_name)
            final_df.to_parquet(parquet_path, engine='pyarrow', compression='snappy')
            
            print(f"✅ 저장 완료: {parquet_path} (Shape: {final_df.shape})")
            return final_df
            
        except Exception as e:
            print(f"⚠️ 마켓 데이터 처리 오류: {e}")
            import traceback
            traceback.print_exc()
            return price_df
    
    # =========================================================================
    # Covariance Matrix (enhanced from recalculate_covariance.py)
    # =========================================================================
    
    def process_covariance(self, market_data_df: pd.DataFrame, window_days=252) -> List[str]:
        """
        Calculate Covariance Matrix using Adj Close for Total Return.
        Returns list of valid tickers that were included.
        """
        print("📊 공분산 행렬 계산 중 (Adj Close 기반)...")
        
        try:
            # Adj Close 추출 (Total Return 반영)
            if isinstance(market_data_df.columns, pd.MultiIndex):
                if 'Adj Close' in market_data_df.columns.get_level_values(1):
                    price_df = market_data_df.xs('Adj Close', axis=1, level=1)
                elif 'Close' in market_data_df.columns.get_level_values(1):
                    print("   ⚠️ Adj Close 없음, Close 사용")
                    price_df = market_data_df.xs('Close', axis=1, level=1)
                else:
                    print("   ❌ 가격 데이터 없음")
                    return []
            else:
                price_df = market_data_df
            
            # 수익률 계산
            returns = price_df.pct_change()
            
            # 최근 N일 필터
            recent_returns = returns.tail(window_days)
            
            # NaN 없는 티커만 사용
            valid_returns = recent_returns.dropna(axis=1, how='any')
            
            if valid_returns.empty:
                print("   ❌ 유효한 티커 없음")
                return []
            
            # 공분산 계산
            cov_matrix = valid_returns.cov()
            
            # 저장
            npy_path = os.path.join(self.output_dir, 'covariance.npy')
            np.save(npy_path, cov_matrix.values)
            
            valid_tickers = valid_returns.columns.tolist()
            print(f"✅ 공분산 저장: {npy_path} (Shape: {cov_matrix.shape}, {len(valid_tickers)}개 티커)")
            
            return valid_tickers
            
        except Exception as e:
            print(f"⚠️ 공분산 계산 오류: {e}")
            return []
    
    # =========================================================================
    # Metrics Calculation
    # =========================================================================
    
    def calculate_metrics_for_ticker(self, ticker_series: pd.Series) -> Dict[str, float]:
        """
        Calculate singular metrics for a price series.
        """
        try:
            series = ticker_series.dropna()
            if len(series) < 30:
                return {}
            
            start_price = series.iloc[0]
            end_price = series.iloc[-1]
            
            # CAGR
            days = (series.index[-1] - series.index[0]).days
            years = days / 365.25
            cagr = 0.0
            if years > 0 and start_price > 0:
                cagr = (end_price / start_price) ** (1 / years) - 1
            
            # Volatility
            daily_ret = series.pct_change().dropna()
            volatility = daily_ret.std() * np.sqrt(252)
            
            # Max Drawdown
            peak = series.cummax()
            drawdown = (series - peak) / peak
            max_drawdown = drawdown.min()
            
            return {
                "cagr_price_5y": round(cagr, 4),
                "volatility": round(volatility, 4),
                "max_drawdown": round(max_drawdown, 4)
            }
        except:
            return {}
    
    # =========================================================================
    # Summary Generation
    # =========================================================================
    
    def process_full_summary(
        self, 
        valid_tickers_for_cov: List[str], 
        price_df: pd.DataFrame, 
        original_data: List[Dict],
        file_name='etf_summary.json'
    ):
        """
        Generate final summary JSON with accurate dividend yields.
        """
        print("📋 Summary 생성 중...")
        
        # 원본 데이터 맵
        orig_map = {item.get('ticker', item.get('Ticker', '')): item for item in original_data}
        
        summary_list = []
        
        for idx, ticker in enumerate(valid_tickers_for_cov):
            if idx % 50 == 0:
                print(f"   [{idx}/{len(valid_tickers_for_cov)}] 처리 중...")
            
            base_obj = orig_map.get(ticker)
            if not base_obj:
                continue
            
            try:
                # 가격 시리즈 추출
                if isinstance(price_df.columns, pd.MultiIndex):
                    if (ticker, 'Adj Close') in price_df.columns:
                        series = price_df[(ticker, 'Adj Close')]
                    elif (ticker, 'Close') in price_df.columns:
                        series = price_df[(ticker, 'Close')]
                    else:
                        series = pd.Series()
                else:
                    series = price_df.get(ticker, pd.Series())
                
                # 메트릭 계산
                metrics = self.calculate_metrics_for_ticker(series)
                
                # 배당 스케줄
                schedule = base_obj.get('dividend_schedule', base_obj.get('Dividend Schedule Summary', []))
                
                # 정확한 배당 수익률 계산
                div_yield, _ = self.calculate_dividend_yield(schedule, series)
                
                # 기존 수익률이 있고 유효하면 사용, 아니면 계산된 값 사용
                base_yield = base_obj.get('dividend_yield', 0)
                if base_yield and 0 < base_yield < 100:
                    final_yield = base_yield
                else:
                    final_yield = div_yield
                
                new_entry = {
                    "Ticker": ticker,
                    "Name": base_obj.get("Name", base_obj.get("ticker", ticker)),
                    "Key Metrics": {
                        "current_dividend_yield": final_yield,
                        **metrics
                    },
                    "Dividend Schedule Summary": schedule
                }
                
                summary_list.append(new_entry)
                
            except Exception as e:
                print(f"   ⚠️ {ticker} 처리 오류: {e}")
        
        # 레버리지 ETF 필터링
        summary_list, removed = self.filter_leveraged_etfs(summary_list)
        
        # 저장
        json_path = os.path.join(self.output_dir, file_name)
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(summary_list, f, indent=4, ensure_ascii=False)
        
        print(f"✅ Summary 저장: {json_path} ({len(summary_list)}개)")
    
    # =========================================================================
    # Full Pipeline
    # =========================================================================
    
    def run_full_pipeline(
        self,
        market_cap_threshold: int = 300_000_000,
        period: str = "5y",
        use_cache: bool = False
    ) -> None:
        """
        전체 데이터 수집 및 처리 파이프라인을 실행합니다.
        
        Args:
            market_cap_threshold: Minimum market cap for ETF filtering
            period: Historical data period (e.g., "5y", "3y")
            use_cache: If True, skip download and use existing data
        """
        print("=" * 60)
        print("🚀 Financial Data Pipeline 시작")
        print("=" * 60)
        
        # 1. ETF 유니버스 수집
        print("\n[1/5] ETF 유니버스 수집...")
        etfs = self.get_etf_universe()
        
        # 2. 시가총액 필터링
        print("\n[2/5] 시가총액 필터링...")
        filtered_etfs = self.filter_by_market_cap(etfs, market_cap_threshold)
        
        if not filtered_etfs:
            print("❌ 조건을 만족하는 ETF가 없습니다.")
            return
        
        # 3. 배당 데이터 수집
        print("\n[3/5] 배당 데이터 수집...")
        dividend_data = self.get_dividend_data(filtered_etfs)
        
        # 4. 가격 데이터 다운로드
        print("\n[4/5] 가격 데이터 다운로드...")
        tickers = [d['ticker'] for d in dividend_data]
        price_df = self.download_data_batch(tickers, period=period)
        
        if price_df.empty:
            print("❌ 가격 데이터 다운로드 실패")
            return
        
        # 5. 데이터 처리 및 저장
        print("\n[5/5] 데이터 처리 및 저장...")
        market_data = self.process_market_data(price_df)
        valid_tickers = self.process_covariance(market_data)
        self.process_full_summary(valid_tickers, price_df, dividend_data)
        
        print("\n" + "=" * 60)
        print("✅ 파이프라인 완료!")
        print("=" * 60)


if __name__ == "__main__":
    import sys
    from db import db_manager
    
    # CLI 인자 처리
    if len(sys.argv) > 1 and sys.argv[1] == '--update-prices':
        # MongoDB 업데이트 모드
        print("=" * 60)
        print("📊 해외 배당 ETF 데이터 MongoDB 업데이트")
        print("=" * 60)
        
        # 1. etf_summary.json → dividend_etf_summary
        summary_path = 'dividend_data/etf_summary.json'
        if os.path.exists(summary_path):
            print(f"\n[1/2] {summary_path} → dividend_etf_summary...")
            with open(summary_path, 'r', encoding='utf-8') as f:
                summary_data = json.load(f)
            
            # 컬렉션 비우고 새로 삽입
            db_manager.dividend_etf_summary.delete_many({})
            if summary_data:
                db_manager.dividend_etf_summary.insert_many(summary_data)
            print(f"   ✅ {len(summary_data)}개 ETF 정보 업로드 완료")
        else:
            print(f"   ⚠️ {summary_path} 파일 없음")
        
        # 2. market_data.parquet → dividend_etf_prices
        parquet_path = 'dividend_data/market_data.parquet'
        if os.path.exists(parquet_path):
            print(f"\n[2/2] {parquet_path} → dividend_etf_prices...")
            df = pd.read_parquet(parquet_path)
            
            # MultiIndex 처리
            if isinstance(df.columns, pd.MultiIndex):
                # 티커별 가격 데이터 추출
                tickers = df.columns.get_level_values(0).unique()
                
                db_manager.dividend_etf_prices.delete_many({})
                uploaded = 0
                
                for ticker in tickers:
                    try:
                        ticker_data = df[ticker].copy()
                        if 'Adj Close' in ticker_data.columns:
                            prices = ticker_data[['Adj Close', 'Close']].dropna(how='all')
                        elif 'Close' in ticker_data.columns:
                            prices = ticker_data[['Close']].dropna(how='all')
                        else:
                            continue
                        
                        if prices.empty:
                            continue
                        
                        prices = prices.reset_index()
                        prices['Date'] = prices['Date'].dt.strftime('%Y-%m-%d')
                        prices_list = prices.to_dict('records')
                        
                        doc = {
                            'ticker': ticker,
                            'prices': prices_list,
                            'updated_at': datetime.now()
                        }
                        db_manager.dividend_etf_prices.insert_one(doc)
                        uploaded += 1
                        
                    except Exception as e:
                        print(f"   ⚠️ {ticker} 처리 오류: {e}")
                
                print(f"   ✅ {uploaded}개 ETF 가격 데이터 업로드 완료")
            else:
                print("   ⚠️ MultiIndex 형식이 아닙니다")
        else:
            print(f"   ⚠️ {parquet_path} 파일 없음")
        
        print("\n" + "=" * 60)
        print("✅ MongoDB 업데이트 완료!")
        print("=" * 60)
    else:
        # 기본 모드: 전체 파이프라인 실행
        manager = FinancialDataManager(output_dir='dividend_data')
        manager.run_full_pipeline()
