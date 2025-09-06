import pandas as pd
import numpy as np
from scipy.stats import norm
from typing import Dict, List, Tuple
import os
import json


class PortfolioAnalytics:
    """포트폴리오 분석을 위한 핵심 계산 클래스"""
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = data_dir
        self.price_data_dir = os.path.join(data_dir, "fund_prices")
        
    def load_price_data(self, tickers: List[str]) -> pd.DataFrame:
        """티커 리스트로부터 가격 데이터를 로드"""
        all_prices = []
        
        for ticker in tickers:
            file_path = os.path.join(self.price_data_dir, f"{ticker}.csv")
            if os.path.exists(file_path):
                try:
                    df = pd.read_csv(
                        file_path, 
                        skiprows=3, 
                        header=None,
                        names=['Date', 'Close', 'High', 'Low', 'Open', 'Volume'],
                        index_col='Date',
                        parse_dates=True,
                        on_bad_lines='skip'
                    )
                    
                    df = df[df.index.notna()]
                    df = df[~df.index.duplicated(keep='first')]
                    df['Adj Close'] = df['Close']
                    
                    if not df.empty and 'Adj Close' in df.columns:
                        price_series = df[['Adj Close']].rename(columns={'Adj Close': ticker})
                        all_prices.append(price_series)
                        
                except Exception:
                    continue
        
        if not all_prices:
            raise ValueError("유효한 가격 데이터를 가진 ETF가 없습니다.")
            
        price_df = pd.concat(all_prices, axis=1, join='outer', sort=True)
        return self.clean_price_data(price_df)
    
    def clean_price_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """가격 데이터 정제"""
        # 무한값을 NaN으로 변환
        df = df.replace([np.inf, -np.inf], np.nan)
        
        # 0 또는 음수 가격을 NaN으로 변환
        df = df.where(df > 0, np.nan)
        
        # 극단적인 일일 수익률 제거 (±30% 초과)
        returns = df.pct_change()
        outlier_mask = (returns.abs() > 0.3)
        df = df.where(~outlier_mask, np.nan)
        
        # forward fill로 단기 결측값 보완 (최대 3일)
        df = df.ffill(limit=3)
        
        # 연속된 NaN이 너무 많은 행 제거
        df = df.dropna(thresh=len(df.columns) * 0.8)
        
        return df.dropna()
    
    def calculate_expected_returns(self, prices: pd.DataFrame) -> pd.Series:
        """기대수익률 계산 (연율화)"""
        returns = prices.pct_change().dropna()
        mean_returns = returns.mean() * 252  # 연율화
        return mean_returns.replace([np.inf, -np.inf], np.nan).fillna(0)
    
    def calculate_covariance_matrix(self, prices: pd.DataFrame) -> pd.DataFrame:
        """공분산 행렬 계산 (연율화)"""
        returns = prices.pct_change().dropna()
        return returns.cov() * 252  # 연율화
    
    def calculate_correlation_matrix(self, prices: pd.DataFrame) -> pd.DataFrame:
        """상관계수 행렬 계산"""
        returns = prices.pct_change().dropna()
        return returns.corr()
    
    def calculate_volatility(self, prices: pd.DataFrame) -> pd.Series:
        """변동성 계산 (연율화)"""
        returns = prices.pct_change().dropna()
        return returns.std() * np.sqrt(252)  # 연율화
    
    def calculate_sharpe_ratio(self, expected_returns: pd.Series, volatility: pd.Series, 
                             risk_free_rate: float = 0.02) -> pd.Series:
        """샤프 비율 계산"""
        return (expected_returns - risk_free_rate) / volatility
    
    def calculate_portfolio_metrics(self, weights: Dict[str, float], prices: pd.DataFrame, 
                                  risk_free_rate: float = 0.02) -> Dict[str, float]:
        """포트폴리오 성과지표 계산"""
        try:
            # 일일 수익률 계산
            daily_returns = prices.pct_change().dropna()
            
            # 이상값 제거
            daily_returns = daily_returns.replace([np.inf, -np.inf], np.nan)
            daily_returns = daily_returns[daily_returns.abs() <= 0.3]
            daily_returns = daily_returns.dropna()
            
            if daily_returns.empty or len(daily_returns) < 10:
                return {
                    "expected_annual_return": 0.0,
                    "annual_volatility": 0.1,
                    "sharpe_ratio": 0.0
                }
            
            # 포트폴리오 일일 수익률
            portfolio_daily_returns = sum(weights.get(col, 0) * daily_returns[col] 
                                        for col in daily_returns.columns 
                                        if col in weights and not pd.isna(daily_returns[col]).all())
            
            if isinstance(portfolio_daily_returns, pd.Series):
                portfolio_daily_returns = portfolio_daily_returns.dropna()
            
            if len(portfolio_daily_returns) == 0:
                return {
                    "expected_annual_return": 0.0,
                    "annual_volatility": 0.1,
                    "sharpe_ratio": 0.0
                }
            
            # 연율화 계산 (252 거래일 기준)
            mean_daily_return = np.mean(portfolio_daily_returns)
            daily_vol = np.std(portfolio_daily_returns, ddof=1)
            
            # NaN 또는 무한값 체크
            if np.isnan(mean_daily_return) or np.isinf(mean_daily_return):
                mean_daily_return = 0.0
            if np.isnan(daily_vol) or np.isinf(daily_vol) or daily_vol <= 0:
                daily_vol = 0.01
            
            # 연율화
            annual_return = mean_daily_return * 252 + risk_free_rate
            annual_vol = daily_vol * np.sqrt(252)
            
            # 변동성 상한선 설정 (100%)
            annual_vol = min(annual_vol, 1.0)
            
            # 샤프 비율
            sharpe = (annual_return - risk_free_rate) / annual_vol if annual_vol > 0 else 0.0
            
            return {
                "expected_annual_return": float(annual_return),
                "annual_volatility": float(annual_vol),
                "sharpe_ratio": float(sharpe)
            }
            
        except Exception as e:
            print(f"포트폴리오 성과 계산 오류: {str(e)}")
            return {
                "expected_annual_return": 0.0,
                "annual_volatility": 0.1,
                "sharpe_ratio": 0.0
            }
    
    def calculate_var(self, annual_return: float, annual_vol: float, 
                     confidence_levels: List[float] = [0.99, 0.95, 0.90]) -> Dict:
        """Value at Risk 계산"""
        var_results = {}
        
        for confidence in confidence_levels:
            z_score = norm.ppf(1 - confidence)
            var_1year = annual_return + z_score * annual_vol
            
            confidence_pct = int((1 - confidence) * 100)
            var_results[f"var_{confidence_pct}pct"] = {
                "confidence_level": f"{confidence_pct}%",
                "var_1year": round(var_1year * 100, 2),
                "var_1year_display": f"{var_1year:.1%}"
            }
        
        return {
            "portfolio_return": round(annual_return * 100, 2),
            "portfolio_volatility": round(annual_vol * 100, 2),
            "var_calculations": var_results
        }
    
    def calculate_shortfall_risk(self, annual_return: float, annual_vol: float, 
                               max_years: int = 20) -> Dict:
        """Shortfall Risk 계산"""
        shortfall_results = []
        
        for years in range(1, max_years + 1):
            expected_return_n_years = annual_return * years
            volatility_n_years = annual_vol * np.sqrt(years)
            
            if volatility_n_years > 0:
                z_score = (0 - expected_return_n_years) / volatility_n_years
                loss_probability = norm.cdf(z_score)
            else:
                loss_probability = 0.0 if expected_return_n_years > 0 else 1.0
            
            shortfall_results.append({
                "years": years,
                "expected_return": round(expected_return_n_years * 100, 2),
                "volatility": round(volatility_n_years * 100, 2),
                "loss_probability": round(loss_probability * 100, 2),
                "loss_probability_display": f"{loss_probability:.1%}"
            })
        
        return {
            "portfolio_annual_return": round(annual_return * 100, 2),
            "portfolio_annual_volatility": round(annual_vol * 100, 2),
            "shortfall_risk_by_years": shortfall_results
        }