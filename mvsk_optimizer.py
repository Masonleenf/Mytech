"""
MVSK Portfolio Optimizer
========================
Higher-Order Moment Portfolio Optimization using Mean-Variance-Skewness-Kurtosis Framework.

This implementation is designed for portfolios containing assets with asymmetric payoff profiles
(e.g., Covered Call ETFs) where traditional MVO fails to capture tail risks.

Mathematical Framework:
    max_w [ α · U_MVSK(w) + (1 - α) · Y(w) ]
    
    U_MVSK(w) = E(Rp) - (λ1/2)·σp² + (λ2/3)·Sp - (λ3/4)·Kp

Author: Quantitative Portfolio Engine
Version: 1.0
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm, skew, kurtosis
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')


# =============================================================================
# Data Classes for Clean Output
# =============================================================================

@dataclass
class PortfolioMoments:
    """Container for portfolio moment statistics."""
    mean: float           # Annualized expected return
    variance: float       # Annualized variance
    volatility: float     # Annualized volatility (std dev)
    skewness: float       # Portfolio skewness
    kurtosis: float       # Portfolio excess kurtosis


@dataclass
class RiskMetrics:
    """Container for risk metrics."""
    sharpe_ratio: float           # Traditional Sharpe
    adjusted_sharpe_ratio: float  # Cornish-Fisher adjusted
    modified_var_95: float        # Modified VaR at 95%
    modified_cvar_95: float       # Modified CVaR at 95%


@dataclass
class OptimizationResult:
    """Complete optimization result."""
    weights: Dict[str, float]
    moments: PortfolioMoments
    risk_metrics: RiskMetrics
    dividend_yield: float
    mvsk_utility: float
    composite_score: float
    success: bool
    message: str


# =============================================================================
# Moment Calculator Module
# =============================================================================

class MomentCalculator:
    """
    Calculates higher-order co-moments for portfolio optimization.
    
    Computes:
        - Mean vector (μ) with James-Stein Shrinkage for recency bias reduction
        - Covariance matrix (Σ) - N x N
        - Co-skewness matrix (M3) - N x N²
        - Co-kurtosis matrix (M4) - N x N³
    
    References:
        Jondeau, E., & Rockinger, M. (2006). "Optimal Portfolio Allocation
        under Higher Moments." European Financial Management.
        
        James, W. & Stein, C. (1961). "Estimation with Quadratic Loss."
    """
    
    def __init__(
        self, 
        returns: pd.DataFrame, 
        annualization_factor: int = 252,
        shrinkage_intensity: float = 0.5,
        target_return: float = 0.08,
        hard_cap: float = 0.15
    ):
        """
        Initialize with historical returns data.
        
        Args:
            returns: DataFrame of daily returns (T x N), columns are asset names.
            annualization_factor: Trading days per year (default 252).
            shrinkage_intensity: James-Stein shrinkage factor (0=raw, 1=all market mean).
            target_return: Conservative market equilibrium return assumption (default 8%).
            hard_cap: Absolute maximum allowed return before shrinkage (default 15%).
        """
        self.returns = returns.values if isinstance(returns, pd.DataFrame) else returns
        self.asset_names = list(returns.columns) if isinstance(returns, pd.DataFrame) else None
        self.T, self.N = self.returns.shape
        self.ann_factor = annualization_factor
        
        # James-Stein parameters
        self.shrinkage_intensity = shrinkage_intensity
        self.target_return = target_return
        self.hard_cap = hard_cap
        
        # Demean returns for moment calculations
        self.means = np.mean(self.returns, axis=0)
        self.demeaned = self.returns - self.means
        
        # Cache computed moments
        self._cov_matrix = None
        self._coskew_matrix = None
        self._cokurt_matrix = None
        self._adjusted_mu = None  # Cache for adjusted returns
    
    def calculate_adjusted_returns(self) -> np.ndarray:
        """
        Calculate expected returns using James-Stein Shrinkage + Hard Capping.
        
        This reduces recency bias by pulling extreme HIGH returns towards a conservative
        market equilibrium return (8%), while hard-capping at 15% as a safety brake.
        
        IMPORTANT: "Do Not Shrink Upwards" Principle
        - Only apply shrinkage if raw_mu > target_return (overperforming assets)
        - If raw_mu <= target_return (underperforming/decaying assets), keep raw_mu as is
        - This prevents negative drift assets from incorrectly looking positive
        
        Mathematical Framework:
            If x̄ > μ_target:
                μ̂_JS = (1 - w) · min(x̄, hard_cap) + w · μ_target
            Else:
                μ̂_JS = x̄  (keep as-is, conservative principle)
        
        Returns:
            np.ndarray: Adjusted expected returns (annualized).
        """
        if self._adjusted_mu is not None:
            return self._adjusted_mu
        
        # Step 1: Calculate raw annualized returns
        raw_mu = self.means * self.ann_factor
        
        # Step 2: Initialize adjusted returns as copy of raw
        adj_mu = raw_mu.copy()
        
        # Step 3: Apply Hard Cap + Shrinkage ONLY to overperformers
        # "Do Not Shrink Upwards" - only shrink if raw > target
        for i in range(len(raw_mu)):
            if raw_mu[i] > self.target_return:
                # Cap at hard_cap first (e.g., 15%)
                capped = min(raw_mu[i], self.hard_cap)
                # Then shrink towards target (8%)
                adj_mu[i] = (1 - self.shrinkage_intensity) * capped + self.shrinkage_intensity * self.target_return
            # else: keep raw_mu[i] as-is (underperforming/decaying assets)
        
        self._adjusted_mu = adj_mu
        return adj_mu
    
    def get_mean_vector(self, use_shrinkage: bool = True) -> np.ndarray:
        """
        Return annualized mean returns.
        
        Args:
            use_shrinkage: If True, apply James-Stein shrinkage (recommended).
                          If False, return raw historical returns.
        
        Returns:
            np.ndarray: Annualized expected returns.
        """
        if use_shrinkage:
            return self.calculate_adjusted_returns()
        return self.means * self.ann_factor
    
    def get_raw_mean_vector(self) -> np.ndarray:
        """Return raw annualized mean returns without shrinkage (for diagnostics)."""
        return self.means * self.ann_factor
    
    def get_covariance_matrix(self) -> np.ndarray:
        """Return annualized covariance matrix (N x N)."""
        if self._cov_matrix is None:
            self._cov_matrix = np.cov(self.returns, rowvar=False) * self.ann_factor
        return self._cov_matrix
    
    def get_coskewness_matrix(self) -> np.ndarray:
        """
        Calculate co-skewness matrix M3 of shape (N, N²).
        
        M3[i, j*N + k] = E[(ri - μi)(rj - μj)(rk - μk)] / (σi·σj·σk)
        
        This is the "unfolded" co-skewness tensor for efficient computation.
        """
        if self._coskew_matrix is None:
            N = self.N
            T = self.T
            
            # Standardize returns
            stds = np.std(self.returns, axis=0, ddof=1)
            stds[stds == 0] = 1e-10  # Avoid division by zero
            standardized = self.demeaned / stds
            
            # Initialize co-skewness matrix
            M3 = np.zeros((N, N * N))
            
            # Compute co-skewness elements
            for i in range(N):
                for j in range(N):
                    for k in range(N):
                        # E[zi * zj * zk]
                        coskew_ijk = np.mean(
                            standardized[:, i] * standardized[:, j] * standardized[:, k]
                        )
                        M3[i, j * N + k] = coskew_ijk
            
            self._coskew_matrix = M3
        
        return self._coskew_matrix
    
    def get_cokurtosis_matrix(self) -> np.ndarray:
        """
        Calculate co-kurtosis matrix M4 of shape (N, N³).
        
        M4[i, j*N² + k*N + l] = E[(ri - μi)(rj - μj)(rk - μk)(rl - μl)] / (σi·σj·σk·σl) - 3
        
        Note: Subtracting 3 gives excess kurtosis (normal distribution = 0).
        """
        if self._cokurt_matrix is None:
            N = self.N
            T = self.T
            
            # Standardize returns
            stds = np.std(self.returns, axis=0, ddof=1)
            stds[stds == 0] = 1e-10
            standardized = self.demeaned / stds
            
            # Initialize co-kurtosis matrix
            M4 = np.zeros((N, N * N * N))
            
            # Compute co-kurtosis elements (this is O(N^4) - can be slow for large N)
            for i in range(N):
                for j in range(N):
                    for k in range(N):
                        for l in range(N):
                            # E[zi * zj * zk * zl] - 3 (for excess kurtosis)
                            cokurt_ijkl = np.mean(
                                standardized[:, i] * standardized[:, j] * 
                                standardized[:, k] * standardized[:, l]
                            ) - 3.0  # Excess kurtosis adjustment
                            M4[i, j * N * N + k * N + l] = cokurt_ijkl
            
            self._cokurt_matrix = M4
        
        return self._cokurt_matrix
    
    def get_individual_moments(self) -> pd.DataFrame:
        """Return individual asset moment statistics."""
        stats = []
        for i in range(self.N):
            asset_returns = self.returns[:, i]
            stats.append({
                'Asset': self.asset_names[i] if self.asset_names else f'Asset_{i}',
                'Mean (Ann.)': np.mean(asset_returns) * self.ann_factor,
                'Volatility (Ann.)': np.std(asset_returns, ddof=1) * np.sqrt(self.ann_factor),
                'Skewness': skew(asset_returns),
                'Excess Kurtosis': kurtosis(asset_returns, fisher=True)
            })
        return pd.DataFrame(stats)


# =============================================================================
# MVSK Optimizer Engine
# =============================================================================

class MVSKOptimizer:
    """
    Mean-Variance-Skewness-Kurtosis Portfolio Optimizer.
    
    Maximizes a composite objective combining MVSK utility and dividend yield:
        max_w [ α · U_MVSK(w) + (1-α) · Y(w) ]
    
    where U_MVSK uses Taylor series expansion of investor utility.
    """
    
    def __init__(
        self,
        returns: pd.DataFrame,
        dividend_yields: np.ndarray,
        lambdas: Tuple[float, float, float] = (1.0, 2.0, 2.0),
        risk_free_rate: float = 0.04
    ):
        """
        Initialize the MVSK optimizer.
        
        Args:
            returns: DataFrame of historical daily returns (T x N).
            dividend_yields: Array of annual dividend yields (as decimals, e.g., 0.05 for 5%).
            lambdas: Tuple of (λ1, λ2, λ3) risk aversion coefficients.
                     λ1: Variance aversion
                     λ2: Skewness preference (positive = prefers positive skew)
                     λ3: Kurtosis aversion (positive = dislikes fat tails)
            risk_free_rate: Annual risk-free rate for Sharpe calculations.
        """
        self.returns = returns
        self.dividend_yields = np.array(dividend_yields)
        self.lambda1, self.lambda2, self.lambda3 = lambdas
        self.rf = risk_free_rate
        
        # Initialize moment calculator
        self.moment_calc = MomentCalculator(returns)
        self.N = self.moment_calc.N
        self.asset_names = self.moment_calc.asset_names
        
        # Pre-compute moments
        self.mu = self.moment_calc.get_mean_vector()
        self.Sigma = self.moment_calc.get_covariance_matrix()
        self.M3 = self.moment_calc.get_coskewness_matrix()
        self.M4 = self.moment_calc.get_cokurtosis_matrix()
        
        # Payment matrix for monthly constraint (optional)
        self.payment_matrix = None
    
    def set_payment_matrix(self, payment_matrix: np.ndarray):
        """
        Set the monthly payment coverage matrix for dividend timing constraints.
        
        Args:
            payment_matrix: (12, N) matrix where [m, i] = yield contribution of asset i in month m.
        """
        self.payment_matrix = payment_matrix
    
    def _portfolio_variance(self, w: np.ndarray) -> float:
        """Calculate portfolio variance: w'Σw"""
        return float(w @ self.Sigma @ w)
    
    def _portfolio_skewness(self, w: np.ndarray) -> float:
        """
        Calculate portfolio skewness using co-skewness matrix.
        
        Sp = w' · M3 · (w ⊗ w)
        """
        # Kronecker product: w ⊗ w gives (N²,) vector
        w_kron = np.kron(w, w)
        return float(w @ self.M3 @ w_kron)
    
    def _portfolio_kurtosis(self, w: np.ndarray) -> float:
        """
        Calculate portfolio excess kurtosis using co-kurtosis matrix.
        
        Kp = w' · M4 · (w ⊗ w ⊗ w)
        """
        # Triple Kronecker product: w ⊗ w ⊗ w gives (N³,) vector
        w_kron3 = np.kron(np.kron(w, w), w)
        return float(w @ self.M4 @ w_kron3)
    
    def _mvsk_utility(self, w: np.ndarray) -> float:
        """
        Calculate MVSK utility based on Taylor expansion.
        
        U = E(Rp) - (λ1/2)·σp² + (λ2/3)·Sp - (λ3/4)·Kp
        
        Note: Positive skewness is GOOD (we add it).
              High kurtosis is BAD (we subtract it).
        """
        expected_return = float(self.mu @ w)
        variance = self._portfolio_variance(w)
        skewness = self._portfolio_skewness(w)
        kurtosis = self._portfolio_kurtosis(w)
        
        utility = (
            expected_return
            - (self.lambda1 / 2) * variance
            + (self.lambda2 / 3) * skewness  # Prefer positive skew
            - (self.lambda3 / 4) * kurtosis  # Penalize fat tails
        )
        
        return utility
    
    def _dividend_yield(self, w: np.ndarray) -> float:
        """Calculate portfolio dividend yield."""
        return float(self.dividend_yields @ w)
    
    def _objective(self, w: np.ndarray, alpha: float) -> float:
        """
        Combined objective function (to maximize).
        
        Returns negative for minimization by scipy.
        """
        mvsk_util = self._mvsk_utility(w)
        div_yield = self._dividend_yield(w)
        
        composite = alpha * mvsk_util + (1 - alpha) * div_yield
        
        return -composite  # Negative for minimization
    
    def optimize(
        self,
        alpha: float = 0.5,
        max_weight: float = 0.25,
        min_weight: float = 0.0,
        enforce_monthly_coverage: bool = False,
        min_nav_growth: Optional[float] = None,  # NEW: NAV preservation constraint
        verbose: bool = True
    ) -> OptimizationResult:
        """
        Run the MVSK optimization.
        
        Args:
            alpha: Trade-off between MVSK utility (1.0) and dividend yield (0.0).
            max_weight: Maximum weight per asset.
            min_weight: Minimum weight per asset (0 for long-only).
            enforce_monthly_coverage: If True and payment_matrix is set, enforce monthly dividends.
            min_nav_growth: Minimum portfolio price drift (NAV growth) constraint.
                           0.0 = breakeven (no NAV erosion)
                           0.05 = require 5% annual price appreciation
                           -0.03 = allow up to 3% annual NAV decline
                           None = no constraint (default)
            verbose: Print optimization progress.
        
        Returns:
            OptimizationResult with weights, moments, and metrics.
        """
        if verbose:
            print(f"Starting MVSK Optimization...")
            print(f"  α = {alpha:.2f} | λ = ({self.lambda1}, {self.lambda2}, {self.lambda3})")
            print(f"  Universe: {self.N} assets | Weight bounds: [{min_weight:.2f}, {max_weight:.2f}]")
            if min_nav_growth is not None:
                print(f"  NAV Constraint: Price Drift ≥ {min_nav_growth*100:+.1f}%")
        
        # Pre-calculate drift vector: Price Drift = Expected Return - Dividend Yield
        drift_vector = self.mu - self.dividend_yields
        
        if verbose and min_nav_growth is not None:
            print(f"\n  [Drift Analysis]")
            for i, name in enumerate(self.asset_names[:5]):  # Show first 5
                print(f"    {name}: Return={self.mu[i]*100:.1f}%, Yield={self.dividend_yields[i]*100:.1f}%, "
                      f"Drift={drift_vector[i]*100:+.1f}%")
            if len(self.asset_names) > 5:
                print(f"    ... and {len(self.asset_names)-5} more assets")
        
        # Constraints
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}  # Fully invested
        ]
        
        # NAV Preservation Constraint: sum(w_i * drift_i) >= min_nav_growth
        if min_nav_growth is not None:
            # Use closure to capture drift_vector and min_nav_growth
            def nav_constraint(w, dv=drift_vector, target=min_nav_growth):
                return np.dot(w, dv) - target
            
            constraints.append({
                'type': 'ineq',
                'fun': nav_constraint
            })
        
        # Monthly coverage constraints
        if enforce_monthly_coverage and self.payment_matrix is not None:
            epsilon = 1e-4
            for m in range(12):
                constraints.append({
                    'type': 'ineq',
                    'fun': lambda w, m=m: np.dot(self.payment_matrix[m], w) - epsilon
                })
        
        # Bounds
        bounds = [(min_weight, max_weight) for _ in range(self.N)]
        
        # Initial guess: equal weight
        w0 = np.ones(self.N) / self.N
        
        # Optimization
        result = minimize(
            self._objective,
            w0,
            args=(alpha,),
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'disp': verbose, 'maxiter': 500, 'ftol': 1e-8}
        )
        
        if not result.success and verbose:
            print(f"Warning: Optimization may not have converged. {result.message}")
        
        # Extract optimal weights
        w_opt = result.x
        w_opt[w_opt < 1e-6] = 0  # Clean up tiny weights
        w_opt = w_opt / w_opt.sum()  # Re-normalize
        
        # Calculate final metrics
        moments = self._calculate_portfolio_moments(w_opt)
        risk_metrics = self._calculate_risk_metrics(w_opt, moments)
        div_yield = self._dividend_yield(w_opt)
        mvsk_util = self._mvsk_utility(w_opt)
        composite = alpha * mvsk_util + (1 - alpha) * div_yield
        
        # Create weight dictionary
        weight_dict = {}
        for i, w in enumerate(w_opt):
            if w > 0.001:
                name = self.asset_names[i] if self.asset_names else f"Asset_{i}"
                weight_dict[name] = round(w, 5)
        
        # Sort by weight descending
        weight_dict = dict(sorted(weight_dict.items(), key=lambda x: -x[1]))
        
        return OptimizationResult(
            weights=weight_dict,
            moments=moments,
            risk_metrics=risk_metrics,
            dividend_yield=round(div_yield * 100, 2),  # As percentage
            mvsk_utility=round(mvsk_util, 4),
            composite_score=round(composite, 4),
            success=result.success,
            message=result.message
        )
    
    def _calculate_portfolio_moments(self, w: np.ndarray) -> PortfolioMoments:
        """Calculate all portfolio moments."""
        mean = float(self.mu @ w)
        variance = self._portfolio_variance(w)
        volatility = np.sqrt(variance)
        skewness = self._portfolio_skewness(w)
        kurtosis = self._portfolio_kurtosis(w)
        
        return PortfolioMoments(
            mean=round(mean * 100, 2),           # As percentage
            variance=round(variance * 100, 4),   # As percentage
            volatility=round(volatility * 100, 2),  # As percentage
            skewness=round(skewness, 4),
            kurtosis=round(kurtosis, 4)
        )
    
    def _calculate_risk_metrics(self, w: np.ndarray, moments: PortfolioMoments) -> RiskMetrics:
        """Calculate advanced risk metrics using Cornish-Fisher expansion."""
        
        # Convert back from percentage
        mean = moments.mean / 100
        vol = moments.volatility / 100
        skew_val = moments.skewness
        kurt_val = moments.kurtosis
        
        # Traditional Sharpe Ratio
        sharpe = (mean - self.rf) / vol if vol > 0 else 0
        
        # Cornish-Fisher Expansion for z-score adjustment
        # z_cf = z + (z² - 1)·S/6 + (z³ - 3z)·K/24 - (2z³ - 5z)·S²/36
        z_95 = norm.ppf(0.05)  # -1.645 for 95% VaR
        
        cf_adjustment = (
            z_95
            + (z_95**2 - 1) * skew_val / 6
            + (z_95**3 - 3*z_95) * kurt_val / 24
            - (2*z_95**3 - 5*z_95) * (skew_val**2) / 36
        )
        
        # Modified VaR (95%)
        modified_var = -(mean + cf_adjustment * vol)
        
        # Adjusted Sharpe Ratio (penalized for negative skew, high kurtosis)
        # ASR = SR × [1 + (S/6)×SR - (K/24)×SR²]
        asr = sharpe * (1 + (skew_val / 6) * sharpe - (kurt_val / 24) * (sharpe**2))
        
        # Modified CVaR (approximate)
        # Using simple adjustment: CVaR ≈ VaR × (1 + K/4) for fat tails
        modified_cvar = modified_var * (1 + abs(kurt_val) / 4)
        
        return RiskMetrics(
            sharpe_ratio=round(sharpe, 4),
            adjusted_sharpe_ratio=round(asr, 4),
            modified_var_95=round(modified_var * 100, 2),  # As percentage
            modified_cvar_95=round(modified_cvar * 100, 2)
        )


# =============================================================================
# Reporting Utilities
# =============================================================================

def print_optimization_report(result: OptimizationResult, title: str = "MVSK Optimization Report"):
    """Pretty print the optimization results."""
    
    print("\n" + "=" * 60)
    print(f" {title}")
    print("=" * 60)
    
    print(f"\n[Status] {'✓ Success' if result.success else '⚠ Warning'}: {result.message}")
    
    print("\n[Portfolio Allocation]")
    print("-" * 40)
    for asset, weight in result.weights.items():
        print(f"  {asset:<20} {weight*100:>8.2f}%")
    
    print("\n[Portfolio Moments]")
    print("-" * 40)
    print(f"  Expected Return:     {result.moments.mean:>8.2f}%")
    print(f"  Volatility:          {result.moments.volatility:>8.2f}%")
    print(f"  Skewness:            {result.moments.skewness:>8.4f}")
    print(f"  Excess Kurtosis:     {result.moments.kurtosis:>8.4f}")
    
    print("\n[Risk Metrics]")
    print("-" * 40)
    print(f"  Sharpe Ratio:        {result.risk_metrics.sharpe_ratio:>8.4f}")
    print(f"  Adjusted Sharpe:     {result.risk_metrics.adjusted_sharpe_ratio:>8.4f}")
    print(f"  Modified VaR (95%):  {result.risk_metrics.modified_var_95:>8.2f}%")
    print(f"  Modified CVaR (95%): {result.risk_metrics.modified_cvar_95:>8.2f}%")
    
    print("\n[Objective Values]")
    print("-" * 40)
    print(f"  Dividend Yield:      {result.dividend_yield:>8.2f}%")
    print(f"  MVSK Utility:        {result.mvsk_utility:>8.4f}")
    print(f"  Composite Score:     {result.composite_score:>8.4f}")
    
    print("\n" + "=" * 60)


# =============================================================================
# Demo with Synthetic Data
# =============================================================================

def generate_synthetic_data(
    n_days: int = 504,  # ~2 years
    seed: int = 42
) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Generate synthetic return data including assets with different skewness profiles.
    
    Returns:
        Tuple of (returns DataFrame, dividend yields array)
    """
    np.random.seed(seed)
    
    assets = {
        # Traditional Assets (roughly normal)
        'SPY': {'mu': 0.10, 'sigma': 0.16, 'skew': 0.0, 'div': 0.015},
        'QQQ': {'mu': 0.14, 'sigma': 0.22, 'skew': 0.0, 'div': 0.005},
        'AGG': {'mu': 0.03, 'sigma': 0.04, 'skew': 0.0, 'div': 0.035},
        
        # Covered Call ETFs (negative skewness, capped upside)
        'JEPI': {'mu': 0.06, 'sigma': 0.10, 'skew': -0.8, 'div': 0.085},
        'QYLD': {'mu': 0.02, 'sigma': 0.12, 'skew': -1.2, 'div': 0.120},
        'XYLD': {'mu': 0.04, 'sigma': 0.11, 'skew': -1.0, 'div': 0.100},
        
        # High Dividend Value
        'SCHD': {'mu': 0.08, 'sigma': 0.14, 'skew': -0.2, 'div': 0.035},
        'VYM':  {'mu': 0.07, 'sigma': 0.13, 'skew': -0.1, 'div': 0.030},
    }
    
    returns_data = {}
    dividends = []
    
    for name, params in assets.items():
        # Generate base normal returns
        daily_mu = params['mu'] / 252
        daily_sigma = params['sigma'] / np.sqrt(252)
        base_returns = np.random.normal(daily_mu, daily_sigma, n_days)
        
        # Apply skewness transformation (Johnson SU approximation)
        target_skew = params['skew']
        if target_skew != 0:
            # Simple skewing: multiply by exp(normal) to create asymmetry
            skew_factor = np.random.normal(0, 0.02, n_days)
            if target_skew < 0:
                # Negative skew: larger negative tails
                base_returns = np.where(
                    base_returns < 0,
                    base_returns * (1 + abs(target_skew) * 0.5),
                    base_returns * (1 - abs(target_skew) * 0.2)
                )
            else:
                base_returns = np.where(
                    base_returns > 0,
                    base_returns * (1 + target_skew * 0.5),
                    base_returns
                )
        
        returns_data[name] = base_returns
        dividends.append(params['div'])
    
    returns_df = pd.DataFrame(returns_data)
    dividend_arr = np.array(dividends)
    
    return returns_df, dividend_arr


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print(" MVSK Portfolio Optimizer - Demonstration")
    print(" (Handling Covered Call ETFs with Asymmetric Payoffs)")
    print("=" * 70)
    
    # 1. Generate synthetic data
    print("\n[1] Generating synthetic market data...")
    returns_df, dividend_yields = generate_synthetic_data()
    print(f"    Generated {len(returns_df)} days of data for {len(returns_df.columns)} assets")
    
    # 2. Analyze individual asset moments
    print("\n[2] Individual Asset Moments:")
    moment_calc = MomentCalculator(returns_df)
    asset_stats = moment_calc.get_individual_moments()
    print(asset_stats.to_string(index=False))
    
    # 3. Run Traditional MVO-style optimization (alpha=1, no skew/kurt penalty)
    print("\n[3] Running Traditional MVO (ignoring higher moments)...")
    mvo_optimizer = MVSKOptimizer(
        returns_df, 
        dividend_yields,
        lambdas=(1.0, 0.0, 0.0),  # Only variance penalty
        risk_free_rate=0.04
    )
    mvo_result = mvo_optimizer.optimize(alpha=0.7, max_weight=0.3, verbose=False)
    print_optimization_report(mvo_result, "Traditional MVO Result")
    
    # 4. Run MVSK optimization (with skew/kurt penalties)
    print("\n[4] Running MVSK Optimization (penalizing negative skew & fat tails)...")
    mvsk_optimizer = MVSKOptimizer(
        returns_df,
        dividend_yields,
        lambdas=(1.0, 2.0, 2.0),  # Full MVSK penalties
        risk_free_rate=0.04
    )
    mvsk_result = mvsk_optimizer.optimize(alpha=0.7, max_weight=0.3, verbose=False)
    print_optimization_report(mvsk_result, "MVSK Optimization Result")
    
    # 5. Comparison Summary
    print("\n" + "=" * 70)
    print(" COMPARISON: MVO vs MVSK")
    print("=" * 70)
    print(f"\n{'Metric':<25} {'MVO':>15} {'MVSK':>15} {'Diff':>12}")
    print("-" * 70)
    print(f"{'Covered Call Weight':<25} "
          f"{sum(w for k, w in mvo_result.weights.items() if k in ['JEPI','QYLD','XYLD'])*100:>14.1f}% "
          f"{sum(w for k, w in mvsk_result.weights.items() if k in ['JEPI','QYLD','XYLD'])*100:>14.1f}%")
    print(f"{'Portfolio Skewness':<25} {mvo_result.moments.skewness:>15.4f} {mvsk_result.moments.skewness:>15.4f}")
    print(f"{'Portfolio Kurtosis':<25} {mvo_result.moments.kurtosis:>15.4f} {mvsk_result.moments.kurtosis:>15.4f}")
    print(f"{'Adjusted Sharpe':<25} {mvo_result.risk_metrics.adjusted_sharpe_ratio:>15.4f} {mvsk_result.risk_metrics.adjusted_sharpe_ratio:>15.4f}")
    print(f"{'Modified VaR (95%)':<25} {mvo_result.risk_metrics.modified_var_95:>14.2f}% {mvsk_result.risk_metrics.modified_var_95:>14.2f}%")
    print(f"{'Dividend Yield':<25} {mvo_result.dividend_yield:>14.2f}% {mvsk_result.dividend_yield:>14.2f}%")
    
    print("\n" + "=" * 70)
    print(" Key Insight: MVSK reduces allocation to negatively-skewed covered call")
    print(" ETFs (QYLD, XYLD) because their tail risk is not captured by variance alone.")
    print("=" * 70 + "\n")
