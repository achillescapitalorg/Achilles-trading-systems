"""
Goldman Sachs-Level Volatility Modeling Framework
==================================================
Professional-grade volatility models including:
- GARCH(p, q) and Extended GARCH variants
- EGARCH (Exponential GARCH)
- GJR-GARCH (Asymmetric GARCH)
- Heston Stochastic Volatility Model
- Realized Volatility with High-Frequency Data
- Volatility Forecasting and Risk Metrics

Author: Quantitative Research Team
"""

import numpy as np
import pandas as pd
from scipy import stats, optimize
from scipy.stats import norm, t
from typing import Tuple, Dict, Optional, List, Union
from dataclasses import dataclass
from enum import Enum
import warnings


class VolatilityModelType(Enum):
    """Supported volatility model types."""
    GARCH = "garch"
    EGARCH = "egarch"
    GJR_GARCH = "gjr_garch"
    HESTON = "heston"
    REALIZED = "realized"
    PARKINSON = "parkinson"
    GARBECK_KLASS = "garbeck_klass"
    YANG_ZHANG = "yang_zhang"


@dataclass
class ModelParameters:
    """Container for model parameters."""
    omega: float = 0.0  # Constant term
    alpha: float = 0.0  # ARCH term
    beta: float = 0.0  # GARCH term
    gamma: float = 0.0  # Leverage/asymmetry term
    nu: float = 2.0  # Degrees of freedom for t-distribution
    kappa: float = 0.0  # Mean reversion speed (Heston)
    theta: float = 0.0  # Long-run variance (Heston)
    xi: float = 0.0  # Vol of vol (Heston)
    rho: float = 0.0  # Correlation (Heston)
    p: int = 1  # GARCH order
    q: int = 1  # ARCH order


@dataclass
class VolatilityForecast:
    """Container for volatility forecasts and metrics."""
    current_volatility: float
    forecast_1day: float
    forecast_5day: float
    forecast_10day: float
    forecast_22day: float
    long_run_volatility: float
    half_life: float  # Days to revert to long-run mean
    model_type: str
    log_likelihood: float
    bic: float
    aic: float


class VolatilityModels:
    """
    Professional Volatility Modeling Class
    
    Implements industry-standard volatility models used by investment banks
    and quantitative hedge funds for risk management and trading.
    """
    
    def __init__(self, returns: np.ndarray, prices: Optional[np.ndarray] = None):
        """
        Initialize the volatility model.
        
        Parameters
        ----------
        returns : np.ndarray
            Log returns or simple returns
        prices : np.ndarray, optional
            Price series for high-frequency volatility estimators
        """
        self.returns = np.asarray(returns, dtype=np.float64)
        self.prices = np.asarray(prices, dtype=np.float64) if prices is not None else None
        self.n = len(self.returns)
        self.params: Optional[ModelParameters] = None
        self.fitted = False
        self._cache = {}
        
    def compute_returns(self, prices: np.ndarray, method: str = 'log') -> np.ndarray:
        """
        Compute returns from price series.
        
        Parameters
        ----------
        prices : np.ndarray
            Price series
        method : str
            'log' for log returns, 'simple' for simple returns
            
        Returns
        -------
        np.ndarray
            Return series
        """
        prices = np.asarray(prices, dtype=np.float64)
        if method == 'log':
            return np.diff(np.log(prices))
        else:
            return np.diff(prices) / prices[:-1]
    
    # =========================================================================
    # GARCH(p, q) Model
    # =========================================================================
    
    def garch_log_likelihood(self, params: np.ndarray, returns: np.ndarray, 
                             p: int = 1, q: int = 1) -> float:
        """
        Compute GARCH(p, q) log-likelihood.
        
        Parameters
        ----------
        params : np.ndarray
            [omega, alpha_1, ..., alpha_p, beta_1, ..., beta_q]
        returns : np.ndarray
            Return series
        p, q : int
            GARCH and ARCH orders
        """
        omega = params[0]
        alphas = params[1:p+1]
        betas = params[p+1:p+q+1]
        
        n = len(returns)
        sigma2 = np.zeros(n)
        sigma2[0] = np.var(returns)
        
        for t in range(1, n):
            lag = min(t, max(p, q))
            start = t - lag
            arch_term = sum(alphas[i] * returns[start + i]**2 
                          for i in range(min(q, lag)))
            garch_term = sum(betas[i] * sigma2[start + i] 
                           for i in range(min(p, lag)))
            sigma2[t] = omega + arch_term + garch_term
            
        sigma2 = np.maximum(sigma2, 1e-10)
        ll = -0.5 * np.sum(np.log(sigma2) + returns**2 / sigma2)
        return -ll  # Negative for minimization
    
    def fit_garch(self, p: int = 1, q: int = 1, 
                  distribution: str = 'normal') -> ModelParameters:
        """
        Fit GARCH(p, q) model using Maximum Likelihood Estimation.
        
        Parameters
        ----------
        p : int
            GARCH order (lag of conditional variance)
        q : int
            ARCH order (lag of squared returns)
        distribution : str
            'normal' or 't' for Student's t innovations
            
        Returns
        -------
        ModelParameters
            Fitted model parameters
        """
        returns = self.returns - np.mean(self.returns)  # Demeaned returns
        
        # Initial parameter guesses
        initial_var = np.var(returns)
        omega_init = initial_var * 0.1
        alpha_init = 0.1
        beta_init = 0.8
        
        # Build initial parameter vector
        init_params = [omega_init] + [alpha_init] * q + [beta_init] * p
        
        # Constraints: omega > 0, alpha > 0, beta > 0, sum(alpha + beta) < 1
        bounds = [(1e-8, None)]  # omega
        bounds += [(0, 0.5)] * q  # alphas
        bounds += [(0, 0.99)] * p  # betas
        
        result = optimize.minimize(
            self.garch_log_likelihood,
            init_params,
            args=(returns, p, q),
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 1000, 'ftol': 1e-10}
        )
        
        if not result.success:
            warnings.warn(f"GARCH optimization did not converge: {result.message}")
        
        opt_params = result.x
        self.params = ModelParameters(
            omega=opt_params[0],
            alpha=opt_params[1] if q > 0 else 0,
            beta=opt_params[1+q] if p > 0 else 0,
            p=p, q=q
        )
        
        # Compute conditional variance for diagnostics
        self._compute_conditional_variance(opt_params, p, q)
        self._compute_fit_statistics(result.fun, 2 + p + q)
        
        self.fitted = True
        return self.params
    
    def _compute_conditional_variance(self, params: np.ndarray, p: int, q: int):
        """Compute the full conditional variance series."""
        returns = self.returns - np.mean(self.returns)
        n = len(returns)
        sigma2 = np.zeros(n)
        sigma2[0] = np.var(returns)
        
        omega = params[0]
        alphas = params[1:p+1]
        betas = params[p+1:p+q+1]
        
        for t in range(1, n):
            lag = min(t, max(p, q))
            start = t - lag
            arch_term = sum(alphas[i] * returns[start + i]**2 
                          for i in range(min(q, lag)))
            garch_term = sum(betas[i] * sigma2[start + i] 
                           for i in range(min(p, lag)))
            sigma2[t] = omega + arch_term + garch_term
            
        self.conditional_variance = sigma2
        self.conditional_volatility = np.sqrt(sigma2)
        
    def _compute_fit_statistics(self, neg_ll: float, n_params: int):
        """Compute AIC, BIC, and other fit statistics."""
        self.log_likelihood = -neg_ll
        self.aic = 2 * n_params - 2 * self.log_likelihood
        self.bic = n_params * np.log(self.n) - 2 * self.log_likelihood
        
    # =========================================================================
    # EGARCH Model (Exponential GARCH)
    # =========================================================================
    
    def egarch_log_likelihood(self, params: np.ndarray, returns: np.ndarray) -> float:
        """
        Compute EGARCH(1,1) log-likelihood.
        
        EGARCH captures leverage effects: negative returns increase volatility
        more than positive returns of the same magnitude.
        
        Model: log(sigma²_t) = ω + α(|z_{t-1}| - E|z_{t-1}|) + γ*z_{t-1} + β*log(sigma²_{t-1})
        """
        omega, alpha, gamma, beta = params
        
        n = len(returns)
        log_sigma2 = np.zeros(n)
        log_sigma2[0] = np.log(np.var(returns))
        
        # E|z| for standard normal ≈ 0.7979
        expected_abs_z = np.sqrt(2 / np.pi)
        
        for t in range(1, n):
            z = returns[t-1] / np.exp(log_sigma2[t-1] / 2)
            log_sigma2[t] = (omega + 
                           alpha * (np.abs(z) - expected_abs_z) + 
                           gamma * z + 
                           beta * log_sigma2[t-1])
            
        sigma2 = np.exp(log_sigma2)
        sigma2 = np.maximum(sigma2, 1e-10)
        ll = -0.5 * np.sum(np.log(sigma2) + returns**2 / sigma2)
        return -ll
    
    def fit_egarch(self) -> ModelParameters:
        """
        Fit EGARCH(1,1) model.
        
        Returns
        -------
        ModelParameters
            Fitted model parameters including leverage term (gamma)
        """
        returns = self.returns - np.mean(self.returns)
        
        # Initial parameters: omega, alpha, gamma, beta
        init_params = [0.0, 0.1, -0.1, 0.9]
        
        # Constraints
        bounds = [
            (None, None),    # omega (can be negative in log space)
            (0, 0.5),        # alpha (must be positive)
            (-0.5, 0.5),     # gamma (leverage effect)
            (0, 0.99)        # beta (persistence)
        ]
        
        result = optimize.minimize(
            self.egarch_log_likelihood,
            init_params,
            args=(returns,),
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 1000, 'ftol': 1e-10}
        )
        
        if not result.success:
            warnings.warn(f"EGARCH optimization did not converge: {result.message}")
        
        opt_params = result.x
        self.params = ModelParameters(
            omega=opt_params[0],
            alpha=opt_params[1],
            gamma=opt_params[2],
            beta=opt_params[3]
        )
        
        self._compute_egarch_variance(opt_params)
        self._compute_fit_statistics(result.fun, 4)
        
        self.fitted = True
        return self.params
    
    def _compute_egarch_variance(self, params: np.ndarray):
        """Compute EGARCH conditional variance series."""
        returns = self.returns - np.mean(self.returns)
        omega, alpha, gamma, beta = params
        
        n = len(returns)
        log_sigma2 = np.zeros(n)
        log_sigma2[0] = np.log(np.var(returns))
        expected_abs_z = np.sqrt(2 / np.pi)
        
        for t in range(1, n):
            z = returns[t-1] / np.exp(log_sigma2[t-1] / 2)
            log_sigma2[t] = (omega + 
                           alpha * (np.abs(z) - expected_abs_z) + 
                           gamma * z + 
                           beta * log_sigma2[t-1])
        
        self.conditional_variance = np.exp(log_sigma2)
        self.conditional_volatility = np.sqrt(self.conditional_variance)
        
    # =========================================================================
    # GJR-GARCH Model (Glosten-Jagannathan-Runkle)
    # =========================================================================
    
    def gjr_garch_log_likelihood(self, params: np.ndarray, returns: np.ndarray) -> float:
        """
        Compute GJR-GARCH(1,1) log-likelihood.
        
        GJR-GARCH captures asymmetric volatility:
        σ²_t = ω + α*r²_{t-1} + γ*I_{t-1}*r²_{t-1} + β*σ²_{t-1}
        where I = 1 if r < 0, else 0
        """
        omega, alpha, gamma, beta = params
        
        n = len(returns)
        sigma2 = np.zeros(n)
        sigma2[0] = np.var(returns)
        
        for t in range(1, n):
            r_sq = returns[t-1]**2
            indicator = 1.0 if returns[t-1] < 0 else 0.0
            sigma2[t] = (omega + 
                        alpha * r_sq + 
                        gamma * indicator * r_sq + 
                        beta * sigma2[t-1])
            sigma2[t] = max(sigma2[t], 1e-10)
            
        ll = -0.5 * np.sum(np.log(sigma2) + returns**2 / sigma2)
        return -ll
    
    def fit_gjr_garch(self) -> ModelParameters:
        """
        Fit GJR-GARCH(1,1) model.
        
        Returns
        -------
        ModelParameters
            Fitted model parameters including asymmetry term (gamma)
        """
        returns = self.returns - np.mean(self.returns)
        
        init_params = [0.0001, 0.1, 0.1, 0.8]
        
        bounds = [
            (1e-8, None),    # omega
            (0, 0.5),        # alpha
            (0, 0.5),        # gamma (asymmetry)
            (0, 0.99)        # beta
        ]
        
        result = optimize.minimize(
            self.gjr_garch_log_likelihood,
            init_params,
            args=(returns,),
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 1000, 'ftol': 1e-10}
        )
        
        if not result.success:
            warnings.warn(f"GJR-GARCH optimization did not converge: {result.message}")
        
        opt_params = result.x
        self.params = ModelParameters(
            omega=opt_params[0],
            alpha=opt_params[1],
            gamma=opt_params[2],
            beta=opt_params[3]
        )
        
        self._compute_gjr_variance(opt_params)
        self._compute_fit_statistics(result.fun, 4)
        
        self.fitted = True
        return self.params
    
    def _compute_gjr_variance(self, params: np.ndarray):
        """Compute GJR-GARCH conditional variance series."""
        returns = self.returns - np.mean(self.returns)
        omega, alpha, gamma, beta = params
        
        n = len(returns)
        sigma2 = np.zeros(n)
        sigma2[0] = np.var(returns)
        
        for t in range(1, n):
            r_sq = returns[t-1]**2
            indicator = 1.0 if returns[t-1] < 0 else 0.0
            sigma2[t] = omega + alpha * r_sq + gamma * indicator * r_sq + beta * sigma2[t-1]
            sigma2[t] = max(sigma2[t], 1e-10)
            
        self.conditional_variance = sigma2
        self.conditional_volatility = np.sqrt(sigma2)
        
    # =========================================================================
    # Heston Stochastic Volatility Model
    # =========================================================================
    
    def fit_heston_mle(self, method: str = 'gmm') -> ModelParameters:
        """
        Fit Heston Stochastic Volatility Model.
        
        The Heston model describes volatility as a mean-reverting square-root process:
        dS_t = μS_t dt + √v_t S_t dW_t^S
        dv_t = κ(θ - v_t)dt + ξ√v_t dW_t^v
        dW_t^S * dW_t^v = ρ dt
        
        Parameters
        ----------
        method : str
            'gmm' for Generalized Method of Moments, 'mle' for MLE
        
        Returns
        -------
        ModelParameters
            Fitted Heston parameters
        """
        returns = self.returns
        
        # Method of Moments estimation
        if method == 'gmm':
            # Realized variance
            realized_var = np.var(returns)
            
            # Variance of squared returns (for vol of vol)
            var_sq = np.var(returns**2)
            
            # Autocorrelation of squared returns (for mean reversion)
            sq_returns = returns**2
            autocorr = np.corrcoef(sq_returns[:-1], sq_returns[1:])[0, 1]
            autocorr = np.clip(autocorr, -0.99, 0.99)
            
            # Leverage effect (correlation between returns and volatility changes)
            vol_changes = np.diff(np.sqrt(np.convolve(returns**2, np.ones(5)/5, mode='valid')))
            if len(vol_changes) > 1:
                min_len = min(len(returns)-1, len(vol_changes))
                leverage_corr = np.corrcoef(returns[1:min_len+1], vol_changes[:min_len])[0, 1]
                leverage_corr = np.clip(leverage_corr, -0.99, 0.99)
            else:
                leverage_corr = -0.5
            
            # Estimate parameters from moments
            kappa = -np.log(autocorr) if autocorr > 0 else 0.5
            theta = realized_var
            xi = np.sqrt(2 * kappa * var_sq / realized_var) if realized_var > 0 else 0.3
            rho = leverage_corr
            omega = kappa * theta
            
            self.params = ModelParameters(
                kappa=max(kappa, 0.01),
                theta=max(theta, 1e-6),
                xi=max(xi, 0.01),
                rho=np.clip(rho, -0.99, 0.99),
                omega=omega
            )
        
        self.fitted = True
        return self.params
    
    def heston_simulation(self, S0: float, T: float, n_steps: int = 252, 
                         n_paths: int = 10000, seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simulate asset prices and volatility under Heston model.
        
        Parameters
        ----------
        S0 : float
            Initial asset price
        T : float
            Time horizon (in years)
        n_steps : int
            Number of time steps
        n_paths : int
            Number of simulation paths
        seed : int
            Random seed
            
        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Simulated price paths and volatility paths
        """
        if not self.fitted or self.params is None:
            raise ValueError("Must fit Heston model first using fit_heston_mle()")
        
        np.random.seed(seed)
        
        params = self.params
        dt = T / n_steps
        
        # Initialize arrays
        S = np.zeros((n_paths, n_steps + 1))
        v = np.zeros((n_paths, n_steps + 1))
        S[:, 0] = S0
        v[:, 0] = params.theta  # Start at long-run variance
        
        # Correlated Brownian motions
        Z1 = np.random.standard_normal((n_paths, n_steps))
        Z2 = np.random.standard_normal((n_paths, n_steps))
        Z2 = params.rho * Z1 + np.sqrt(1 - params.rho**2) * Z2
        
        # Full truncation scheme for variance (prevents negative variance)
        for t in range(n_steps):
            v[:, t+1] = v[:, t] + params.kappa * (params.theta - v[:, t]) * dt + \
                       params.xi * np.sqrt(np.maximum(v[:, t], 0)) * np.sqrt(dt) * Z2[:, t]
            v[:, t+1] = np.maximum(v[:, t+1], 0)  # Full truncation
            
            S[:, t+1] = S[:, t] * np.exp(-0.5 * np.maximum(v[:, t], 0) * dt + 
                                         np.sqrt(np.maximum(v[:, t], 0) * dt) * Z1[:, t])
        
        return S, v
    
    # =========================================================================
    # High-Frequency Volatility Estimators
    # =========================================================================
    
    def realized_volatility(self, returns: Optional[np.ndarray] = None, 
                           window: int = 1) -> float:
        """
        Compute Realized Volatility from high-frequency returns.
        
        RV = sqrt(sum of squared returns)
        
        Parameters
        ----------
        returns : np.ndarray, optional
            High-frequency returns (intraday)
        window : int
            Number of periods to aggregate
            
        Returns
        -------
        float
            Realized volatility
        """
        if returns is None:
            returns = self.returns
            
        return np.sqrt(np.sum(returns**2))
    
    def parkinson_volatility(self, high: np.ndarray, low: np.ndarray, 
                            window: int = 21) -> np.ndarray:
        """
        Parkinson Volatility Estimator using High-Low range.
        
        More efficient than close-to-close volatility (up to 5x).
        σ² = (1 / (4N*ln(2))) * sum(ln(H/L)²)
        
        Parameters
        ----------
        high : np.ndarray
            High prices
        low : np.ndarray
            Low prices
        window : int
            Rolling window size
            
        Returns
        -------
        np.ndarray
            Parkinson volatility estimates
        """
        high = np.asarray(high)
        low = np.asarray(low)
        
        log_hl = np.log(high / low)
        log_hl_sq = log_hl ** 2
        
        n = len(log_hl_sq)
        parkinson_var = np.zeros(n)
        parkinson_var[:window] = np.nan
        
        for i in range(window, n):
            parkinson_var[i] = np.sum(log_hl_sq[i-window:i]) / (4 * window * np.log(2))
            
        return np.sqrt(parkinson_var)
    
    def garman_klass_volatility(self, open_p: np.ndarray, high: np.ndarray,
                                low: np.ndarray, close: np.ndarray,
                                window: int = 21) -> np.ndarray:
        """
        Garman-Klass Volatility Estimator (OHLC).
        
        Uses open, high, low, close prices for improved efficiency.
        σ² = 0.5*ln(H/L)² - (2*ln(2)-1)*ln(C/O)²
        
        Parameters
        ----------
        open_p : np.ndarray
            Open prices
        high : np.ndarray
            High prices
        low : np.ndarray
            Low prices
        close : np.ndarray
            Close prices
        window : int
            Rolling window size
            
        Returns
        -------
        np.ndarray
            Garman-Klass volatility estimates
        """
        open_p = np.asarray(open_p)
        high = np.asarray(high)
        low = np.asarray(low)
        close = np.asarray(close)
        
        log_hl = np.log(high / low)
        log_co = np.log(close / open_p)
        
        gk_var = 0.5 * log_hl**2 - (2 * np.log(2) - 1) * log_co**2
        
        n = len(gk_var)
        result = np.zeros(n)
        result[:window] = np.nan
        
        for i in range(window, n):
            result[i] = np.sqrt(np.mean(gk_var[i-window:i]))
            
        return result
    
    def yang_zhang_volatility(self, open_p: np.ndarray, high: np.ndarray,
                              low: np.ndarray, close: np.ndarray,
                              window: int = 21) -> np.ndarray:
        """
        Yang-Zhang Volatility Estimator.
        
        Most efficient estimator, handles overnight jumps and drift.
        Combines overnight volatility, open-to-close volatility, and range.
        
        Parameters
        ----------
        open_p : np.ndarray
            Open prices
        high : np.ndarray
            High prices
        low : np.ndarray
            Low prices
        close : np.ndarray
            Close prices
        window : int
            Rolling window size
            
        Returns
        -------
        np.ndarray
            Yang-Zhang volatility estimates
        """
        open_p = np.asarray(open_p)
        high = np.asarray(high)
        low = np.asarray(low)
        close = np.asarray(close)
        
        n = len(close)
        
        # Overnight volatility (close to open)
        log_oc = np.log(open_p / np.roll(close, 1))
        log_oc[0] = 0
        
        # Open-to-close volatility
        log_co = np.log(close / open_p)
        
        # Range component (Garman-Klass style)
        log_hl = np.log(high / low)
        
        # Yang-Zhang weight factor
        k = 0.34 / (1.34 + (window + 1) / (window - 1))
        
        # Overnight variance
        sigma_oc_sq = np.zeros(n)
        for i in range(window, n):
            sigma_oc_sq[i] = np.var(log_oc[max(0, i-window):i])
            
        # Open-to-close variance  
        sigma_co_sq = np.zeros(n)
        for i in range(window, n):
            sigma_co_sq[i] = np.var(log_co[max(0, i-window):i])
            
        # Range-based variance
        sigma_r_sq = np.zeros(n)
        for i in range(window, n):
            sigma_r_sq[i] = np.sum(log_hl[max(0, i-window):i]**2) / (4 * window * np.log(2))
            
        # Combined Yang-Zhang volatility
        yz_var = sigma_oc_sq + k * sigma_co_sq + (1 - k) * sigma_r_sq
        yz_var = np.maximum(yz_var, 0)
        
        return np.sqrt(yz_var)
    
    # =========================================================================
    # Volatility Forecasting
    # =========================================================================
    
    def forecast_volatility(self, horizon: int = 1) -> float:
        """
        Forecast volatility at specified horizon.
        
        Parameters
        ----------
        horizon : int
            Forecast horizon in days
            
        Returns
        -------
        float
            Forecasted annualized volatility
        """
        if not self.fitted or self.params is None:
            raise ValueError("Must fit a model first")
        
        params = self.params
        
        # Long-run variance
        if hasattr(params, 'omega') and hasattr(params, 'alpha') and hasattr(params, 'beta'):
            persistence = params.alpha + params.beta
            if persistence >= 1:
                long_run_var = np.var(self.returns)
            else:
                long_run_var = params.omega / (1 - persistence)
        else:
            long_run_var = np.var(self.returns)
        
        # Current variance
        current_var = self.conditional_variance[-1] if hasattr(self, 'conditional_variance') else np.var(self.returns)
        
        # Forecast using mean reversion
        if hasattr(params, 'beta') and params.beta > 0:
            forecast_var = long_run_var + (params.beta ** horizon) * (current_var - long_run_var)
        else:
            forecast_var = long_run_var
            
        return np.sqrt(forecast_var * 252)  # Annualized
    
    def get_volatility_forecast(self) -> VolatilityForecast:
        """
        Get comprehensive volatility forecast with multiple horizons.
        
        Returns
        -------
        VolatilityForecast
            Complete forecast object with metrics
        """
        if not self.fitted:
            raise ValueError("Must fit a model first")
        
        params = self.params
        
        # Long-run variance
        if params.alpha + params.beta < 1:
            long_run_var = params.omega / (1 - params.alpha - params.beta)
        else:
            long_run_var = np.var(self.returns)
            
        long_run_vol = np.sqrt(long_run_var * 252)
        current_vol = np.sqrt(self.conditional_variance[-1] * 252) if hasattr(self, 'conditional_variance') else np.std(self.returns) * np.sqrt(252)
        
        # Persistence and half-life
        persistence = params.alpha + params.beta
        half_life = np.log(0.5) / np.log(persistence) if persistence > 0 and persistence < 1 else np.inf
        
        forecasts = {}
        for h in [1, 5, 10, 22]:
            if persistence < 1:
                forecast_var = long_run_var + (persistence ** h) * (self.conditional_variance[-1] - long_run_var)
            else:
                forecast_var = self.conditional_variance[-1]
            forecasts[h] = np.sqrt(max(forecast_var, 0) * 252)
        
        return VolatilityForecast(
            current_volatility=current_vol,
            forecast_1day=forecasts[1],
            forecast_5day=forecasts[5],
            forecast_10day=forecasts[10],
            forecast_22day=forecasts[22],
            long_run_volatility=long_run_vol,
            half_life=half_life,
            model_type=self.params.__class__.__name__,
            log_likelihood=getattr(self, 'log_likelihood', 0),
            bic=getattr(self, 'bic', 0),
            aic=getattr(self, 'aic', 0)
        )
    
    # =========================================================================
    # Risk Metrics
    # =========================================================================
    
    def value_at_risk(self, confidence: float = 0.95, 
                     position_value: float = 1e6,
                     horizon: int = 1) -> float:
        """
        Calculate Value at Risk (VaR).
        
        Parameters
        ----------
        confidence : float
            Confidence level (e.g., 0.95 for 95%)
        position_value : float
            Portfolio value
        horizon : int
            Time horizon in days
            
        Returns
        -------
        float
            VaR in currency units
        """
        vol = self.forecast_volatility(horizon) / np.sqrt(252)
        vol = vol * np.sqrt(horizon)
        
        z_score = norm.ppf(1 - confidence)
        var = position_value * vol * abs(z_score)
        
        return var
    
    def expected_shortfall(self, confidence: float = 0.95,
                          position_value: float = 1e6,
                          horizon: int = 1) -> float:
        """
        Calculate Expected Shortfall (CVaR).
        
        Parameters
        ----------
        confidence : float
            Confidence level
        position_value : float
            Portfolio value
        horizon : int
            Time horizon in days
            
        Returns
        -------
        float
            Expected Shortfall in currency units
        """
        vol = self.forecast_volatility(horizon) / np.sqrt(252)
        vol = vol * np.sqrt(horizon)
        
        z_score = norm.ppf(1 - confidence)
        es = position_value * vol * norm.pdf(z_score) / (1 - confidence)
        
        return es
    
    # =========================================================================
    # Model Diagnostics
    # =========================================================================
    
    def diagnostic_check(self) -> Dict[str, float]:
        """
        Perform model diagnostic checks.
        
        Returns
        -------
        Dict[str, float]
            Diagnostic statistics
        """
        if not hasattr(self, 'conditional_variance'):
            return {}
            
        # Standardized residuals
        returns_demeaned = self.returns - np.mean(self.returns)
        std_residuals = returns_demeaned / np.sqrt(self.conditional_variance)
        
        # Remove NaN values
        std_residuals = std_residuals[~np.isnan(std_residuals)]
        
        diagnostics = {
            'mean_std_residual': np.mean(std_residuals),
            'std_std_residual': np.std(std_residuals),
            'skewness': stats.skew(std_residuals),
            'kurtosis': stats.kurtosis(std_residuals),
            'jarque_bera_stat': stats.jarque_bera(std_residuals)[0],
            'jarque_bera_pvalue': stats.jarque_bera(std_residuals)[1],
            'arch_lm_stat': self._arch_lm_test(std_residuals, lags=10),
        }
        
        return diagnostics
    
    def _arch_lm_test(self, residuals: np.ndarray, lags: int = 10) -> float:
        """
        ARCH LM test for remaining heteroskedasticity.
        
        Parameters
        ----------
        residuals : np.ndarray
            Standardized residuals
        lags : int
            Number of lags to test
            
        Returns
        -------
        float
            LM test statistic
        """
        n = len(residuals)
        sq_residuals = residuals**2
        
        # Regress squared residuals on lagged squared residuals
        X = np.zeros((n - lags, lags + 1))
        X[:, 0] = 1  # Intercept
        
        for i in range(lags):
            X[:, i+1] = sq_residuals[i:n-lags+i]
            
        y = sq_residuals[lags:]
        
        # OLS
        try:
            beta = np.linalg.lstsq(X, y, rcond=None)[0]
            y_pred = X @ beta
            ssr = np.sum((y - y_pred)**2)
            sst = np.sum((y - np.mean(y))**2)
            r_squared = 1 - ssr / sst
            lm_stat = (n - lags) * r_squared
        except:
            lm_stat = 0.0
            
        return lm_stat


def create_volatility_model(returns: np.ndarray, 
                           prices: Optional[np.ndarray] = None) -> VolatilityModels:
    """
    Factory function to create a volatility model instance.
    
    Parameters
    ----------
    returns : np.ndarray
        Return series
    prices : np.ndarray, optional
        Price series for OHLC estimators
        
    Returns
    -------
    VolatilityModels
        Initialized volatility model
    """
    return VolatilityModels(returns, prices)


if __name__ == "__main__":
    # Example usage
    np.random.seed(42)
    
    # Simulate returns with volatility clustering
    n = 2520  # 10 years of daily data
    omega, alpha, beta = 0.00001, 0.1, 0.85
    
    sigma2 = np.zeros(n)
    returns = np.zeros(n)
    sigma2[0] = 0.0004
    
    for t in range(1, n):
        sigma2[t] = omega + alpha * returns[t-1]**2 + beta * sigma2[t-1]
        returns[t] = np.random.normal(0, np.sqrt(sigma2[t]))
    
    # Fit models
    vol_model = VolatilityModels(returns)
    
    print("=" * 60)
    print("GOLDMAN SACHS-LEVEL VOLATILITY MODELING FRAMEWORK")
    print("=" * 60)
    
    # GARCH
    print("\n[GARCH(1,1) Model]")
    params = vol_model.fit_garch()
    print(f"  Omega: {params.omega:.6f}")
    print(f"  Alpha: {params.alpha:.4f}")
    print(f"  Beta:  {params.beta:.4f}")
    print(f"  Persistence: {params.alpha + params.beta:.4f}")
    
    forecast = vol_model.get_volatility_forecast()
    print(f"\n  Current Vol: {forecast.current_volatility:.2f}%")
    print(f"  1-Day Forecast: {forecast.forecast_1day:.2f}%")
    print(f"  22-Day Forecast: {forecast.forecast_22day:.2f}%")
    print(f"  Long-Run Vol: {forecast.long_run_volatility:.2f}%")
    
    # EGARCH
    print("\n[EGARCH Model - Captures Leverage Effects]")
    vol_model2 = VolatilityModels(returns)
    params = vol_model2.fit_egarch()
    print(f"  Gamma (Leverage): {params.gamma:.4f}")
    print(f"  Interpretation: {'Negative' if params.gamma < 0 else 'Positive'} leverage effect")
    
    # GJR-GARCH
    print("\n[GJR-GARCH Model - Asymmetric Volatility]")
    vol_model3 = VolatilityModels(returns)
    params = vol_model3.fit_gjr_garch()
    print(f"  Gamma (Asymmetry): {params.gamma:.4f}")
    
    # Risk Metrics
    print("\n[Risk Metrics - $1M Position]")
    var_95 = vol_model.value_at_risk(confidence=0.95, position_value=1e6)
    es_95 = vol_model.expected_shortfall(confidence=0.95, position_value=1e6)
    print(f"  95% VaR (1-day): ${var_95:,.2f}")
    print(f"  95% ES (1-day): ${es_95:,.2f}")
    
    # Diagnostics
    print("\n[Model Diagnostics]")
    diag = vol_model.diagnostic_check()
    print(f"  Standardized Residual Std: {diag.get('std_std_residual', 'N/A'):.4f}")
    print(f"  Jarque-Bera p-value: {diag.get('jarque_bera_pvalue', 'N/A'):.4f}")
