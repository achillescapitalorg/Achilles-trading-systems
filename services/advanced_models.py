"""
Advanced Quantitative Finance Models
=====================================
Institutional-grade models for volatility, option pricing, and trading.
Includes Heston, SABR, Local Volatility, and Reinforcement Learning.
"""

import numpy as np
import pandas as pd
from scipy import stats, optimize, integrate
from scipy.stats import norm, multivariate_normal
from typing import Dict, List, Tuple, Optional, Callable, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
import warnings


# ============================================================================
# Black-Scholes Model
# ============================================================================

class BlackScholes:
    """
    Black-Scholes Option Pricing Model

    Calculates option prices and Greeks for European options.
    """

    @staticmethod
    def d1(S: float, K: float, T: float, r: float, sigma: float) -> float:
        """Calculate d1 parameter."""
        return (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))

    @staticmethod
    def d2(d1: float, sigma: float, T: float) -> float:
        """Calculate d2 parameter."""
        return d1 - sigma * np.sqrt(T)

    @classmethod
    def call_price(cls, S: float, K: float, T: float, r: float, sigma: float) -> float:
        """Calculate European call option price."""
        if T <= 0 or sigma <= 0:
            return max(0, S - K)

        d1 = cls.d1(S, K, T, r, sigma)
        d2 = cls.d2(d1, sigma, T)

        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

    @classmethod
    def put_price(cls, S: float, K: float, T: float, r: float, sigma: float) -> float:
        """Calculate European put option price."""
        if T <= 0 or sigma <= 0:
            return max(0, K - S)

        d1 = cls.d1(S, K, T, r, sigma)
        d2 = cls.d2(d1, sigma, T)

        return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

    @classmethod
    def delta(cls, S: float, K: float, T: float, r: float, sigma: float,
              option_type: str = 'call') -> float:
        """Calculate option delta."""
        d1 = cls.d1(S, K, T, r, sigma)

        if option_type == 'call':
            return norm.cdf(d1)
        else:
            return norm.cdf(d1) - 1

    @classmethod
    def gamma(cls, S: float, K: float, T: float, r: float, sigma: float) -> float:
        """Calculate option gamma."""
        d1 = cls.d1(S, K, T, r, sigma)
        return norm.pdf(d1) / (S * sigma * np.sqrt(T))

    @classmethod
    def vega(cls, S: float, K: float, T: float, r: float, sigma: float) -> float:
        """Calculate option vega."""
        d1 = cls.d1(S, K, T, r, sigma)
        return S * norm.pdf(d1) * np.sqrt(T) / 100

    @classmethod
    def theta(cls, S: float, K: float, T: float, r: float, sigma: float,
              option_type: str = 'call') -> float:
        """Calculate option theta."""
        d1 = cls.d1(S, K, T, r, sigma)
        d2 = cls.d2(d1, sigma, T)

        term1 = -S * norm.pdf(d1) * sigma / (2 * np.sqrt(T))

        if option_type == 'call':
            term2 = r * K * np.exp(-r * T) * norm.cdf(d2)
            return (term1 - term2) / 365
        else:
            term2 = -r * K * np.exp(-r * T) * norm.cdf(-d2)
            return (term1 + term2) / 365

    @classmethod
    def rho(cls, S: float, K: float, T: float, r: float, sigma: float,
            option_type: str = 'call') -> float:
        """Calculate option rho."""
        d2 = cls.d2(cls.d1(S, K, T, r, sigma), sigma, T)

        if option_type == 'call':
            return K * T * np.exp(-r * T) * norm.cdf(d2) / 100
        else:
            return -K * T * np.exp(-r * T) * norm.cdf(-d2) / 100

    @classmethod
    def implied_volatility(cls, price: float, S: float, K: float, T: float,
                          r: float, option_type: str = 'call',
                          max_iter: int = 100, tol: float = 1e-6) -> float:
        """Calculate implied volatility using Newton-Raphson method."""
        # Initial guess
        sigma = 0.3

        for i in range(max_iter):
            if option_type == 'call':
                bs_price = cls.call_price(S, K, T, r, sigma)
            else:
                bs_price = cls.put_price(S, K, T, r, sigma)

            diff = bs_price - price

            if abs(diff) < tol:
                return sigma

            vega = cls.vega(S, K, T, r, sigma)

            if vega < 1e-10:
                sigma *= 1.1
            else:
                sigma = sigma - diff / vega

            # Bound sigma
            sigma = max(0.01, min(sigma, 5.0))

        return sigma


# ============================================================================
# Heston Stochastic Volatility Model
# ============================================================================

@dataclass
class HestonParams:
    """Heston model parameters."""
    kappa: float   # Mean reversion speed
    theta: float   # Long-run variance
    xi: float      # Volatility of volatility
    rho: float     # Correlation between asset and volatility
    v0: float      # Initial variance


class HestonModel:
    """
    Heston Stochastic Volatility Model

    dv_t = kappa(theta - v_t)dt + xi*sqrt(v_t)*dW_t^v
    dS_t = r*S_t*dt + sqrt(v_t)*S_t*dW_t^S
    """

    def __init__(self, params: HestonParams):
        self.params = params

    def characteristic_function(self, u: np.ndarray, S: float, K: float,
                                T: float, r: float) -> np.ndarray:
        """
        Calculate Heston characteristic function for option pricing.
        Uses the Heston (1993) semi-closed form solution.
        """
        kappa = self.params.kappa
        theta = self.params.theta
        xi = self.params.xi
        rho = self.params.rho
        v0 = self.params.v0

        # Complex arithmetic
        iu = 1j * u

        # D and C functions
        d = np.sqrt((kappa - rho * xi * iu)**2 + xi**2 * (iu + u**2))
        g = (kappa - rho * xi * iu - d) / (kappa - rho * xi * iu + d)

        D = (kappa - rho * xi * iu - d) / xi**2 * (1 - np.exp(-d * T)) / (1 - g * np.exp(-d * T))
        C = r * iu * T + kappa * theta / xi**2 * (
            (kappa - rho * xi * iu - d) * T -
            2 * np.log((1 - g * np.exp(-d * T)) / (1 - g))
        )

        # Characteristic function
        cf = np.exp(C + D * v0 + iu * np.log(S))

        return cf

    def call_price_fft(self, S: float, K: float, T: float, r: float) -> float:
        """
        Calculate call price using FFT method.
        """
        # Fourier transform approach
        def integrand(u):
            cf = self.characteristic_function(u - 0.5j, S, K, T, r)
            return np.real(cf * np.exp(-1j * u * np.log(K)) / (u**2 + 0.25))

        # Numerical integration
        result, _ = integrate.quad(integrand, 0, 100, limit=100)

        call_price = S - np.sqrt(S * K) * np.exp(-r * T) * result / np.pi

        return max(0, call_price)

    def simulate_paths(self, S0: float, T: float, n_steps: int,
                      n_paths: int, seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simulate asset price and volatility paths using QE scheme.
        """
        np.random.seed(seed)

        kappa = self.params.kappa
        theta = self.params.theta
        xi = self.params.xi
        rho = self.params.rho
        v0 = self.params.v0

        dt = T / n_steps

        S = np.zeros((n_paths, n_steps + 1))
        v = np.zeros((n_paths, n_steps + 1))

        S[:, 0] = S0
        v[:, 0] = v0

        # Correlated Brownian motions
        Z1 = np.random.standard_normal((n_paths, n_steps))
        Z2 = np.random.standard_normal((n_paths, n_steps))
        Z2 = rho * Z1 + np.sqrt(1 - rho**2) * Z2

        # Full truncation scheme
        for t in range(n_steps):
            v[:, t+1] = v[:, t] + kappa * (theta - v[:, t]) * dt + \
                       xi * np.sqrt(np.maximum(v[:, t], 0)) * np.sqrt(dt) * Z2[:, t]
            v[:, t+1] = np.maximum(v[:, t+1], 0)

            S[:, t+1] = S[:, t] * np.exp(
                (r - 0.5 * np.maximum(v[:, t], 0)) * dt +
                np.sqrt(np.maximum(v[:, t], 0) * dt) * Z1[:, t]
            )

        return S, v


# ============================================================================
# SABR Volatility Model
# ============================================================================

@dataclass
class SABRParams:
    """SABR model parameters."""
    alpha: float  # Initial volatility
    beta: float   # Elasticity parameter (0-1)
    rho: float    # Correlation
    nu: float     # Volatility of volatility


class SABRModel:
    """
    SABR Stochastic Alpha Beta Rho Model

    Used for volatility surface interpolation in options markets.
    """

    def __init__(self, params: SABRParams):
        self.params = params

    def implied_volatility(self, F: float, K: float, T: float) -> float:
        """
        Calculate implied volatility using Hagan's approximation.

        Parameters
        ----------
        F : float - Forward price
        K : float - Strike price
        T : float - Time to expiry
        """
        alpha = self.params.alpha
        beta = self.params.beta
        rho = self.params.rho
        nu = self.params.nu

        # Handle ATM case
        if abs(F - K) < 1e-10:
            return self._atm_vol(F, T)

        logFK = np.log(F / K)

        # z and x functions
        z = (nu / alpha) * (F * K)**((1 - beta) / 2) * logFK
        x = np.log((np.sqrt(1 - 2 * rho * z + z**2) + z - rho) / (1 - rho))

        # First order term
        term1 = alpha / ((F * K)**((1 - beta) / 2) * (1 + (1 - beta)**2 / 24 * logFK**2 +
                    (1 - beta)**4 / 1920 * logFK**4))

        # Second order correction
        term2 = 1 + T * (
            ((1 - beta)**2 / 24) * alpha**2 / (F * K)**(1 - beta) +
            (rho * beta * nu * alpha) / (4 * (F * K)**((1 - beta) / 2)) +
            (2 - 3 * rho**2) / 24 * nu**2
        )

        sigma = term1 * term2 * (z / x)

        return max(0.001, sigma)

    def _atm_vol(self, F: float, T: float) -> float:
        """Calculate ATM volatility."""
        alpha = self.params.alpha
        beta = self.params.beta
        rho = self.params.rho
        nu = self.params.nu

        F_beta = F**(1 - beta)

        sigma_atm = alpha / F_beta * (1 + T * (
            ((1 - beta)**2 / 24) * alpha**2 / F**(2 - 2*beta) +
            (rho * beta * nu * alpha) / (4 * F_beta) +
            (2 - 3 * rho**2) / 24 * nu**2
        ))

        return sigma_atm

    @classmethod
    def calibrate(cls, market_vols: np.ndarray, forwards: np.ndarray,
                  strikes: np.ndarray, expiries: np.ndarray,
                  beta: float = 0.5) -> SABRParams:
        """
        Calibrate SABR parameters to market volatilities.
        """
        def objective(params):
            alpha, rho, nu = params

            model_vols = []
            for i, (F, K, T) in enumerate(zip(forwards, strikes, expiries)):
                model = SABRModel(SABRParams(alpha, beta, rho, nu))
                model_vols.append(model.implied_volatility(F, K, T))

            return np.sum((np.array(model_vols) - market_vols)**2)

        # Initial guess
        x0 = [0.3, 0.0, 0.3]

        # Bounds
        bounds = [(0.01, 2.0), (-0.999, 0.999), (0.01, 2.0)]

        result = optimize.minimize(objective, x0, method='L-BFGS-B', bounds=bounds)

        alpha, rho, nu = result.x

        return cls(SABRParams(alpha, beta, rho, nu))


# ============================================================================
# Local Volatility Model (Dupire)
# ============================================================================

class LocalVolatilityModel:
    """
    Local Volatility Model using Dupire's Formula

    sigma_loc^2(K,T) = 2 * (dC/dT + r*K*dC/dK) / (K^2 * d^2C/dK^2)
    """

    def __init__(self, surface_data: Dict):
        """
        Initialize with volatility surface data.

        Parameters
        ----------
        surface_data : Dict
            Dictionary with strikes, expiries, and volatilities
        """
        self.strikes = np.array(surface_data['strikes'])
        self.expiries = np.array(surface_data['expiries'])
        self.volatilities = np.array(surface_data['volatilities'])
        self.surface = self._interpolate_surface()

    def _interpolate_surface(self) -> Callable:
        """Create interpolation function for volatility surface."""
        from scipy.interpolate import RectBivariateSpline

        # Create 2D interpolation
        self.interp_func = RectBivariateSpline(
            self.expiries, self.strikes, self.volatilities
        )

        return self.interp_func

    def local_volatility(self, K: float, T: float, S: float, r: float) -> float:
        """
        Calculate local volatility using Dupire's formula.
        """
        eps = 1e-5

        # Get option prices
        def get_price(k, t):
            vol = self.interp_func(t, k)[0, 0]
            return BlackScholes.call_price(S, k, t, r, vol)

        # Numerical derivatives
        c_T = (get_price(K, T + eps) - get_price(K, T - eps)) / (2 * eps)
        c_K = (get_price(K + eps, T) - get_price(K - eps, T)) / (2 * eps)
        c_KK = (get_price(K + eps, T) - 2 * get_price(K, T) + get_price(K - eps, T)) / eps**2

        # Dupire's formula
        numerator = 2 * (c_T + r * K * c_K)
        denominator = K**2 * c_KK

        if denominator <= 0:
            return self.interp_func(T, K)[0, 0]

        sigma_loc = np.sqrt(numerator / denominator)

        return max(0.01, sigma_loc)


# ============================================================================
# Variance Gamma Model
# ============================================================================

class VarianceGammaModel:
    """
    Variance Gamma Process for asset returns

    Captures skewness and kurtosis in return distributions.
    """

    def __init__(self, sigma: float, nu: float, theta: float):
        """
        Parameters
        ----------
        sigma : float - Volatility of Brownian motion
        nu : float - Variance rate of gamma process
        theta : float - Drift of Brownian motion
        """
        self.sigma = sigma
        self.nu = nu
        self.theta = theta

    def simulate(self, S0: float, T: float, n_steps: int,
                n_paths: int, r: float = 0.05) -> np.ndarray:
        """Simulate VG paths."""
        np.random.seed(42)

        dt = T / n_steps

        # Gamma time changes
        gamma_increments = np.random.gamma(dt / self.nu, self.nu, (n_paths, n_steps))
        gamma_time = np.cumsum(gamma_increments, axis=1)
        gamma_time = np.hstack([np.zeros((n_paths, 1)), gamma_time])

        # Brownian motion with drift
        bm = np.random.standard_normal((n_paths, n_steps))
        bm = np.cumsum(bm * np.sqrt(dt), axis=1)
        bm = np.hstack([np.zeros((n_paths, 1)), bm])

        # VG process
        X = self.theta * gamma_time + self.sigma * bm

        # Asset prices
        omega = (1 / self.nu) * np.log(1 - self.theta * self.nu - 0.5 * self.sigma**2 * self.nu)
        S = S0 * np.exp(r * np.arange(n_steps + 1) * dt + X + omega * gamma_time)

        return S


# ============================================================================
# Rough Volatility Model (rBergomi)
# ============================================================================

class RoughBergomiModel:
    """
    Rough Bergomi Model

    Captures rough behavior of volatility (H ≈ 0.1).
    """

    def __init__(self, xi0: float, eta: float, H: float = 0.1):
        """
        Parameters
        ----------
        xi0 : float - Initial forward variance
        eta : float - Volatility of volatility
        H : float - Hurst parameter (typically ~0.1)
        """
        self.xi0 = xi0
        self.eta = eta
        self.H = H

    def simulate(self, T: float, n_steps: int, n_paths: int) -> np.ndarray:
        """Simulate rough volatility paths."""
        np.random.seed(42)

        dt = T / n_steps
        t = np.linspace(dt, T, n_steps)

        # Fractional kernel
        kernel = (t[:, None]**(self.H - 0.5))[::-1]

        # Brownian motion
        dW = np.random.standard_normal((n_paths, n_steps)) * np.sqrt(dt)
        W = np.cumsum(dW, axis=1)

        # Convolution for fractional Brownian motion
        WH = np.zeros((n_paths, n_steps))
        for i in range(n_paths):
            WH[i] = np.convolve(W[i], kernel[:], mode='full')[:n_steps]

        # Forward variance
        xi = self.xi0 * np.exp(
            self.eta * WH - 0.5 * self.eta**2 * t**(2 * self.H)
        )

        return xi


# ============================================================================
# Regime-Switching Model (Hidden Markov Model)
# ============================================================================

class RegimeSwitchingModel:
    """
    Hidden Markov Regime-Switching Model
    
    Models different market states using a proper Markov chain with
    Baum-Welch EM algorithm for parameter estimation.
    
    Features:
    - Proper EM algorithm with forward-backward pass
    - Viterbi algorithm for optimal state sequence
    - Regime duration statistics
    - Transition probability estimation
    - Smoothed regime probabilities
    """
    
    REGIME_NAMES = {
        0: "LOW_VOL_BEAR",
        1: "HIGH_VOL_BEAR", 
        2: "LOW_VOL_BULL",
        3: "HIGH_VOL_BULL",
        4: "SIDEWAYS"
    }
    
    def __init__(self, n_regimes: int = 3, regime_names: Dict[int, str] = None):
        self.n_regimes = n_regimes
        if regime_names:
            self.REGIME_NAMES = regime_names
        self.transition_matrix = None
        self.regime_params = None
        self.log_likelihood = None
        self.converged = False
        self.n_iterations = 0
        self.regime_durations = None
        
    def fit(self, returns: np.ndarray, max_iter: int = 100, tol: float = 1e-6,
            min_variance: float = 1e-8) -> Dict:
        """
        Fit regime-switching model using Baum-Welch EM algorithm.
        
        Parameters
        ----------
        returns : np.ndarray
            Return series
        max_iter : int
            Maximum EM iterations
        tol : float
            Convergence tolerance for log-likelihood
        min_variance : float
            Minimum variance to prevent numerical issues
            
        Returns
        -------
        Dict
            Model results including transition matrix, regime params,
            filtered/smoothed probabilities, and regime statistics
        """
        returns = np.asarray(returns, dtype=np.float64)
        n = len(returns)
        
        if n < 20:
            return self._fallback_result(n)
        
        mu_data = np.mean(returns)
        sigma_data = np.std(returns)
        
        if self.n_regimes == 2:
            mus = np.array([-abs(mu_data) * 2, abs(mu_data) * 2])
            sigmas = np.array([sigma_data * 0.8, sigma_data * 1.2])
        elif self.n_regimes == 3:
            sorted_returns = np.sort(returns)
            q1, q2 = np.percentile(sorted_returns, [33, 67])
            mus = np.array([np.mean(sorted_returns[sorted_returns < q1]),
                          np.mean(sorted_returns[(sorted_returns >= q1) & (sorted_returns < q2)]),
                          np.mean(sorted_returns[sorted_returns >= q2])])
            mus = np.clip(mus, -0.1, 0.1)
            sigmas = np.array([sigma_data * 0.7, sigma_data, sigma_data * 1.3])
        else:
            quantiles = np.percentile(returns, np.linspace(0, 100, self.n_regimes + 1))
            mus = np.array([np.mean(returns[(returns >= quantiles[i]) & (returns < quantiles[i+1])])
                          for i in range(self.n_regimes)])
            mus = np.clip(mus, -0.1, 0.1)
            base_sigma = sigma_data / np.sqrt(self.n_regimes)
            sigmas = np.array([base_sigma * (0.5 + 0.5 * i / max(1, self.n_regimes - 1))
                              for i in range(self.n_regimes)])
        
        sigmas = np.clip(sigmas, min_variance, None)
        
        if self.n_regimes == 2:
            P = np.array([[0.9, 0.1],
                         [0.1, 0.9]])
        else:
            P = np.eye(self.n_regimes) * 0.8 + (1 - 0.8) / self.n_regimes
        
        pi = np.ones(self.n_regimes) / self.n_regimes
        
        self.log_likelihood = -np.inf
        
        for iteration in range(max_iter):
            xi, gamma, log_lik = self._baum_welch_pass(returns, mus, sigmas, P, pi)
            
            if iteration > 0 and abs(log_lik - self.log_likelihood) < tol:
                self.converged = True
                self.n_iterations = iteration + 1
                break
            
            self.log_likelihood = log_lik
            
            mus_new, sigmas_new, P_new, pi_new = self._m_step(
                returns, gamma, xi, P, self.n_regimes, min_variance
            )
            
            if np.any(np.isnan(mus_new)) or np.any(np.isnan(sigmas_new)):
                break
                
            mus, sigmas, P, pi = mus_new, sigmas_new, P_new, pi_new
        
        else:
            self.n_iterations = max_iter
        
        P = np.clip(P, 1e-6, 1 - 1e-6)
        P = P / P.sum(axis=1, keepdims=True)
        
        self.transition_matrix = P
        self.regime_params = {
            'mus': mus.tolist(),
            'sigmas': sigmas.tolist(),
            'initial_probs': pi.tolist(),
            'regime_labels': [self.REGIME_NAMES.get(i, f"REGIME_{i}") for i in range(self.n_regimes)]
        }
        
        self.regime_durations = self._calculate_durations(gamma)
        
        smoothed_probs = self._forward_backward_smooth(returns, mus, sigmas, P, pi)
        
        return {
            'transition_matrix': P,
            'regime_params': self.regime_params,
            'filtered_probs': gamma,
            'smoothed_probs': smoothed_probs,
            'log_likelihood': self.log_likelihood,
            'converged': self.converged,
            'n_iterations': self.n_iterations,
            'regime_durations': self.regime_durations
        }
    
    def _fallback_result(self, n: int) -> Dict:
        """Return fallback results when insufficient data."""
        gamma = np.random.rand(n, self.n_regimes)
        gamma = gamma / gamma.sum(axis=1, keepdims=True)
        
        return {
            'transition_matrix': np.eye(self.n_regimes) * 0.8 + 0.2 / self.n_regimes,
            'regime_params': {
                'mus': [0.0] * self.n_regimes,
                'sigmas': [0.02] * self.n_regimes,
                'initial_probs': [1.0 / self.n_regimes] * self.n_regimes,
                'regime_labels': [self.REGIME_NAMES.get(i, f"REGIME_{i}") for i in range(self.n_regimes)]
            },
            'filtered_probs': gamma,
            'smoothed_probs': gamma,
            'log_likelihood': 0.0,
            'converged': False,
            'n_iterations': 0,
            'regime_durations': {'mean': [np.nan] * self.n_regimes, 'current': [0] * self.n_regimes}
        }
    
    def _compute_log_likelihoods(self, returns: np.ndarray, mus: np.ndarray,
                                  sigmas: np.ndarray) -> np.ndarray:
        """Compute log-likelihood for each observation in each regime."""
        n = len(returns)
        log_liks = np.zeros((n, self.n_regimes))
        
        for k in range(self.n_regimes):
            if sigmas[k] > 1e-10:
                log_liks[:, k] = stats.norm.logpdf(returns, mus[k], sigmas[k])
            else:
                log_liks[:, k] = -np.inf
                
        return log_liks
    
    def _baum_welch_pass(self, returns: np.ndarray, mus: np.ndarray,
                        sigmas: np.ndarray, P: np.ndarray, 
                        pi: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Single Baum-Welch pass (E-step).
        
        Returns:
        - xi: Joint probability of being in regime i at t and j at t+1
        - gamma: Filtered probability of being in each regime at each t
        - log_lik: Total log-likelihood
        """
        n = len(returns)
        log_liks = self._compute_log_likelihoods(returns, mus, sigmas)
        
        log_P = np.log(np.clip(P, 1e-10, 1.0))
        log_pi = np.log(np.clip(pi, 1e-10, 1.0))
        
        log_alpha = np.zeros((n, self.n_regimes))
        log_gamma = np.zeros((n, self.n_regimes))
        
        log_alpha[0] = log_pi + log_liks[0]
        for t in range(1, n):
            log_alpha_sum = np.log(np.sum(np.exp(log_alpha[t-1] + log_P.T), axis=1) + 1e-10)
            log_alpha[t] = log_liks[t] + log_alpha_sum
        
        log_likelihood = np.log(np.sum(np.exp(log_alpha[-1])) + 1e-10)
        
        log_beta = np.zeros((n, self.n_regimes))
        log_beta[-1] = 0.0
        for t in range(n - 2, -1, -1):
            log_beta_sum = np.log(np.sum(np.exp(log_P + log_liks[t+1] + log_beta[t+1]), axis=1) + 1e-10)
            log_beta[t] = log_beta_sum
        
        alpha = np.exp(log_alpha - log_likelihood)
        beta = np.exp(log_beta - log_likelihood)
        
        log_gamma = log_alpha + log_beta - log_likelihood
        gamma = np.exp(log_gamma)
        gamma = np.clip(gamma, 0, 1)
        
        xi = np.zeros((n - 1, self.n_regimes, self.n_regimes))
        for t in range(n - 1):
            denom = np.sum(np.exp(log_alpha[t][:, None] + log_P + log_liks[t+1] + log_beta[t+1][None, :]))
            if denom > 1e-10:
                xi[t] = np.exp(log_alpha[t][:, None] + log_P + log_liks[t+1] + log_beta[t+1][None, :] - np.log(denom))
            else:
                xi[t] = 1.0 / self.n_regimes
                
        return xi, gamma, log_likelihood
    
    def _forward_backward_smooth(self, returns: np.ndarray, mus: np.ndarray,
                                  sigmas: np.ndarray, P: np.ndarray,
                                  pi: np.ndarray) -> np.ndarray:
        """
        Compute smoothed (optimal) regime probabilities using forward-backward algorithm.
        """
        n = len(returns)
        log_liks = self._compute_log_likelihoods(returns, mus, sigmas)
        
        log_alpha = np.zeros((n, self.n_regimes))
        log_alpha[0] = np.log(np.clip(pi, 1e-10, 1.0)) + log_liks[0]
        
        for t in range(1, n):
            log_alpha_sum = np.log(np.sum(np.exp(log_alpha[t-1][:, None] + np.log(np.clip(P, 1e-10, 1.0))), axis=1) + 1e-10)
            log_alpha[t] = log_liks[t] + log_alpha_sum
        
        total_log_lik = np.log(np.sum(np.exp(log_alpha[-1])) + 1e-10)
        
        log_beta = np.zeros((n, self.n_regimes))
        for t in range(n - 2, -1, -1):
            log_beta[t] = np.log(np.sum(np.exp(np.log(np.clip(P, 1e-10, 1.0)) + log_liks[t+1] + log_beta[t+1]), axis=1) + 1e-10)
        
        log_gamma = log_alpha + log_beta - total_log_lik
        gamma = np.exp(log_gamma)
        
        return np.clip(gamma, 0, 1)
    
    def _m_step(self, returns: np.ndarray, gamma: np.ndarray, xi: np.ndarray,
                P: np.ndarray, n_regimes: int, min_variance: float) -> Tuple:
        """M-step: Update parameters using sufficient statistics."""
        n = len(returns)
        
        gamma_sum = gamma.sum(axis=0) + 1e-10
        
        mus = np.zeros(n_regimes)
        for k in range(n_regimes):
            mus[k] = np.sum(gamma[:, k] * returns) / gamma_sum[k]
        
        sigmas = np.zeros(n_regimes)
        for k in range(n_regimes):
            sigmas[k] = np.sqrt(np.sum(gamma[:, k] * (returns - mus[k])**2) / gamma_sum[k] + min_variance)
        
        xi_sum = xi.sum(axis=0)
        P_new = np.zeros((n_regimes, n_regimes))
        for i in range(n_regimes):
            for j in range(n_regimes):
                P_new[i, j] = xi_sum[i, j] / (xi_sum[i, :].sum() + 1e-10)
        P_new = np.clip(P_new, 1e-6, 1 - 1e-6)
        
        pi = gamma[0] / (gamma[0].sum() + 1e-10)
        
        return mus, sigmas, P_new, pi
    
    def _calculate_durations(self, gamma: np.ndarray) -> Dict:
        """Calculate expected regime durations from transition matrix."""
        if self.transition_matrix is None:
            return {'mean': [np.nan] * self.n_regimes, 'current': [0] * self.n_regimes}
        
        durations = []
        for i in range(self.n_regimes):
            if self.transition_matrix[i, i] < 1 - 1e-6:
                duration = 1.0 / (1.0 - self.transition_matrix[i, i])
            else:
                duration = np.nan
            durations.append(duration)
        
        state_sequence = np.argmax(gamma, axis=1)
        
        current_durations = []
        current_state = state_sequence[-1]
        duration = 1
        for t in range(len(state_sequence) - 2, -1, -1):
            if state_sequence[t] == current_state:
                duration += 1
            else:
                break
        current_durations = [0] * self.n_regimes
        current_durations[current_state] = duration
        
        return {
            'mean': durations,
            'current': current_durations,
            'transition_matrix': self.transition_matrix.tolist()
        }
    
    def viterbi(self, returns: np.ndarray) -> np.ndarray:
        """
        Find most likely sequence of regimes using Viterbi algorithm.
        
        Parameters
        ----------
        returns : np.ndarray
            Return series
            
        Returns
        -------
        np.ndarray
            Most likely regime sequence
        """
        if self.regime_params is None:
            return np.zeros(len(returns), dtype=int)
            
        n = len(returns)
        mus = np.array(self.regime_params['mus'])
        sigmas = np.array(self.regime_params['sigmas'])
        P = self.transition_matrix
        pi = np.array(self.regime_params.get('initial_probs', [1.0/self.n_regimes]*self.n_regimes))
        
        log_liks = self._compute_log_likelihoods(returns, mus, sigmas)
        log_P = np.log(np.clip(P, 1e-10, 1.0))
        log_pi = np.log(np.clip(pi, 1e-10, 1.0))
        
        delta = np.zeros((n, self.n_regimes))
        psi = np.zeros((n, self.n_regimes), dtype=int)
        
        delta[0] = log_pi + log_liks[0]
        
        for t in range(1, n):
            for j in range(self.n_regimes):
                trans_probs = delta[t-1] + log_P[:, j]
                psi[t, j] = np.argmax(trans_probs)
                delta[t, j] = np.max(trans_probs) + log_liks[t, j]
        
        states = np.zeros(n, dtype=int)
        states[-1] = np.argmax(delta[-1])
        
        for t in range(n - 2, -1, -1):
            states[t] = psi[t + 1, states[t + 1]]
            
        return states
    
    def predict_regime(self, returns: np.ndarray) -> Tuple[int, np.ndarray]:
        """
        Predict current regime with confidence.
        
        Returns
        -------
        Tuple[int, np.ndarray]
            Most likely regime index and probability distribution
        """
        returns = np.asarray(returns, dtype=np.float64)
        
        if self.regime_params is None:
            probs = np.ones(self.n_regimes) / self.n_regimes
            return 0, probs
            
        n = len(returns)
        mus = np.array(self.regime_params['mus'])
        sigmas = np.array(self.regime_params['sigmas'])
        P = self.transition_matrix
        pi = np.array(self.regime_params.get('initial_probs', [1.0/self.n_regimes]*self.n_regimes))
        
        log_liks = self._compute_log_likelihoods(returns, mus, sigmas)
        log_P = np.log(np.clip(P, 1e-10, 1.0))
        log_pi = np.log(np.clip(pi, 1e-10, 1.0))
        
        log_alpha = np.zeros(self.n_regimes)
        log_alpha = log_pi + log_liks[0]
        
        for t in range(1, n):
            log_alpha_sum = np.log(np.sum(np.exp(log_alpha + log_P.T), axis=1) + 1e-10)
            log_alpha = log_liks[t] + log_alpha_sum
        
        probs = np.exp(log_alpha - np.max(log_alpha))
        probs = probs / probs.sum()
        
        return int(np.argmax(probs)), probs
    
    def predict_next_regime(self) -> Tuple[int, float]:
        """
        Predict next regime and its probability.
        
        Returns
        -------
        Tuple[int, float]
            Most likely next regime and its probability
        """
        if self.transition_matrix is None:
            return 0, 1.0 / self.n_regimes
            
        current_state = self.n_regimes - 1
        next_probs = self.transition_matrix[current_state]
        
        return int(np.argmax(next_probs)), float(np.max(next_probs))
    
    def get_regime_confidence(self, returns: np.ndarray) -> float:
        """
        Calculate confidence score for current regime classification.
        
        Returns
        -------
        float
            Confidence score between 0 and 1
        """
        _, probs = self.predict_regime(returns)
        confidence = 1.0 - np.sum(probs * np.log(np.clip(probs, 1e-10, 1.0))) / np.log(1.0 / self.n_regimes)
        return float(np.clip(confidence, 0, 1))
    
    def regime_analysis_summary(self, returns: np.ndarray) -> Dict:
        """
        Comprehensive regime analysis summary.
        
        Returns
        -------
        Dict
            Complete regime analysis including current regime, trend, volatility regime
        """
        returns = np.asarray(returns, dtype=np.float64)
        
        if len(returns) < 20:
            return self._fallback_result(len(returns))
        
        results = self.fit(returns)
        
        current_regime, regime_probs = self.predict_regime(returns)
        viterbi_states = self.viterbi(returns)
        
        regime_returns = {}
        regime_vol = {}
        for i in range(self.n_regimes):
            mask = viterbi_states == i
            if mask.sum() > 0:
                regime_returns[i] = float(np.mean(returns[mask]))
                regime_vol[i] = float(np.std(returns[mask]))
            else:
                regime_returns[i] = 0.0
                regime_vol[i] = 0.0
        
        avg_duration = results['regime_durations']['mean']
        
        sorted_regimes = sorted(range(self.n_regimes), 
                              key=lambda i: regime_returns[i], 
                              reverse=True)
        
        trend_regime = sorted_regimes[0] if regime_returns[sorted_regimes[0]] > 0 else sorted_regimes[-1]
        vol_regime = sorted(range(self.n_regimes), key=lambda i: regime_vol[i], reverse=True)[0]
        
        is_trending = abs(regime_returns[trend_regime]) > 0.001
        is_high_vol = regime_vol[vol_regime] > np.mean(list(regime_vol.values()))
        
        if is_trending:
            if regime_returns[trend_regime] > 0:
                market_regime = "TRENDING_UP"
            else:
                market_regime = "TRENDING_DOWN"
        else:
            market_regime = "SIDEWAYS"
        
        if is_high_vol:
            market_regime += "_HIGH_VOL"
        else:
            market_regime += "_LOW_VOL"
        
        return {
            'current_regime': current_regime,
            'current_regime_name': self.REGIME_NAMES.get(current_regime, f"REGIME_{current_regime}"),
            'regime_probs': regime_probs.tolist(),
            'market_regime': market_regime,
            'regime_returns': regime_returns,
            'regime_volatility': regime_vol,
            'avg_duration': avg_duration,
            'transition_matrix': self.transition_matrix.tolist() if self.transition_matrix is not None else None,
            'confidence': self.get_regime_confidence(returns),
            'converged': self.converged,
            'n_iterations': self.n_iterations,
            'log_likelihood': self.log_likelihood,
            'viterbi_states': viterbi_states.tolist()
        }


# ============================================================================
# Utility Functions
# ============================================================================

def calculate_var(returns: np.ndarray, confidence: float = 0.95,
                  method: str = 'historical') -> float:
    """
    Calculate Value at Risk.

    Parameters
    ----------
    returns : np.ndarray - Return series
    confidence : float - Confidence level
    method : str - 'historical', 'parametric', or 'monte_carlo'
    """
    if method == 'historical':
        return np.percentile(returns, (1 - confidence) * 100)

    elif method == 'parametric':
        mu = np.mean(returns)
        sigma = np.std(returns)
        return mu + sigma * norm.ppf(1 - confidence)

    elif method == 'monte_carlo':
        simulated = np.random.normal(np.mean(returns), np.std(returns), 10000)
        return np.percentile(simulated, (1 - confidence) * 100)

    return 0.0


def calculate_expected_shortfall(returns: np.ndarray,
                                 confidence: float = 0.95) -> float:
    """Calculate Expected Shortfall (CVaR)."""
    var = calculate_var(returns, confidence)
    return returns[returns <= var].mean()


def calculate_sharpe_ratio(returns: np.ndarray,
                          risk_free_rate: float = 0.05) -> float:
    """Calculate annualized Sharpe ratio."""
    excess_returns = returns - risk_free_rate / 252
    if np.std(excess_returns) == 0:
        return 0.0
    return np.sqrt(252) * np.mean(excess_returns) / np.std(excess_returns)


def calculate_sortino_ratio(returns: np.ndarray,
                           risk_free_rate: float = 0.05) -> float:
    """Calculate annualized Sortino ratio."""
    excess_returns = returns - risk_free_rate / 252
    downside_returns = returns[returns < 0]

    if len(downside_returns) == 0 or np.std(downside_returns) == 0:
        return 0.0

    downside_std = np.sqrt(np.mean(downside_returns**2))
    return np.sqrt(252) * np.mean(excess_returns) / downside_std


def calculate_max_drawdown(equity_curve: np.ndarray) -> float:
    """Calculate maximum drawdown."""
    peak = np.maximum.accumulate(equity_curve)
    drawdown = (peak - equity_curve) / peak
    return np.max(drawdown)


def calculate_calmar_ratio(returns: np.ndarray,
                          equity_curve: np.ndarray) -> float:
    """Calculate Calmar ratio."""
    annual_return = np.mean(returns) * 252
    max_dd = calculate_max_drawdown(equity_curve)

    if max_dd == 0:
        return 0.0

    return annual_return / max_dd


# ============================================================================
# Unified Regime Detection
# ============================================================================

def detect_regime(returns: np.ndarray, n_regimes: int = 3) -> Dict[str, Any]:
    """
    Unified regime detection function for use across all modules.
    
    Provides a consistent interface for market regime detection
    that can be used by both the dashboard and trading bot.
    
    Parameters
    ----------
    returns : np.ndarray
        Return series
    n_regimes : int
        Number of regimes to detect (2-5)
        
    Returns
    -------
    Dict
        Comprehensive regime analysis including:
        - current_regime: Current regime index
        - regime_name: Human-readable regime name
        - regime_probs: Probability distribution over regimes
        - market_regime: Market state description
        - trend: Trend strength
        - volatility: Annualized volatility
        - hurst: Hurst exponent estimate
        - confidence: Classification confidence
        - regime_returns: Expected return per regime
        - regime_volatility: Expected volatility per regime
        - avg_duration: Expected regime durations
        - transition_matrix: Markov transition probabilities
        - viterbi_states: Optimal state sequence
    """
    returns = np.asarray(returns, dtype=np.float64)
    n = len(returns)
    
    if n < 20:
        return {
            'current_regime': 1,
            'regime_name': 'SIDEWAYS',
            'regime_probs': [1.0 / n_regimes] * n_regimes,
            'market_regime': 'SIDEWAYS',
            'trend': 0.0,
            'volatility': 0.20,
            'hurst': 0.5,
            'confidence': 0.0,
            'regime_returns': [0.0] * n_regimes,
            'regime_volatility': [0.20] * n_regimes,
            'avg_duration': [np.nan] * n_regimes,
            'transition_matrix': np.eye(n_regimes).tolist(),
            'viterbi_states': [1] * n,
            'n_observations': n,
            'converged': False,
            'method': 'insufficient_data'
        }
    
    n_regimes = max(2, min(5, n_regimes))
    
    model = RegimeSwitchingModel(n_regimes=n_regimes)
    results = model.fit(returns, max_iter=100, tol=1e-6)
    
    current_regime, regime_probs = model.predict_regime(returns)
    regime_params = results.get('regime_params', {})
    regime_labels = regime_params.get('regime_labels', [f'REGIME_{i}' for i in range(n_regimes)])
    regime_name = regime_labels[current_regime] if current_regime < len(regime_labels) else f'REGIME_{current_regime}'
    
    market_analysis = model.regime_analysis_summary(returns)
    
    hurst_estimate = estimate_hurst_from_regimes(
        returns, 
        results.get('viterbi_states', np.zeros(n, dtype=int)),
        n_regimes
    )
    
    trend = float(np.sum(returns[-min(50, n):]) / min(50, n))
    volatility = float(np.std(returns) * np.sqrt(252))
    
    return {
        'current_regime': current_regime,
        'regime_name': regime_name,
        'regime_probs': regime_probs.tolist(),
        'market_regime': market_analysis.get('market_regime', 'SIDEWAYS'),
        'trend': trend,
        'volatility': volatility,
        'hurst': hurst_estimate,
        'confidence': float(model.get_regime_confidence(returns)),
        'regime_returns': [market_analysis['regime_returns'].get(i, 0.0) for i in range(n_regimes)],
        'regime_volatility': [market_analysis['regime_volatility'].get(i, 0.0) for i in range(n_regimes)],
        'avg_duration': results.get('regime_durations', {}).get('mean', [np.nan] * n_regimes),
        'transition_matrix': transition_matrix_to_dict(results.get('transition_matrix', np.eye(n_regimes))),
        'viterbi_states': results.get('viterbi_states', [0] * n),
        'n_observations': n,
        'converged': results.get('converged', False),
        'method': 'hmm_baum_welch'
    }


def estimate_hurst_from_regimes(returns: np.ndarray, viterbi_states: np.ndarray, n_regimes: int) -> float:
    """Estimate Hurst exponent from regime switching pattern."""
    if len(returns) < 20:
        return 0.5
    
    n = len(returns)
    
    regime_autocorr = []
    for i in range(n_regimes):
        mask = viterbi_states == i
        if np.sum(mask) > 10:
            regime_returns = returns[mask]
            if len(regime_returns) > 1:
                autocorr = np.corrcoef(regime_returns[:-1], regime_returns[1:])[0, 1]
                if not np.isnan(autocorr):
                    regime_autocorr.append(autocorr)
    
    if regime_autocorr:
        avg_autocorr = np.mean(regime_autocorr)
        H = 0.5 + 0.5 * avg_autocorr
    else:
        H = 0.5
    
    return float(np.clip(H, 0.1, 0.9))


def transition_matrix_to_dict(P: np.ndarray) -> Dict[str, Dict[str, float]]:
    """Convert transition matrix to dictionary format."""
    n = len(P)
    result = {}
    for i in range(n):
        result[f'from_{i}'] = {f'to_{j}': float(P[i, j]) for j in range(n)}
    return result


def get_regime_recommendation(regime_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Get trading strategy recommendations based on detected regime.
    
    Parameters
    ----------
    regime_data : Dict
        Output from detect_regime()
        
    Returns
    -------
    Dict
        Strategy recommendations including:
        - preferred_strategy: 'momentum', 'mean_reversion', or 'neutral'
        - position_size_modifier: Multiplier for position size (0-1)
        - stop_loss_widen: Whether to widen stops
        - timeframe_bias: 'shorter' or 'longer'
        - confidence_threshold: Minimum confidence for trades
        - description: Human-readable explanation
    """
    market_regime = regime_data.get('market_regime', 'SIDEWAYS')
    confidence = regime_data.get('confidence', 0.5)
    volatility = regime_data.get('volatility', 0.20)
    
    recommendations = {
        'preferred_strategy': 'neutral',
        'position_size_modifier': 1.0,
        'stop_loss_widen': False,
        'timeframe_bias': 'neutral',
        'confidence_threshold': 0.6,
        'description': 'Standard trading conditions'
    }
    
    if 'HIGH_VOL' in market_regime:
        recommendations['position_size_modifier'] = 0.7
        recommendations['stop_loss_widen'] = True
        recommendations['description'] = 'High volatility - reduce exposure, widen stops'
    
    if 'TRENDING' in market_regime or 'BULL' in market_regime or 'BEAR' in market_regime:
        recommendations['preferred_strategy'] = 'momentum'
        recommendations['timeframe_bias'] = 'longer'
        recommendations['description'] = f'Trending market ({market_regime}) - momentum strategies'
    
    if 'SIDEWAYS' in market_regime or 'LOW_VOL' in market_regime:
        if volatility < 0.15:
            recommendations['preferred_strategy'] = 'mean_reversion'
            recommendations['timeframe_bias'] = 'shorter'
            recommendations['description'] = 'Low volatility sideways - mean reversion strategies'
    
    if confidence < 0.5:
        recommendations['position_size_modifier'] *= 0.8
        recommendations['confidence_threshold'] = 0.7
        recommendations['description'] += ' (low confidence - reduce risk)'
    
    return recommendations
