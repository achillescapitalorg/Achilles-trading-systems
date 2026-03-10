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
from typing import Dict, List, Tuple, Optional, Callable
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
# Regime-Switching Model
# ============================================================================

class RegimeSwitchingModel:
    """
    Markov Regime-Switching Model

    Models different market states (bull, bear, high vol, low vol).
    """

    def __init__(self, n_regimes: int = 2):
        self.n_regimes = n_regimes
        self.transition_matrix = None
        self.regime_params = None

    def fit(self, returns: np.ndarray, max_iter: int = 100) -> Dict:
        """
        Fit regime-switching model using EM algorithm.
        """
        n = len(returns)

        # Initialize parameters
        mus = np.array([np.mean(returns), -np.mean(returns)])
        sigmas = np.array([np.std(returns), np.std(returns) * 2])
        probs = np.ones(self.n_regimes) / self.n_regimes

        # Transition matrix
        P = np.ones((self.n_regimes, self.n_regimes)) * 0.5
        P += np.eye(self.n_regimes) * 0.5

        # EM algorithm
        for iteration in range(max_iter):
            # E-step: Calculate regime probabilities
            gamma = self._e_step(returns, mus, sigmas, P, probs)

            # M-step: Update parameters
            mus_new, sigmas_new, P_new, probs_new = self._m_step(
                returns, gamma, self.n_regimes
            )

            # Check convergence
            if np.max(np.abs(mus_new - mus)) < 1e-6:
                break

            mus, sigmas, P, probs = mus_new, sigmas_new, P_new, probs_new

        self.transition_matrix = P
        self.regime_params = {'mus': mus, 'sigmas': sigmas, 'probs': probs}

        return {
            'transition_matrix': P,
            'regime_params': self.regime_params,
            'regime_probs': gamma
        }

    def _e_step(self, returns: np.ndarray, mus: np.ndarray,
                sigmas: np.ndarray, P: np.ndarray,
                probs: np.ndarray) -> np.ndarray:
        """E-step: Calculate posterior regime probabilities."""
        n = len(returns)
        gamma = np.zeros((n, self.n_regimes))

        for t in range(n):
            likelihoods = np.array([
                stats.norm.pdf(returns[t], mus[k], sigmas[k])
                for k in range(self.n_regimes)
            ])
            gamma[t] = probs * likelihoods
            gamma[t] /= gamma[t].sum()

        return gamma

    def _m_step(self, returns: np.ndarray, gamma: np.ndarray,
                n_regimes: int) -> Tuple:
        """M-step: Update model parameters."""
        n = len(returns)

        # Update means
        mus = np.array([
            np.sum(gamma[:, k] * returns) / gamma[:, k].sum()
            for k in range(n_regimes)
        ])

        # Update volatilities
        sigmas = np.array([
            np.sqrt(np.sum(gamma[:, k] * (returns - mus[k])**2) / gamma[:, k].sum())
            for k in range(n_regimes)
        ])

        # Update transition probabilities
        P = np.ones((n_regimes, n_regimes)) / n_regimes

        # Update regime probabilities
        probs = gamma.mean(axis=0)

        return mus, sigmas, P, probs

    def predict_regime(self, returns: np.ndarray) -> int:
        """Predict current market regime."""
        if self.regime_params is None:
            return 0

        mus = self.regime_params['mus']
        sigmas = self.regime_params['sigmas']

        current_return = returns[-1]
        likelihoods = [
            stats.norm.pdf(current_return, mus[k], sigmas[k])
            for k in range(self.n_regimes)
        ]

        return int(np.argmax(likelihoods))


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
