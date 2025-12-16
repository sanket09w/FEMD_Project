import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq, minimize

# --- BLACK-SCHOLES & IV ---
def black_scholes_price(S, K, T, r, q, sigma, option_type='CE'):
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type == 'CE':
        return S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)

def get_implied_volatility(market_price, S, K, T, r, q, option_type='CE'):
    if market_price <= 0: return np.nan
    intrinsic = max(S * np.exp(-q * T) - K * np.exp(-r * T), 0) if option_type == 'CE' else max(K * np.exp(-r * T) - S * np.exp(-q * T), 0)
    if market_price <= intrinsic + 0.5: return np.nan
    
    def objective(sigma):
        return black_scholes_price(S, K, T, r, q, sigma, option_type) - market_price
    try:
        return brentq(objective, 0.01, 5.0)
    except:
        return np.nan

# --- SVI MODEL ---
def raw_svi_model(k, a, b, rho, m, sigma):
    """Returns Total Variance w(k) = sigma_bs^2 * T"""
    return a + b * (rho * (k - m) + np.sqrt((k - m)**2 + sigma**2))

def calibrate_svi(strikes, ivs, T, S):
    k_data = np.log(np.array(strikes) / S)
    w_data = (np.array(ivs) ** 2) * T
    
    def objective(params):
        a, b, rho, m, sigma_val = params
        w_model = raw_svi_model(k_data, a, b, rho, m, sigma_val)
        if np.any(w_model < 0): return 1e6
        return np.sum((w_model - w_data)**2)

    initial_guess = [0.01, 0.1, -0.5, 0.0, 0.1]
    bounds = [(None, None), (0.001, None), (-0.99, 0.99), (None, None), (0.001, None)]
    
    result = minimize(objective, initial_guess, bounds=bounds, method='L-BFGS-B')
    return result.x