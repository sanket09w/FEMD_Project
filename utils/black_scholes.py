import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq

def black_scholes_price(S, K, T, r, q, sigma, option_type='CE'):
    """Standard Black-Scholes Pricing Formula"""
    if T <= 0: return max(0, S-K) if option_type == 'CE' else max(0, K-S)
    
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    if option_type == 'CE':
        return S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)

def get_implied_volatility(market_price, S, K, T, r, q, option_type='CE'):
    """Root-finding to get IV from Price"""
    if market_price <= 0: return np.nan
    
    # Intrinsic check
    intrinsic = max(S * np.exp(-q * T) - K * np.exp(-r * T), 0) if option_type == 'CE' else max(K * np.exp(-r * T) - S * np.exp(-q * T), 0)
    if market_price <= intrinsic + 0.5: return np.nan
    
    def objective(sigma):
        return black_scholes_price(S, K, T, r, q, sigma, option_type) - market_price
    
    try:
        return brentq(objective, 0.001, 5.0)
    except:
        return np.nan