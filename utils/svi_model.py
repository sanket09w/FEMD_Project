import numpy as np
from scipy.optimize import minimize

def raw_svi_model(k, a, b, rho, m, sigma):
    """Returns Total Variance w(k) = a + b * (rho*(k-m) + sqrt((k-m)^2 + sigma^2))"""
    return a + b * (rho * (k - m) + np.sqrt((k - m)**2 + sigma**2))

def svi_derivatives(k, params):
    """Computes w, w', and w'' for arbitrage checks and Dupire"""
    a, b, rho, m, sigma = params
    discr = np.sqrt((k - m)**2 + sigma**2)
    w = a + b * (rho * (k - m) + discr)
    
    dw_dk = b * (rho + (k - m) / discr)
    d2w_dk2 = b * (1 / discr - (k - m)**2 / discr**3) 
    return w, dw_dk, d2w_dk2

def check_butterfly_arbitrage(k_grid, params):
    """Returns True if probability density is non-negative everywhere"""
    w, dw, d2w = svi_derivatives(k_grid, params)
    w = np.maximum(w, 1e-6) # Avoid div/0
    
    # g(k) is proportional to risk-neutral density
    g = (1 - k_grid * dw / (2 * w))**2 - (dw**2 / 4) * (0.25 + 1/w) + d2w / 2
    return np.all(g >= 0), g

def calibrate_svi(strikes, ivs, T, S):
    k_data = np.log(np.array(strikes) / S)
    w_data = (np.array(ivs) ** 2) * T
    
    def objective(params):
        a, b, rho, m, sigma_val = params
        w_model = raw_svi_model(k_data, *params)
        error = np.sum((w_model - w_data)**2)
        
        # Soft Constraints
        if np.any(w_model < 0): error += 1e6
        if b < 0 or sigma_val < 0: error += 1e6
        if abs(rho) > 1: error += 1e6
        return error

    initial_guess = [0.01, 0.1, -0.5, 0.0, 0.1]
    bounds = [(None, None), (0.001, None), (-0.999, 0.999), (None, None), (0.001, None)]
    
    try:
        result = minimize(objective, initial_guess, bounds=bounds, method='L-BFGS-B')
        return result.x
    except:
        return initial_guess