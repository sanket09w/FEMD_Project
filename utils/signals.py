import numpy as np
from utils.black_scholes import black_scholes_price
from utils.svi_model import raw_svi_model
from utils.dupire import compute_local_vol

def calculate_fair_metrics(S, K, T, r, q, svi_params, market_price):
    """
    Combines SVI Fair Price, Dupire Local Vol, and Z-Score into one result.
    """
    k = np.log(K / S)
    
    # 1. Fair Price Calculation
    w_fair = raw_svi_model(k, *svi_params)
    sigma_fair = np.sqrt(max(w_fair, 1e-6) / T)
    fair_price = black_scholes_price(S, K, T, r, q, sigma_fair, 'CE')
    
    # 2. Local Volatility
    local_vol = compute_local_vol(k, T, svi_params)
    
    # 3. Z-Score Calculation
    # We assume 'noise' in the market is roughly 5% of the option price
    # A Z-score > 1.96 means the deviation is statistically significant
    gap = market_price - fair_price
    std_dev_proxy = fair_price * 0.05 
    
    if std_dev_proxy == 0: 
        z_score = 0 
    else:
        z_score = gap / std_dev_proxy
    
    return fair_price, sigma_fair, local_vol, z_score