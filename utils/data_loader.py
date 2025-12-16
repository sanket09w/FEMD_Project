import pandas as pd
import numpy as np
from nselib import derivatives
from datetime import datetime
from utils.math_model import black_scholes_price # Importing from your sibling file

def generate_synthetic_data(spot=24000):
    """Generates a multi-expiry surface for 3D plotting"""
    data = []
    # Create expiries: 7 days, 1 month, 2 months, 3 months, 6 months
    days_to_expiry = [7, 30, 60, 90, 180]
    
    for days in days_to_expiry:
        T = days / 365.0
        # Strikes range depends on time (volatility cone)
        # Further out = wider strikes
        range_pct = 0.05 + (0.05 * (days/30)) 
        min_strike = int(spot * (1 - range_pct))
        max_strike = int(spot * (1 + range_pct))
        strikes = np.arange(min_strike, max_strike, 100)
        
        for K in strikes:
            moneyness = np.log(K / spot)
            
            # Surface Formula: 
            # 1. Smile: Quadratic in moneyness (k^2)
            # 2. Term Structure: IV increases/decreases with time (sqrt(T))
            
            # Base vol + Smile + Term structure tilt
            fake_iv = 0.14 + (0.5 * moneyness**2) + (0.02 * np.log(T*10))
            
            # Add noise
            fake_iv += np.random.normal(0, 0.002)
            
            # Calculate Price
            price = black_scholes_price(spot, K, T, 0.1, 0.0, fake_iv, 'CE')
            
            data.append({
                'StrikePrice': K,
                'ExpiryDays': days,      # New Column for Y-Axis
                'Time_T': T,
                'CE_LTP': round(price, 2),
                'PE_LTP': 0,
                'ExpiryDate': f"In {days} Days"
            })
            
    return pd.DataFrame(data)

def fetch_nse_data(symbol, expiry=None):
    """Fetches and cleans live NSE data"""
    try:
        raw = derivatives.nse_live_option_chain(symbol)
        # Clean numeric columns
        cols = ['StrikePrice', 'CE_LTP', 'PE_LTP', 'underlyingValue']
        for c in cols:
            if c in raw.columns:
                raw[c] = pd.to_numeric(raw[c].astype(str).str.replace(',', ''), errors='coerce')
        return raw
    except Exception as e:
        print(f"Error: {e}")
        return pd.DataFrame()