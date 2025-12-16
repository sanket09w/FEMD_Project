import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq, minimize
from nselib import derivatives
from datetime import datetime, timedelta
import plotly.graph_objects as go

# --- CONFIGURATION ---
st.set_page_config(page_title="FEMD Engine: SVI Calibration", layout="wide")
st.title("Step 4: SVI Surface Calibration")

# --- 1. CORE MATH: BLACK-SCHOLES & IV ---
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

# --- 2. NEW MATH: SVI MODEL ---
def raw_svi_model(k, a, b, rho, m, sigma):
    """
    Returns Total Variance w(k) = sigma_bs^2 * T
    Formula: a + b * (rho * (k - m) + sqrt((k - m)^2 + sigma^2))
    """
    return a + b * (rho * (k - m) + np.sqrt((k - m)**2 + sigma**2))

def calibrate_svi(strikes, ivs, T, S):
    # 1. Prepare Data
    # k = log-moneyness = ln(Strike / Spot)
    # Note: Professional SVI uses ln(K/Forward), but ln(K/S) is fine for MVP
    k_data = np.log(np.array(strikes) / S)
    
    # w = Total Variance = IV^2 * T
    w_data = (np.array(ivs) ** 2) * T
    
    # 2. Define Optimization Objective (Minimize Error)
    def objective(params):
        a, b, rho, m, sigma_val = params
        w_model = raw_svi_model(k_data, a, b, rho, m, sigma_val)
        # Penalty for negative variance
        if np.any(w_model < 0): return 1e6
        return np.sum((w_model - w_data)**2) # Sum of Squared Errors

    # 3. Initial Guesses & Bounds
    # a, b, rho, m, sigma
    initial_guess = [0.01, 0.1, -0.5, 0.0, 0.1]
    bounds = [
        (None, None),   # a
        (0.001, None),  # b > 0
        (-0.99, 0.99),  # -1 < rho < 1
        (None, None),   # m
        (0.001, None)   # sigma > 0
    ]

    result = minimize(objective, initial_guess, bounds=bounds, method='L-BFGS-B')
    return result.x # Returns [a, b, rho, m, sigma]

# --- 3. DATA HANDLERS ---
def generate_synthetic_data(spot=24000):
    """Generates fake option chain when NSE is down"""
    strikes = np.arange(spot-1000, spot+1000, 100)
    data = []
    # Create a "fake" smile: IV is higher for OTM Puts and OTM Calls
    for K in strikes:
        moneyness = K / spot
        # Parabolic shape for IV (Vol Smile)
        fake_iv = 0.15 + 0.5 * (np.log(moneyness))**2 
        
        # Add some random noise to simulate real market messiness
        fake_iv += np.random.normal(0, 0.005) 
        
        # Calculate Option Price from this fake IV
        price = black_scholes_price(spot, K, 30/365, 0.1, 0.0, fake_iv, 'CE')
        
        data.append({
            'StrikePrice': K,
            'CE_LTP': round(price, 2),
            'PE_LTP': 0, # Ignored for this demo
            'ExpiryDate': 'Dummy-Expiry'
        })
    return pd.DataFrame(data)

# --- 4. UI LOGIC ---
st.sidebar.header("Data Source")
data_source = st.sidebar.radio("Select Mode", ["Test Mode (Synthetic)", "Live NSE Data"])

spot_price = 24000.0
T = 30/365.0 # Default 30 days
df = pd.DataFrame()

# A. FETCH DATA
if data_source == "Test Mode (Synthetic)":
    st.warning("Running in TEST MODE with Synthetic Data (NSE bypassed)")
    df = generate_synthetic_data(spot_price)
    
else:
    # Attempt Live Fetch
    symbol = st.sidebar.selectbox("Symbol", ["NIFTY", "BANKNIFTY"])
    try:
        raw = derivatives.nse_live_option_chain(symbol)
        spot_price = float(str(raw['underlyingValue'].iloc[0]).replace(',', ''))
        # Simple cleaning
        cols = ['StrikePrice', 'CE_LTP']
        for c in cols: raw[c] = pd.to_numeric(raw[c].astype(str).str.replace(',', ''), errors='coerce')
        
        # Filter Expiry
        exps = sorted(list(set(raw['ExpiryDate'])))
        sel_exp = st.sidebar.selectbox("Expiry", exps)
        df = raw[raw['ExpiryDate'] == sel_exp].copy()
        
        # Recalculate T
        exp_dt = datetime.strptime(sel_exp, "%d-%b-%Y")
        T = (exp_dt - datetime.now()).days / 365.0
        if T<=0: T=1/365.0
        
    except Exception as e:
        st.error(f"Live Data Failed: {e}. Switch to Test Mode.")

# B. PROCESS & CALIBRATE
if not df.empty:
    st.metric("Spot Price", spot_price)
    
    # 1. Calculate Raw IVs
    strikes = []
    market_ivs = []
    
    for i, row in df.iterrows():
        iv = get_implied_volatility(row['CE_LTP'], spot_price, row['StrikePrice'], T, 0.1, 0.0, 'CE')
        if not np.isnan(iv) and 0.01 < iv < 2.0: # Filter garbage
            strikes.append(row['StrikePrice'])
            market_ivs.append(iv)
            
    # 2. Fit SVI Model
    if len(strikes) > 5:
        params = calibrate_svi(strikes, market_ivs, T, spot_price)
        a_opt, b_opt, rho_opt, m_opt, sigma_opt = params
        
        # 3. Generate Smooth Curve Points
        smooth_strikes = np.linspace(min(strikes), max(strikes), 100)
        k_smooth = np.log(smooth_strikes / spot_price)
        w_smooth = raw_svi_model(k_smooth, *params)
        
        # Convert Total Variance (w) back to Implied Vol (IV)
        # IV = sqrt(w / T)
        svi_ivs = np.sqrt(w_smooth / T)
        
        # 4. PLOT
        fig = go.Figure()
        # Raw Market Data (Dots)
        fig.add_trace(go.Scatter(x=strikes, y=market_ivs, mode='markers', name='Market IV (Raw)', marker=dict(color='red', size=8)))
        # SVI Model (Line)
        fig.add_trace(go.Scatter(x=smooth_strikes, y=svi_ivs, mode='lines', name='SVI Fair Value', line=dict(color='cyan', width=3)))
        
        fig.update_layout(title="Volatility Surface: Market vs SVI Model", xaxis_title="Strike", yaxis_title="Implied Volatility", template="plotly_dark")
        st.plotly_chart(fig)
        
        # 5. SHOW PARAMETERS
        st.info(f"SVI Parameters calibrated: a={a_opt:.4f}, b={b_opt:.4f}, rho={rho_opt:.4f}, m={m_opt:.4f}, sigma={sigma_opt:.4f}")
    else:
        st.error("Not enough valid data points to calibrate SVI.")