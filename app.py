import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime

# --- IMPORTS FROM OUR NEW MODULES ---
from utils.math_model import get_implied_volatility, calibrate_svi, raw_svi_model
from utils.data_loader import generate_synthetic_data, fetch_nse_data

st.set_page_config(page_title="FEMD Engine", layout="wide")
st.title("FEMD: Volatility Surface Engine")

# --- UI CONTROLS ---
st.sidebar.header("Configuration")
data_source = st.sidebar.radio("Data Source", ["Test Mode (Synthetic)", "Live NSE Data"])
symbol = st.sidebar.selectbox("Symbol", ["NIFTY", "BANKNIFTY"])

# --- DATA LOADING ---
spot_price = 24000.0
T = 30/365.0
df = pd.DataFrame()

if data_source == "Test Mode (Synthetic)":
    st.info("Using Synthetic Data Generator")
    df = generate_synthetic_data(spot_price)
else:
    # Live Fetch Logic
    with st.spinner("Connecting to NSE..."):
        raw_df = fetch_nse_data(symbol)
        if not raw_df.empty:
            spot_price = raw_df['underlyingValue'].iloc[0]
            # Expiry Selection
            exps = sorted(list(set(raw_df['ExpiryDate'])))
            sel_exp = st.sidebar.selectbox("Select Expiry", exps)
            df = raw_df[raw_df['ExpiryDate'] == sel_exp].copy()
            
            # Recalculate T
            exp_dt = datetime.strptime(sel_exp, "%d-%b-%Y")
            T = (exp_dt - datetime.now()).days / 365.0
            if T <= 0: T = 1/365.0
        else:
            st.error("Failed to fetch live data.")

# ---# ... (Top part of app.py stays the same) ...

# --- PROCESSING & PLOTTING ---
if not df.empty:
    st.metric("Underlying Spot", f"{spot_price:,.2f}")

    # Check if we have multiple expiries (for 3D) or just one (for 2D)
    unique_expiries = df['ExpiryDate'].unique()
    
    # --- 3D SURFACE VIEW (If multiple expiries exist) ---
    if len(unique_expiries) > 1:
        st.subheader("3D Volatility Surface")
        
        # Prepare 3D Data
        x_strikes = []
        y_time = []
        z_iv = []
        
        # We need to calculate IVs for all rows first
        df['Calculated_IV'] = df.apply(
            lambda row: get_implied_volatility(
                row['CE_LTP'], spot_price, row['StrikePrice'], 
                row.get('Time_T', T), # Use row-specific T if available
                0.1, 0.0, 'CE'
            ), axis=1
        )
        
        # Filter clean data
        clean_df = df.dropna(subset=['Calculated_IV'])
        clean_df = clean_df[(clean_df['Calculated_IV'] > 0.01) & (clean_df['Calculated_IV'] < 2.0)]
        
        # Plotly 3D Scatter
        fig = go.Figure(data=[go.Scatter3d(
            x=clean_df['StrikePrice'],
            y=clean_df['ExpiryDays'], # or Time_T
            z=clean_df['Calculated_IV'],
            mode='markers',
            marker=dict(
                size=4,
                color=clean_df['Calculated_IV'],
                colorscale='Viridis',
                opacity=0.8
            )
        )])
        
        fig.update_layout(
            scene = dict(
                xaxis_title='Strike Price',
                yaxis_title='Days to Expiry',
                zaxis_title='Implied Volatility'
            ),
            margin=dict(l=0, r=0, b=0, t=0),
            height=600
        )
        st.plotly_chart(fig, use_container_width=True)

    # --- 2D SLICE VIEW (Standard SVI) ---
    else:
        # (This is your existing 2D Logic)
        st.subheader(f"2D Volatility Smile ({unique_expiries[0]})")
        
        strikes, market_ivs = [], []
        for _, row in df.iterrows():
            iv = get_implied_volatility(row['CE_LTP'], spot_price, row['StrikePrice'], T, 0.1, 0.0, 'CE')
            if not np.isnan(iv) and 0.01 < iv < 2.0:
                strikes.append(row['StrikePrice'])
                market_ivs.append(iv)
                
        if len(strikes) > 5:
            params = calibrate_svi(strikes, market_ivs, T, spot_price)
            smooth_strikes = np.linspace(min(strikes), max(strikes), 100)
            k_smooth = np.log(smooth_strikes / spot_price)
            w_smooth = raw_svi_model(k_smooth, *params)
            svi_ivs = np.sqrt(w_smooth / T)
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=strikes, y=market_ivs, mode='markers', name='Market IV'))
            fig.add_trace(go.Scatter(x=smooth_strikes, y=svi_ivs, mode='lines', name='SVI Fit', line=dict(color='cyan')))
            st.plotly_chart(fig, use_container_width=True)