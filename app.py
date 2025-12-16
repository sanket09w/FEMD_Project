import streamlit as st
import pandas as pd
import nselib
from nselib import derivatives

# 1. Page Config
st.set_page_config(page_title="NSE Fair-Value Engine", layout="wide")
st.title("Step 1: Raw Data Fetcher")

# 2. Sidebar for Inputs
st.sidebar.header("Select Parameters")
symbol = st.sidebar.selectbox("Select Index", ["NIFTY", "BANKNIFTY", "FINNIFTY"])

# 3. Fetch Data Function
@st.cache_data(ttl=60)  # Cache for 60 seconds to prevent banning
def get_raw_data(symbol):
    try:
        # Fetch live data from NSE
        df = derivatives.nse_live_option_chain(symbol)
        return df
    except Exception as e:
        st.error(f"Error fetching data from NSE: {e}")
        return pd.DataFrame()

# 4. Main Execution
st.write(f"Fetching data for **{symbol}**...")
raw_df = get_raw_data(symbol)

if not raw_df.empty:
    # Get all unique expiry dates available in the chain
    available_expiries = sorted(list(set(raw_df['ExpiryDate'])))
    
    # Let user select expiry
    selected_expiry = st.sidebar.selectbox("Select Expiry", available_expiries)
    
    # Filter data for that expiry
    filtered_df = raw_df[raw_df['ExpiryDate'] == selected_expiry]
    
    # Display
    st.success(f"Found {len(filtered_df)} contracts for expiry {selected_expiry}")
    st.dataframe(filtered_df)
else:
    st.warning("No data received. Market might be closed or connection failed.")