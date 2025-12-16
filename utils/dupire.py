import numpy as np
from utils.svi_model import svi_derivatives

def compute_local_vol(k, T, params):
    """
    Computes Local Volatility using Dupire's formula on the SVI surface.
    """
    # 1. Get Spatial Derivatives from SVI
    w, dw_dk, d2w_dk2 = svi_derivatives(k, params)
    
    # 2. Time Derivative assumption (Flat term structure approximation for MVP)
    # Ideally, we would fit SVI for two time slices T1 and T2 and compute (w2-w1)/(T2-T1)
    # Here, we assume linear variance growth: dw/dT = w / T
    dw_dT = w / T 
    
    # 3. Dupire Formula numerator/denominator
    numerator = dw_dT
    denominator = 1 - (k/w)*dw_dk + 0.25*(-0.25 - 1/w + (k**2)/(w**2))*(dw_dk**2) + 0.5*d2w_dk2
    
    # Clamp denominator to avoid explosion
    denominator = np.maximum(denominator, 1e-4)
    
    local_var = numerator / denominator
    return np.sqrt(np.maximum(local_var, 0))