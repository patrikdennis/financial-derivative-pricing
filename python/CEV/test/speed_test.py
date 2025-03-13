############################################
# timing_test.py
############################################
import time
import numpy as np
import pandas as pd
from python.CEV.CEVPricer import CEVoptionPricer

def timing_monte_carlo(pricer, n_euler, sims_list):
    """
    Measures computation time for the call option price
    as we vary the number of Monte Carlo paths.
    """
    records = []
    for N in sims_list:
        start_time = time.time()
        call_price, std_err = pricer.monte_carlo_call(n_euler, N)
        elapsed = time.time() - start_time
        
        records.append({
            'N_sims': N,
            'Call_Price': call_price,
            'Std_Err': std_err,
            'Time_s': elapsed
        })
    df = pd.DataFrame(records)
    return df

def timing_finite_difference(pricer, n_fd, m_list):
    """
    Measures computation time for the call option price
    as we vary the number of space steps m, with n fixed.
    """
    records = []
    for m_fd in m_list:
        start_time = time.time()
        call_price_fd, _, _, _ = pricer.crank_nicolson_call(n_fd, m_fd)
        elapsed = time.time() - start_time
        
        records.append({
            'm_steps': m_fd,
            'Call_Price_FD': call_price_fd,
            'Time_s': elapsed
        })
    df = pd.DataFrame(records)
    return df

def main():
    s0      = 30.0
    sigma   = 0.1
    r       = 0.02
    K       = 40.0
    T       = 0.5
    delta   = 1.0    # => Black–Scholes PDE
    X       = 4 * K
    
    # Instantiate the pricer once (we'll reuse it).
    pricer = CEVoptionPricer(s0, sigma, r, K, T, delta, X)
    
    # --- 1) Monte Carlo Timing with n_euler=50, vary N_sims:
    n_euler = 50
    sims_list = [1_000, 10_000, 100_000, 1_000_000]
    df_mc = timing_monte_carlo(pricer, n_euler, sims_list)
    print("\n--- Monte Carlo Timing (Euler steps=50) ---")
    print(df_mc.to_string(index=False, float_format=lambda x: f"{x:8.4f}"))
    
    # --- 2) Finite Difference Timing with n_fd=126, vary m:
    n_fd = 126
    m_list = [30, 100, 500, 2000,5000,10000]
    df_fd = timing_finite_difference(pricer, n_fd, m_list)
    print("\n--- Finite Difference Timing (time steps=126) ---")
    print(df_fd.to_string(index=False, float_format=lambda x: f"{x:8.4f}"))

if __name__ == "__main__":
    main()

