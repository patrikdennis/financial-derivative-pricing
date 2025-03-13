import numpy as np
import pandas as pd
import time
from python.CEV.CEVPricer import CEVoptionPricer

def compare_fd_mc():
    """
    Compare the Finite Difference (FD) method and the Monte Carlo (MC) method
    for pricing European call/put for different values of sigma. Also measure
    and compare the computational times of both methods.
    """
    # Fixed parameters
    s0_list = range(30,51,5)      # you can adjust or extend as desired
    sigma_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6,0.7,0.8,0.9,1.0]
    r = 0.02
    K = 40
    T = 1.0

    s0_list = [100,110]
    K = 100
    
    # FD grid parameters
    n_fd = 150    # time steps
    m_fd = 10_000    # space steps
    X = 4 * K     # domain cutoff
    delta = 1.0   # set to 1 for the Black–Scholes PDE

    # MC parameters
    n_steps_mc = 50   # number of Euler steps for [0,T]
    n_sims_mc  = 100000
    
    records = []
    
    for s0 in s0_list:
        for sig in sigma_list:
            # ============ Finite Difference ============ #
            pricer = CEVoptionPricer(s0, sig, r, K, T, delta, X)
            
            # Time the call FD
            start_fd_call = time.time()
            fd_call_price, _, _, _ = pricer.crank_nicolson_call(n_fd, m_fd)
            fd_call_time = time.time() - start_fd_call
            
            # Time the put FD
            start_fd_put = time.time()
            fd_put_price, _, _, _ = pricer.crank_nicolson_put(n_fd, m_fd)
            fd_put_time = time.time() - start_fd_put
            
            # ============ Monte Carlo ============ #
            # Time the call MC
            start_mc_call = time.time()
            mc_call_price, mc_call_std = pricer.monte_carlo_call(n_steps_mc, n_sims_mc)
            mc_call_time = time.time() - start_mc_call
            
            # Time the put MC
            start_mc_put = time.time()
            mc_put_price, mc_put_std = pricer.monte_carlo_put(n_steps_mc,n_sims_mc)
            mc_put_time = time.time() - start_mc_put
            
            # Record results in a dictionary
            records.append({
                'S0'              : s0,
                'sigma'           : sig,
                'FD_Call'         : np.round(fd_call_price,5),
                'FD_Put'          : np.round(fd_put_price,5),
                #'FD_Call_Time_s'  : fd_call_time,
                #'FD_Put_Time_s'   : fd_put_time,
                'MC_Call'         : np.round(mc_call_price,5),
                'MC_Put'          : np.round(mc_put_price,5),
                #'MC_Call_StdErr'  : mc_call_std,
                #'MC_Put_StdErr'   : mc_put_std,
                #'MC_Call_Time_s'  : mc_call_time,
                #'MC_Put_Time_s'   : mc_put_time
            })
    
    # Create a DataFrame with all results
    df = pd.DataFrame(records)
    print("\nComparison of FD vs. MC for various sigmas and S0 values:\n")
    print(df.to_string(index=False, float_format=lambda x: f"{x:8.5f}"))
    
    return df

if __name__ == "__main__":
   df = compare_fd_mc()
   df.to_csv('comparison_CEV.txt', index = None, sep=" ")
