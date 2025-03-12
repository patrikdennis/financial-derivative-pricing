############################################
# timing_asian.py
############################################
import time
import pandas as pd
from python.asian.AsianOptionPricer import AsianOptionPricer

def time_monte_carlo(pricer, n_mc, N_values):
    """
    Measures computation time of monte_carlo_ac (control-variate Asian call)
    as we vary the number of simulated paths.
    
    Parameters
    ----------
    pricer  : AsianOptionPricer object
    n_mc    : int, number of time steps in the MC discretization
    N_values: list of int, # of (non-antithetic) paths to generate
    
    Returns
    -------
    df_mc : pd.DataFrame with columns:
       [N_sims, Price, StdDev, Time_s]
    """
    records = []
    for N in N_values:
        start_time = time.time()
        price, stddev = pricer.monte_carlo_ac(n_mc, N)
        elapsed = time.time() - start_time
        
        records.append({
            'N_sims': N,
            'MC_Price': price,
            'MC_StdDev': stddev,
            'MC_Time_s': elapsed
        })
    return pd.DataFrame(records)

def time_finite_difference(pricer, n_fd, m_values):
    """
    Measures computation time of crank_nicolson_ac (Asian call)
    as we vary the number of spatial steps m.
    
    Parameters
    ----------
    pricer  : AsianOptionPricer object
    n_fd    : int, # of time steps for FD
    m_values: list of int, # of space steps to test
    
    Returns
    -------
    df_fd : pd.DataFrame with columns:
       [m_steps, FD_Price, FD_Time_s]
    """
    records = []
    for m_fd in m_values:
        start_time = time.time()
        price_fd, sol, space, time_grid = pricer.crank_nicolson_ac(n_fd, m_fd)
        elapsed = time.time() - start_time
        
        records.append({
            'm_steps': m_fd,
            'FD_Price': price_fd,
            'FD_Time_s': elapsed
        })
    return pd.DataFrame(records)

def main():
    """
    This script mimics the C++ approach of measuring timing for:
      1) Monte Carlo with different N_sims, while n=50 Euler steps
      2) Finite Difference with different m, while n_fd=126 time steps
    """
    # Match the parameters in your C++ snippet:
    s0    = 30.0
    sigma = 0.1
    r     = 0.02
    K     = 40.0
    T     = 0.5
    
    # Instantiate the Asian pricer with these parameters
    pricer = AsianOptionPricer(s0, sigma, r, K, T)

    # 1) Monte Carlo timing: vary N_sims in {1k, 10k, 100k, 1M}, keep n=50
    n_mc = 50
    N_values = [1000, 10000, 100000, 1000000]
    df_mc = time_monte_carlo(pricer, n_mc, N_values)
    print("\n--- Monte Carlo Timing (Asian Call w/ control variate) ---\n"
          f"(Parameters: s0={s0}, sigma={sigma}, r={r}, K={K}, T={T}, n_mc={n_mc})\n")
    print(df_mc.to_string(index=False, float_format=lambda x: f"{x:8.4f}"))
    
    # 2) Finite Difference timing: vary m in {30, 100, 500, 2000}, keep n_fd=126
    n_fd = 126
    m_values = [30, 100, 500, 2000,5000, 10000]
    df_fd = time_finite_difference(pricer, n_fd, m_values)
    print("\n--- Finite Difference Timing (Asian Call, Crank–Nicolson) ---\n"
          f"(Parameters: s0={s0}, sigma={sigma}, r={r}, K={K}, T={T}, n_fd={n_fd})\n")
    print(df_fd.to_string(index=False, float_format=lambda x: f"{x:8.4f}"))

if __name__ == "__main__":
    main()

