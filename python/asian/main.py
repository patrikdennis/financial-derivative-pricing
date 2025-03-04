import time
import numpy as np
import pandas as pd
from python.asian.AsianOptionPricer import AsianOptionPricer
from utils.asian.verification import verify_arithmetic_parity
from utils.plotting import (
    plot_parity_vs_S0,
    plot_parity_vs_sigma,
    plot_surface,
    plot_cpu_time,
    plot_price_difference,
    plot_FDS_vs_CVMC_S0,
    plot_FDS_vs_CVMC_sigma
)
def main():
    # Option and market parameters
    r = 0.02
    K = 40
    T = 1/2
    S0_list = [80, 90, 100, 110, 120]
    S0_list = [element + 1 for element in range(29,35)]
    sigma_list = [0.1, 0.2, 0.3, 0.4, 0.5]

    # FD and MC parameters
    n_FD = 126   # time steps for finite difference
    m_FD = 100   # space steps for finite difference
    n_MC = 50    # time steps for Monte Carlo
    N_MC = 10000 # number of Monte Carlo paths

    numS0 = len(S0_list)
    numSig = len(sigma_list)

    # Preallocate arrays for different methods
    call_geo = np.zeros((numS0, numSig))
    put_geo  = np.zeros((numS0, numSig))
    call_FD = np.zeros((numS0, numSig))
    put_FD  = np.zeros((numS0, numSig))
    call_MC = np.zeros((numS0, numSig))
    put_MC  = np.zeros((numS0, numSig))
    time_FD = np.zeros((numS0, numSig))
    time_MC = np.zeros((numS0, numSig))

    # Loop over S0 and sigma values
    for i, s0 in enumerate(S0_list):
        for j, sigma in enumerate(sigma_list):
            # Closed-form (geometric) prices (for reference & control variate)
            call_geo[i, j] = AsianOptionPricer.closed_formula_ac_geo(s0, sigma, r, K, T)
            put_geo[i, j]  = AsianOptionPricer.closed_formula_ap_geo(s0, sigma, r, K, T)

            pricer = AsianOptionPricer(s0, sigma, r, K, T)
            start = time.time()
            call_price_FD, _, _, _ = pricer.crank_nicolson_ac(n_FD, m_FD)
            t1 = time.time() - start
            start = time.time()
            put_price_FD, _, _, _ = pricer.crank_nicolson_ap(n_FD, m_FD)
            t2 = time.time() - start
            time_FD[i, j] = t1 + t2
            call_FD[i, j] = call_price_FD
            put_FD[i, j]  = put_price_FD

            start = time.time()
            call_price_MC, _ = pricer.monte_carlo_ac(n_MC, N_MC)
            t3 = time.time() - start
            start = time.time()
            put_price_MC, _ = pricer.monte_carlo_ap(n_MC, N_MC)
            t4 = time.time() - start
            time_MC[i, j] = t3 + t4
            call_MC[i, j] = call_price_MC
            put_MC[i, j]  = put_price_MC

    # ----- 1. Verify Arithmetic Put–Call Parity -----
    parityTable = verify_arithmetic_parity(S0_list, sigma_list, call_MC, put_MC, call_FD, put_FD, r, T, K)
    print("Arithmetic Put–Call Parity Comparison for Asian Options")
    print(parityTable)

    # ----- 2. Generate Plots -----
    plot_parity_vs_S0(parityTable, sigma_val=0.1)
    plot_parity_vs_sigma(parityTable, S0_val=30)
    plot_surface(S0_list, sigma_list, call_FD, 'Asian Call Price (FD)', 'Asian Call Price (FD)')
    plot_surface(S0_list, sigma_list, put_FD, 'Asian Put Price (FD)', 'Asian Put Price (FD)')
    S0_index = np.argmin(np.abs(np.array(S0_list) - 100))
    plot_cpu_time(sigma_list, time_FD, time_MC, S0_index)
    # e) FD vs. MC price difference for call options at S0 = 100
    plot_price_difference(sigma_list, call_FD, call_MC, S0_index)
    # f) Additional Plots: FDS vs. CVMC for varying S0 and sigma
    plot_FDS_vs_CVMC_S0(S0_list, sigma_list, call_FD, put_FD, call_MC, put_MC, K=40, sigma_fixed=0.1)
    plot_FDS_vs_CVMC_sigma(S0_list, sigma_list, call_FD, put_FD, call_MC, put_MC, K=40, S0_fixed=30)

if __name__ == '__main__':
    main()
