#!/usr/bin/env python3
import numpy as np
import pandas as pd
from python.european.european_pricing import EuropeanOptionPricer
from utils.european.verification import verify_european_parity 
from utils.plotting import plot_parity_vs_S0, plot_parity_vs_sigma

def main():
    # Market parameters
    r = 0.05
    K = 100
    T = 1.0
    n = 100    # time steps
    m = 200    # space steps
    X = 300.0  # sufficiently large upper bound for S

    # Range of initial prices and volatilities to test
    S0_list = [80, 90, 100, 110, 120]
    sigma_list = [0.1, 0.2, 0.3, 0.4, 0.5]

    parity_df = verify_european_parity(S0_list, sigma_list, r, K, T, n, m, X)
    print("European Put–Call Parity Verification:")
    print(parity_df)

    # Plot parity vs. S0 for fixed sigma (e.g., sigma=0.2)
    plot_parity_vs_S0(parity_df, sigma_val=0.2, computation_methods = ['FD'])
    # Plot parity vs. sigma for fixed S0 (e.g., S0=100)
    plot_parity_vs_sigma(parity_df, S0_val=100, computation_methods = ['FD'])

if __name__ == '__main__':
    main()
