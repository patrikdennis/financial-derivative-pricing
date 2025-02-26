#!/usr/bin/env python3
import numpy as np
import pandas as pd
from asian_pricing import AsianOptionPricer
from utils.asian.verification import verify_arithmetic_parity
from utils.plotting import plot_parity_vs_S0, plot_parity_vs_sigma

def main():
    r = 0.05
    K = 100
    T = 1
    S0_list = [80, 90, 100, 110, 120]
    sigma_list = [0.1, 0.2, 0.3, 0.4, 0.5]
    n_MC = 50; N_MC = 10000
    n_FD = 100; m_FD = 100

    numS0 = len(S0_list)
    numSig = len(sigma_list)
    call_MC = np.zeros((numS0, numSig))
    put_MC = np.zeros((numS0, numSig))
    call_FD = np.zeros((numS0, numSig))
    put_FD = np.zeros((numS0, numSig))

    for i, s0 in enumerate(S0_list):
        for j, sigma in enumerate(sigma_list):
            pricer = AsianOptionPricer(s0, sigma, r, K, T)
            call_MC[i, j], _ = pricer.monte_carlo_ac(n_MC, N_MC)
            put_MC[i, j], _ = pricer.monte_carlo_ap(n_MC, N_MC)
            call_FD[i, j], _, _, _ = pricer.crank_nicolson_ac(n_FD, m_FD)
            put_FD[i, j], _, _, _ = pricer.crank_nicolson_ap(n_FD, m_FD)

    parityTable = verify_arithmetic_parity(S0_list, sigma_list, call_MC, put_MC, call_FD, put_FD, r, T, K)
    print("Arithmetic Put–Call Parity Comparison for Asian Options")
    print(parityTable)
    
    plot_parity_vs_S0(parityTable, sigma_val=0.2, computation_methods = ['MC', 'FD'])
    plot_parity_vs_sigma(parityTable, S0_val=100, computation_methods = ['MC', 'FD'])

if __name__ == '__main__':
    main()
