import numpy as np
import pandas as pd
from utils.verification import verify_arithmetic_parity
from utils.plotting import plot_parity_vs_S0, plot_parity_vs_sigma

def main():
    # Option and market parameters
    r = 0.05
    K = 100
    T = 1
    S0_list = [80, 90, 100, 110, 120]
    sigma_list = [0.1, 0.2, 0.3, 0.4, 0.5]

    # For demonstration, here we simulate some dummy data.
    numS0, numSig = len(S0_list), len(sigma_list)
    
    call_MC = np.random.uniform(5, 15, (numS0, numSig))
    put_MC = np.random.uniform(3, 10, (numS0, numSig))
    call_FD = np.random.uniform(5, 15, (numS0, numSig))
    put_FD = np.random.uniform(3, 10, (numS0, numSig))
    
    parityTable = verify_arithmetic_parity(S0_list, sigma_list, call_MC, put_MC, call_FD, put_FD, r, T, K)
    print("Arithmetic Put–Call Parity Comparison for Asian Options")
    print(parityTable)
    
    # Plot parity vs. S0 for a fixed sigma value
    plot_parity_vs_S0(parityTable, sigma_val=0.2)
    # Plot parity vs. sigma for a fixed S0 value
    plot_parity_vs_sigma(parityTable, S0_val=100)

if __name__ == '__main__':
    main()
