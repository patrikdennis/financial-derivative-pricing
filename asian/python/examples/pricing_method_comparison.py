
import numpy as np
from asian_pricing import AsianOptionPricer
from utils.plotting import plot_price_difference, plot_FDS_vs_CVMC_S0, plot_FDS_vs_CVMC_sigma

def main():
    # Option and market parameters for comparison
    r = 0.05; K = 100; T = 1
    S0_list = [80, 90, 100, 110, 120]
    sigma_list = [0.1, 0.2, 0.3, 0.4, 0.5]
    numS0, numSig = len(S0_list), len(sigma_list)
    call_FD = np.zeros((numS0, numSig))
    put_FD = np.zeros((numS0, numSig))
    call_MC = np.zeros((numS0, numSig))
    put_MC = np.zeros((numS0, numSig))
    n_FD = 100; m_FD = 100; n_MC = 50; N_MC = 10000

    for i, s0 in enumerate(S0_list):
        for j, sigma in enumerate(sigma_list):
            pricer = AsianOptionPricer(s0, sigma, r, K, T)
            call_FD[i, j], _, _, _ = pricer.crank_nicolson_ac(n_FD, m_FD)
            put_FD[i, j], _, _, _ = pricer.crank_nicolson_ap(n_FD, m_FD)
            call_MC[i, j], _ = pricer.monte_carlo_ac(n_MC, N_MC)
            put_MC[i, j], _ = pricer.monte_carlo_ap(n_MC, N_MC)

    # For S0 = 100, plot price difference between FD and MC call prices.
    S0_index = np.argmin(np.abs(np.array(S0_list) - 100))
    plot_price_difference(sigma_list, call_FD, call_MC, S0_index)
    
    # Additional comparisons: FD vs. MC for varying S0 at a fixed sigma.
    plot_FDS_vs_CVMC_S0(S0_list, sigma_list, call_FD, put_FD, call_MC, put_MC, K, sigma_fixed=0.2)
    # And FD vs. MC for varying sigma at a fixed S0.
    plot_FDS_vs_CVMC_sigma(S0_list, sigma_list, call_FD, put_FD, call_MC, put_MC, K, S0_fixed=100)

if __name__ == '__main__':
    main()
