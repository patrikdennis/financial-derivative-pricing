
import time
import numpy as np
from asian_pricing import AsianOptionPricer
from utils.plotting import plot_cpu_time

def main():
    # Option and market parameters
    r = 0.05; K = 100; T = 1
    S0 = 100; sigma = 0.3
    
    # FD and MC parameters
    n_FD = 100; m_FD = 100; n_MC = 50; N_MC = 10000

    pricer = AsianOptionPricer(S0, sigma, r, K, T)
    start = time.time()
    call_price_FD, _, _, _ = pricer.crank_nicolson_ac(n_FD, m_FD)
    put_price_FD, _, _, _ = pricer.crank_nicolson_ap(n_FD, m_FD)
    cpu_FD = time.time() - start

    start = time.time()
    call_price_MC, _ = pricer.monte_carlo_ac(n_MC, N_MC)
    put_price_MC, _ = pricer.monte_carlo_ap(n_MC, N_MC)
    cpu_MC = time.time() - start

    # For demonstration, assume we did this for several sigma values at S0 = 100.
    sigma_list = [0.1, 0.2, 0.3, 0.4, 0.5]
    numSig = len(sigma_list)
    time_FD = np.zeros(numSig)
    time_MC = np.zeros(numSig)
    for idx, sig in enumerate(sigma_list):
        pr = AsianOptionPricer(S0, sig, r, K, T)
        start = time.time()
        pr.crank_nicolson_ac(n_FD, m_FD)
        pr.crank_nicolson_ap(n_FD, m_FD)
        time_FD[idx] = time.time() - start

        start = time.time()
        pr.monte_carlo_ac(n_MC, N_MC)
        pr.monte_carlo_ap(n_MC, N_MC)
        time_MC[idx] = time.time() - start

    # Plot CPU times versus sigma for S0 = 100.
    S0_index = 0  # only one S0, so index 0
    # We need to reshape time arrays to 2D to use our plotting function.
    time_FD = time_FD.reshape(1, numSig)
    time_MC = time_MC.reshape(1, numSig)
    plot_cpu_time(sigma_list, time_FD, time_MC, S0_index)

if __name__ == '__main__':
    main()
