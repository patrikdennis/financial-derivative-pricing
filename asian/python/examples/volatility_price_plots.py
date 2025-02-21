
import numpy as np
from asian_pricing import AsianOptionPricer
from utils.plotting import plot_surface

def main():
    # Option and market parameters for surface plots
    r = 0.05; K = 100; T = 1
    S0_list = [80, 90, 100, 110, 120]
    sigma_list = [0.1, 0.2, 0.3, 0.4, 0.5]
    numS0, numSig = len(S0_list), len(sigma_list)
    call_FD = np.zeros((numS0, numSig))
    put_FD = np.zeros((numS0, numSig))

    n_FD = 100; m_FD = 100

    for i, s0 in enumerate(S0_list):
        for j, sigma in enumerate(sigma_list):
            pricer = AsianOptionPricer(s0, sigma, r, K, T)
            call_FD[i, j], _, _, _ = pricer.crank_nicolson_ac(n_FD, m_FD)
            put_FD[i, j], _, _, _ = pricer.crank_nicolson_ap(n_FD, m_FD)

    # Plot 3D surfaces for call and put prices (FD)
    plot_surface(S0_list, sigma_list, call_FD, 'Asian Call Price (FD)', 'Asian Call Price (FD)')
    plot_surface(S0_list, sigma_list, put_FD, 'Asian Put Price (FD)', 'Asian Put Price (FD)')

if __name__ == '__main__':
    main()
