#!/usr/bin/env python3
import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D 
import time
import pandas as pd

class AsianOptionPricer:
    def __init__(self, s0: float, sigma: float, r :float, K :float, T :float ) --> None:
        
        # inital stock price
        self.s0= s0      
        
        # volitiliy
        self.sigma= sigma
        
        # interest rate
        self.r = r
        
        # strike price
        self.K = K
        
        # time to maturity
        self.T = T

    @staticmethod
    def closed_formula_ac_geo(s0, sigma, r, K, T):
        """Closed-form solution for geometric Asian call option."""
        dStar = (T / 2) * (r - sigma**2 / 6)
        d = (np.log(s0 / K) + (T / 2) * (r + sigma**2 / 6)) / (sigma * np.sqrt(T / 3))
        price = np.exp(dStar) * s0 * st.norm.cdf(d) - K * st.norm.cdf(d - sigma * np.sqrt(T / 3))
        return price

    @staticmethod
    def closed_formula_ap_geo(s0, sigma, r, K, T):
        """Closed-form solution for geometric Asian put option."""
        dStar = (T / 2) * (r - sigma**2 / 6)
        d = (np.log(s0 / K) + (T / 2) * (r + sigma**2 / 6)) / (sigma * np.sqrt(T / 3))
        price = -np.exp(dStar) * s0 * st.norm.cdf(-d) + K * st.norm.cdf(-d + sigma * np.sqrt(T / 3))
        return price

    def stock_path(self, n, N):
        """
        Simulate stock paths using antithetic variates.
        Returns an array of shape (2*N, n+1), where each row is one simulated path.
        """
        h = self.T / n
        # Generate N x n random normals
        w = np.random.randn(N, n)
        # Antithetic variates: stack -w and w (resulting in 2N x n)
        W = np.vstack([-w, w])
        # Cumulative sum along each row, with proper drift
        cum_steps = np.tile(np.arange(1, n + 1), (2 * N, 1))
        drift = (self.r - self.sigma**2 / 2) * h * cum_steps
        diffusion = self.sigma * np.sqrt(h) * np.cumsum(W, axis=1)
        paths = self.s0 * np.exp(drift + diffusion)
        # Prepend the initial price as the first column
        paths = np.hstack((self.s0 * np.ones((2 * N, 1)), paths))
        return paths

    def monte_carlo_ac_crude(self, n, N):
        """Monte Carlo (crude) pricing for Asian call option."""
        paths = self.stock_path(n, N)
        avg_price = np.mean(paths, axis=1)
        payoff = np.maximum(avg_price - self.K, 0)
        price = np.exp(-self.r * self.T) * np.mean(payoff)
        deviation = np.std(payoff) / np.sqrt(2 * N)
        return price, deviation

    def monte_carlo_ap_crude(self, n, N):
        """Monte Carlo (crude) pricing for Asian put option."""
        paths = self.stock_path(n, N)
        avg_price = np.mean(paths, axis=1)
        payoff = np.maximum(self.K - avg_price, 0)
        price = np.exp(-self.r * self.T) * np.mean(payoff)
        deviation = np.std(payoff) / np.sqrt(2 * N)
        return price, deviation

    def monte_carlo_ac(self, n, N):
        """
        Monte Carlo pricing for Asian call option with control variate.
        Uses the closed-form geometric call as the control variate.
        """
        detValueCV = AsianOptionPricer.closed_formula_ac_geo(self.s0, self.sigma, self.r, self.K, self.T)
        paths = self.stock_path(n, N)
        avg_price = np.mean(paths, axis=1)
        payoff = np.maximum(avg_price - self.K, 0)
        geo_price = np.exp(np.mean(np.log(paths), axis=1))
        payoffCV = np.maximum(geo_price - self.K, 0)
        price = np.exp(-self.r * self.T) * (np.mean(payoff) - np.mean(payoffCV) + detValueCV)
        deviation = np.std(payoff - payoffCV + detValueCV) / np.sqrt(2 * N)
        return price, deviation

    def monte_carlo_ap(self, n, N):
        """
        Monte Carlo pricing for Asian put option with control variate.
        Uses the closed-form geometric put as the control variate.
        """
        detValueCV = AsianOptionPricer.closed_formula_ap_geo(self.s0, self.sigma, self.r, self.K, self.T)
        paths = self.stock_path(n, N)
        avg_price = np.mean(paths, axis=1)
        payoff = np.maximum(self.K - avg_price, 0)
        geo_price = np.exp(np.mean(np.log(paths), axis=1))
        payoffCV = np.maximum(self.K - geo_price, 0)
        price = np.exp(-self.r * self.T) * (np.mean(payoff - payoffCV) + detValueCV)
        deviation = np.std(payoff - payoffCV + detValueCV) / np.sqrt(2 * N)
        return price, deviation

    def crank_nicolson_ac(self, n, m):
        """
        Finite Difference (Crank–Nicolson) pricing for Asian call option.
        Returns price, solution grid, space grid, and time grid.
        """
        dt = self.T / n
        Z0 = (1 - np.exp(-self.r * self.T)) / (self.r * self.T) - self.K * np.exp(-self.r * self.T) / self.s0
        dz = 2 * (np.abs(Z0) + 1) / m
        d_val = dt / dz**2

        time_grid = np.linspace(0, self.T, n + 1)
        space = np.linspace(-(np.abs(Z0) + 1), (np.abs(Z0) + 1), m + 1)
        Z0index = np.argmin(np.abs(space - Z0))
        space = space - space[Z0index] + Z0

        sol = np.zeros((n + 1, m + 1))
        sol[0, :] = np.maximum(space, 0)
        sol[:, 0] = 0
        sol[:, -1] = space[-1]

        Q = (1 - np.exp(-self.r * time_grid)) / (self.r * self.T)

        def alpha(i, j):
            return d_val * self.sigma**2 * (Q[i] - space[j])**2 / 4

        for i in range(1, n + 1):
            A = np.zeros((m + 1, m + 1))
            B = np.zeros((m + 1, m + 1))
            A[0, 0] = 1
            A[-1, -1] = 1
            B[0, 0] = 1
            B[-1, -1] = 1
            for j in range(1, m):
                A[j, j - 1] = -alpha(i, j)
                A[j, j] = 1 + 2 * alpha(i, j)
                A[j, j + 1] = -alpha(i, j)

                B[j, j - 1] = alpha(i - 1, j)
                B[j, j] = 1 - 2 * alpha(i - 1, j)
                B[j, j + 1] = alpha(i - 1, j)
            sol[i, :] = np.linalg.solve(A, B.dot(sol[i - 1, :]))
        price = self.s0 * sol[-1, Z0index]
        return price, sol, space, time_grid

    def crank_nicolson_ap(self, n, m):
        """
        Finite Difference (Crank–Nicolson) pricing for Asian put option.
        Returns price, solution grid, space grid, and time grid.
        """
        dt = self.T / n
        Z0 = -(1 - np.exp(-self.r * self.T)) / (self.r * self.T) + self.K * np.exp(-self.r * self.T) / self.s0
        dz = 2 * (np.abs(Z0) + 1) / m
        d_val = dt / dz**2

        time_grid = np.linspace(0, self.T, n + 1)
        space = np.linspace(-(np.abs(Z0) + 1), (np.abs(Z0) + 1), m + 1)
        Z0index = np.argmin(np.abs(space - Z0))
        space = space - space[Z0index] + Z0

        sol = np.zeros((n + 1, m + 1))
        sol[0, :] = np.maximum(space, 0)
        sol[:, 0] = 0
        sol[:, -1] = space[-1]

        Q = (1 - np.exp(-self.r * time_grid)) / (self.r * self.T)

        def alpha(i, j):
            return d_val * self.sigma**2 * (Q[i] - space[j])**2 / 4

        for i in range(1, n + 1):
            A = np.zeros((m + 1, m + 1))
            B = np.zeros((m + 1, m + 1))
            A[0, 0] = 1
            A[-1, -1] = 1
            B[0, 0] = 1
            B[-1, -1] = 1
            for j in range(1, m):
                A[j, j - 1] = -alpha(i, j)
                A[j, j] = 1 + 2 * alpha(i, j)
                A[j, j + 1] = -alpha(i, j)

                B[j, j - 1] = alpha(i - 1, j)
                B[j, j] = 1 - 2 * alpha(i - 1, j)
                B[j, j + 1] = alpha(i - 1, j)
            sol[i, :] = np.linalg.solve(A, B.dot(sol[i - 1, :]))
        price = self.s0 * sol[-1, Z0index]
        return price, sol, space, time_grid

# End of AsianOptionPricer class

def main():
    # Option and market parameters
    r = 0.05
    K = 100
    T = 1
    S0_list = [80, 90, 100, 110, 120]
    sigma_list = [0.1, 0.2, 0.3, 0.4, 0.5]

    # Parameters for finite difference and Monte Carlo
    n_FD = 100   # time steps for finite difference
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

            # Finite Difference (Crank–Nicolson) pricing
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

            # Monte Carlo with control variate pricing
            start = time.time()
            call_price_MC, _ = pricer.monte_carlo_ac(n_MC, N_MC)
            t3 = time.time() - start
            start = time.time()
            put_price_MC, _ = pricer.monte_carlo_ap(n_MC, N_MC)
            t4 = time.time() - start
            time_MC[i, j] = t3 + t4
            call_MC[i, j] = call_price_MC
            put_MC[i, j]  = put_price_MC

    # --------------------------
    # 1. Arithmetic Put–Call Parity Verification and Table
    # The theoretical parity (at t=0) for arithmetic Asian options:
    #   Call - Put = exp(-rT)*(((exp(rT)-1)/(rT))*S0 - K)
    parity_results = []
    for i, s0 in enumerate(S0_list):
        parity_theory = np.exp(-r * T) * ((((np.exp(r * T) - 1) / (r * T)) * s0) - K)
        for j, sigma in enumerate(sigma_list):
            lhs_MC = call_MC[i, j] - put_MC[i, j]
            lhs_FD = call_FD[i, j] - put_FD[i, j]
            err_MC = lhs_MC - parity_theory
            err_FD = lhs_FD - parity_theory
            parity_results.append([s0, sigma, lhs_MC, lhs_FD, parity_theory, err_MC, err_FD])
    parity_results = np.array(parity_results)
    colNames = ['S0', 'sigma', '(Call-Put)MC', '(Call-Put)FD', 'Theory', 'ErrorMC', 'ErrorFD']
    parityTable = pd.DataFrame(parity_results, columns=colNames)
    print("Arithmetic Put–Call Parity Comparison for Asian Options")
    print(parityTable)

    # --------------------------
    # 2. Example Plots: Arithmetic Put–Call Parity
    # a) Parity vs. S0 for fixed sigma (e.g., sigma = 0.2)
    sigma_plot = 0.2
    temp_sigma = parityTable[np.abs(parityTable['sigma'] - sigma_plot) < 1e-8].sort_values('S0')
    plt.figure(figsize=(8, 6))
    plt.plot(temp_sigma['S0'], temp_sigma['(Call-Put)MC'], 'r-o', linewidth=1.5, label='MC: Call-Put')
    plt.plot(temp_sigma['S0'], temp_sigma['(Call-Put)FD'], 'b-o', linewidth=1.5, label='FD: Call-Put')
    plt.plot(temp_sigma['S0'], temp_sigma['Theory'], 'k--', linewidth=1.5, label='Theory')
    plt.xlabel('Initial Stock Price S0')
    plt.ylabel('Call - Put')
    plt.title(f'Arithmetic Put-Call Parity @ sigma = {sigma_plot}')
    plt.legend(loc='best')
    plt.grid(True)

    # b) Parity vs. sigma for fixed S0 (e.g., S0 = 100)
    S0_plot = 100
    temp_S0 = parityTable[np.abs(parityTable['S0'] - S0_plot) < 1e-8].sort_values('sigma')
    plt.figure(figsize=(8, 6))
    plt.plot(temp_S0['sigma'], temp_S0['(Call-Put)MC'], 'r-o', linewidth=1.5, label='MC: Call-Put')
    plt.plot(temp_S0['sigma'], temp_S0['(Call-Put)FD'], 'b-o', linewidth=1.5, label='FD: Call-Put')
    plt.plot(temp_S0['sigma'], temp_S0['Theory'], 'k--', linewidth=1.5, label='Theory')
    plt.xlabel('Volatility sigma')
    plt.ylabel('Call - Put')
    plt.title(f'Arithmetic Put-Call Parity @ S0 = {S0_plot}')
    plt.legend(loc='best')
    plt.grid(True)

    # --------------------------
    # 3. Surface Plots of Asian Option Prices (Finite Difference)
    # a) Asian Call Price surface (FD)
    S0_arr = np.array(S0_list)
    sigma_arr = np.array(sigma_list)
    S0_grid, sigma_grid = np.meshgrid(S0_arr, sigma_arr, indexing='ij')
    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(S0_grid, sigma_grid, call_FD, cmap='viridis', edgecolor='none')
    ax.set_xlabel('S0')
    ax.set_ylabel('sigma')
    ax.set_zlabel('Asian Call Price (FD)')
    ax.set_title('Asian Call Price (Finite Difference)')
    
    # b) Asian Put Price surface (FD)
    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(S0_grid, sigma_grid, put_FD, cmap='plasma', edgecolor='none')
    ax.set_xlabel('S0')
    ax.set_ylabel('sigma')
    ax.set_zlabel('Asian Put Price (FD)')
    ax.set_title('Asian Put Price (Finite Difference)')

    # --------------------------
    # 4. Verification of Geometric Put–Call Parity
    put_call_error_geo = np.zeros((numS0, numSig))
    parity_RHS_geo = np.zeros((numS0, numSig))
    for i, s0 in enumerate(S0_list):
        for j, sigma in enumerate(sigma_list):
            lhs_geo = call_geo[i, j] - put_geo[i, j]
            rhs_geo = np.exp((T / 2) * (r - sigma**2 / 6)) * s0 - K
            parity_RHS_geo[i, j] = rhs_geo
            put_call_error_geo[i, j] = np.abs(lhs_geo - rhs_geo)
    max_error_geo = np.max(put_call_error_geo)
    print(f"Maximum put-call parity error (geometric closed-form): {max_error_geo:e}")

    # --------------------------
    # 5. Comparison: Finite Difference vs. Monte Carlo (for S0 = 100)
    idx_S0 = np.argmin(np.abs(np.array(S0_list) - 100))
    comp_data = {
        'Sigma': sigma_list,
        'Call_FD': call_FD[idx_S0, :],
        'Call_MC': call_MC[idx_S0, :],
        'Abs_Difference': np.abs(call_FD[idx_S0, :] - call_MC[idx_S0, :])
    }
    comp_table = pd.DataFrame(comp_data)
    print("Comparison of Asian Call Prices for S0 = 100:")
    print(comp_table)

    # --------------------------
    # 6. Efficiency Comparison (CPU Timing) for S0 = 100
    avg_time_FD = np.mean(time_FD)
    avg_time_MC = np.mean(time_MC)
    print(f"Average CPU Time for Finite Difference method: {avg_time_FD:.4f} seconds")
    print(f"Average CPU Time for Monte Carlo (Control Variate) method: {avg_time_MC:.4f} seconds")
    
    plt.figure(figsize=(8,6))
    plt.plot(sigma_list, time_FD[idx_S0, :], 'bo-', linewidth=2, label='Finite Difference')
    plt.plot(sigma_list, time_MC[idx_S0, :], 'rs-', linewidth=2, label='Monte Carlo (CV)')
    plt.xlabel('Volatility sigma')
    plt.ylabel('CPU Time (seconds)')
    plt.legend(loc='best')
    plt.title('CPU Time Comparison for S0 = 100')
    plt.grid(True)

    # --------------------------
    # 7. FD vs. MC Price Difference (Call Option) for S0 = 100
    plt.figure(figsize=(8,6))
    plt.plot(sigma_list, call_FD[idx_S0, :] - call_MC[idx_S0, :], 'k*-', linewidth=2)
    plt.xlabel('Volatility sigma')
    plt.ylabel('Price Difference (FD - MC)')
    plt.title('Difference between FD and MC Asian Call Prices for S0 = 100')
    plt.grid(True)

    # --------------------------
    # 8. Additional Plots: FDS vs CVMC for Varying S0 and sigma
    # a) Plot vs. S0 for fixed sigma (e.g., sigma = 0.2)
    sigma_fixed = 0.2
    idx_sigma = np.argmin(np.abs(np.array(sigma_list) - sigma_fixed))
    plt.figure(figsize=(8,6))
    plt.plot(S0_list, call_FD[:, idx_sigma], 'r-o', linewidth=1.5, label='FDS Call')
    plt.plot(S0_list, put_FD[:, idx_sigma], 'b-o', linewidth=1.5, label='FDS Put')
    plt.plot(S0_list, call_MC[:, idx_sigma], 'r--*', linewidth=1.5, label='CVMC Call')
    plt.plot(S0_list, put_MC[:, idx_sigma], 'b--*', linewidth=1.5, label='CVMC Put')
    plt.axhline(y=K, color='k', linestyle='--', label='Strike Price')
    plt.xlabel('Initial Stock Price S0')
    plt.ylabel('Option Price P(0)')
    plt.title(f'Comparison vs. S0 @ sigma = {sigma_fixed}')
    plt.legend(loc='best')
    plt.grid(True)

    # b) Plot vs. sigma for fixed S0 (e.g., S0 = 100)
    S0_fixed = 100
    idx_S0 = np.argmin(np.abs(np.array(S0_list) - S0_fixed))
    plt.figure(figsize=(8,6))
    plt.plot(sigma_list, call_FD[idx_S0, :], 'r-o', linewidth=1.5, label='FDS Call')
    plt.plot(sigma_list, put_FD[idx_S0, :], 'b-o', linewidth=1.5, label='FDS Put')
    plt.plot(sigma_list, call_MC[idx_S0, :], 'r--*', linewidth=1.5, label='CVMC Call')
    plt.plot(sigma_list, put_MC[idx_S0, :], 'b--*', linewidth=1.5, label='CVMC Put')
    plt.axhline(y=K, color='k', linestyle='--', label='Strike Price')
    plt.xlabel('Volatility sigma')
    plt.ylabel('Option Price P(0)')
    plt.title(f'Comparison vs. sigma @ S0 = {S0_fixed}')
    plt.legend(loc='best')
    plt.grid(True)

    plt.show()

if __name__ == '__main__':
    main()
