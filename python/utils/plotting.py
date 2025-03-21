import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # for 3D plotting

def plot_parity_vs_S0(parityTable, sigma_val, computation_methods=['MC', 'FD']):
    temp = parityTable[np.abs(parityTable['sigma'] - sigma_val) < 1e-8].sort_values('S0')
    plt.figure(figsize=(8,6))
    if 'MC' in computation_methods:
        plt.plot(temp['S0'], temp['(Call-Put)MC'], 'r-o', label='MC: Call-Put', linewidth=1.5)
    if 'FD' in computation_methods:
        plt.plot(temp['S0'], temp['(Call-Put)FD'], 'b-o', label='FD: Call-Put', linewidth=1.5)
    plt.plot(temp['S0'], temp['Theory'], 'k--', label='Theory', linewidth=1.5)
    plt.xlabel('Initial Stock Price S0')
    plt.ylabel('Call - Put')
    plt.title(f'Put-Call Parity @ sigma = {sigma_val}')
    plt.legend(loc='best')
    plt.grid(True)
    plt.show()

def plot_parity_vs_sigma(parityTable, S0_val, computation_methods=['MC', 'FD']):
    temp = parityTable[np.abs(parityTable['S0'] - S0_val) < 1e-8].sort_values('sigma')
    plt.figure(figsize=(8,6))
    if 'MC' in computation_methods:
        plt.plot(temp['sigma'], temp['(Call-Put)MC'], 'r-o', label='MC: Call-Put', linewidth=1.5)
    if 'FD' in computation_methods:
        plt.plot(temp['sigma'], temp['(Call-Put)FD'], 'b-o', label='FD: Call-Put', linewidth=1.5)
    plt.plot(temp['sigma'], temp['Theory'], 'k--', label='Theory', linewidth=1.5)
    plt.xlabel('Volatility sigma')
    plt.ylabel('Call - Put')
    plt.title(f'Put-Call Parity @ S0 = {S0_val}')
    plt.legend(loc='best')
    plt.grid(True)
    plt.show()


def plot_surface(S0_list, sigma_list, Z, title, zlabel):
    S0_arr = np.array(S0_list)
    sigma_arr = np.array(sigma_list)
    S0_grid, sigma_grid = np.meshgrid(S0_arr, sigma_arr, indexing='ij')
    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(S0_grid, sigma_grid, Z, cmap='viridis', edgecolor='none')
    ax.set_xlabel('S0')
    ax.set_ylabel('sigma')
    ax.set_zlabel(zlabel)
    ax.set_title(title)
    plt.show()

def plot_cpu_time(sigma_list, time_FD, time_MC, S0_index):
    plt.figure(figsize=(8,6))
    plt.plot(sigma_list, time_FD[S0_index, :], 'bo-', linewidth=2, label='Finite Difference')
    plt.plot(sigma_list, time_MC[S0_index, :], 'rs-', linewidth=2, label='Monte Carlo (CV)')
    plt.xlabel('Volatility sigma')
    plt.ylabel('CPU Time (seconds)')
    plt.legend(loc='best')
    plt.title('CPU Time Comparison for S0 = 100')
    plt.grid(True)
    plt.show()

def plot_price_difference(sigma_list, call_FD, call_MC, S0_index):
    plt.figure(figsize=(8,6))
    plt.plot(sigma_list, call_FD[S0_index, :] - call_MC[S0_index, :], 'k*-', linewidth=2)
    plt.xlabel('Volatility sigma')
    plt.ylabel('Price Difference (FD - MC)')
    plt.title('Difference between FD and MC Call Prices for S0 = 100')
    plt.grid(True)
    plt.show()

def plot_FDS_vs_CVMC_S0(S0_list, sigma_list, call_FD, put_FD, call_MC, put_MC, K, sigma_fixed):
    idx_sigma = np.argmin(np.abs(np.array(sigma_list) - sigma_fixed))
    plt.figure(figsize=(8,6))
    plt.plot(S0_list, call_FD[:, idx_sigma], 'r-o', label='FDS Call', linewidth=1.5)
    plt.plot(S0_list, put_FD[:, idx_sigma], 'b-o', label='FDS Put', linewidth=1.5)
    plt.plot(S0_list, call_MC[:, idx_sigma], 'r--*', label='CVMC Call', linewidth=1.5)
    plt.plot(S0_list, put_MC[:, idx_sigma], 'b--*', label='CVMC Put', linewidth=1.5)
    plt.axhline(y=K, color='k', linestyle='--', label='Strike Price')
    plt.xlabel('Initial Stock Price S0')
    plt.ylabel('Option Price P(0)')
    plt.title(f'Comparison vs. S0 @ sigma = {sigma_fixed}')
    plt.legend(loc='best')
    plt.grid(True)
    plt.show()

def plot_FDS_vs_CVMC_sigma(S0_list, sigma_list, call_FD, put_FD, call_MC, put_MC, K, S0_fixed):
    idx_S0 = np.argmin(np.abs(np.array(S0_list) - S0_fixed))
    plt.figure(figsize=(8,6))
    plt.plot(sigma_list, call_FD[idx_S0, :], 'r-o', label='FDS Call', linewidth=1.5)
    plt.plot(sigma_list, put_FD[idx_S0, :], 'b-o', label='FDS Put', linewidth=1.5)
    plt.plot(sigma_list, call_MC[idx_S0, :], 'r--*', label='CVMC Call', linewidth=1.5)
    plt.plot(sigma_list, put_MC[idx_S0, :], 'b--*', label='CVMC Put', linewidth=1.5)
    plt.axhline(y=K, color='k', linestyle='--', label='Strike Price')
    plt.xlabel('Volatility sigma')
    plt.ylabel('Option Price P(0)')
    plt.title(f'Comparison vs. sigma @ S0 = {S0_fixed}')
    plt.legend(loc='best')
    plt.grid(True)
    plt.show()
