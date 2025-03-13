import numpy as np
import pandas as pd
from utils.CEV.verify import verify_CEV_parity
from utils.plotting import (
    plot_surface,
    plot_parity_vs_S0,
    plot_parity_vs_sigma
)

def main():
    # 1) Define parameter ranges
    #S0_list    = [80, 90, 100, 110, 120]  # Initial stock prices
    S0_list = range(30,51,5)
    sigma_list = [0.10, 0.20, 0.30, 0.40, 0.5] # Volatilities
    r          = 0.02                    # Risk-free rate
    K          = 40.0                   # Strike price
    T          = 1.0                     # Maturity (in years)
    n          = 126                     # Time steps for finite-difference
    m          = 10_000                     # Space steps for finite-difference
    X          = 10 * K                   # Upper bound of the stock-price domain
    delta = 1.0
    # 2) Compute put/call prices and check put–call parity
    df = verify_CEV_parity(S0_list, sigma_list, r, K, T, n, m,delta, X)
    # 'df' columns: [S0, sigma, 'Call', 'Put', '(Call-Put)FD', 'Theory', 'Error']
    # where (Call-Put)FD is the computed difference, 'Theory' = S0 - K e^{-rT}, and 'Error' = difference - theory

    # 3) Pivot the DataFrame to get matrices for surface plots
    #    (Rows = distinct S0, Columns = distinct sigma)
    call_values = df.pivot(index='S0', columns='sigma', values='Call').values
    put_values  = df.pivot(index='S0', columns='sigma', values='Put').values

    # 4) Plot the call and put surfaces w.r.t S0 and sigma
    plot_surface(
        S0_list, sigma_list, call_values,
        title="Call Price (FD) vs. S0 & σ",
        zlabel="Call Price"
    )
    plot_surface(
        S0_list, sigma_list, put_values,
        title="Put Price (FD) vs. S0 & σ",
        zlabel="Put Price"
    )

    # 5) Verify and plot put–call parity for a fixed sigma
    #    (This will plot (Call - Put)FD vs. the theoretical line)
    chosen_sigma = 0.1
    plot_parity_vs_S0(df, sigma_val=chosen_sigma, computation_methods=['FD'])

    # 6) Alternatively, verify and plot put–call parity for a fixed S0
    chosen_S0 = 30
    plot_parity_vs_sigma(df, S0_val=chosen_S0, computation_methods=['FD'])

    print(df)
    df.to_csv('parity_cev.txt', index = None, sep = ' ')

if __name__ == "__main__":
    main()

