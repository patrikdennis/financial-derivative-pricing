#!/usr/bin/env python3
import numpy as np
import pandas as pd
from european_pricing import EuropeanOptionPricer  

def verify_european_parity(S0_list, sigma_list, r, K, T, n, m, X):
    """
    For each S0 and sigma, compute the European call and put prices using the
    Crank–Nicolson finite difference method, then verify the put–call parity:
       C - P = S0 - K * exp(-rT).
    
    Returns a pandas DataFrame with columns:
       S0, sigma, Call, Put, (Call-Put), Theory, Error
    """
    numS0 = len(S0_list)
    numSigma = len(sigma_list)
    
    call_prices = np.zeros((numS0, numSigma))
    put_prices  = np.zeros((numS0, numSigma))
    
    # Loop over combinations of S0 and sigma:
    for i, s0 in enumerate(S0_list):
        for j, sigma in enumerate(sigma_list):
            pricer = EuropeanOptionPricer(s0, sigma, r, K, T, n, m, X)
            call_price, _, _, _ = pricer.crank_nicolson_call(n, m)
            put_price, _, _, _  = pricer.crank_nicolson_put(n, m)
            call_prices[i, j] = call_price
            put_prices[i, j]  = put_price

    # Build parity table records:
    records = []
    for i, s0 in enumerate(S0_list):
        # Theoretical parity: C - P = S0 - K*exp(-rT)
        theory = s0 - K * np.exp(-r*T)
        for j, sigma in enumerate(sigma_list):
            diff = call_prices[i, j] - put_prices[i, j]
            error = diff - theory
            records.append([s0, sigma, call_prices[i, j], put_prices[i, j], diff, theory, error])
    
    df = pd.DataFrame(records, columns=['S0', 'sigma', 'Call', 'Put', '(Call-Put)', 'Theory', 'Error'])
    return df

if __name__ == '__main__':
    # Parameters:
    r = 0.05
    K = 100
    T = 1.0
    n = 100      # time steps
    m = 200      # space steps
    X = 300.0    # Upper bound for S (should be sufficiently large)

    S0_list = [80, 90, 100, 110, 120]
    sigma_list = [0.1, 0.2, 0.3, 0.4, 0.5]

    parity_df = verify_european_parity(S0_list, sigma_list, r, K, T, n, m, X)
    print("European Put–Call Parity Verification:")
    print(parity_df)
