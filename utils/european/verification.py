import numpy as np
import pandas as pd


def verify_european_parity(S0_list, sigma_list, r, K, T, n, m, X):
    """
    For each S0 and sigma, compute the European call and put prices using the
    Crank–Nicolson finite difference method, then verify the put–call parity:
    
         C - P = S0 - K * exp(-rT)
    
    Returns a pandas DataFrame with columns:
         S0, sigma, Call, Put, (Call-Put), Theory, Error
    """
    numS0 = len(S0_list)
    numSig = len(sigma_list)
    
    call_prices = np.zeros((numS0, numSig))
    put_prices  = np.zeros((numS0, numSig))
    
    # Loop over combinations of S0 and sigma:
    for i, s0 in enumerate(S0_list):_
        for j, sigma in enumerate(sigma_list):
            pricer = EuropeanOptionPricer(s0, sigma, r, K, T, n, m, X)
            call_price, _, _, _ = pricer.crank_nicolson_call(n, m)
            put_price, _, _, _  = pricer.crank_nicolson_put(n, m)
            call_prices[i, j] = call_price
            put_prices[i, j]  = put_price

    records = []
    # For European options, put–call parity is: C - P = S0 - K * exp(-rT)
    for i, s0 in enumerate(S0_list):
        theory = s0 - K * np.exp(-r * T)
        for j, sigma in enumerate(sigma_list):
            diff = call_prices[i, j] - put_prices[i, j]
            error = diff - theory
            records.append([s0, sigma, call_prices[i, j], put_prices[i, j], diff, theory, error])
    
    df = pd.DataFrame(records, columns=['S0', 'sigma', 'Call', 'Put', '(Call-Put)', 'Theory', 'Error'])
    return df