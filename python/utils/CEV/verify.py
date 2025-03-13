import numpy as np
import pandas as pd
from python.CEV.CEVPricer import CEVoptionPricer

def verify_CEV_parity(S0_list, sigma_list, r, K, T, n, m, delta, X):
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
    for i, s0 in enumerate(S0_list):
        for j, sigma in enumerate(sigma_list):
            pricer = CEVoptionPricer(s0, sigma, r, K, T, delta, X)
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
            records.append([s0,
                            sigma,
                            np.round(call_prices[i, j],5),
                            np.round(put_prices[i, j],5), 
                            np.round(diff,5),
                            np.round(theory,5),
                            np.round(error,5)])
    
    df = pd.DataFrame(records, columns=['S0', 'sigma', 'Call', 'Put', '(Call-Put)FD', 'Theory', 'Error'])
    return df
