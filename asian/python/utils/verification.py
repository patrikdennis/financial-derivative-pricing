import numpy as np
import pandas as pd

def verify_arithmetic_parity(S0_list, sigma_list, call_MC, put_MC, call_FD, put_FD, r, T, K):
    """
    Verifies the arithmetic Asian put–call parity and returns a DataFrame.
    
    The theoretical parity (at t=0) is given by:
      Call - Put = exp(-rT)*(((exp(rT)-1)/(rT))*S0 - K)
    """
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
    return parityTable
