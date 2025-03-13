import numpy as np
import scipy.stats as st
from CEVPricer  import CEVoptionPricer  # your FD implementation file

def black_scholes_call_put(s0, K, r, sigma, T):
    """
    Computes the exact Black-Scholes European call and put option prices.
    
    Parameters
    ----------
    s0 : float
        Initial stock price.
    K : float
        Strike price.
    r : float
        Risk-free interest rate.
    sigma : float
        Volatility of the underlying.
    T : float
        Time to maturity (in years).
        
    Returns
    -------
    call : float
        The Black-Scholes call option price.
    put : float
        The Black-Scholes put option price.
    """
    d1 = (np.log(s0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    call = s0 * st.norm.cdf(d1) - K * np.exp(-r * T) * st.norm.cdf(d2)
    put = K * np.exp(-r * T) * st.norm.cdf(-d2) - s0 * st.norm.cdf(-d1)
    return call, put

def compare_fd_bs():
    """
    Compares the finite difference (FD) solution from the CEV option pricer 
    (with delta=1, i.e. Black-Scholes) with the exact Black-Scholes solution.
    
    Also prints the absolute errors between the two methods.
    """
    # Parameters for the option
    s0 = 30.0
    sigma = 0.1
    r = 0.02
    K = 40.0
    T = 1.0
    delta = 1.0   # when delta = 1, the CEV model reduces to Black-Scholes
    X = 5 * s0    # choose an upper bound for the asset price domain

    # FD grid settings
    n = 126  # number of time steps
    m = 10000  # number of space steps

    # Instantiate the CEV option pricer (which implements FD methods)
    pricer = CEVoptionPricer(s0, sigma, r, K, T, delta, X)
    
    # Compute FD prices using Crank-Nicolson for a call and a put
    fd_call_price, sol_call, space, time = pricer.crank_nicolson_call(n, m)
    fd_put_price, sol_put, space, time = pricer.crank_nicolson_put(n, m)
    
    # Compute the exact Black-Scholes prices
    bs_call, bs_put = black_scholes_call_put(s0, K, r, sigma, T)
    
    # Print and compare the results
    print("Comparison of FD and Exact Black-Scholes Prices (δ = 1):\n")
    print(f"FD Call Price         : {fd_call_price:.4f}")
    print(f"Black-Scholes Call    : {bs_call:.4f}")
    print(f"Call Price Error      : {fd_call_price - bs_call:.4e}\n")
    
    print(f"FD Put Price          : {fd_put_price:.4f}")
    print(f"Black-Scholes Put     : {bs_put:.4f}")
    print(f"Put Price Error       : {fd_put_price - bs_put:.4e}")

if __name__ == "__main__":
    compare_fd_bs()
