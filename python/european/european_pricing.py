import numpy as np
import scipy.stats as st
import scipy.linalg

class EuropeanOptionPricer:
    def __init__(self, s0: float, sigma: float, r: float, K: float, T: float, n: float, m: float, X: float) -> None:
        """
        Parameters:
          s0    : initial stock price
          sigma : volatility (sigma)
          r     : risk-free interest rate
          K     : strike price
          T     : time to maturity
          n     : number of time steps
          m     : number of space steps
          X     : sufficiently large upper bound for stock price (S in [0, X])
        """
        self.s0 = s0          
        self.sigma = sigma    
        self.r = r            
        self.K = K            
        self.T = T
        self.n = n
        self.m = m
        self.X = X

    def alpha(self, x, d, space):
        """Helper method for second-derivative coefficient in the PDE."""
        return 0.25 * d * (self.sigma**2) * (space[x]**2)

    def beta(self, x, d, space, dx):
        """Helper method for first-derivative coefficient in the PDE."""
        return 0.25 * d * self.r * space[x] * dx

    def crank_nicolson_call(self, n, m):
        """
        Crank–Nicolson finite difference for a European Call.

        Returns:
          (price, sol, space, time)
          where:
            price : the option value at (t=0, S=s0),
            sol   : shape (n+1, m+1) with row 0 => t=T, row n => t=0,
            space : array of length (m+1),
            time  : array of length (n+1).
        """
        dt = self.T / n
        dx = self.X / m
        d  = dt / (dx**2)

        time  = np.linspace(0, self.T, n+1)
        space = np.linspace(0, self.X, m+1)

        # zero-based index for S0
        priceIndex = int(round((m / self.X)*self.s0))

        # solution matrix, shape (n+1, m+1)
        sol = np.zeros((n+1, m+1))

        # Tri-di matrices A, B
        A = np.zeros((m+1, m+1))
        B = np.zeros((m+1, m+1))

        # Boundary conditions in A, B
        A[0,0] = 1.0
        A[m,m] = 1.0
        B[0,0] = 1.0
        B[m,m] = 1.0

        # Terminal condition at t=T => row 0
        sol[0,:] = np.maximum(space - self.K, 0.0)

        # Boundary in S: C(t,0)=0 => sol[:,0] = 0
        sol[:,0] = 0.0
        # For a call: at S=X => X - K e^{-r t}
        sol[:,m] = self.X - self.K*np.exp(-self.r*time)

        # Fill interior of A, B
        for k_ in range(1, m):
            a_ = self.alpha(k_, d, space)
            b_ = self.beta(k_, d, space, dx)

            A[k_,k_-1] =  b_ - a_
            A[k_,k_]   =  1.0 + (self.r*dt/2.0) + 2.0*a_
            A[k_,k_+1] = -b_ - a_

            B[k_,k_-1] = -b_ + a_
            B[k_,k_]   =  1.0 - (self.r*dt/2.0) - 2.0*a_
            B[k_,k_+1] =  b_ + a_

        # invert A once
        invA = np.linalg.inv(A)

        # Time stepping from row=1..n => t from T->0
        for i in range(1, n+1):
            prevRow = sol[i-1,:]
            rhs = B @ prevRow

            # update boundary at S=X
            rhs[m] = self.X - self.K*np.exp(-self.r*time[i])

            newRow = np.linalg.solve(A, rhs)
            sol[i,:] = newRow

        # final price => row n => t=0
        price = sol[n, priceIndex]
        return price, sol, space, time

    def crank_nicolson_put(self, n, m):
        """
        Crank–Nicolson finite difference for a European Put.

        Returns:
          (price, sol, space, time)
          where:
            price : the option value at (t=0, S=s0),
            sol   : shape (n+1, m+1) with row 0 => t=T, row n => t=0,
            space : array of length (m+1),
            time  : array of length (n+1).
        """
        dt = self.T / n
        dx = self.X / m
        d  = dt / (dx**2)

        time  = np.linspace(0, self.T, n+1)
        space = np.linspace(0, self.X, m+1)

        priceIndex = int(round((m / self.X)*self.s0))

        sol = np.zeros((n+1, m+1))

        A = np.zeros((m+1, m+1))
        B = np.zeros((m+1, m+1))

        A[0,0] = 1.0
        A[m,m] = 1.0
        B[0,0] = 1.0
        B[m,m] = 1.0

        # Terminal condition at t=T => payoff for put = max(K - S, 0)
        sol[0,:] = np.maximum(self.K - space, 0.0)

        # Boundary: P(t,0)=K e^{-r t}, P(t, X)=0
        sol[:,0] = self.K * np.exp(-self.r*time)
        sol[:,m] = 0.0

        # Fill interior
        for k_ in range(1, m):
            a_ = self.alpha(k_, d, space)
            b_ = self.beta(k_, d, space, dx)

            A[k_,k_-1] =  b_ - a_
            A[k_,k_]   =  1.0 + (self.r*dt/2.0) + 2.0*a_
            A[k_,k_+1] = -b_ - a_

            B[k_,k_-1] = -b_ + a_
            B[k_,k_]   =  1.0 - (self.r*dt/2.0) - 2.0*a_
            B[k_,k_+1] =  b_ + a_

        invA = np.linalg.inv(A)

        for i in range(1, n+1):
            prevRow = sol[i-1,:]
            rhs = B @ prevRow

            # update boundary at S=0
            rhs[0] = self.K*np.exp(-self.r*time[i])
            # S=X => 0
            rhs[m] = 0.0

            newRow = np.linalg.solve(A, rhs)
            sol[i,:] = newRow

        price = sol[n, priceIndex]
        return price, sol, space, time


if __name__ == "__main__":
    # Example usage
    s0    = 100.0
    sigma = 0.2
    r     = 0.05
    K     = 100.0
    T     = 1.0
    n     = 100
    m     = 200
    X     = 300.0

    # Create instance and compute call
    pricer = EuropeanOptionPricer(s0, sigma, r, K, T, n, m, X)
    call_price, sol_call, space_call, time_call = pricer.crank_nicolson_call(n, m)
    print(f"European Call Price: {call_price:.4f}")

    # Compute put
    put_price, sol_put, space_put, time_put = pricer.crank_nicolson_put(n, m)
    print(f"European Put Price:  {put_price:.4f}")
