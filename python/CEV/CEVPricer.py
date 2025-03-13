import numpy as np
import scipy.stats as st
import scipy.linalg

class CEVoptionPricer:
    def __init__(self, 
                 s0: float, 
                 sigma: float,
                 r: float,
                 K: float, 
                 T: float, 
                 delta: float, 
                 X: float) -> None:
        """
        Parameters:
          s0    : initial stock price
          sigma : volatility
          r     : risk-free interest rate
          K     : strike price
          T     : time to maturity
          delta : elasticity parameter in the CEV model
          X     : sufficiently large upper bound for stock price (S in [0, X])
        """
        self.s0 = s0          
        self.sigma = sigma    
        self.r = r            
        self.K = K            
        self.T = T
        self.delta = delta
        self.X = X

    def thomas_algorithm(self, a, b, c, d_vec):
        """
        Solve a tridiagonal system Ax = d_vec using the Thomas algorithm.
        
        Parameters:
            a: sub-diagonal (length n-1)
            b: main diagonal (length n)
            c: super-diagonal (length n-1)
            d_vec: right-hand side vector (length n)
            
        Returns:
            x: solution vector (length n)
        """
        n = len(b)
        cp = np.zeros(n-1)
        dp = np.zeros(n)
        
        cp[0] = c[0] / b[0]
        dp[0] = d_vec[0] / b[0]
        
        for i in range(1, n-1):
            denom = b[i] - a[i-1] * cp[i-1]
            cp[i] = c[i] / denom
            dp[i] = (d_vec[i] - a[i-1] * dp[i-1]) / denom
        
        dp[n-1] = (d_vec[n-1] - a[n-2] * dp[n-2]) / (b[n-1] - a[n-2] * cp[n-2])
        
        x = np.zeros(n)
        x[-1] = dp[-1]
        for i in range(n-2, -1, -1):
            x[i] = dp[i] - cp[i] * x[i+1]
        
        return x

    def crank_nicolson_call(self, n, m):
        """
        Crank–Nicolson finite difference for a CEV based European Call using 
        the Thomas algorithm to solve the tridiagonal systems.

        Returns:
          (price, sol, space, time)
          where:
            price : the option value at (t=0, S=s0),
            sol   : grid of shape (n+1, m+1) with row 0 => t=T, row n => t=0,
            space : spatial grid (stock prices),
            time  : time grid.
        """
        dt = self.T / n
        dx = self.X / m
        d = dt / (dx**2)

        time = np.linspace(0, self.T, n+1)
        space = np.linspace(0, self.X, m+1)
        priceIndex = int(round((m / self.X) * self.s0))
        
        # Initialize solution grid and set terminal & boundary conditions
        sol = np.zeros((n+1, m+1))
        sol[0, :] = np.maximum(space - self.K, 0.0)  # payoff at maturity
        sol[:, 0] = 0.0                              # S=0, call value = 0
        for i in range(n+1):
            sol[i, m] = self.X - self.K * np.exp(-self.r * time[i])  # S=X boundary
        
        # Precompute the coefficients of the matrix A (for interior nodes j=1,..,m-1)
        # Note: unknown vector is sol[i,1:m]
        a = np.zeros(m-2)  # sub-diagonal
        b = np.zeros(m-1)  # main diagonal
        c = np.zeros(m-2)  # super-diagonal
        for j in range(1, m):
            idx = j - 1

            A_left  = ((1/4)*d*self.r*space[j]*dx) - ((1/4)*d*(self.sigma**2)*(space[j]**(2*self.delta)))
            A_diag  = 1 + ( (1/2)*d*(self.sigma**2)*(space[j]**(2*self.delta)))
            A_right = -((1/4)*d*self.r*space[j]*dx) - ((1/4)*d*(self.sigma**2)*(space[j]**(2*self.delta)))

            b[idx] = A_diag
            if j > 1:
                a[idx-1] = A_left
            if j < m-1:
                c[idx] = A_right

        # Time-stepping backward (from T to 0)
        for i in range(1, n+1):
            rhs = np.zeros(m-1)
            for j in range(1, m):

                # Coefficients for the explicit part (matrix B)
                B_left  = -((1/4)*d*self.r*space[j]*dx) + ((1/4)*d*(self.sigma**2)*(space[j]**(2*self.delta)))
                B_diag  = 1 - (1/2)*d*(self.sigma**2)*(space[j]**(2*self.delta))
                B_right = ((1/4)*d*self.r*space[j]*dx) +( (1/4)*d*(self.sigma**2)*(space[j]**(2*self.delta)))
                rhs[j-1] = (B_left * sol[i-1, j-1] + 
                            B_diag * sol[i-1, j] + 
                            B_right * sol[i-1, j+1])
            # Adjust for boundary contribution at S=X (j = m-1)
            # sol[i, m] is known from the boundary condition.
            rhs[-1] -= c[-1] * sol[i, m]
            
            # Solve tridiagonal system for interior nodes using the Thomas algorithm
            sol[i, 1:m] = self.thomas_algorithm(a, b, c, rhs)
        
        price = sol[n, priceIndex]
        return price, sol, space, time

    def crank_nicolson_put(self, n, m):
        """
        Crank–Nicolson finite difference for a CEV based European Put using 
        the Thomas algorithm to solve the tridiagonal systems.

        Returns:
          (price, sol, space, time)
          where:
            price : the option value at (t=0, S=s0),
            sol   : grid of shape (n+1, m+1) with row 0 => t=T, row n => t=0,
            space : spatial grid (stock prices),
            time  : time grid.
        """
        dt = self.T / n
        dx = self.X / m
        d = dt / (dx**2)

        time = np.linspace(0, self.T, n+1)
        space = np.linspace(0, self.X, m+1)
        priceIndex = int(round((m / self.X) * self.s0))
        
        # Initialize solution grid and set terminal & boundary conditions
        sol = np.zeros((n+1, m+1))
        sol[0, :] = np.maximum(self.K - space, 0.0)  # payoff at maturity
        for i in range(n+1):
            sol[i, 0] = self.K * np.exp(-self.r * time[i])  # S=0 boundary for put
        sol[:, m] = 0.0                                   # S=X, put value = 0
        
        # Precompute coefficients for matrix A (for interior nodes j=1,..,m-1)
        a = np.zeros(m-2)  # sub-diagonal
        b = np.zeros(m-1)  # main diagonal
        c = np.zeros(m-2)  # super-diagonal
        for j in range(1, m):
            idx = j - 1
            
            A_left  = ((1/4)*d*self.r*space[j]*dx) - ((1/4)*d*(self.sigma**2)*(space[j]**(2*self.delta)))
            A_diag  = 1 + ( (1/2)*d*(self.sigma**2)*(space[j]**(2*self.delta)))
            A_right = -((1/4)*d*self.r*space[j]*dx) - ((1/4)*d*(self.sigma**2)*(space[j]**(2*self.delta)))

            b[idx] = A_diag
            if j > 1:
                a[idx-1] = A_left
            if j < m-1:
                c[idx] = A_right

        # Time stepping backward (from T to 0)
        for i in range(1, n+1):
            rhs = np.zeros(m-1)
            for j in range(1, m):
                # Coefficients for the explicit part (matrix B)
                B_left  = -(1/4)*d*self.r*space[j]*dx + (1/4)*d*(self.sigma**2)*(space[j]**(2*self.delta))
                B_diag  = 1 - (1/2)*d*(self.sigma**2)*(space[j]**(2*self.delta))
                B_right = (1/4)*d*self.r*space[j]*dx + (1/4)*d*(self.sigma**2)*(space[j]**(2*self.delta))
                rhs[j-1] = (B_left * sol[i-1, j-1] +
                            B_diag * sol[i-1, j] +
                            B_right * sol[i-1, j+1])
            # Adjust for left boundary contribution (j = 1)
            A_left_first = (1/4)*d*self.r*space[1]*dx - (1/4)*d*(self.sigma**2)*(space[1]**(2*self.delta))
            rhs[0] -= A_left_first * sol[i, 0]
            # For j = m-1, right boundary is 0 so no adjustment needed.
            
            # Solve the tridiagonal system for interior nodes
            sol[i, 1:m] = self.thomas_algorithm(a, b, c, rhs)
            sol[i, m] = 0.0  # enforce right boundary explicitly
        
        price = sol[n, priceIndex]
        return price, sol, space, time
    
        
    def euler_maruyama_paths(self, n, N, antithetic=True):
        """
        Generate simulated stock paths using the Euler–Maruyama scheme for
        the (constant-volatility) SDE:
        
            dS(t) = r*S(t)*dt + sigma*S(t)*dW(t).
        
        Parameters
        ----------
        s0        : float
            Initial stock price.
        sigma     : float
            Constant volatility parameter.
        r         : float
            Risk-free interest rate.
        T         : float
            Time to maturity.
        n   : int
            Number of time steps (partitions of [0, T]).
        N    : int
            Number of simulated paths (excluding antithetics).
        antithetic: bool
            If True, will generate antithetic variates, effectively doubling
            the total number of simulated paths.
        
        Returns
        -------
        paths : ndarray of shape (n_paths, n_steps+1)
            Simulated paths. Each row is one path in time from t=0 to t=T.
        """
        dt = self.T / n
        # Brownian increments (n_sims x n_steps)
        dW = np.random.normal(0, np.sqrt(dt), size=(N, n))
        if antithetic:
            # double up for antithetic variance reduction
            dW = np.vstack([dW, -dW])
        
        n_paths = dW.shape[0]
        
        # Initialize array of paths
        paths = np.zeros((n_paths, n + 1))
        paths[:, 0] = self.s0
        
        # Euler–Maruyama iteration
        for i in range(n):
            paths[:, i+1] = (
                paths[:, i]
                + self.r * paths[:, i] * dt
                + self.sigma * paths[:, i] * dW[:, i]
            )
        
        return paths

    def monte_carlo_call(self, n, N):
        """
        Prices a European call option using a basic Monte Carlo approach
        (Euler–Maruyama path generation).
        
        Parameters
        ----------
        s0     : float
            Initial stock price.
        sigma  : float
            Volatility parameter (constant in Black–Scholes).
        r      : float
            Risk-free interest rate.
        K      : float
            Strike price.
        T      : float
            Time to maturity.
        n: int
            Number of time steps in Euler discretization.
        N : int
            Number of simulated paths (excluding antithetic).
            
        Returns
        -------
        call_price : float
            The Monte Carlo estimate of the call option price.
        std_error  : float
            The standard error of the Monte Carlo estimate.
        """
        # Generate simulated paths
        paths = self.euler_maruyama_paths(n, N, antithetic=True)
        S_T = paths[:, -1]  # terminal prices
        
        # Compute discounted payoff
        payoffs = np.maximum(S_T - self.K, 0.0)
        discounted = np.exp(-self.r * self.T) * payoffs
        
        # Monte Carlo estimator
        call_price = np.mean(discounted)
        std_error = np.std(discounted, ddof=1) / np.sqrt(len(discounted))
        
        return call_price, std_error

    def monte_carlo_put(self, n, N):
        """
        Prices a European put option using a basic Monte Carlo approach
        (Euler–Maruyama path generation).
        
        Parameters
        ----------
        s0     : float
            Initial stock price.
        sigma  : float
            Volatility parameter (constant in Black–Scholes).
        r      : float
            Risk-free interest rate.
        K      : float
            Strike price.
        T      : float
            Time to maturity.
        n     : int
            Number of time steps in Euler discretization.
        N     : int
            Number of simulated paths (excluding antithetic).
            
        Returns
        -------
        put_price : float
            The Monte Carlo estimate of the put option price.
        std_error : float
            The standard error of the Monte Carlo estimate.
        """
        # Generate simulated paths
        paths = self.euler_maruyama_paths(n, N, antithetic=True)
        S_T = paths[:, -1]  # terminal prices
        
        # Compute discounted payoff
        payoffs = np.maximum(self.K - S_T, 0.0)
        discounted = np.exp(-self.r * self.T) * payoffs
        
        # Monte Carlo estimator
        put_price = np.mean(discounted)
        std_error = np.std(discounted, ddof=1) / np.sqrt(len(discounted))
        
        return put_price, std_error
