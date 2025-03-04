import numpy as np
import scipy.stats as st

class AsianOptionPricer:
    def __init__(self, s0: float, sigma: float, r: float, K: float, T: float) -> None:
        self.s0 = s0          # Initial stock price
        self.sigma = sigma    # Volatility
        self.r = r            # Risk-free rate
        self.K = K            # Strike price
        self.T = T            # Time to maturity

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
        w = np.random.randn(N, n)
        W = np.vstack([-w, w])
        cum_steps = np.tile(np.arange(1, n + 1), (2 * N, 1))
        drift = (self.r - self.sigma**2 / 2) * h * cum_steps
        diffusion = self.sigma * np.sqrt(h) * np.cumsum(W, axis=1)
        paths = self.s0 * np.exp(drift + diffusion)
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

    def alpha(self, time, i, z, d):
        """
        Compute the alpha coefficient using:
            alpha = d * sigma^2 * (gamma - z)^2 / 4,
        where gamma = (1 - exp(-r * time[i])) / (r * T).
        """
        gamma_val = (1 - np.exp(-self.r * time[i])) / (self.r * self.T)
        return d * (self.sigma ** 2) * (gamma_val - z) ** 2 / 4

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

    def crank_nicolson_ac(self, n, m):
        """
        Finite Difference (Crank–Nicolson) pricing for Asian call option.
        Uses the Thomas algorithm to solve the tridiagonal system for interior nodes.
        
        Returns:
            price, solution grid, space grid, and time grid.
        """
        dt = self.T / n
        Z0 = (1 - np.exp(-self.r * self.T)) / (self.r * self.T) - self.K * np.exp(-self.r * self.T) / self.s0
        half_range = abs(Z0) + 1
        dz = 2 * half_range / m
        d_val = dt / dz**2

        time_grid = np.linspace(0, self.T, n + 1)
        space = np.linspace(-half_range, half_range, m + 1)
        Z0index = np.argmin(np.abs(space - Z0))
        shift = space[Z0index] - Z0
        space = space - shift

        sol = np.zeros((n + 1, m + 1))
        for j in range(m + 1):
            sol[0, j] = np.maximum(space[j], 0)

        # Set boundary conditions for all time levels
        for i in range(n + 1):
            sol[i, 0] = 0
            sol[i, -1] = space[-1]

        # Time stepping
        for i in range(1, n + 1):
            # Build the tridiagonal system for interior nodes (j = 1, ..., m-1)
            d_vec_interior = np.zeros(m - 1)
            a = np.zeros(m - 2)  # sub-diagonal
            b = np.zeros(m - 1)  # main diagonal
            c = np.zeros(m - 2)  # super-diagonal

            for j in range(1, m):
                alpha_i_j = self.alpha(time_grid, i, space[j], d_val)
                alpha_i_minus1_j = self.alpha(time_grid, i-1, space[j], d_val)
                idx = j - 1
                b[idx] = 1 + 2 * alpha_i_j
                if idx > 0:
                    a[idx-1] = -alpha_i_j
                if idx < m - 2:
                    c[idx] = -alpha_i_j

                # Right-hand side from explicit part (matrix B)
                if j == 1:
                    d_vec_interior[idx] = (1 - 2 * alpha_i_minus1_j) * sol[i-1, j] + alpha_i_minus1_j * sol[i-1, j+1]
                elif j == m - 1:
                    d_vec_interior[idx] = alpha_i_minus1_j * sol[i-1, j-1] + (1 - 2 * alpha_i_minus1_j) * sol[i-1, j]
                else:
                    d_vec_interior[idx] = (alpha_i_minus1_j * sol[i-1, j-1] +
                                           (1 - 2 * alpha_i_minus1_j) * sol[i-1, j] +
                                           alpha_i_minus1_j * sol[i-1, j+1])
            
            sol_interior = self.thomas_algorithm(a, b, c, d_vec_interior)
            sol[i, 1:m] = sol_interior

        price = self.s0 * sol[-1, Z0index]
        return price, sol, space, time_grid

    def crank_nicolson_ap(self, n, m):
        """
        Finite Difference (Crank–Nicolson) pricing for Asian put option.
        Uses the Thomas algorithm to solve the tridiagonal system for interior nodes.
        
        Returns:
            price, solution grid, space grid, and time grid.
        """
        dt = self.T / n
        Z0 = -(1 - np.exp(-self.r * self.T)) / (self.r * self.T) + (self.K * np.exp(-self.r * self.T)) / self.s0
        half_range = abs(Z0) + 1
        dz = 2 * half_range / m
        d_val = dt / dz**2

        time_grid = np.linspace(0, self.T, n + 1)
        space = np.linspace(-half_range, half_range, m + 1)
        Z0index = np.argmin(np.abs(space - Z0))
        shift = space[Z0index] - Z0
        space = space - shift

        sol = np.zeros((n + 1, m + 1))
        for j in range(m + 1):
            sol[0, j] = np.maximum(space[j], 0)
        for i in range(n + 1):
            sol[i, 0] = 0
            sol[i, -1] = -space[-1]

        for i in range(1, n + 1):
            d_vec_interior = np.zeros(m - 1)
            a = np.zeros(m - 2)
            b = np.zeros(m - 1)
            c = np.zeros(m - 2)
            for j in range(1, m):
                alpha_i_j = self.alpha(time_grid, i, space[j], d_val)
                alpha_i_minus1_j = self.alpha(time_grid, i-1, space[j], d_val)
                idx = j - 1
                b[idx] = 1 + 2 * alpha_i_j
                if idx > 0:
                    a[idx-1] = -alpha_i_j
                if idx < m - 2:
                    c[idx] = -alpha_i_j

                if j == 1:
                    d_vec_interior[idx] = (1 - 2 * alpha_i_minus1_j) * sol[i-1, j] + alpha_i_minus1_j * sol[i-1, j+1]
                elif j == m - 1:
                    d_vec_interior[idx] = alpha_i_minus1_j * sol[i-1, j-1] + (1 - 2 * alpha_i_minus1_j) * sol[i-1, j]
                else:
                    d_vec_interior[idx] = (alpha_i_minus1_j * sol[i-1, j-1] +
                                           (1 - 2 * alpha_i_minus1_j) * sol[i-1, j] +
                                           alpha_i_minus1_j * sol[i-1, j+1])
            sol_interior = self.thomas_algorithm(a, b, c, d_vec_interior)
            sol[i, 1:m] = sol_interior

        price = self.s0 * sol[-1, Z0index]
        return price, sol, space, time_grid

