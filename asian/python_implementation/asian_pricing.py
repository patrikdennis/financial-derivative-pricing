#!/usr/bin/env python3
import numpy as np
import scipy.stats as st

class AsianOptionPricer:
    def __init__(self, s0: float, sigma: float, r: float, K: float, T: float) -> None:
        self.s0 = s0          # initial stock price
        self.sigma = sigma    # volatility
        self.r = r            # risk-free rate
        self.K = K            # strike price
        self.T = T            # time to maturity

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

    def _alpha(self, d_val, Q, space, i, j):
        """Helper method for finite difference methods (replaces nested function)."""
        return d_val * self.sigma**2 * (Q[i] - space[j])**2 / 4

    def crank_nicolson_ac(self, n, m):
        """
        Finite Difference (Crank–Nicolson) pricing for Asian call option.
        Returns price, solution grid, space grid, and time grid.
        """
        dt = self.T / n
        Z0 = (1 - np.exp(-self.r * self.T)) / (self.r * self.T) - self.K * np.exp(-self.r * self.T) / self.s0
        dz = 2 * (np.abs(Z0) + 1) / m
        d_val = dt / dz**2

        time_grid = np.linspace(0, self.T, n + 1)
        space = np.linspace(-(np.abs(Z0) + 1), (np.abs(Z0) + 1), m + 1)
        Z0index = np.argmin(np.abs(space - Z0))
        space = space - space[Z0index] + Z0

        sol = np.zeros((n + 1, m + 1))
        sol[0, :] = np.maximum(space, 0)
        sol[:, 0] = 0
        sol[:, -1] = space[-1]

        Q = (1 - np.exp(-self.r * time_grid)) / (self.r * self.T)

        for i in range(1, n + 1):
            A = np.zeros((m + 1, m + 1))
            B = np.zeros((m + 1, m + 1))
            A[0, 0] = 1
            A[-1, -1] = 1
            B[0, 0] = 1
            B[-1, -1] = 1
            for j in range(1, m):
                A[j, j - 1] = -self._alpha(d_val, Q, space, i, j)
                A[j, j] = 1 + 2 * self._alpha(d_val, Q, space, i, j)
                A[j, j + 1] = -self._alpha(d_val, Q, space, i, j)
                B[j, j - 1] = self._alpha(d_val, Q, space, i - 1, j)
                B[j, j] = 1 - 2 * self._alpha(d_val, Q, space, i - 1, j)
                B[j, j + 1] = self._alpha(d_val, Q, space, i - 1, j)
            sol[i, :] = np.linalg.solve(A, B.dot(sol[i - 1, :]))
        price = self.s0 * sol[-1, Z0index]
        return price, sol, space, time_grid

    def crank_nicolson_ap(self, n, m):
        """
        Finite Difference (Crank–Nicolson) pricing for Asian put option.
        Returns price, solution grid, space grid, and time grid.
        """
        dt = self.T / n
        Z0 = -(1 - np.exp(-self.r * self.T)) / (self.r * self.T) + self.K * np.exp(-self.r * self.T) / self.s0
        dz = 2 * (np.abs(Z0) + 1) / m
        d_val = dt / dz**2

        time_grid = np.linspace(0, self.T, n + 1)
        space = np.linspace(-(np.abs(Z0) + 1), (np.abs(Z0) + 1), m + 1)
        Z0index = np.argmin(np.abs(space - Z0))
        space = space - space[Z0index] + Z0

        sol = np.zeros((n + 1, m + 1))
        sol[0, :] = np.maximum(space, 0)
        sol[:, 0] = 0
        sol[:, -1] = space[-1]

        Q = (1 - np.exp(-self.r * time_grid)) / (self.r * self.T)

        for i in range(1, n + 1):
            A = np.zeros((m + 1, m + 1))
            B = np.zeros((m + 1, m + 1))
            A[0, 0] = 1
            A[-1, -1] = 1
            B[0, 0] = 1
            B[-1, -1] = 1
            for j in range(1, m):
                A[j, j - 1] = -self._alpha(d_val, Q, space, i, j)
                A[j, j] = 1 + 2 * self._alpha(d_val, Q, space, i, j)
                A[j, j + 1] = -self._alpha(d_val, Q, space, i, j)
                B[j, j - 1] = self._alpha(d_val, Q, space, i - 1, j)
                B[j, j] = 1 - 2 * self._alpha(d_val, Q, space, i - 1, j)
                B[j, j + 1] = self._alpha(d_val, Q, space, i - 1, j)
            sol[i, :] = np.linalg.solve(A, B.dot(sol[i - 1, :]))
        price = self.s0 * sol[-1, Z0index]
        return price, sol, space, time_grid
