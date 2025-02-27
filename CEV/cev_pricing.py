import numpy as np
import scipy.stats as st

class CEVOptionPricer:
    def __init__(self, s0: float, sigma: float, K:float, T:float) -> None:
        self.s0 = s0
        self.sigma = sigma
        self.r = r 
        self.K = K 
        self.T = T
    
    def alpha(self, d_val, Q, space, i, j):
        """Helper method for finite differnce methods"""
    
    def d_operator(self, d)
    def crank_nicolson_call(self, n,m):
        
        priceIndex = np.round((m/X) * S0 + 1)
        dt = self.T / n 
        dx = X / m
        d = dt / dx**2
         
        time_grid = np.linspace(0,self.T, n + 1)
        space = np.linspace(0, X, dx)
        Z0index = np.argmin(np.abs(space - Z0))
        space = space - space[Z0index] + Z0
        
        sol = np.zeros((n + 1, m + 1))
        
        
        
        # SOLVE Q = ?
        
        for i in range(1, n + 1):
            A = np.zeros((m + 1, m + 1))
            B = np.zeros((m + 1, m + 1))
            # A[0,0] = 1
            # A[-1, -1] = 1
            # B[0, 0] = 1
            # B[-1, -1] = 1
            
            for j in range(1,m):
                A[j, j - 1] = -self._alpa(d_val, Q, space, i, j)
                A[j, j] = 
        return price, sol, space, time_grid
    
    
    def crank_nicolson_put(self, n,m):
        return price, sol, space, time_grid
    
    