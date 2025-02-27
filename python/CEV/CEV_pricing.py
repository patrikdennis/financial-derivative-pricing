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
        self.delta = delta
        self.X = X

    def crank_nicolson_call(self, n, m):
        """
        Crank–Nicolson finite difference for a CEV based on European Call.

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
        A[0, 0] = 1.0
        A[m, m] = 1.0
        B[0, 0] = 1.0
        B[m, m] = 1.0

        # Terminal condition at t=T => row 0
        sol[0, :] = np.maximum(space - self.K, 0.0)

        # Boundary in S: C(t,0)=0 => sol[:,0] = 0
        sol[:, 0] = 0.0
        # For a call: at S=X => X - K e^{-r t}
        sol[:, m] = self.X - self.K*np.exp(-self.r*time)



        for k in range(1, m):
            A[k, k - 1] = ( (1/4)*d*self.r*space[k]*dx) - (1/4*d*(self.sigma**2)*(space[k]**(2*self.delta)))
            A[k, k + 1] = -( (1/4)*d*self.r*space[k]*dx) - (1/4*d*(self.sigma**2)*(space[k]**(2*self.delta)))
            A[k, k] = 1 + ( (1/2)*d*(self.sigma**2)*(space[k]**(2*self.delta)))

            B[k, k] = 1 - ( (1/2)*d*(self.sigma**2)*(space[k]**(2*self.delta)))
            B[k, k + 1] = ( (1/4)*d*self.r*space[k]*dx) + ( (1/4)*d*(self.sigma**2)*(space[k]**(2*self.delta)))
            B[k, k - 1] = -( (1/4)*d*self.r*space[k]*dx) + ( (1/4)*d*(self.sigma**2)*(space[k]**(2*self.delta)))

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
        Crank–Nicolson finite difference for CEV model with European Put.

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
        d = dt / (dx**2)

        time = np.linspace(0, self.T, n+1)
        space = np.linspace(0, self.X, m+1)

        priceIndex = int(round((m / self.X)*self.s0))

        sol = np.zeros((n+1, m+1))

        A = np.zeros((m+1, m+1))
        B = np.zeros((m+1, m+1))

        A[0, 0] = 1.0
        A[m, m] = 1.0
        B[0, 0] = 1.0
        B[m, m] = 1.0

        # Terminal condition at t=T => payoff for put = max(K - S, 0)
        sol[0, :] = np.maximum(self.K - space, 0.0)

        # Boundary: P(t,0)=K e^{-r t}, P(t, X)=0
        sol[:, 0] = self.K * np.exp(-self.r*time)
        sol[:, m] = 0.0

        # Fill interior
        for k in range(1, m):
            A[k, k - 1] = ( (1/4)*d*self.r*space[k]*dx) - (1/4*d*(self.sigma**2)*(space[k]**(2*self.delta)))
            A[k, k + 1] = -( (1/4)*d*self.r*space[k]*dx) - (1/4*d*(self.sigma**2)*(space[k]**(2*self.delta)))
            A[k, k] = 1 + ( (1/2)*d*(self.sigma**2)*(space[k]**(2*self.delta)))

            B[k, k] = 1 - ( (1/2)*d*(self.sigma**2)*(space[k]**(2*self.delta)))
            B[k, k + 1] = ( (1/4)*d*self.r*space[k]*dx) + ( (1/4)*d*(self.sigma**2)*(space[k]**(2*self.delta)))
            B[k, k - 1] = -( (1/4)*d*self.r*space[k]*dx) + ( (1/4)*d*(self.sigma**2)*(space[k]**(2*self.delta)))


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


