function Path = StockPath(s0, sig, r, T, n,N)
    % Function for simulating stock paths . Returns a matrix , where the columns
    % consist of simulated stock paths. Yields twice the number of requested
    % simulations , since Antithetic variates are used . %
    % Variables:
    %−S0>=0. Initial stock price.
    % − sig >= 0. Volatility .
    %−r>=0. Risk−free interest rate.
    %−T>= 0. Time to maturity.
    % − n > 0, Integer. Number of subintervals with respect to time. % − N > 0, Integer . Number of simulated paths .

    h = T/n;
    w = randn(N,n);
    W = [-w;w];
    q = ones(2*N,n);
    Path = s0*exp((r-sig^2/2)*h.*cumsum(q') + sig*sqrt(h)*cumsum(W'));
    Path = [s0*ones(1,2*N); Path];
end