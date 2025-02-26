function [price, deviation] = MonteCarlo_AP_crude(s0, sig, r, K, T, n, N)
    stockPath = StockPath(s0,sig,r,T,n,N);
    payOff = max(0, K - mean(stockPath));
    price = exp(-r*T)*mean(payOff);
    deviation = std(payOff)/sqrt(2*N);
end
