function [price, deviation] = MonteCarlo_AC_crude(s0, sig, r, K, T, n, N)
    stockPath = StockPath(s0, sig, r, T, n,N);
    payOff = max(0, mean(stockPath)-K);
    price = exp(-r*T)*mean(payOff);
    deviation = std(payOff)/sqrt(2*N);
end
