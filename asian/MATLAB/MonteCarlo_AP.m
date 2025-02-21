function [price, deviation] = MonteCarlo_AP(s0, sig, r, K, T,n,N)
    detValueCV = ClosedFormula_AP_geo(s0, sig, r, K, T);
    stockPath = StockPath(s0, sig, r, T, n, N);
    payOff = max(0, K-mean(stockPath));
    payOffCV = max(K-geomean(stockPath), 0);
    price = exp(-r*T)*(mean(payOff - payOffCV) + detValueCV);
    deviation = std(payOff - payOffCV + detValueCV) / sqrt(2*N);
end
