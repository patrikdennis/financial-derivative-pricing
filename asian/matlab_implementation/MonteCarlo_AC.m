function [price, deviation] = MonteCarlo_AC(s0, sig, r, K, T,n,N)
    detValueCV = ClosedFormula_AC_geo(s0,sig,r,K,T);
    stockPath = StockPath(s0,sig,r,T,n,N);
    payOff = max(0,mean(stockPath)-K);
    payOffCV = max(geomean(stockPath) - K, 0);
    price = exp(-r*T)*(mean(payOff) - mean(payOffCV) + detValueCV);
    deviation = std(payOff - payOffCV + detValueCV)/sqrt(2*N);
end
