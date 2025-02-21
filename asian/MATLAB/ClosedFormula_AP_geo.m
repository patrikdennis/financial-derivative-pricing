function price = ClosedFormula_AP_geo(s0,sig,r,K,T)
    dStar = (T/2)*(r-sig.^2/6);
    d = (log(s0/K) + (T/2)*(r+sig.^2/6)) / (sig*sqrt(T/3));
    price = -exp(dStar)*s0*normcdf(-d) + K*normcdf(-d+sig*sqrt(T/3));
end
