#ifndef OPTION_PRICING_H
#define OPTION_PRICING_H

#include <vector>

class AsianOptionPricer {
public:
    double s0;
    double sigma;
    double r;
    double K;
    double T;
    
    AsianOptionPricer(double s0, double sigma, double r, double K, double T);
    
    static double closedFormulaACGeo(double s0, double sigma, double r, double K, double T);
    static double closedFormulaAPGeo(double s0, double sigma, double r, double K, double T);
    
    // Returns a 2D vector with dimensions [2*N][n+1]
    std::vector<std::vector<double> > stockPath(int n, int N);
    
    double monteCarloACCrude(int n, int N, double &deviation);
    double monteCarloAPCrude(int n, int N, double &deviation);
    double monteCarloAC(int n, int N, double &deviation);
    double monteCarloAP(int n, int N, double &deviation);
    
    // The crank–nicolson methods output the computed solution grid, space grid, and time grid.
    double crankNicolsonAC(int n, int m, std::vector<std::vector<double> > &sol, std::vector<double> &space, std::vector<double> &time_grid);
    double crankNicolsonAP(int n, int m, std::vector<std::vector<double> > &sol, std::vector<double> &space, std::vector<double> &time_grid);
};

#endif
