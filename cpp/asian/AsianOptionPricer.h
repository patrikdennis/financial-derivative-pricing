#ifndef ASIANOPTIONPRICER_H
#define ASIANOPTIONPRICER_H

#include <vector>
using std::vector;
using std::pair;

class AsianOptionPricer {
public:
    // Option parameters.
    double s0;     // initial stock price
    double sigma;  // volatility
    double r;      // risk-free rate
    double K;      // strike price
    double T;      // time to maturity

    // Constructor.
    AsianOptionPricer(double s0_, double sigma_, double r_, double K_, double T_);

    // Closed-form (geometric) Asian option formulas.
    double closedFormulaACGeo();
    double closedFormulaAPGeo();

    // Monte Carlo simulation:
    // Generates stock paths using antithetic variates.
    vector<vector<double>> stockPath(int n, int N);
    
    // Crude Monte Carlo pricing.
    pair<double, double> monteCarloACCrude(int n, int N);
    pair<double, double> monteCarloAPCrude(int n, int N);
    
    // Monte Carlo pricing with control variate.
    pair<double, double> monteCarloACControl(int n, int N);
    pair<double, double> monteCarloAPControl(int n, int N);

    double monteCarloAC(int n, int N, double& dev);
    double monteCarloAP(int n, int N, double& dev);

    // Finite Difference: Crank-Nicolson for Asian options.
    // These methods output the solution grid, space grid, and time grid.
    double crankNicolsonAC(int n, int m, vector<vector<double>>& sol,
                           vector<double>& space, vector<double>& time_grid);
    double crankNicolsonAP(int n, int m, vector<vector<double>>& sol,
                           vector<double>& space, vector<double>& time_grid);

private:
    // Helper: finite-difference coefficient.
    double alpha(const vector<double>& time, int i, double z, double d);

    // Thomas algorithm for solving tridiagonal systems.
    vector<double> thomasAlgorithm(const vector<double>& a,
                                   const vector<double>& b,
                                   const vector<double>& c,
                                   const vector<double>& d_vec);
};

#endif  // ASIANOPTIONPRICER_H

