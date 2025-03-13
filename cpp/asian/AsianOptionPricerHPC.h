#ifndef ASIANOPTIONPRICER_HPC_H
#define ASIANOPTIONPRICER_HPC_H

#include <vector>
#include <utility>

// AsianOptionPricer class encapsulates pricing methods for Asian options.
class AsianOptionPricerHPC {
public:
    // Constructor.
    AsianOptionPricerHPC(double s0, double sigma, double r, double K, double T);

    // Closed-form pricing methods for geometric Asian options.
    double closedFormulaACGeo();
    double closedFormulaAPGeo();

    // Monte Carlo simulation to generate stock paths.
    // The method returns a 2D vector where each simulation produces two paths
    // using antithetic variates.
    std::vector<std::vector<double>> stockPath(int n, int N);

    // Monte Carlo pricing methods (crude approach).
    std::pair<double, double> monteCarloACCrude(int n, int N);
    std::pair<double, double> monteCarloAPCrude(int n, int N);

    // Monte Carlo pricing methods with control variate technique.
    std::pair<double, double> monteCarloACControl(int n, int N);
    std::pair<double, double> monteCarloAPControl(int n, int N);

    // Convenience functions returning the price while storing the deviation.
    double monteCarloAC(int n, int N, double& dev);
    double monteCarloAP(int n, int N, double& dev);

    // Finite Difference pricing using the Crank-Nicolson method.
    // The solution grid, space grid, and time grid are returned via reference.
    double crankNicolsonAC(int n, int m, std::vector<std::vector<double>>& sol,
                           std::vector<double>& space, std::vector<double>& time_grid);
    double crankNicolsonAP(int n, int m, std::vector<std::vector<double>>& sol,
                           std::vector<double>& space, std::vector<double>& time_grid);

private:
    // Model parameters.
    double s0;    // Initial stock price.
    double sigma; // Volatility.
    double r;     // Risk-free interest rate.
    double K;     // Strike price.
    double T;     // Time to maturity.

    // Helper function for the finite difference method.
    double alpha(const std::vector<double>& time, int i, double z, double d);

    // Thomas algorithm for solving tridiagonal systems.
    std::vector<double> thomasAlgorithm(const std::vector<double>& a,
                                        const std::vector<double>& b,
                                        const std::vector<double>& c,
                                        const std::vector<double>& d_vec);
};

#endif // ASIANOPTIONPRICER_HPC_H

