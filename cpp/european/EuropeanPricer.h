#ifndef EUROPEANPRICER_H
#define EUROPEANPRICER_H

#include <vector>

class EuropeanPricer {
public:
    // s0    : initial stock price
    // sigma : volatility
    // r     : risk-free interest rate
    // K     : strike price
    // T     : time to maturity
    // n     : number of time steps for finite differences
    // m     : number of space steps for finite differences
    // X     : sufficiently large upper bound for stock price (S in [0, X])
    EuropeanPricer(double s0, double sigma, double r, double K, double T, int n, int m, double X);

    // Finite difference pricing via the Crank–Nicolson scheme for a European Call.
    // Outputs:
    //   price : option value at t=0 and S = s0,
    //   sol   : grid of shape (n+1) x (m+1) (row 0: t=T, row n: t=0),
    //   space : spatial grid (stock prices),
    //   time  : time grid.
    void crankNicolsonCall(int n, int m, double &price,
                             std::vector<std::vector<double>> &sol,
                             std::vector<double> &space,
                             std::vector<double> &time);

    // Finite difference pricing via the Crank–Nicolson scheme for a European Put.
    void crankNicolsonPut(int n, int m, double &price,
                            std::vector<std::vector<double>> &sol,
                            std::vector<double> &space,
                            std::vector<double> &time);

private:
    double s0;
    double sigma;
    double r;
    double K;
    double T;
    int nFD; // default number of time steps 
    int mFD; // default number of space steps
    double X;

    // Helper methods for PDE coefficients.
    // α(x) = 0.25 * d * sigma² * (space[x])²
    double alpha(int x, double d, const std::vector<double> &space);
    // β(x) = 0.25 * d * r * space[x] * dx
    double beta(int x, double d, const std::vector<double> &space, double dx);

    // Thomas algorithm to solve a tridiagonal system.
    // lower: sub-diagonal (length n-1)
    // diag : main diagonal (length n)
    // upper: super-diagonal (length n-1)
    // d_vec: right-hand side vector (length n)
    // Returns solution vector (length n).
    std::vector<double> thomasAlgorithm(const std::vector<double> &lower,
                                          const std::vector<double> &diag,
                                          const std::vector<double> &upper,
                                          const std::vector<double> &d_vec);
};

#endif // EUROPEANPRICER_H

