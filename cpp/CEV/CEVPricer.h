#ifndef CEVPRICER_H
#define CEVPRICER_H

#include <vector>

class CEVPricer {
public:
    // Constructor.
    // s0    : initial stock price
    // sigma : volatility
    // r     : risk-free interest rate
    // K     : strike price
    // T     : time to maturity
    // delta : elasticity parameter in the CEV model
    // X     : sufficiently large upper bound for stock price (S in [0, X])
    CEVPricer(double s0, double sigma, double r, double K, double T, double delta, double X);

    // Finite difference pricing via the Crank–Nicolson scheme.
    // On output:
    //   price : option value at (t=0, S=s0)
    //   sol   : grid (n+1 x m+1) with row 0 corresponding to maturity (t=T)
    //           and row n corresponding to t=0.
    //   space : spatial grid (stock prices)
    //   time  : time grid
    void crankNicolsonCall(int n, int m, double& price,
                           std::vector<std::vector<double>>& sol,
                           std::vector<double>& space,
                           std::vector<double>& time);

    void crankNicolsonPut(int n, int m, double& price,
                          std::vector<std::vector<double>>& sol,
                          std::vector<double>& space,
                          std::vector<double>& time);

    // Monte Carlo pricing using Euler–Maruyama simulation.
    // The simulation uses antithetic variates if 'antithetic' is true.
    // For the call and put methods, the price and the standard error are returned via reference.
    void eulerMaruyamaPaths(int n, int N, bool antithetic,
                            std::vector<std::vector<double>>& paths);

    void monteCarloCall(int n, int N, double& price, double& std_error);
    void monteCarloPut(int n, int N, double& price, double& std_error);

private:
    double s0;
    double sigma;
    double r;
    double K;
    double T;
    double delta;
    double X;

    // Solve a tridiagonal system A x = d using the Thomas algorithm.
    // a: sub-diagonal (length n-1)
    // b: main diagonal (length n)
    // c: super-diagonal (length n-1)
    // d_vec: right-hand side (length n)
    // Returns: solution vector x (length n)
    std::vector<double> thomasAlgorithm(const std::vector<double>& a,
                                          const std::vector<double>& b,
                                          const std::vector<double>& c,
                                          const std::vector<double>& d_vec);
};

#endif // CEVPRICER_H

