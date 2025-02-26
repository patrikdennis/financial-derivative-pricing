#include "option_pricing.h"
#include <cmath>
#include <vector>
#include <random>
#include <numeric>
#include <algorithm>

// Helper: Normal CDF using error function
static double normcdf(double x) {
    return 0.5 * std::erfc(-x / std::sqrt(2.0));
}


// Helper: Solve linear system A*x = b using Gaussian elimination with partial pivoting.
std::vector<double> solveLinearSystem(std::vector<std::vector<double>> A, std::vector<double> b) {
    int n = A.size();
    for (int i = 0; i < n; i++) {
        // Partial pivoting
        int pivot = i;
        double max_val = std::abs(A[i][i]);
        for (int j = i + 1; j < n; j++) {
            if (std::abs(A[j][i]) > max_val) {
                max_val = std::abs(A[j][i]);
                pivot = j;
            }
        }
        if (std::abs(A[pivot][i]) < 1e-12)
            throw std::runtime_error("Zero pivot encountered in solveLinearSystem");
        if (pivot != i) {
            std::swap(A[i], A[pivot]);
            std::swap(b[i], b[pivot]);
        }
        // Elimination
        for (int j = i + 1; j < n; j++) {
            double factor = A[j][i] / A[i][i];
            for (int k = i; k < n; k++) {
                A[j][k] -= factor * A[i][k];
            }
            b[j] -= factor * b[i];
        }
    }
    // Back substitution
    std::vector<double> x(n, 0.0);
    for (int i = n - 1; i >= 0; i--) {
        double sum = 0.0;
        for (int j = i + 1; j < n; j++) {
            sum += A[i][j] * x[j];
        }
        x[i] = (b[i] - sum) / A[i][i];
    }
    return x;
}

AsianOptionPricer::AsianOptionPricer(double s0, double sigma, double r, double K, double T)
    : s0(s0), sigma(sigma), r(r), K(K), T(T)
{ }

double AsianOptionPricer::closedFormulaACGeo(double s0, double sigma, double r, double K, double T) {
    double dStar = (T / 2.0) * (r - sigma*sigma/6.0);
    double d = (std::log(s0/K) + (T/2.0)*(r + sigma*sigma/6.0)) / (sigma * std::sqrt(T/3.0));
    return std::exp(dStar)*s0*normcdf(d) - K*normcdf(d - sigma*std::sqrt(T/3.0));
}

double AsianOptionPricer::closedFormulaAPGeo(double s0, double sigma, double r, double K, double T) {
    double dStar = (T / 2.0) * (r - sigma*sigma/6.0);
    double d = (std::log(s0/K) + (T/2.0)*(r + sigma*sigma/6.0)) / (sigma * std::sqrt(T/3.0));
    return -std::exp(dStar)*s0*normcdf(-d) + K*normcdf(-d + sigma*std::sqrt(T/3.0));
}

std::vector<std::vector<double>> AsianOptionPricer::stockPath(int n, int N) {
    double h = T / n;
    int totalPaths = 2 * N;
    std::vector<std::vector<double>> paths(totalPaths, std::vector<double>(n+1, 0.0));
    std::mt19937 rng(std::random_device{}());
    std::normal_distribution<double> dist(0.0, 1.0);
    
    for (int i = 0; i < N; i++){
        std::vector<double> noise(n), noiseAnti(n);
        for (int j = 0; j < n; j++){
            noise[j] = dist(rng);
            noiseAnti[j] = -noise[j];
        }
        paths[i][0] = s0;
        paths[i+N][0] = s0;
        double sum1 = 0, sum2 = 0;
        for (int j = 0; j < n; j++){
            sum1 += noise[j];
            sum2 += noiseAnti[j];
            double exponent1 = (r - sigma*sigma/2.0)*h*(j+1) + sigma*std::sqrt(h)*sum1;
            double exponent2 = (r - sigma*sigma/2.0)*h*(j+1) + sigma*std::sqrt(h)*sum2;
            paths[i][j+1] = s0 * std::exp(exponent1);
            paths[i+N][j+1] = s0 * std::exp(exponent2);
        }
    }
    return paths;
}

double AsianOptionPricer::monteCarloACCrude(int n, int N, double &deviation) {
    auto paths = stockPath(n, N);
    int total = paths.size();
    std::vector<double> avg(total, 0.0);
    for (int i = 0; i < total; i++){
        double sum = 0;
        for (double val : paths[i]) sum += val;
        avg[i] = sum / paths[i].size();
    }
    std::vector<double> payoff(total, 0.0);
    for (int i = 0; i < total; i++){
        payoff[i] = std::max(avg[i] - K, 0.0);
    }
    double meanPayoff = std::accumulate(payoff.begin(), payoff.end(), 0.0) / total;
    deviation = std::sqrt(std::accumulate(payoff.begin(), payoff.end(), 0.0,
                  [meanPayoff](double sum, double x){ return sum + (x-meanPayoff)*(x-meanPayoff); }) / total);
    return std::exp(-r*T) * meanPayoff;
}

double AsianOptionPricer::monteCarloAPCrude(int n, int N, double &deviation) {
    auto paths = stockPath(n, N);
    int total = paths.size();
    std::vector<double> avg(total, 0.0);
    for (int i = 0; i < total; i++){
        double sum = 0;
        for (double val : paths[i]) sum += val;
        avg[i] = sum / paths[i].size();
    }
    std::vector<double> payoff(total, 0.0);
    for (int i = 0; i < total; i++){
        payoff[i] = std::max(K - avg[i], 0.0);
    }
    double meanPayoff = std::accumulate(payoff.begin(), payoff.end(), 0.0) / total;
    deviation = std::sqrt(std::accumulate(payoff.begin(), payoff.end(), 0.0,
                  [meanPayoff](double sum, double x){ return sum + (x-meanPayoff)*(x-meanPayoff); }) / total);
    return std::exp(-r*T) * meanPayoff;
}


// Monte Carlo pricing for Asian call option with control variate.
double AsianOptionPricer::monteCarloAC(int n, int N, double &deviation) {
    // Compute the closed-form geometric Asian call value for control variate.
    double detValueCV = AsianOptionPricer::closedFormulaACGeo(s0, sigma, r, K, T);
    // Simulate stock paths: dimensions 2*N x (n+1)
    auto paths = stockPath(n, N);
    int total = paths.size(); // should be 2*N

    // Compute arithmetic average of each path.
    std::vector<double> avg(total, 0.0);
    for (int i = 0; i < total; i++) {
        double sum = 0.0;
        for (double val : paths[i]) {
            sum += val;
        }
        avg[i] = sum / paths[i].size();
    }
    // Compute payoff for call: max( arithmetic average - K, 0 )
    std::vector<double> payoff(total, 0.0);
    for (int i = 0; i < total; i++) {
        payoff[i] = std::max(avg[i] - K, 0.0);
    }
    double sum_payoff = std::accumulate(payoff.begin(), payoff.end(), 0.0);
    double mean_payoff = sum_payoff / total;

    // Compute geometric average for each path.
    std::vector<double> geo(total, 0.0);
    for (int i = 0; i < total; i++) {
        double sum_log = 0.0;
        for (double val : paths[i]) {
            sum_log += std::log(val);
        }
        geo[i] = std::exp(sum_log / paths[i].size());
    }
    // Compute payoff for the control variate: max( geo average - K, 0 )
    std::vector<double> payoffCV(total, 0.0);
    for (int i = 0; i < total; i++) {
        payoffCV[i] = std::max(geo[i] - K, 0.0);
    }
    double sum_payoffCV = std::accumulate(payoffCV.begin(), payoffCV.end(), 0.0);
    double mean_payoffCV = sum_payoffCV / total;

    // Adjusted price using control variate:
    double price = std::exp(-r * T) * (mean_payoff - mean_payoffCV + detValueCV);

    // Compute deviation: standard deviation of (payoff - payoffCV + detValueCV)
    std::vector<double> diff(total, 0.0);
    for (int i = 0; i < total; i++) {
        diff[i] = payoff[i] - payoffCV[i] + detValueCV;
    }
    double sum_diff = std::accumulate(diff.begin(), diff.end(), 0.0);
    double mean_diff = sum_diff / total;
    double sq_sum = 0.0;
    for (double d : diff) {
        sq_sum += (d - mean_diff) * (d - mean_diff);
    }
    double std_dev = std::sqrt(sq_sum / total);
    deviation = std_dev / std::sqrt(2 * N);

    return price;
}

// Monte Carlo pricing for Asian put option with control variate.
double AsianOptionPricer::monteCarloAP(int n, int N, double &deviation) {
    // Compute the closed-form geometric Asian put value for control variate.
    double detValueCV = AsianOptionPricer::closedFormulaAPGeo(s0, sigma, r, K, T);
    // Simulate stock paths.
    auto paths = stockPath(n, N);
    int total = paths.size(); // should be 2*N

    // Compute arithmetic average for each path.
    std::vector<double> avg(total, 0.0);
    for (int i = 0; i < total; i++) {
        double sum = 0.0;
        for (double val : paths[i]) {
            sum += val;
        }
        avg[i] = sum / paths[i].size();
    }
    // Payoff for put: max( K - arithmetic average, 0 )
    std::vector<double> payoff(total, 0.0);
    for (int i = 0; i < total; i++) {
        payoff[i] = std::max(K - avg[i], 0.0);
    }
    double sum_payoff = std::accumulate(payoff.begin(), payoff.end(), 0.0);
    double mean_payoff = sum_payoff / total;

    // Compute geometric average for each path.
    std::vector<double> geo(total, 0.0);
    for (int i = 0; i < total; i++) {
        double sum_log = 0.0;
        for (double val : paths[i]) {
            sum_log += std::log(val);
        }
        geo[i] = std::exp(sum_log / paths[i].size());
    }
    // Payoff for control variate: max( K - geo average, 0 )
    std::vector<double> payoffCV(total, 0.0);
    for (int i = 0; i < total; i++) {
        payoffCV[i] = std::max(K - geo[i], 0.0);
    }
    double sum_payoffCV = std::accumulate(payoffCV.begin(), payoffCV.end(), 0.0);
    double mean_payoffCV = sum_payoffCV / total;

    // Adjusted price:
    double price = std::exp(-r * T) * (mean_payoff - mean_payoffCV + detValueCV);

    // Compute deviation:
    std::vector<double> diff(total, 0.0);
    for (int i = 0; i < total; i++) {
        diff[i] = payoff[i] - payoffCV[i] + detValueCV;
    }
    double sum_diff = std::accumulate(diff.begin(), diff.end(), 0.0);
    double mean_diff = sum_diff / total;
    double sq_sum = 0.0;
    for (double d : diff) {
        sq_sum += (d - mean_diff) * (d - mean_diff);
    }
    double std_dev = std::sqrt(sq_sum / total);
    deviation = std_dev / std::sqrt(2 * N);

    return price;
}
double AsianOptionPricer::crankNicolsonAC(int n, int m, 
    std::vector<std::vector<double>> &sol, 
    std::vector<double> &space, 
    std::vector<double> &time_grid) 
{
    double dt = T / n;
    double Z0 = (1 - std::exp(-r * T)) / (r * T) - K * std::exp(-r * T) / s0;
    double dz = 2 * (std::abs(Z0) + 1) / m;
    double d_val = dt / (dz * dz);

    // Build time grid:
    time_grid.resize(n + 1);
    for (int i = 0; i <= n; i++)
        time_grid[i] = i * dt;
    // Build space grid:
    space.resize(m + 1);
    for (int j = 0; j <= m; j++)
        space[j] = -(std::abs(Z0) + 1) + j * (2 * (std::abs(Z0) + 1) / m);
    int Z0index = 0;
    double minDiff = std::abs(space[0] - Z0);
    for (int j = 1; j <= m; j++) {
        double diff = std::abs(space[j] - Z0);
        if (diff < minDiff) { minDiff = diff; Z0index = j; }
    }
    for (int j = 0; j <= m; j++)
        space[j] = space[j] - space[Z0index] + Z0;

    // Initialize solution grid: at t=0, sol[0][j] = max(space[j], 0)
    sol.resize(n + 1, std::vector<double>(m + 1, 0.0));
    sol[0] = space;  
    for (int i = 0; i <= n; i++){
        sol[i][0] = 0.0;
        sol[i][m] = space[m];
    }
    // Precompute Q vector: Q[i] = (1 - exp(-r*t_i)) / (r*T)
    std::vector<double> Q(n + 1, 0.0);
    for (int i = 0; i <= n; i++)
        Q[i] = (1 - std::exp(-r * time_grid[i])) / (r * T);

    // Define lambda for alpha:
    auto alpha = [=](int i, int j) -> double {
        return d_val * sigma * sigma * std::pow(Q[i] - space[j], 2) / 4.0;
    };

    // Time stepping:
    for (int i = 1; i <= n; i++){
        // Build matrices A and B (size: (m+1)x(m+1))
        std::vector<std::vector<double>> A(m + 1, std::vector<double>(m + 1, 0.0));
        std::vector<std::vector<double>> B(m + 1, std::vector<double>(m + 1, 0.0));
        A[0][0] = 1.0; A[m][m] = 1.0;
        B[0][0] = 1.0; B[m][m] = 1.0;
        for (int j = 1; j < m; j++){
            A[j][j - 1] = -alpha(i, j);
            A[j][j] = 1 + 2 * alpha(i, j);
            A[j][j + 1] = -alpha(i, j);
            B[j][j - 1] = alpha(i - 1, j);
            B[j][j] = 1 - 2 * alpha(i - 1, j);
            B[j][j + 1] = alpha(i - 1, j);
        }
        // Compute right-hand side: rhs = B * sol[i-1]
        std::vector<double> rhs(m + 1, 0.0);
        for (int j = 0; j <= m; j++){
            double sum = 0.0;
            for (int k = 0; k <= m; k++){
                sum += B[j][k] * sol[i - 1][k];
            }
            rhs[j] = sum;
        }
        // Solve A * x = rhs using the helper function.
        std::vector<double> x = solveLinearSystem(A, rhs);
        sol[i] = x;
    }
    return s0 * sol[n][Z0index];
}

double AsianOptionPricer::crankNicolsonAP(int n, int m, 
    std::vector<std::vector<double>> &sol, 
    std::vector<double> &space, 
    std::vector<double> &time_grid) 
{
    double dt = T / n;
    // Alternate definition for put: Z0 = -(1 - exp(-rT))/(rT) + K*exp(-rT)/s0
    double Z0 = -(1 - std::exp(-r * T)) / (r * T) + K * std::exp(-r * T) / s0;
    double dz = 2 * (std::abs(Z0) + 1) / m;
    double d_val = dt / (dz * dz);

    // Build time grid:
    time_grid.resize(n + 1);
    for (int i = 0; i <= n; i++)
        time_grid[i] = i * dt;
    // Build space grid:
    space.resize(m + 1);
    for (int j = 0; j <= m; j++)
        space[j] = -(std::abs(Z0) + 1) + j * (2 * (std::abs(Z0) + 1) / m);
    int Z0index = 0;
    double minDiff = std::abs(space[0] - Z0);
    for (int j = 1; j <= m; j++){
        double diff = std::abs(space[j] - Z0);
        if (diff < minDiff) { minDiff = diff; Z0index = j; }
    }
    for (int j = 0; j <= m; j++)
        space[j] = space[j] - space[Z0index] + Z0;

    // Initialize solution grid: sol[0][j] = max(space[j], 0)
    sol.resize(n + 1, std::vector<double>(m + 1, 0.0));
    for (int j = 0; j <= m; j++){
        sol[0][j] = std::max(space[j], 0.0);
    }
    // Set boundary conditions: for all i, sol[i][0] = 0 and sol[i][m] = space[m]
    for (int i = 0; i <= n; i++){
        sol[i][0] = 0.0;
        sol[i][m] = space[m];
    }
    
    // Precompute Q vector: Q[i] = (1 - exp(-r*t_i))/(r*T)
    std::vector<double> Q(n + 1, 0.0);
    for (int i = 0; i <= n; i++){
        Q[i] = (1 - std::exp(-r * time_grid[i])) / (r * T);
    }
    
    // Define lambda for alpha:
    auto alpha = [=](int i, int j) -> double {
        return d_val * sigma * sigma * std::pow(Q[i] - space[j], 2) / 4.0;
    };
    
    // Time stepping: for each time step, solve A * x = B * sol[i-1]
    for (int i = 1; i <= n; i++){
        std::vector<std::vector<double>> A(m + 1, std::vector<double>(m + 1, 0.0));
        std::vector<std::vector<double>> B(m + 1, std::vector<double>(m + 1, 0.0));
        A[0][0] = 1.0; A[m][m] = 1.0;
        B[0][0] = 1.0; B[m][m] = 1.0;
        for (int j = 1; j < m; j++){
            A[j][j - 1] = -alpha(i, j);
            A[j][j] = 1 + 2 * alpha(i, j);
            A[j][j + 1] = -alpha(i, j);
            B[j][j - 1] = alpha(i - 1, j);
            B[j][j] = 1 - 2 * alpha(i - 1, j);
            B[j][j + 1] = alpha(i - 1, j);
        }
        // Compute rhs = B * sol[i-1]
        std::vector<double> rhs(m + 1, 0.0);
        for (int j = 0; j <= m; j++){
            double sum = 0.0;
            for (int k = 0; k <= m; k++){
                sum += B[j][k] * sol[i - 1][k];
            }
            rhs[j] = sum;
        }
        // Solve A*x = rhs
        std::vector<double> x = solveLinearSystem(A, rhs);
        sol[i] = x;
    }
    return s0 * sol[n][Z0index];
}
