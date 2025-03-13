#include "AsianOptionPricerHPC.h"
#include <cmath>
#include <algorithm>
#include <random>
#include <stdexcept>
#ifdef _OPENMP
#include <omp.h>
#endif

// For the normal CDF.
static double norm_cdf(double x) {
    return 0.5 * erfc(-x / std::sqrt(2.0));
}

// Constructor.
AsianOptionPricerHPC::AsianOptionPricerHPC(double s0_, double sigma_, double r_, double K_, double T_)
    : s0(s0_), sigma(sigma_), r(r_), K(K_), T(T_) {}

// Closed-form solution for geometric Asian call option.
double AsianOptionPricerHPC::closedFormulaACGeo() {
    double dStar = (T / 2.0) * (r - sigma * sigma / 6.0);
    double d = (std::log(s0 / K) + (T / 2.0) * (r + sigma * sigma / 6.0)) /
               (sigma * std::sqrt(T / 3.0));
    double price = std::exp(dStar) * s0 * norm_cdf(d) -
                   K * norm_cdf(d - sigma * std::sqrt(T / 3.0));
    return price;
}

// Closed-form solution for geometric Asian put option.
double AsianOptionPricerHPC::closedFormulaAPGeo() {
    double dStar = (T / 2.0) * (r - sigma * sigma / 6.0);
    double d = (std::log(s0 / K) + (T / 2.0) * (r + sigma * sigma / 6.0)) /
               (sigma * std::sqrt(T / 3.0));
    double price = -std::exp(dStar) * s0 * norm_cdf(-d) +
                   K * norm_cdf(-d + sigma * std::sqrt(T / 3.0));
    return price;
}

// Modified stockPath function using OpenMP for parallelization.
// Each simulation is independent so we can parallelize the loop over N simulations.
std::vector<std::vector<double>> AsianOptionPricerHPC::stockPath(int n, int N) {
    double h = T / n;
    // Pre-allocate a container for all paths (each simulation produces two paths with antithetic variates)
    std::vector<std::vector<double>> paths(2 * N, std::vector<double>(n + 1, 0.0));

    // Parallelize the outer loop over simulations.
#ifdef _OPENMP
    #pragma omp parallel for
#endif
    for (int i = 0; i < N; i++) {
        // Create a thread-local random number generator.
        std::random_device rd;
#ifdef _OPENMP
        unsigned int seed = rd() + i + omp_get_thread_num();
#else
        unsigned int seed = rd() + i;
#endif
        std::mt19937 gen(seed);
        std::normal_distribution<> dist(0.0, 1.0);

        std::vector<double> path1(n + 1, 0.0), path2(n + 1, 0.0);
        path1[0] = s0;
        path2[0] = s0;
        double cum1 = 0.0, cum2 = 0.0;
        for (int j = 1; j <= n; j++) {
            double w = dist(gen);
            double w_anti = -w;
            cum1 += w;
            cum2 += w_anti;
            double drift = (r - 0.5 * sigma * sigma) * h;
            double diffusion = sigma * std::sqrt(h);
            path1[j] = s0 * std::exp(j * drift + diffusion * cum1);
            path2[j] = s0 * std::exp(j * drift + diffusion * cum2);
        }
        // Store the two generated paths in the pre-allocated container.
        paths[2 * i]     = path1;
        paths[2 * i + 1] = path2;
    }
    return paths;
}

// Monte Carlo crude pricing for Asian call option.
std::pair<double, double> AsianOptionPricerHPC::monteCarloACCrude(int n, int N) {
    std::vector<std::vector<double>> paths = stockPath(n, N);
    std::vector<double> payoffs;
    for (const auto& path : paths) {
        double sum = 0.0;
        for (double price : path)
            sum += price;
        double avg = sum / path.size();
        payoffs.push_back(std::max(avg - K, 0.0));
    }
    double sum_payoffs = 0.0;
    for (double p : payoffs)
        sum_payoffs += p;
    double price = std::exp(-r * T) * (sum_payoffs / payoffs.size());

    double mean = sum_payoffs / payoffs.size();
    double sq_sum = 0.0;
    for (double p : payoffs)
        sq_sum += (p - mean) * (p - mean);
    double deviation = std::sqrt(sq_sum / payoffs.size()) / std::sqrt(payoffs.size());
    return {price, deviation};
}

// Monte Carlo crude pricing for Asian put option.
std::pair<double, double> AsianOptionPricerHPC::monteCarloAPCrude(int n, int N) {
    std::vector<std::vector<double>> paths = stockPath(n, N);
    std::vector<double> payoffs;
    for (const auto& path : paths) {
        double sum = 0.0;
        for (double price : path)
            sum += price;
        double avg = sum / path.size();
        payoffs.push_back(std::max(K - avg, 0.0));
    }
    double sum_payoffs = 0.0;
    for (double p : payoffs)
        sum_payoffs += p;
    double price = std::exp(-r * T) * (sum_payoffs / payoffs.size());

    double mean = sum_payoffs / payoffs.size();
    double sq_sum = 0.0;
    for (double p : payoffs)
        sq_sum += (p - mean) * (p - mean);
    double deviation = std::sqrt(sq_sum / payoffs.size()) / std::sqrt(payoffs.size());
    return {price, deviation};
}

// Monte Carlo pricing with control variate for Asian call option.
std::pair<double, double> AsianOptionPricerHPC::monteCarloACControl(int n, int N) {
    double detValueCV = closedFormulaACGeo();
    std::vector<std::vector<double>> paths = stockPath(n, N);
    std::vector<double> payoff, payoffCV;
    for (const auto& path : paths) {
        double sum = 0.0, sum_log = 0.0;
        for (double price : path) {
            sum += price;
            sum_log += std::log(price);
        }
        double arith_avg = sum / path.size();
        double geo_avg = std::exp(sum_log / path.size());
        payoff.push_back(std::max(arith_avg - K, 0.0));
        payoffCV.push_back(std::max(geo_avg - K, 0.0));
    }
    double mean_payoff = 0.0, mean_payoffCV = 0.0;
    for (size_t i = 0; i < payoff.size(); i++) {
        mean_payoff += payoff[i];
        mean_payoffCV += payoffCV[i];
    }
    mean_payoff /= payoff.size();
    mean_payoffCV /= payoff.size();

    double price = std::exp(-r * T) * (mean_payoff - mean_payoffCV + detValueCV);

    double sq_sum = 0.0;
    for (size_t i = 0; i < payoff.size(); i++) {
        double diff = (payoff[i] - payoffCV[i] + detValueCV) - (mean_payoff - mean_payoffCV + detValueCV);
        sq_sum += diff * diff;
    }
    double deviation = std::sqrt(sq_sum / payoff.size()) / std::sqrt(payoff.size());
    return {price, deviation};
}

// Monte Carlo pricing with control variate for Asian put option.
std::pair<double, double> AsianOptionPricerHPC::monteCarloAPControl(int n, int N) {
    double detValueCV = closedFormulaAPGeo();
    std::vector<std::vector<double>> paths = stockPath(n, N);
    std::vector<double> payoff, payoffCV;
    for (const auto& path : paths) {
        double sum = 0.0, sum_log = 0.0;
        for (double price : path) {
            sum += price;
            sum_log += std::log(price);
        }
        double arith_avg = sum / path.size();
        double geo_avg = std::exp(sum_log / path.size());
        payoff.push_back(std::max(K - arith_avg, 0.0));
        payoffCV.push_back(std::max(K - geo_avg, 0.0));
    }
    double mean_payoff = 0.0, mean_payoffCV = 0.0;
    for (size_t i = 0; i < payoff.size(); i++) {
        mean_payoff += payoff[i];
        mean_payoffCV += payoffCV[i];
    }
    mean_payoff /= payoff.size();
    mean_payoffCV /= payoff.size();

    double price = std::exp(-r * T) * (mean_payoff - mean_payoffCV + detValueCV);

    double sq_sum = 0.0;
    for (size_t i = 0; i < payoff.size(); i++) {
        double diff = (payoff[i] - payoffCV[i] + detValueCV) - (mean_payoff - mean_payoffCV + detValueCV);
        sq_sum += diff * diff;
    }
    double deviation = std::sqrt(sq_sum / payoff.size()) / std::sqrt(payoff.size());
    return {price, deviation};
}

double AsianOptionPricerHPC::monteCarloAC(int n, int N, double& dev) {
    auto result = monteCarloACControl(n, N);
    dev = result.second;
    return result.first;
}

double AsianOptionPricerHPC::monteCarloAP(int n, int N, double& dev) {
    auto result = monteCarloAPControl(n, N);
    dev = result.second;
    return result.first;
}

// Helper: Compute alpha coefficient.
double AsianOptionPricerHPC::alpha(const std::vector<double>& time, int i, double z, double d) {
    double gamma_val = (1.0 - std::exp(-r * time[i])) / (r * T);
    return d * (sigma * sigma) * std::pow(gamma_val - z, 2) / 4.0;
}

// Thomas algorithm for solving a tridiagonal system.
std::vector<double> AsianOptionPricerHPC::thomasAlgorithm(const std::vector<double>& a,
                                                       const std::vector<double>& b,
                                                       const std::vector<double>& c,
                                                       const std::vector<double>& d_vec) {
    int n = b.size();
    std::vector<double> cp(n - 1, 0.0), dp(n, 0.0), x(n, 0.0);
    cp[0] = c[0] / b[0];
    dp[0] = d_vec[0] / b[0];
    for (int i = 1; i < n - 1; i++) {
        double denom = b[i] - a[i - 1] * cp[i - 1];
        cp[i] = c[i] / denom;
        dp[i] = (d_vec[i] - a[i - 1] * dp[i - 1]) / denom;
    }
    dp[n - 1] = (d_vec[n - 1] - a[n - 2] * dp[n - 2]) / (b[n - 1] - a[n - 2] * cp[n - 2]);
    x[n - 1] = dp[n - 1];
    for (int i = n - 2; i >= 0; i--) {
        x[i] = dp[i] - cp[i] * x[i + 1];
    }
    return x;
}

// Finite Difference (Crank-Nicolson) for Asian call option using Thomas algorithm.
double AsianOptionPricerHPC::crankNicolsonAC(int n, int m, std::vector<std::vector<double>>& sol,
                                          std::vector<double>& space, std::vector<double>& time_grid) {
    double dt = T / n;
    double Z0 = (1.0 - std::exp(-r * T)) / (r * T) - K * std::exp(-r * T) / s0;
    double half_range = std::fabs(Z0) + 1;
    double dz = 2 * half_range / m;
    double d_val = dt / (dz * dz);

    // Build time grid.
    time_grid.resize(n + 1);
    for (int i = 0; i <= n; i++) {
        time_grid[i] = i * dt;
    }
    // Build space grid.
    space.resize(m + 1);
    for (int j = 0; j <= m; j++) {
        space[j] = -half_range + j * dz;
    }
    // Shift grid so that one node equals Z0.
    int Z0index = 0;
    double minDiff = std::fabs(space[0] - Z0);
    for (int j = 1; j <= m; j++) {
        double diff = std::fabs(space[j] - Z0);
        if (diff < minDiff) {
            minDiff = diff;
            Z0index = j;
        }
    }
    double shift = space[Z0index] - Z0;
    for (int j = 0; j <= m; j++) {
        space[j] -= shift;
    }

    // Initialize solution grid with initial condition.
    sol.assign(n + 1, std::vector<double>(m + 1, 0.0));
    for (int j = 0; j <= m; j++) {
        sol[0][j] = std::max(space[j], 0.0);
    }
    // Boundary conditions.
    for (int i = 0; i <= n; i++) {
        sol[i][0] = 0.0;
        sol[i][m] = space[m];
    }

    // Time stepping.
    for (int i = 1; i <= n; i++) {
        int interior_size = m - 1;
        std::vector<double> d_vec_interior(interior_size, 0.0);
        std::vector<double> a(interior_size - 1, 0.0);
        std::vector<double> b(interior_size, 0.0);
        std::vector<double> c(interior_size - 1, 0.0);
        for (int j = 1; j < m; j++) {
            double alpha_i_j = alpha(time_grid, i, space[j], d_val);
            double alpha_i_minus1_j = alpha(time_grid, i - 1, space[j], d_val);
            int idx = j - 1;
            b[idx] = 1.0 + 2.0 * alpha_i_j;
            if (idx > 0)
                a[idx - 1] = -alpha_i_j;
            if (idx < interior_size - 1)
                c[idx] = -alpha_i_j;
            if (j == 1)
                d_vec_interior[idx] = (1.0 - 2.0 * alpha_i_minus1_j) * sol[i - 1][j] +
                                      alpha_i_minus1_j * sol[i - 1][j + 1];
            else if (j == m - 1)
                d_vec_interior[idx] = alpha_i_minus1_j * sol[i - 1][j - 1] +
                                      (1.0 - 2.0 * alpha_i_minus1_j) * sol[i - 1][j];
            else
                d_vec_interior[idx] = alpha_i_minus1_j * sol[i - 1][j - 1] +
                                      (1.0 - 2.0 * alpha_i_minus1_j) * sol[i - 1][j] +
                                      alpha_i_minus1_j * sol[i - 1][j + 1];
        }
        std::vector<double> sol_interior = thomasAlgorithm(a, b, c, d_vec_interior);
        for (int j = 1; j < m; j++) {
            sol[i][j] = sol_interior[j - 1];
        }
    }
    double price = s0 * sol[n][Z0index];
    return price;
}

// Finite Difference (Crank-Nicolson) for Asian put option using Thomas algorithm.
double AsianOptionPricerHPC::crankNicolsonAP(int n, int m, std::vector<std::vector<double>>& sol,
                                          std::vector<double>& space, std::vector<double>& time_grid) {
    double dt = T / n;
    double Z0 = -(1.0 - std::exp(-r * T)) / (r * T) + (K * std::exp(-r * T)) / s0;
    double half_range = std::fabs(Z0) + 1;
    double dz = 2 * half_range / m;
    double d_val = dt / (dz * dz);

    // Build time grid.
    time_grid.resize(n + 1);
    for (int i = 0; i <= n; i++) {
        time_grid[i] = i * dt;
    }
    // Build space grid.
    space.resize(m + 1);
    for (int j = 0; j <= m; j++) {
        space[j] = -half_range + j * dz;
    }
    // Shift grid so that one node equals Z0.
    int Z0index = 0;
    double minDiff = std::fabs(space[0] - Z0);
    for (int j = 1; j <= m; j++) {
        double diff = std::fabs(space[j] - Z0);
        if (diff < minDiff) {
            minDiff = diff;
            Z0index = j;
        }
    }
    double shift = space[Z0index] - Z0;
    for (int j = 0; j <= m; j++) {
        space[j] -= shift;
    }

    sol.assign(n + 1, std::vector<double>(m + 1, 0.0));
    for (int j = 0; j <= m; j++) {
        sol[0][j] = std::max(space[j], 0.0);
    }
    for (int i = 0; i <= n; i++) {
        sol[i][0] = 0.0;
        sol[i][m] = space[m];
    }
    for (int i = 1; i <= n; i++) {
        int interior_size = m - 1;
        std::vector<double> d_vec_interior(interior_size, 0.0);
        std::vector<double> a(interior_size - 1, 0.0);
        std::vector<double> b(interior_size, 0.0);
        std::vector<double> c(interior_size - 1, 0.0);
        for (int j = 1; j < m; j++) {
            double alpha_i_j = alpha(time_grid, i, space[j], d_val);
            double alpha_i_minus1_j = alpha(time_grid, i - 1, space[j], d_val);
            int idx = j - 1;
            b[idx] = 1.0 + 2.0 * alpha_i_j;
            if (idx > 0)
                a[idx - 1] = -alpha_i_j;
            if (idx < interior_size - 1)
                c[idx] = -alpha_i_j;
            if (j == 1)
                d_vec_interior[idx] = (1.0 - 2.0 * alpha_i_minus1_j) * sol[i - 1][j] +
                                      alpha_i_minus1_j * sol[i - 1][j + 1];
            else if (j == m - 1)
                d_vec_interior[idx] = alpha_i_minus1_j * sol[i - 1][j - 1] +
                                      (1.0 - 2.0 * alpha_i_minus1_j) * sol[i - 1][j];
            else
                d_vec_interior[idx] = alpha_i_minus1_j * sol[i - 1][j - 1] +
                                      (1.0 - 2.0 * alpha_i_minus1_j) * sol[i - 1][j] +
                                      alpha_i_minus1_j * sol[i - 1][j + 1];
        }
        std::vector<double> sol_interior = thomasAlgorithm(a, b, c, d_vec_interior);
        for (int j = 1; j < m; j++) {
            sol[i][j] = sol_interior[j - 1];
        }
    }
    double price = s0 * sol[n][Z0index];
    return price;
}

