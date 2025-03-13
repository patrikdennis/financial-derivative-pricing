#include "CEVPricer.h"
#include <vector>
#include <cmath>
#include <algorithm>
#include <random>
#include <stdexcept>

// Constructor.
CEVPricer::CEVPricer(double s0, double sigma, double r, double K, double T, double delta, double X)
    : s0(s0), sigma(sigma), r(r), K(K), T(T), delta(delta), X(X) {}

// Thomas algorithm: solves a tridiagonal system.
std::vector<double> CEVPricer::thomasAlgorithm(const std::vector<double>& a,
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

// Crank–Nicolson finite difference for European Call.
void CEVPricer::crankNicolsonCall(int n, int m, double& price,
                                  std::vector<std::vector<double>>& sol,
                                  std::vector<double>& space,
                                  std::vector<double>& time) {
    double dt = T / n;
    double dx = X / m;
    double d = dt / (dx * dx);

    // Build time grid: row 0 corresponds to t=T, row n corresponds to t=0.
    time.resize(n + 1);
    for (int i = 0; i <= n; i++) {
        time[i] = i * dt;
    }
    // Build space grid.
    space.resize(m + 1);
    for (int j = 0; j <= m; j++) {
        space[j] = j * dx;
    }
    // Determine index corresponding to s0.
    int priceIndex = static_cast<int>(std::round(s0 / dx));
    priceIndex = std::max(0, std::min(priceIndex, m));

    // Initialize solution grid.
    sol.assign(n + 1, std::vector<double>(m + 1, 0.0));
    // Terminal condition at t=T: payoff = max(S - K, 0)
    for (int j = 0; j <= m; j++) {
        sol[0][j] = std::max(space[j] - K, 0.0);
    }
    // Boundary conditions:
    // S=0: call value = 0.
    for (int i = 0; i <= n; i++) {
        sol[i][0] = 0.0;
    }
    // S=X: call value = X - K * exp(-r * t)
    for (int i = 0; i <= n; i++) {
        sol[i][m] = X - K * std::exp(-r * time[i]);
    }

    // Precompute coefficients for interior nodes j=1,...,m-1.
    int interiorSize = m - 1;
    std::vector<double> a, b, c;
    if (interiorSize > 0) {
        a.resize(interiorSize - 1, 0.0);
        b.resize(interiorSize, 0.0);
        c.resize(interiorSize - 1, 0.0);
        for (int j = 1; j <= m - 1; j++) {
            int idx = j - 1;
            double S = space[j];
            double A_left  = (0.25 * d * r * S * dx) - (0.25 * d * sigma * sigma * std::pow(S, 2 * delta));
            double A_diag  = 1.0 + (0.5 * d * sigma * sigma * std::pow(S, 2 * delta));
            double A_right = - (0.25 * d * r * S * dx) - (0.25 * d * sigma * sigma * std::pow(S, 2 * delta));
            b[idx] = A_diag;
            if (j > 1) {
                a[idx - 1] = A_left;
            }
            if (j < m - 1) {
                c[idx] = A_right;
            }
        }
    }

    // Time stepping backward (from t=T to t=0).
    for (int i = 1; i <= n; i++) {
        std::vector<double> rhs(interiorSize, 0.0);
        for (int j = 1; j <= m - 1; j++) {
            double S = space[j];
            double B_left  = - (0.25 * d * r * S * dx) + (0.25 * d * sigma * sigma * std::pow(S, 2 * delta));
            double B_diag  = 1.0 - (0.5 * d * sigma * sigma * std::pow(S, 2 * delta));
            double B_right = (0.25 * d * r * S * dx) + (0.25 * d * sigma * sigma * std::pow(S, 2 * delta));
            rhs[j - 1] = B_left * sol[i - 1][j - 1] +
                         B_diag * sol[i - 1][j] +
                         B_right * sol[i - 1][j + 1];
        }
        // Adjust for the boundary contribution at S = X.
        if (interiorSize > 0)
            rhs[interiorSize - 1] -= c.back() * sol[i][m];
        // Solve tridiagonal system.
        if (interiorSize > 0) {
            std::vector<double> sol_interior = thomasAlgorithm(a, b, c, rhs);
            for (int j = 1; j <= m - 1; j++) {
                sol[i][j] = sol_interior[j - 1];
            }
        }
    }
    // Option price at t=0 is in the last row.
    price = sol[n][priceIndex];
}

// Crank–Nicolson finite difference for European Put.
void CEVPricer::crankNicolsonPut(int n, int m, double& price,
                                 std::vector<std::vector<double>>& sol,
                                 std::vector<double>& space,
                                 std::vector<double>& time) {
    double dt = T / n;
    double dx = X / m;
    double d = dt / (dx * dx);

    // Build grids.
    time.resize(n + 1);
    for (int i = 0; i <= n; i++) {
        time[i] = i * dt;
    }
    space.resize(m + 1);
    for (int j = 0; j <= m; j++) {
        space[j] = j * dx;
    }
    int priceIndex = static_cast<int>(std::round(s0 / dx));
    priceIndex = std::max(0, std::min(priceIndex, m));

    // Initialize solution grid.
    sol.assign(n + 1, std::vector<double>(m + 1, 0.0));
    // Terminal condition at t=T: payoff = max(K - S, 0)
    for (int j = 0; j <= m; j++) {
        sol[0][j] = std::max(K - space[j], 0.0);
    }
    // Boundary conditions:
    // At S=0: put value = K * exp(-r*t)
    for (int i = 0; i <= n; i++) {
        sol[i][0] = K * std::exp(-r * time[i]);
    }
    // At S=X: put value = 0.
    for (int i = 0; i <= n; i++) {
        sol[i][m] = 0.0;
    }

    // Precompute coefficients for interior nodes.
    int interiorSize = m - 1;
    std::vector<double> a, b, c;
    if (interiorSize > 0) {
        a.resize(interiorSize - 1, 0.0);
        b.resize(interiorSize, 0.0);
        c.resize(interiorSize - 1, 0.0);
        for (int j = 1; j <= m - 1; j++) {
            int idx = j - 1;
            double S = space[j];
            double A_left  = (0.25 * d * r * S * dx) - (0.25 * d * sigma * sigma * std::pow(S, 2 * delta));
            double A_diag  = 1.0 + (0.5 * d * sigma * sigma * std::pow(S, 2 * delta));
            double A_right = - (0.25 * d * r * S * dx) - (0.25 * d * sigma * sigma * std::pow(S, 2 * delta));
            b[idx] = A_diag;
            if (j > 1) {
                a[idx - 1] = A_left;
            }
            if (j < m - 1) {
                c[idx] = A_right;
            }
        }
    }

    // Time stepping.
    for (int i = 1; i <= n; i++) {
        std::vector<double> rhs(interiorSize, 0.0);
        for (int j = 1; j <= m - 1; j++) {
            double S = space[j];
            double B_left  = - (0.25 * d * r * S * dx) + (0.25 * d * sigma * sigma * std::pow(S, 2 * delta));
            double B_diag  = 1.0 - (0.5 * d * sigma * sigma * std::pow(S, 2 * delta));
            double B_right = (0.25 * d * r * S * dx) + (0.25 * d * sigma * sigma * std::pow(S, 2 * delta));
            rhs[j - 1] = B_left * sol[i - 1][j - 1] +
                         B_diag * sol[i - 1][j] +
                         B_right * sol[i - 1][j + 1];
        }
        // Adjust for left boundary at S=0.
        double A_left_first = (0.25 * d * r * space[1] * dx) - (0.25 * d * sigma * sigma * std::pow(space[1], 2 * delta));
        if (!rhs.empty())
            rhs[0] -= A_left_first * sol[i][0];
        // Solve tridiagonal system.
        if (interiorSize > 0) {
            std::vector<double> sol_interior = thomasAlgorithm(a, b, c, rhs);
            for (int j = 1; j <= m - 1; j++) {
                sol[i][j] = sol_interior[j - 1];
            }
        }
    }
    price = sol[n][priceIndex];
}

// Euler–Maruyama simulation for the SDE: dS = r*S*dt + sigma*S*dW.
void CEVPricer::eulerMaruyamaPaths(int n, int N, bool antithetic,
                                   std::vector<std::vector<double>>& paths) {
    double dt = T / n;
    int totalPaths = antithetic ? 2 * N : N;
    paths.assign(totalPaths, std::vector<double>(n + 1, 0.0));
    // Set initial condition.
    for (int i = 0; i < totalPaths; i++) {
        paths[i][0] = s0;
    }
    // Set up random number generation.
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<double> dist(0.0, std::sqrt(dt));
    
    // For each simulation.
    for (int i = 0; i < N; i++) {
        std::vector<double> dW(n, 0.0);
        for (int j = 0; j < n; j++) {
            dW[j] = dist(gen);
        }
        // First set: use dW as is.
        for (int j = 0; j < n; j++) {
            paths[i][j + 1] = paths[i][j] + r * paths[i][j] * dt + sigma * paths[i][j] * dW[j];
        }
        // If antithetic, second set uses negative increments.
        if (antithetic) {
            int i2 = i + N;
            for (int j = 0; j < n; j++) {
                paths[i2][j + 1] = paths[i2][j] + r * paths[i2][j] * dt + sigma * paths[i2][j] * (-dW[j]);
            }
        }
    }
}

// Monte Carlo pricing for European Call using Euler–Maruyama.
void CEVPricer::monteCarloCall(int n, int N, double& price, double& std_error) {
    std::vector<std::vector<double>> paths;
    eulerMaruyamaPaths(n, N, true, paths);
    int totalPaths = paths.size();
    std::vector<double> discounted(totalPaths, 0.0);
    for (int i = 0; i < totalPaths; i++) {
        double ST = paths[i][n];
        double payoff = std::max(ST - K, 0.0);
        discounted[i] = std::exp(-r * T) * payoff;
    }
    double sum = 0.0;
    for (double v : discounted) {
        sum += v;
    }
    price = sum / totalPaths;
    double variance = 0.0;
    for (double v : discounted) {
        variance += (v - price) * (v - price);
    }
    if (totalPaths > 1)
        variance /= (totalPaths - 1);
    std_error = std::sqrt(variance) / std::sqrt(totalPaths);
}

// Monte Carlo pricing for European Put using Euler–Maruyama.
void CEVPricer::monteCarloPut(int n, int N, double& price, double& std_error) {
    std::vector<std::vector<double>> paths;
    eulerMaruyamaPaths(n, N, true, paths);
    int totalPaths = paths.size();
    std::vector<double> discounted(totalPaths, 0.0);
    for (int i = 0; i < totalPaths; i++) {
        double ST = paths[i][n];
        double payoff = std::max(K - ST, 0.0);
        discounted[i] = std::exp(-r * T) * payoff;
    }
    double sum = 0.0;
    for (double v : discounted) {
        sum += v;
    }
    price = sum / totalPaths;
    double variance = 0.0;
    for (double v : discounted) {
        variance += (v - price) * (v - price);
    }
    if (totalPaths > 1)
        variance /= (totalPaths - 1);
    std_error = std::sqrt(variance) / std::sqrt(totalPaths);
}

