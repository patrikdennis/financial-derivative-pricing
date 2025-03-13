#include "EuropeanPricer.h"
#include <vector>
#include <cmath>
#include <stdexcept>
#include <algorithm>

// Constructor.
EuropeanPricer::EuropeanPricer(double s0, double sigma, double r, double K, double T, int n, int m, double X)
    : s0(s0), sigma(sigma), r(r), K(K), T(T), nFD(n), mFD(m), X(X) {}

// Helper: α coefficient
double EuropeanPricer::alpha(int x, double d, const std::vector<double> &space) {
    return 0.25 * d * (sigma * sigma) * (space[x] * space[x]);
}

// Helper: β coefficient
double EuropeanPricer::beta(int x, double d, const std::vector<double> &space, double dx) {
    return 0.25 * d * r * space[x] * dx;
}

// Thomas algorithm for solving a tridiagonal system
std::vector<double> EuropeanPricer::thomasAlgorithm(const std::vector<double> &lower,
                                                    const std::vector<double> &diag,
                                                    const std::vector<double> &upper,
                                                    const std::vector<double> &d_vec) {
    int n = diag.size();
    std::vector<double> cp(n - 1, 0.0), dp(n, 0.0), x(n, 0.0);
    cp[0] = upper[0] / diag[0];
    dp[0] = d_vec[0] / diag[0];
    for (int i = 1; i < n - 1; i++) {
        double denom = diag[i] - lower[i - 1] * cp[i - 1];
        cp[i] = upper[i] / denom;
        dp[i] = (d_vec[i] - lower[i - 1] * dp[i - 1]) / denom;
    }
    dp[n - 1] = (d_vec[n - 1] - lower[n - 2] * dp[n - 2]) / (diag[n - 1] - lower[n - 2] * cp[n - 2]);
    x[n - 1] = dp[n - 1];
    for (int i = n - 2; i >= 0; i--) {
        x[i] = dp[i] - cp[i] * x[i + 1];
    }
    return x;
}

// Crank–Nicolson for European Call
void EuropeanPricer::crankNicolsonCall(int n, int m, double &price,
                                       std::vector<std::vector<double>> &sol,
                                       std::vector<double> &space,
                                       std::vector<double> &time) {
    double dt = T / n;
    double dx = X / m;
    double d = dt / (dx * dx);

    // Build time grid
    time.resize(n + 1);
    for (int i = 0; i <= n; i++) {
        time[i] = i * dt;
    }
    // Build space grid
    space.resize(m + 1);
    for (int j = 0; j <= m; j++) {
        space[j] = j * dx;
    }
    // Determine the index corresponding to s0
    int priceIndex = static_cast<int>(std::round((m / X) * s0));
    priceIndex = std::max(0, std::min(priceIndex, m));

    // Initialize solution grid: sol[0] corresponds to t = T (terminal condition)
    sol.assign(n + 1, std::vector<double>(m + 1, 0.0));
    // Terminal condition: payoff = max(S - K, 0)
    for (int j = 0; j <= m; j++) {
        sol[0][j] = std::max(space[j] - K, 0.0);
    }
    // Boundary conditions:
    // At S = 0: call value = 0
    for (int i = 0; i <= n; i++) {
        sol[i][0] = 0.0;
    }
    // At S = X: call value = X - K*exp(-r*t)
    for (int i = 0; i <= n; i++) {
        sol[i][m] = X - K * std::exp(-r * time[i]);
    }

    // Build the interior tridiagonal coefficients for A and B.
    int interiorSize = m - 1; // indices 1,..., m-1.
    std::vector<double> A_lower(interiorSize - 1, 0.0);
    std::vector<double> A_diag(interiorSize, 0.0);
    std::vector<double> A_upper(interiorSize - 1, 0.0);
    std::vector<double> B_lower(interiorSize - 1, 0.0);
    std::vector<double> B_diag(interiorSize, 0.0);
    std::vector<double> B_upper(interiorSize - 1, 0.0);
    for (int k = 1; k <= m - 1; k++) {
        int idx = k - 1;
        double a_val = alpha(k, d, space);
        double b_val = beta(k, d, space, dx);
        A_diag[idx] = 1.0 + (r * dt / 2.0) + 2.0 * a_val;
        B_diag[idx] = 1.0 - (r * dt / 2.0) - 2.0 * a_val;
        if (k > 1) {
            A_lower[idx - 1] = b_val - a_val;
            B_lower[idx - 1] = -b_val + a_val;
        }
        if (k < m - 1) {
            A_upper[idx] = -b_val - a_val;
            B_upper[idx] = b_val + a_val;
        }
    }

    // Time stepping (backward in time: row 0 is t=T, row n is t=0)
    for (int i = 1; i <= n; i++) {
        // Compute rhs = B * (sol[i-1]) for interior nodes
        std::vector<double> rhs(interiorSize, 0.0);
        for (int k = 1; k <= m - 1; k++) {
            int idx = k - 1;
            double term = B_diag[idx] * sol[i - 1][k];
            if (k > 1) {
                term += B_lower[idx - 1] * sol[i - 1][k - 1];
            }
            if (k < m - 1) {
                term += B_upper[idx] * sol[i - 1][k + 1];
            }
            rhs[idx] = term;
        }
        // Adjust for the boundary at S = X (right boundary).
        if (interiorSize > 0)
            rhs[interiorSize - 1] -= A_upper.back() * sol[i][m];
        // Solve the tridiagonal system A * newInterior = rhs.
        std::vector<double> newInterior = thomasAlgorithm(A_lower, A_diag, A_upper, rhs);
        // Form new row with boundaries.
        std::vector<double> newRow(m + 1, 0.0);
        newRow[0] = sol[i][0]; // Already set (boundary S=0: 0)
        newRow[m] = X - K * std::exp(-r * time[i]); // Boundary at S = X.
        for (int k = 1; k <= m - 1; k++) {
            newRow[k] = newInterior[k - 1];
        }
        sol[i] = newRow;
    }
    price = sol[n][priceIndex];
}

// Crank–Nicolson for European Put.
void EuropeanPricer::crankNicolsonPut(int n, int m, double &price,
                                      std::vector<std::vector<double>> &sol,
                                      std::vector<double> &space,
                                      std::vector<double> &time) {
    double dt = T / n;
    double dx = X / m;
    double d = dt / (dx * dx);

    // Build grids
    time.resize(n + 1);
    for (int i = 0; i <= n; i++) {
        time[i] = i * dt;
    }
    space.resize(m + 1);
    for (int j = 0; j <= m; j++) {
        space[j] = j * dx;
    }
    int priceIndex = static_cast<int>(std::round((m / X) * s0));
    priceIndex = std::max(0, std::min(priceIndex, m));

    sol.assign(n + 1, std::vector<double>(m + 1, 0.0));
    // Terminal condition: payoff = max(K - S, 0)
    for (int j = 0; j <= m; j++) {
        sol[0][j] = std::max(K - space[j], 0.0);
    }
    // Boundary conditions:
    // At S = 0: put value = K*exp(-r*t)
    for (int i = 0; i <= n; i++) {
        sol[i][0] = K * std::exp(-r * time[i]);
    }
    // At S = X: put value = 0
    for (int i = 0; i <= n; i++) {
        sol[i][m] = 0.0;
    }

    // Build interior tridiagonal coefficients.
    int interiorSize = m - 1;
    std::vector<double> A_lower(interiorSize - 1, 0.0);
    std::vector<double> A_diag(interiorSize, 0.0);
    std::vector<double> A_upper(interiorSize - 1, 0.0);
    std::vector<double> B_lower(interiorSize - 1, 0.0);
    std::vector<double> B_diag(interiorSize, 0.0);
    std::vector<double> B_upper(interiorSize - 1, 0.0);
    for (int k = 1; k <= m - 1; k++) {
        int idx = k - 1;
        double a_val = alpha(k, d, space);
        double b_val = beta(k, d, space, dx);
        A_diag[idx] = 1.0 + (r * dt / 2.0) + 2.0 * a_val;
        B_diag[idx] = 1.0 - (r * dt / 2.0) - 2.0 * a_val;
        if (k > 1) {
            A_lower[idx - 1] = b_val - a_val;
            B_lower[idx - 1] = -b_val + a_val;
        }
        if (k < m - 1) {
            A_upper[idx] = -b_val - a_val;
            B_upper[idx] = b_val + a_val;
        }
    }

    // Time stepping.
    for (int i = 1; i <= n; i++) {
        std::vector<double> rhs(interiorSize, 0.0);
        for (int k = 1; k <= m - 1; k++) {
            int idx = k - 1;
            double term = B_diag[idx] * sol[i - 1][k];
            if (k > 1) {
                term += B_lower[idx - 1] * sol[i - 1][k - 1];
            }
            if (k < m - 1) {
                term += B_upper[idx] * sol[i - 1][k + 1];
            }
            rhs[idx] = term;
        }
        // Adjust for boundaries.
        if (!A_lower.empty())
            rhs[0] = rhs[0] - A_lower[0] * sol[i][0];
        if (!A_upper.empty())
            rhs[interiorSize - 1] = rhs[interiorSize - 1] - A_upper.back() * sol[i][m];
        std::vector<double> newInterior = thomasAlgorithm(A_lower, A_diag, A_upper, rhs);
        std::vector<double> newRow(m + 1, 0.0);
        newRow[0] = K * std::exp(-r * time[i]); // S=0 boundary.
        newRow[m] = 0.0;                        // S=X boundary.
        for (int k = 1; k <= m - 1; k++) {
            newRow[k] = newInterior[k - 1];
        }
        sol[i] = newRow;
    }
    price = sol[n][priceIndex];
}

