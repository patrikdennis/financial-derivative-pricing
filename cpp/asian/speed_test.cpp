#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>
#include "AsianOptionPricer.h"     // Serial version (from AsianOptionPricer.cpp)
#include "AsianOptionPricerHPC.h"  // HPC version (from AsianOptionPricer_hpc.cpp)

// This function runs a Monte Carlo test for both pricers and prints their run times.
void compareMonteCarlo(int n_MC, const std::vector<int>& N_list,
                       double S0, double sigma, double r, double K, double T) {
    std::cout << "\n--- Monte Carlo Timing Comparison ---\n";
    std::cout << std::setw(12) << "N"
              << std::setw(20) << "Serial Time (s)"
              << std::setw(20) << "HPC Time (s)" << "\n";
    
    for (int N_MC : N_list) {
        // Serial pricer
        AsianOptionPricer serialPricer(S0, sigma, r, K, T);
        auto start = std::chrono::high_resolution_clock::now();
        auto serialResult = serialPricer.monteCarloACControl(n_MC, N_MC);
        auto end = std::chrono::high_resolution_clock::now();
        double serialTime = std::chrono::duration<double>(end - start).count();

        // HPC pricer
        AsianOptionPricerHPC hpcPricer(S0, sigma, r, K, T);
        start = std::chrono::high_resolution_clock::now();
        auto hpcResult = hpcPricer.monteCarloACControl(n_MC, N_MC);
        end = std::chrono::high_resolution_clock::now();
        double hpcTime = std::chrono::duration<double>(end - start).count();

        std::cout << std::setw(12) << N_MC
                  << std::setw(20) << serialTime
                  << std::setw(20) << hpcTime << "\n";
    }
}

// This function runs a Finite Difference test for both pricers and prints their run times.
void compareFiniteDifference(int n_FD, const std::vector<int>& m_list,
                             double S0, double sigma, double r, double K, double T) {
    std::cout << "\n--- Finite Difference Timing Comparison (n=" << n_FD << ") ---\n";
    std::cout << std::setw(12) << "m"
              << std::setw(20) << "Serial Time (s)"
              << std::setw(20) << "HPC Time (s)" << "\n";
    
    for (int m_FD : m_list) {
        // Serial pricer
        AsianOptionPricer serialPricer(S0, sigma, r, K, T);
        std::vector<std::vector<double>> sol;
        std::vector<double> space, time_grid;
        auto start = std::chrono::high_resolution_clock::now();
        serialPricer.crankNicolsonAC(n_FD, m_FD, sol, space, time_grid);
        auto end = std::chrono::high_resolution_clock::now();
        double serialTime = std::chrono::duration<double>(end - start).count();

        // HPC pricer
        AsianOptionPricerHPC hpcPricer(S0, sigma, r, K, T);
        sol.clear();
        space.clear();
        time_grid.clear();
        start = std::chrono::high_resolution_clock::now();
        hpcPricer.crankNicolsonAC(n_FD, m_FD, sol, space, time_grid);
        end = std::chrono::high_resolution_clock::now();
        double hpcTime = std::chrono::duration<double>(end - start).count();

        std::cout << std::setw(12) << m_FD
                  << std::setw(20) << serialTime
                  << std::setw(20) << hpcTime << "\n";
    }
}

int main() {
    // Common parameters.
    double r = 0.02;
    double K = 40;
    double T = 0.5;
    double S0 = 30;
    double sigma = 0.1;
    int n_MC = 50;          // Monte Carlo time steps per path.
    int n_FD = 126;         // Finite Difference time steps (fixed).

    // Test Monte Carlo with various numbers of simulations.
    std::vector<int> N_list = {1000, 10000, 100000, 1000000};
    compareMonteCarlo(n_MC, N_list, S0, sigma, r, K, T);

    // Test Finite Difference with various m values (spatial steps).
    std::vector<int> m_list = {30, 100, 500, 2000};
    compareFiniteDifference(n_FD, m_list, S0, sigma, r, K, T);

    std::cout << "\nDone.\n";
    return 0;
}

