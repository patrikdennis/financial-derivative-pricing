#include <iostream>
#include <vector>
#include <chrono>
#include "../AsianOptionPricer.h"
#include <iomanip>

int main() {
    // Common parameters
    double r = 0.02;
    double K = 40;
    double T = 0.5;
    double S0 = 30;
    double sigma = 0.1;     // We'll fix sigma for these tests
    int n_MC = 50;          // MC time steps per path
    int n_FD_default = 126; // default FD time steps

    // 1) Monte Carlo: measure times as we vary N
    // example: N = {1,000; 10,000; 100,000; 1,000,000}
    std::vector<int> N_list = {1000, 10000, 100000, 1000000};
    std::cout << "\n--- Monte Carlo Timing (varying N) ---\n";
    std::cout << "Parameters: S0=" << S0 << ", sigma=" << sigma << ", r=" << r
              << ", K=" << K << ", T=" << T << ", n_MC=" << n_MC << "\n\n";
    std::cout << "   N        Time (s)\n";
    std::cout << "--------------------\n";
    
    for (int N_MC : N_list) {
        AsianOptionPricer pricer(S0, sigma, r, K, T);
        
        auto start = std::chrono::high_resolution_clock::now();
        {
            // We do both call + put once each for a fair total time
            // Control variate or crude—your choice:
            auto c = pricer.monteCarloACControl(n_MC, N_MC);
            //auto p = pricer.monteCarloAPControl(n_MC, N_MC);
        }
        auto end = std::chrono::high_resolution_clock::now();
        double timeSec = std::chrono::duration<double>(end - start).count();
        
        std::cout << std::setw(8) << N_MC << "    " << timeSec << "\n";
    }

    // 2) Finite Difference: measure times as we vary (m, n).
    //    Example combos: (30,126), (100,126), (500,126), (2000,126)
    //    We'll fix n=126 but vary m in {30, 100, 500, 2000}.
    //    Or do the reverse if you prefer.
    std::vector<int> m_list = {30, 100, 500, 2000};
    std::cout << "\n--- Finite Difference Timing (varying m, fixed n=126) ---\n";
    std::cout << "Parameters: S0=" << S0 << ", sigma=" << sigma << ", r=" << r
              << ", K=" << K << ", T=" << T << "\n\n";
    std::cout << " (m, n)       Time (s)\n";
    std::cout << "------------------------\n";
    
    for (int m_FD : m_list) {
        AsianOptionPricer pricer(S0, sigma, r, K, T);
        
        auto start = std::chrono::high_resolution_clock::now();
        {
            std::vector<std::vector<double>> sol;
            std::vector<double> space, time_grid;
            // We'll do call + put once each
            pricer.crankNicolsonAC(n_FD_default, m_FD, sol, space, time_grid);
            //pricer.crankNicolsonAP(n_FD_default, m_FD, sol, space, time_grid);
        }
        auto end = std::chrono::high_resolution_clock::now();
        double timeSec = std::chrono::duration<double>(end - start).count();
        
        std::cout << "(" << m_FD << "," << n_FD_default << ")     " << timeSec << "\n";
    }

    std::cout << "\nDone.\n";
    return 0;
}

