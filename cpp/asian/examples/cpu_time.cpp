#include <iostream>
#include <vector>
#include <chrono>
#include "../option_pricing.h"
// #include "utils/plotting.h" // Not needed for table output

int main(){
    double r = 0.05, K = 100, T = 1, S0 = 100;
    std::vector<double> sigma_list = {0.1, 0.2, 0.3, 0.4, 0.5};
    int n_FD = 100, m_FD = 100, n_MC = 50, N_MC = 10000;
    size_t numSig = sigma_list.size();
    std::vector<double> time_FD(numSig, 0.0), time_MC(numSig, 0.0);
    
    for (size_t i = 0; i < numSig; i++){
        AsianOptionPricer pricer(S0, sigma_list[i], r, K, T);
        
        // Measure FD time using Crank-Nicolson methods.
        auto start = std::chrono::high_resolution_clock::now();
        {
            // We assume that dummy variables are passed for sol and space; adjust if needed.
            std::vector<std::vector<double> > sol;
            std::vector<double> space, time_grid;
            pricer.crankNicolsonAC(n_FD, m_FD, sol, space, time_grid);
            pricer.crankNicolsonAP(n_FD, m_FD, sol, space, time_grid);
        }
        auto end = std::chrono::high_resolution_clock::now();
        time_FD[i] = std::chrono::duration<double>(end - start).count();
        
        // Measure MC time using control variate methods.
        start = std::chrono::high_resolution_clock::now();
        {
            double dev;
            pricer.monteCarloAC(n_MC, N_MC, dev);
            pricer.monteCarloAP(n_MC, N_MC, dev);
        }
        end = std::chrono::high_resolution_clock::now();
        time_MC[i] = std::chrono::duration<double>(end - start).count();
    }
    
    // Print a table of the CPU times.
    std::cout << "CPU Time Comparison for S0 = " << S0 << ", r = " << r << ", K = " << K << ", T = " << T << "\n";
    std::cout << "------------------------------------------------------------\n";
    std::cout << "Sigma\tFinite Difference Time (s)\tMonte Carlo Time (s)\n";
    std::cout << "------------------------------------------------------------\n";
    for (size_t i = 0; i < sigma_list.size(); i++){
        std::cout << sigma_list[i] << "\t" << time_FD[i] << "\t\t\t" << time_MC[i] << "\n";
    }
    std::cout << "------------------------------------------------------------\n";
    
    return 0;
}
