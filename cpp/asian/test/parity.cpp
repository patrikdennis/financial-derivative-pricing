#include <iostream>
#include <vector>
#include "../AsianOptionPricer.h"
#include "../utils/verification.h"

int main() {
    // Option and market parameters
    double r = 0.02;
    double K = 40;
    double T = 0.5;

    // S0_list equivalent to [element + 1 for element in range(29, 35)] => [30, 31, 32, 33, 34, 35]
    std::vector<double> S0_list = {30, 31, 32, 33, 34, 35};

    // sigma_list as shown
    std::vector<double> sigma_list = {0.1, 0.2, 0.3, 0.4, 0.5};

    // Monte Carlo and Finite Difference parameters
    int n_MC = 50, N_MC = 10000;
    int n_FD = 126, m_FD = 10000;

    size_t numS0 = S0_list.size();
    size_t numSig = sigma_list.size();

    // Preallocate 2D arrays (vectors of vectors) for pricing results:
    std::vector<std::vector<double>> call_MC(numS0, std::vector<double>(numSig, 0.0));
    std::vector<std::vector<double>> put_MC(numS0, std::vector<double>(numSig, 0.0));
    std::vector<std::vector<double>> call_FD(numS0, std::vector<double>(numSig, 0.0));
    std::vector<std::vector<double>> put_FD(numS0, std::vector<double>(numSig, 0.0));

    // Loop over each combination of S0 and sigma, and compute prices.
    for (size_t i = 0; i < numS0; i++) {
        for (size_t j = 0; j < numSig; j++) {
            AsianOptionPricer pricer(S0_list[i], sigma_list[j], r, K, T);
            double dev;  // variable to capture deviation (not used in table)
            call_MC[i][j] = pricer.monteCarloAC(n_MC, N_MC, dev);
            put_MC[i][j]  = pricer.monteCarloAP(n_MC, N_MC, dev);
            
            std::vector<std::vector<double>> sol;
            std::vector<double> space, time_grid;
            call_FD[i][j] = pricer.crankNicolsonAC(n_FD, m_FD, sol, space, time_grid);
            put_FD[i][j]  = pricer.crankNicolsonAP(n_FD, m_FD, sol, space, time_grid);
        }
    }

    // Verify the arithmetic put–call parity.
    // The theoretical parity at t=0 is:
    //   Call - Put = exp(-rT) * ( ((exp(rT)-1)/(rT)) * S0 - K )
    std::vector<ParityRecord> parityTable = verifyArithmeticParity(
        S0_list, sigma_list, call_MC, put_MC, call_FD, put_FD, r, T, K
    );

    // Convert the parity table to a formatted string.
    std::string tableStr = parityTableToString(parityTable);

    // Print the table to the console.
    std::cout << tableStr << std::endl;

    return 0;
}
