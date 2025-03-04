#include "verification.h"
#include <cmath>
#include <sstream>
#include <iomanip>
#include <numeric>

// Function to compute the parity table.
std::vector<ParityRecord> verifyArithmeticParity(
    const std::vector<double>& S0_list,
    const std::vector<double>& sigma_list,
    const std::vector<std::vector<double>>& call_MC,
    const std::vector<std::vector<double>>& put_MC,
    const std::vector<std::vector<double>>& call_FD,
    const std::vector<std::vector<double>>& put_FD,
    double r, double T, double K
) {
    std::vector<ParityRecord> table;
    // Loop over each S0 value.
    for (size_t i = 0; i < S0_list.size(); i++) {
        double s0 = S0_list[i];
        // Compute the theoretical parity:
        //    exp(-rT)*(((exp(rT)-1)/(rT))*s0 - K)
        double parity_theory = std::exp(-r * T) * ( ((std::exp(r * T) - 1) / (r * T)) * s0 - K );
        // Loop over each sigma value.
        for (size_t j = 0; j < sigma_list.size(); j++) {
            ParityRecord rec;
            rec.s0 = s0;
            rec.sigma = sigma_list[j];
            double lhs_MC = call_MC[i][j] - put_MC[i][j];
            double lhs_FD = call_FD[i][j] - put_FD[i][j];
            rec.callPutMC = lhs_MC;
            rec.callPutFD = lhs_FD;
            rec.theory    = parity_theory;
            rec.errorMC   = lhs_MC - parity_theory;
            rec.errorFD   = lhs_FD - parity_theory;
            table.push_back(rec);
        }
    }
    return table;
}

std::string parityTableToString(const std::vector<ParityRecord>& table) {
    std::ostringstream ss;

    // We won't set a global format at the start,
    // because we want to set it individually for each column.

    // Column headers (you can keep them at a fixed width).
    ss << "Arithmetic Put-Call Parity Comparison for Asian Options\n";
    ss << std::setw(12) << "S0"
       << std::setw(12) << "sigma"
       << std::setw(15) << "(Call-Put)MC"
       << std::setw(15) << "(Call-Put)FD"
       << std::setw(15) << "Theory"
       << std::setw(15) << "ErrorMC"
       << std::setw(15) << "ErrorFD"
       << "\n";

    // Separator line
    ss << std::string(12 + 12 + 15 + 15 + 15 + 15 + 15, '-') << "\n";

    for (const auto& rec : table) {
        // 1) Print S0 in fixed, say 2 decimal places
        ss << std::fixed << std::setprecision(2)
           << std::setw(12) << rec.s0;

        // 2) Print sigma in fixed, say 2 decimal places
        ss << std::fixed << std::setprecision(2)
           << std::setw(12) << rec.sigma;

        // 3) Print the rest in scientific, e.g. 6 decimal places
        ss << std::scientific << std::setprecision(6)
           << std::setw(15) << rec.callPutMC
           << std::setw(15) << rec.callPutFD
           << std::setw(15) << rec.theory
           << std::setw(15) << rec.errorMC
           << std::setw(15) << rec.errorFD
           << "\n";
    }
    return ss.str();
}

