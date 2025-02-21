#ifndef VERIFICATION_H
#define VERIFICATION_H

#include <vector>
#include <string>

// Structure to hold one row of the parity table.
struct ParityRecord {
    double s0;
    double sigma;
    double callPutMC;   // (Call - Put) from Monte Carlo
    double callPutFD;   // (Call - Put) from Finite Difference
    double theory;      // Theoretical parity value
    double errorMC;     // callPutMC - theory
    double errorFD;     // callPutFD - theory
};

/**
 * @brief Computes the arithmetic Asian put–call parity table.
 * 
 * The theoretical parity at t = 0 for arithmetic-average Asian options is:
 *    Call - Put = exp(-rT)*(((exp(rT)-1)/(rT))*S0 - K)
 * 
 * @param S0_list      Vector of initial stock prices.
 * @param sigma_list   Vector of volatilities.
 * @param call_MC      2D vector (numS0 x numSigma) of Monte Carlo call prices.
 * @param put_MC       2D vector (numS0 x numSigma) of Monte Carlo put prices.
 * @param call_FD      2D vector (numS0 x numSigma) of Finite Difference call prices.
 * @param put_FD       2D vector (numS0 x numSigma) of Finite Difference put prices.
 * @param r            Risk-free interest rate.
 * @param T            Time to maturity.
 * @param K            Strike price.
 * @return std::vector<ParityRecord>  The computed parity table.
 */
std::vector<ParityRecord> verifyArithmeticParity(
    const std::vector<double>& S0_list,
    const std::vector<double>& sigma_list,
    const std::vector<std::vector<double>>& call_MC,
    const std::vector<std::vector<double>>& put_MC,
    const std::vector<std::vector<double>>& call_FD,
    const std::vector<std::vector<double>>& put_FD,
    double r, double T, double K
);

/**
 * @brief Converts the parity table into a formatted string.
 * 
 * Uses i/o manipulators to align columns and set fixed decimal precision.
 * 
 * @param table  Vector of ParityRecord.
 * @return std::string  Formatted table as a string.
 */
std::string parityTableToString(const std::vector<ParityRecord>& table);

#endif
