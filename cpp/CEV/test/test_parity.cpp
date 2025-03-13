#include <iostream>
#include <fstream>
#include <iomanip>
#include <vector>
#include <cmath>
#include "../CEVPricer.h"

using std::cout;
using std::endl;
using std::vector;
using std::setw;
using std::fixed;
using std::setprecision;
using std::ofstream;

int main() {
    // Define parameter ranges.
    // S0: 30, 35, 40, 45, 50
    vector<double> S0_list = {30, 35, 40, 45, 50};
    // sigma: 0.10, 0.20, 0.30, 0.40, 0.50
    vector<double> sigma_list = {0.10, 0.20, 0.30, 0.40, 0.50};

    double r = 0.02;      // Risk-free rate
    double K = 40.0;      // Strike price
    double T = 1.0;       // Maturity (years)
    int n = 126;          // Time steps for finite-difference
    int m = 10000;        // Space steps for finite-difference
    double X = 10 * K;    // Upper bound for the stock price domain
    double delta = 1.0;   // Elasticity parameter in the CEV model

    // Table header
    cout << fixed << setprecision(4);
    cout << setw(8) << "S0" << setw(8) << "sigma"
         << setw(12) << "Call" << setw(12) << "Put"
         << setw(16) << "(Call-Put)FD" << setw(12) << "Theory"
         << setw(12) << "Error" << endl;
    cout << std::string(80, '-') << endl;

    // Loop over S0 and sigma values.
    for (double S0 : S0_list) {
        for (double sigma : sigma_list) {
            // Create a CEVPricer instance for current parameters.
            CEVPricer pricer(S0, sigma, r, K, T, delta, X);

            double callPrice = 0.0;
            double putPrice = 0.0;
            vector<vector<double>> solCall, solPut;
            vector<double> spaceCall, timeCall;
            vector<double> spacePut, timePut;

            // Compute call and put prices using the Crank–Nicolson FD method.
            pricer.crankNicolsonCall(n, m, callPrice, solCall, spaceCall, timeCall);
            pricer.crankNicolsonPut(n, m, putPrice, solPut, spacePut, timePut);

            double parityFD = callPrice - putPrice;
            double parityTheory = S0 - K * std::exp(-r * T);
            double error = parityFD - parityTheory;

            cout << setw(8) << S0 << setw(8) << sigma
                 << setw(12) << callPrice << setw(12) << putPrice
                 << setw(16) << parityFD << setw(12) << parityTheory
                 << setw(12) << error << endl;
        }
    }

    // Write the table to a text file.
    ofstream outfile("parity_cev.txt");
    if (outfile.is_open()) {
        outfile << "S0 sigma Call Put (Call-Put)FD Theory Error\n";
        for (double S0 : S0_list) {
            for (double sigma : sigma_list) {
                CEVPricer pricer(S0, sigma, r, K, T, delta, X);
                double callPrice = 0.0, putPrice = 0.0;
                vector<vector<double>> solCall, solPut;
                vector<double> spaceCall, timeCall;
                vector<double> spacePut, timePut;
                pricer.crankNicolsonCall(n, m, callPrice, solCall, spaceCall, timeCall);
                pricer.crankNicolsonPut(n, m, putPrice, solPut, spacePut, timePut);
                double parityFD = callPrice - putPrice;
                double parityTheory = S0 - K * std::exp(-r * T);
                double error = parityFD - parityTheory;
                outfile << S0 << " " << sigma << " " << callPrice << " " 
                        << putPrice << " " << parityFD << " " << parityTheory 
                        << " " << error << "\n";
            }
        }
        outfile.close();
    } else {
        cout << "Error: Unable to open file for writing." << endl;
    }

    return 0;
}

