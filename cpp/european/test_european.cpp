#include <iostream>
#include <vector>
#include "EuropeanPricer.h"

using std::cout;
using std::endl;
using std::vector;

int main() {
    // Define option parameters.
    double s0 = 100.0;   // initial stock price
    double sigma = 0.2;  // volatility
    double r = 0.05;     // risk-free interest rate
    double K = 100.0;    // strike price
    double T = 1.0;      // time to maturity (years)
    int n = 100;         // number of time steps
    int m = 200;         // number of space steps
    double X = 300.0;    // upper bound for the stock price

    // Create an instance of EuropeanPricer.
    EuropeanPricer pricer(s0, sigma, r, K, T, n, m, X);

    double callPrice = 0.0, putPrice = 0.0;
    vector<vector<double>> solCall, solPut;
    vector<double> space, time;

    // Compute European call and put prices using the Crank-Nicolson method.
    pricer.crankNicolsonCall(n, m, callPrice, solCall, space, time);
    pricer.crankNicolsonPut(n, m, putPrice, solPut, space, time);

    // Output the computed prices.
    cout << "European Call Price: " << callPrice << endl;
    cout << "European Put Price:  " << putPrice << endl;

    return 0;
}

