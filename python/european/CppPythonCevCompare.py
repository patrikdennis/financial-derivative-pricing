import subprocess
import io
import pandas as pd

def run_cpp_pricer():
    """
    Run the C++ executable and capture its output.
    The executable prints CSV output with headers 'Call' and 'Put'.
    """
    # Run the C++ executable.
    result = subprocess.run(["../../cpp/european/cpp_pricer"],
                            stdout=subprocess.PIPE,
                            text=True)
    return result.stdout

def parse_cpp_output(output):
    """
    Parse the CSV output from the C++ executable.
    In this test we have:
    Expected format:
        Call,Put
        10.1234, 5.6789
    """
    df = pd.read_csv(io.StringIO(output))
    call_price = df.iloc[0]["Call"]
    put_price  = df.iloc[0]["Put"]
    return call_price, put_price

def main():
    # Run the C++ pricer and parse its output
    cpp_output = run_cpp_pricer()
    cpp_call, cpp_put = parse_cpp_output(cpp_output)
    
    # Compute Python prices using your Python pricer
    # For this example, we import EuropeanOptionPricer from european_pricing.py
    from european_pricing import EuropeanOptionPricer
    # Set parameters 
    s0 = 100.0
    sigma = 0.2
    r = 0.05
    K = 100.0
    T = 1.0
    n = 100
    m = 200
    X = 300.0
    
    pricer = EuropeanOptionPricer(s0, sigma, r, K, T, n, m, X)
    py_call, sol_call, space_call, time_call = pricer.crank_nicolson_call(n, m)
    py_put, sol_put, space_put, time_put = pricer.crank_nicolson_put(n, m)
    
    # Compare and print results
    print("\n")
    print("C++ Call Price:    {:.4f}".format(cpp_call))
    print("Python Call Price: {:.4f}".format(py_call))
    print("C++ Put Price:     {:.4f}".format(cpp_put))
    print("Python Put Price:  {:.4f}".format(py_put))
    print("\n")
    
if __name__ == "__main__":
    main()

