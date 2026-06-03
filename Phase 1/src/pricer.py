from scipy.stats import norm
from numpy import log, exp, sqrt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

class BSParams:
    def __init__(self, S, K, T, r, sigma):
        """
        Class to hold the parameters for the Black-Scholes option pricing model.
        
        Parameters:
        S: Current stock price
        K: Strike price
        T: Time to maturity (in years)
        r: Risk-free interest rate (annualized)
        sigma: Volatility of the underlying stock (annualized)

        """
        self.S = S
        self.K = K
        self.T = T
        self.r = r
        self.sigma = sigma
    

def bs_price(params,option_type):
    """
    Calculate the Black-Scholes price for a European call or put option.
    
    Parameters:
    params: An instance of BSParams containing the parameters for the option
    option_type: A string, either 'call' or 'put', indicating the type of option
    
    Returns:
    The price of the option
    """

    if params.T <= 0:
        raise ValueError("Time to maturity T must be greater than 0.")
    if params.sigma <= 0:
        raise ValueError("Volatility sigma must be greater than 0.")
    
    S = params.S
    K = params.K
    T = params.T
    r = params.r
    sigma = params.sigma

    d1 = (log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqrt(T))
    d2 = d1 - sigma * sqrt(T)

    if option_type == 'call':
        price = S * norm.cdf(d1) - K * exp(-r * T) * norm.cdf(d2)
    elif option_type == 'put':
        price = K * exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    else:
        raise ValueError("option_type must be 'call' or 'put'")

    return price


def check_put_call_parity(params, tol = 1e-8):
    """
    Check the put-call parity for European options.
    
    Parameters:
    params: An instance of BSParams containing the parameters for the options
    
    Returns:
    A tuple containing the call price, put price, and a boolean indicating if parity holds
    """
    call_price = bs_price(params, 'call')
    put_price = bs_price(params, 'put')

    # Calculate the left-hand side and right-hand side of the put-call parity equation
    lhs = call_price - put_price
    rhs = params.S - params.K * exp(-params.r * params.T)

    parity_holds = abs(lhs - rhs) < tol  # Allow for a small numerical tolerance

    if parity_holds:
        print("Put-call parity holds.")
    else: 
        raise ValueError("Put-call parity does not hold. Check the parameters and calculations.")

    return call_price, put_price, parity_holds

if __name__ == "__main__":
    # Test 1 - Basic test with standard parameters
    params = BSParams(S=100, K=100, T=1, r=0.05, sigma=0.2)
    call_price, put_price, parity_holds = check_put_call_parity(params)
    print(f"Call Price: {call_price:.4f}, Put Price: {put_price:.4f}, Parity Holds: {parity_holds}")

    Maturity_price = np.linspace(90, 120, 100)
    European_Call_price = bs_price(BSParams(S=100, K=100, T=1, r=0.05, sigma=0.2), 'call')
    return_call = [max(S - 100, 0) - European_Call_price for S in Maturity_price]

    plt.figure(figsize=(10, 6))
    plt.plot(Maturity_price, return_call, label='European Call Option Payoff at Maturity')
    plt.axhline(0, color='gray', linestyle='--',label='Break-even Point')  # Line at zero payoff for reference
    plt.axvline(100, color='blue', linestyle='--',label='Strike Price')  # Line at the strike price for reference

    stats_text = (
            f"Strike:      {params.K:.2f}\n"
            f"Initial Price:       {params.S:.2f}\n"
            f"Maturity Time:       {params.T:.2f}\n"
            f"Volatility:       {params.sigma:.2f}\n"
            f"Interest Rate:      {params.r:.2f}\n"
        )
    plt.gca().text(0.98, 0.97, stats_text,transform=plt.gca().transAxes,
        fontsize=9, verticalalignment='top', horizontalalignment='right',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray')
    )
    
    plt.xlabel('Stock Price at Maturity')
    plt.ylabel('Payoff')
    plt.title('European Call Option Payoff at Maturity')
    plt.legend()
    plt.grid(True)
    plt.savefig('reports/pricer_european_call_payoff.png')  # Save the plot as an image file
    plt.show()

    # Test 2 - Price plot for a range of strike prices
    x = np.linspace(60, 140, 20)
    call_prices = [bs_price(BSParams(S=k, K=100, T=1, r=0.05, sigma=0.2), 'call') for k in x]
    put_prices = [bs_price(BSParams(S=k, K=100, T=1, r=0.05, sigma=0.2), 'put') for k in x]

    plt.figure(figsize=(10, 6))
    plt.plot(x, call_prices, label='European Call Option Price', marker='o')
    plt.plot(x, put_prices, label='European Put Option Price', marker='o')
    plt.axvline(x=100, linestyle='--', color='gray',label='Strike Price')  # Plot the Strike price for reference
    plt.xlabel('Initial Price')
    plt.ylabel('Option Price')
    plt.title('European Option Prices vs Initial Price')
    plt.legend()
    plt.savefig('reports/pricer_option_prices_vs_initial_price.png')  # Save the plot as an image file
    plt.grid(True)
    plt.show()

    # Test 3 - Convergence test when the initial stock prices is big and small
    S_Big = 100000
    S_Small = 0.00001

    call_price_big = bs_price(BSParams(S=S_Big, K=100, T=1, r=0.05, sigma=0.2), 'call')
    print(f"Call Price with S={S_Big}: {call_price_big:.4f}")
    put_price_small = bs_price(BSParams(S=S_Small, K=100, T=1, r=0.05, sigma=0.2), 'put')
    print(f"Put Price with S={S_Small}: {put_price_small:.4f}, K*exp(-r*T): {100*exp(-0.05*1):.4f}")

    # Test 4 - Price plot for a range of volatilities
    volatilities = np.linspace(0.05, 0.8, 30)
    Strike_price = 100
    Time_to_maturity = 1
    Interest_rate = 0.05

    for Initial_price in [80, 100, 120]:
        call_prices_vol = [bs_price(BSParams(S=Initial_price, K=Strike_price, T=Time_to_maturity, r=Interest_rate, sigma=sigma), 'call') for sigma in volatilities]
        put_prices_vol  = [bs_price(BSParams(S=Initial_price, K=Strike_price, T=Time_to_maturity, r=Interest_rate, sigma=sigma), 'put')  for sigma in volatilities]

        bs_limit_sigma0_call = max(Initial_price - Strike_price * exp(-Interest_rate * Time_to_maturity), 0)
        bs_limit_sigma0_put  = max(Strike_price * exp(-Interest_rate * Time_to_maturity) - Initial_price, 0)

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(volatilities, call_prices_vol, label='European Call Option Price', marker='o')
        ax.plot(volatilities, put_prices_vol,  label='European Put Option Price',  marker='o')
        ax.scatter([0], [bs_limit_sigma0_call], color='blue', zorder=5, s=30, label=r'$\lim_{\sigma\to 0} C = \max(S - Ke^{-rT}, 0)$')
        ax.scatter([0], [bs_limit_sigma0_put],  color='red',  zorder=5, s=30, label=r'$\lim_{\sigma\to 0} P = \max(Ke^{-rT} - S, 0)$')

        stats_text = (
            f"Strike:        {Strike_price:.2f}\n"
            f"Initial Price: {Initial_price:.2f}\n"
            f"Maturity Time: {Time_to_maturity:.2f}\n"
            f"Interest Rate: {Interest_rate:.2f}\n"
            f"Lim C (sigma=0): {bs_limit_sigma0_call:.2f}\n"
            f"Lim P (sigma=0): {bs_limit_sigma0_put:.2f}"
        )
        ax.text(0.98, 0.25, stats_text, transform=ax.transAxes,
            fontsize=9, verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray'))

        ax.set_xlabel('Volatility')
        ax.set_ylabel('Option Price')
        ax.set_title('European Option Prices vs Volatility')
        ax.legend()
        ax.grid(True)
        plt.savefig(f'reports/pricer_option_prices_vs_volatility_S_{Initial_price}_K_{Strike_price}.png')
        plt.show()

    # Test 5 - Price plot for a range of time to maturity
    times_to_maturity = np.linspace(0.01, 3, 30)
    Strike_price = 100
    Interest_rate = 0.05

    for Initial_price in [80, 100, 120]:
        call_prices_time = [bs_price(BSParams(S=Initial_price, K=Strike_price, T=T, r=Interest_rate, sigma=0.2), 'call') for T in times_to_maturity]
        put_prices_time  = [bs_price(BSParams(S=Initial_price, K=Strike_price, T=T, r=Interest_rate, sigma=0.2), 'put')  for T in times_to_maturity]

        bs_limit_T0_call = max(Initial_price - Strike_price, 0)
        bs_limit_T0_put  = max(Strike_price - Initial_price, 0)

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(times_to_maturity, call_prices_time, label='European Call Option Price', marker='o')
        ax.plot(times_to_maturity, put_prices_time,  label='European Put Option Price',  marker='o')
        ax.scatter([0], [bs_limit_T0_call], color='blue', zorder=5, s=30, label=r'$\lim_{T\to 0} C = \max(S - K, 0)$')
        ax.scatter([0], [bs_limit_T0_put],  color='red',  zorder=5, s=30, label=r'$\lim_{T\to 0} P = \max(K - S, 0)$')

        stats_text = (
            f"Strike:        {Strike_price:.2f}\n"
            f"Initial Price: {Initial_price:.2f}\n"
            f"Volatility:    {0.2:.2f}\n"
            f"Interest Rate: {Interest_rate:.2f}\n"
            f"Lim C (T=0): {bs_limit_T0_call:.2f}\n"
            f"Lim P (T=0): {bs_limit_T0_put:.2f}"
        )
        ax.text(0.98, 0.25, stats_text, transform=ax.transAxes,
            fontsize=9, verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray'))

        ax.set_xlabel('Time to Maturity (Years)')
        ax.set_ylabel('Option Price')
        ax.set_title('European Option Prices vs Time to Maturity')
        ax.legend()
        ax.grid(True)
        plt.savefig(f'reports/pricer_option_prices_vs_time_to_maturity_S_{Initial_price}_K_{Strike_price}.png')
        plt.show()

    # Test 6 - Value table
    print("\nValue Table for European Call Option with initial price S=100, sigma = 0.2:")
    
    strikes = [80, 90, 100, 110, 120]
    maturities = {'1M': 1/12, '3M': 3/12, '6M': 6/12, '1Y': 1.0}

    rows = []
    for K in strikes:
        row = {'Strike': K}
        for label, T in maturities.items():
            p = BSParams(S=100, K=K, T=T, r=0.05, sigma=0.20)
            row[f'Call {label}'] = round(bs_price(p, 'call'), 2)
            row[f'Put {label}']  = round(bs_price(p, 'put'),  2)
        rows.append(row)

    df = pd.DataFrame(rows).set_index('Strike')
    print(df)