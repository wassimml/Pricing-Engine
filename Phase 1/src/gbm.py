import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import lognorm
from scipy import stats

def simulate_gbm(S0, mu, sigma, T, n_steps, n_paths, seed): 
    """
    Simulate Geometric Brownian Motion paths.
    
    This function returns a 2D array where each row represents 
    a simulation path and each column represents a time step.

    Parameters:
    S0 (float): Initial stock price
    mu (float): Drift coefficient
    sigma (float): Volatility coefficient
    T (float): Time to maturity
    n_steps (int): Number of time steps
    n_paths (int): Number of simulation paths
    seed (int): Random seed for reproducibility
    """
    rng = np.random.default_rng(seed)
    
    dt = T / n_steps
    S = np.zeros((n_paths,n_steps + 1))
    S[:, 0] = S0

    for t in range(1, n_steps + 1):
        # Z is a vector of standard normal random variables 
        # for each path at time step t
        Z = rng.standard_normal(n_paths)
        S[:, t] = S[:, t - 1] * np.exp((mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z)

    return S

if __name__ == "__main__":
    # Example usage
    S0 = 100  # Initial stock price
    mu = 0.08  # Drift coefficient
    sigma = 0.2  # Volatility coefficient
    T = 1  # Time to maturity (1 year)
    n_steps = 252  # Number of time steps (trading days in a year)
    n_paths = 10000  # Number of simulation paths
    seed = 56  # Random seed for reproducibility

    simulated_paths = simulate_gbm(S0, mu, sigma, T, n_steps, n_paths, seed)
    print(simulated_paths)

    ###### Plotting the first 10 simulated paths ######

    plt.figure(figsize=(10, 6))
    for i in range(10):
        plt.plot(simulated_paths[i], lw=0.5)

    # Line at the initial stock price for reference
    plt.axhline(S0, color='red', linestyle='--', lw=1, label='Initial Stock Price')
    plt.legend()
    plt.title('Simulated Geometric Brownian Motion Paths')
    plt.xlabel('Time Steps')
    plt.ylabel('Stock Price')
    plt.savefig('reports/gbm_simulation.png')  # Save the plot as an image file
    plt.show()

    ###### Plotting 50 paths with a thicker line, Red color for the paths finishing below the initial stock price, and Green color for the paths finishing above the initial stock price ######
    plt.figure(figsize=(10, 6))
    for i in range(50):
        if simulated_paths[i, -1] < S0:
            plt.plot(simulated_paths[i], lw=1.5, color='lightcoral')
        else:
            plt.plot(simulated_paths[i], lw=1.5, color='lightgreen')
    plt.axhline(S0, color='blue', linestyle='--', lw=1, label='Initial Stock Price')
    plt.legend()
    plt.title('Simulated Geometric Brownian Motion Paths (50 paths)')
    plt.xlabel('Time Steps')
    plt.ylabel('Stock Price')
    plt.savefig('reports/gbm_simulation_50_paths.png')  # Save the plot as an image file
    plt.show()

    ###### Plotting the Histogram of the final stock prices at maturity ######

    # Plotting also the theoretical log-normal distribution fit for the final stock prices
    plt.figure(figsize=(10, 6))
    final_prices = simulated_paths[:, -1]
    plt.hist(final_prices, bins=50, density=True, alpha=0.6, color='g', label='Final Stock Prices')
    
    # Theoretical parameters for the log-normal distribution
    theoretical_mean = np.log(S0) + (mu - 0.5 * sigma**2) * T

    # Var of log-normal distribution
    theoretical_var = sigma * np.sqrt(T)
    
    x = np.linspace(min(final_prices), max(final_prices), 100)
    pdf = lognorm.pdf(x, s=theoretical_var, scale=np.exp(theoretical_mean))

    textstr = '\n'.join((
        r'$\mu=%.2f$' % (theoretical_mean, ),
        r'$\sigma=%.2f$' % (theoretical_var, )))
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    plt.text(0.05, 0.95, textstr, transform=plt.gca().transAxes, fontsize=10, verticalalignment='top', bbox=props)
    plt.plot(x, pdf, 'r-', label='Theoretical Log-normal')
    plt.title('Histogram of Final Stock Prices at Maturity')
    plt.xlabel('Stock Price')
    plt.ylabel('Density')
    plt.legend()
    plt.savefig('reports/gbm_final_prices_histogram.png')  # Save the plot
    plt.show()

    ###### Plotting the histogram of the log returns of the final stock prices at maturity ln(S_T/S_0) ######

    log_returns = np.log(final_prices / S0)
    plt.figure(figsize=(10, 6))
    plt.hist(log_returns, bins=50, density=True, alpha=0.6, color='b', label='Log Returns')

    # Theoretical parameters for the normal distribution of log returns
    mu_LN = (mu - 0.5 * sigma**2) * T
    sigma_LN = sigma * np.sqrt(T)
    x = np.linspace(min(log_returns), max(log_returns), 100)
    pdf = (1 / (sigma_LN * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu_LN) / sigma_LN) ** 2)

    # Textbox with the parameters of the normal distribution
    textstr = '\n'.join((
        r'$\mu=%.2f$' % (mu_LN, ),
        r'$\sigma=%.2f$' % (sigma_LN, )))
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    plt.text(0.05, 0.95, textstr, transform=plt.gca().transAxes, fontsize=10, verticalalignment='top', bbox=props)
    plt.plot(x, pdf, 'r-', label='Theoretical Normal')
    plt.title('Histogram of Log Returns at Maturity')
    plt.xlabel('Log Return')
    plt.ylabel('Density')
    plt.legend()
    plt.savefig('reports/gbm_log_returns_histogram.png')  # Save the plot
    plt.show()

    ###### QQ plot of the log returns against the theoretical normal distribution ######
    plt.figure(figsize=(10, 6)) 
    stats.probplot(log_returns, dist="norm", plot=plt)
    plt.title('QQ Plot of Log Returns vs Theoretical Normal Distribution')

    plt.savefig('reports/gbm_log_returns_qq_plot.png')  # Save the plot
    plt.show()