from dataclasses import dataclass
from pathlib import Path

from option import Option
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import lognorm
from scipy import stats

REPORTS = Path(__file__).parent.parent / "reports"


# ---------------------------------------------------------------------------
# GBM simulator
# ---------------------------------------------------------------------------

@dataclass
class MCEngine:
    """
    Geometric Brownian Motion simulator.

    simulate() returns an array of shape (n_paths, n_steps + 1).
    Column 0 is the initial price S0; subsequent columns are the simulated path.
    Fully vectorised — no Python loop over time steps.
    """

    S0: float
    mu: float
    sigma: float
    T: float
    n_steps: int
    n_paths: int
    seed: int

    def __init__(self, option: Option, n_steps: int = 252, n_paths: int = 100_000, seed: int = 42):
        self.S0 = option.S
        self.mu = option.r
        self.sigma = option.sigma
        self.T = option.T
        self.n_steps = n_steps
        self.n_paths = n_paths
        self.seed = seed
    
    def simulate(self) -> np.ndarray:
        rng = np.random.default_rng(self.seed)
        dt = self.T / self.n_steps
        Z = rng.standard_normal((self.n_paths, self.n_steps))
        log_increments = (self.mu - 0.5 * self.sigma ** 2) * dt + self.sigma * np.sqrt(dt) * Z
        S = self.S0 * np.exp(np.cumsum(log_increments, axis=1))
        return np.hstack([np.full((self.n_paths, 1), self.S0), S])


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    option = Option(S=100, K=100, T=1, r=0.08, sigma=0.2, kind='call')
    sim = MCEngine(option, n_steps=252, n_paths=10_000, seed=56)
    simulated_paths = sim.simulate()
    print(simulated_paths)

    # --- First 10 paths ---
    plt.figure(figsize=(10, 6))
    for i in range(10):
        plt.plot(simulated_paths[i], lw=0.5)
    plt.axhline(sim.S0, color='red', linestyle='--', lw=1, label='Initial Stock Price')
    plt.legend()
    plt.title('Simulated Geometric Brownian Motion Paths')
    plt.xlabel('Time Steps')
    plt.ylabel('Stock Price')
    plt.savefig(REPORTS / 'gbm_simulation.png')
    plt.show()

    # --- 50 paths, coloured by terminal value ---
    plt.figure(figsize=(10, 6))
    for i in range(50):
        color = 'lightcoral' if simulated_paths[i, -1] < sim.S0 else 'lightgreen'
        plt.plot(simulated_paths[i], lw=1.5, color=color)
    plt.axhline(sim.S0, color='blue', linestyle='--', lw=1, label='Initial Stock Price')
    plt.legend()
    plt.title('Simulated Geometric Brownian Motion Paths (50 paths)')
    plt.xlabel('Time Steps')
    plt.ylabel('Stock Price')
    plt.savefig(REPORTS / 'gbm_simulation_50_paths.png')
    plt.show()

    # --- Histogram of terminal prices vs theoretical log-normal ---
    final_prices = simulated_paths[:, -1]
    theoretical_mean = np.log(sim.S0) + (sim.mu - 0.5 * sim.sigma ** 2) * sim.T
    theoretical_std  = sim.sigma * np.sqrt(sim.T)

    plt.figure(figsize=(10, 6))
    plt.hist(final_prices, bins=50, density=True, alpha=0.6, color='g', label='Final Stock Prices')
    x = np.linspace(final_prices.min(), final_prices.max(), 200)
    plt.plot(x, lognorm.pdf(x, s=theoretical_std, scale=np.exp(theoretical_mean)), 'r-', label='Theoretical Log-normal')
    textstr = '\n'.join((r'$\mu=%.2f$' % theoretical_mean, r'$\sigma=%.2f$' % theoretical_std))
    plt.gca().text(0.05, 0.95, textstr, transform=plt.gca().transAxes,
        fontsize=10, verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    plt.title('Histogram of Final Stock Prices at Maturity')
    plt.xlabel('Stock Price')
    plt.ylabel('Density')
    plt.legend()
    plt.savefig(REPORTS / 'gbm_final_prices_histogram.png')
    plt.show()

    # --- Log-returns histogram vs theoretical normal ---
    log_returns = np.log(final_prices / sim.S0)
    mu_LN    = (sim.mu - 0.5 * sim.sigma ** 2) * sim.T
    sigma_LN = sim.sigma * np.sqrt(sim.T)

    plt.figure(figsize=(10, 6))
    plt.hist(log_returns, bins=50, density=True, alpha=0.6, color='b', label='Log Returns')
    x = np.linspace(log_returns.min(), log_returns.max(), 200)
    pdf = (1 / (sigma_LN * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu_LN) / sigma_LN) ** 2)
    plt.plot(x, pdf, 'r-', label='Theoretical Normal')
    textstr = '\n'.join((r'$\mu=%.2f$' % mu_LN, r'$\sigma=%.2f$' % sigma_LN))
    plt.gca().text(0.05, 0.95, textstr, transform=plt.gca().transAxes,
        fontsize=10, verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    plt.title('Histogram of Log Returns at Maturity')
    plt.xlabel('Log Return')
    plt.ylabel('Density')
    plt.legend()
    plt.savefig(REPORTS / 'gbm_log_returns_histogram.png')
    plt.show()

    # --- QQ plot ---
    plt.figure(figsize=(10, 6))
    stats.probplot(log_returns, dist="norm", plot=plt)
    plt.title('QQ Plot of Log Returns vs Theoretical Normal Distribution')
    plt.savefig(REPORTS / 'gbm_log_returns_qq_plot.png')
    plt.show()
