from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

from option import Option
from BSpricer import BSModel

REPORTS = Path(__file__).parent.parent / "reports"

if __name__ == "__main__":
    K, T, r, sigma = 100, 1.0, 0.05, 0.2

    options = [Option(S, K, T, r, sigma, 'call') for S in range(60, 140, 1)]
    bs_model = BSModel()
    # Time values for each european call option
    bs_prices = [bs_model.price(opt) for opt in options]
    time_values = [bs_prices[i] - max(S-K, 0) for i, S in enumerate([opt.S for opt in options])]

    plt.figure(figsize=(10, 5))
    plt.plot([opt.S for opt in options], time_values, marker='o')
    plt.title(f'Time Value of European Call Options for K={K}')
    plt.xlabel('Spot Price (S)')
    plt.ylabel('Time Value')
    plt.grid(True)
    plt.savefig(REPORTS / 'time_value_vs_spot.png', dpi=150)
    plt.show()

    # 3D surface: time value as a function of spot price and maturity
    spots      = np.arange(60, 140, 1)
    maturities = np.linspace(0.05, 2.0, 60)
    S_grid, T_grid = np.meshgrid(spots, maturities)

    d1, d2 = bs_model._d1_d2(S_grid, K, T_grid, r, sigma)
    call_price_grid = S_grid * norm.cdf(d1) - K * np.exp(-r * T_grid) * norm.cdf(d2)
    time_value_grid = call_price_grid - np.maximum(S_grid - K, 0)

    fig = plt.figure(figsize=(11, 8))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(S_grid, T_grid, time_value_grid, cmap='viridis',
                            edgecolor='k', linewidth=0.2, alpha=0.9)
    ax.set_xlabel('Spot Price (S)')
    ax.set_ylabel('Maturity (T, years)')
    ax.set_zlabel('Time Value')
    ax.set_title(f'Time Value Surface - European Call (K={K}, r={r:.0%}, σ={sigma:.0%})')
    fig.colorbar(surf, ax=ax, shrink=0.6, label='Time Value')
    ax.view_init(elev=28, azim=-74)
    plt.tight_layout()
    plt.savefig(REPORTS / 'time_value_surface.png', dpi=150)
    plt.show()