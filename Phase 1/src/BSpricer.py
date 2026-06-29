from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import norm

from option import Option

REPORTS = Path(__file__).parent.parent / "reports"


# ---------------------------------------------------------------------------
# Black-Scholes-Merton pricer
# ---------------------------------------------------------------------------

class BSModel:
    """Closed-form BSM pricer for European vanilla options."""

    @staticmethod
    def _d1_d2(S, K, T, r, sigma):
        sqrtT = np.sqrt(T)
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqrtT)
        return d1, d1 - sigma * sqrtT

    def price(self, opt: Option) -> float:
        d1, d2 = self._d1_d2(opt.S, opt.K, opt.T, opt.r, opt.sigma)
        disc = np.exp(-opt.r * opt.T)
        if opt.kind == 'call':
            return float(opt.S * norm.cdf(d1) - opt.K * disc * norm.cdf(d2))
        return float(opt.K * disc * norm.cdf(-d2) - opt.S * norm.cdf(-d1))

    def price_grid(self, spots: np.ndarray, opt: Option) -> np.ndarray:
        """Vectorised BSM price over a spot array (all other params fixed)."""
        d1, d2 = self._d1_d2(spots, opt.K, opt.T, opt.r, opt.sigma)
        disc = np.exp(-opt.r * opt.T)
        if opt.kind == 'call':
            return spots * norm.cdf(d1) - opt.K * disc * norm.cdf(d2)
        return opt.K * disc * norm.cdf(-d2) - spots * norm.cdf(-d1)

    def check_put_call_parity(self, opt: Option, tol: float = 1e-8) -> tuple:
        call = self.price(Option(opt.S, opt.K, opt.T, opt.r, opt.sigma, 'call'))
        put  = self.price(Option(opt.S, opt.K, opt.T, opt.r, opt.sigma, 'put'))
        lhs, rhs = call - put, opt.S - opt.K * np.exp(-opt.r * opt.T)
        if abs(lhs - rhs) >= tol:
            raise ValueError("Put-call parity does not hold.")
        return call, put, True


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    model = BSModel()

    # Test 1 — put-call parity
    opt = Option(S=100, K=100, T=1, r=0.05, sigma=0.2, kind='call')
    call_price, put_price, _ = model.check_put_call_parity(opt)
    print("Put-call parity holds.")
    print(f"Call Price: {call_price:.4f}, Put Price: {put_price:.4f}")

    # Payoff diagram
    Maturity_price = np.linspace(90, 120, 300)
    return_call = np.maximum(Maturity_price - opt.K, 0) - call_price

    plt.figure(figsize=(10, 6))
    plt.plot(Maturity_price, return_call, label='European Call Option Payoff at Maturity')
    plt.axhline(0, color='gray', linestyle='--', label='Break-even Point')
    plt.axvline(opt.K, color='blue', linestyle='--', label='Strike Price')
    stats_text = (
        f"Strike:      {opt.K:.2f}\n"
        f"Initial Price:       {opt.S:.2f}\n"
        f"Maturity Time:       {opt.T:.2f}\n"
        f"Volatility:       {opt.sigma:.2f}\n"
        f"Interest Rate:      {opt.r:.2f}\n"
    )
    plt.gca().text(0.98, 0.97, stats_text, transform=plt.gca().transAxes,
        fontsize=9, verticalalignment='top', horizontalalignment='right',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray'))
    plt.xlabel('Stock Price at Maturity')
    plt.ylabel('Payoff')
    plt.title('European Call Option Payoff at Maturity')
    plt.legend()
    plt.grid(True)
    plt.savefig(REPORTS / 'pricer_european_call_payoff.png')
    plt.show()

    # Test 2 — price vs spot (vectorised over spot grid)
    S_arr = np.linspace(60, 140, 300)
    call_prices = model.price_grid(S_arr, Option(S=100, K=100, T=1, r=0.05, sigma=0.2, kind='call'))
    put_prices  = model.price_grid(S_arr, Option(S=100, K=100, T=1, r=0.05, sigma=0.2, kind='put'))

    plt.figure(figsize=(10, 6))
    plt.plot(S_arr, call_prices, label='European Call Option Price')
    plt.plot(S_arr, put_prices,  label='European Put Option Price')
    plt.axvline(x=100, linestyle='--', color='gray', label='Strike Price')
    plt.xlabel('Initial Price')
    plt.ylabel('Option Price')
    plt.title('European Option Prices vs Initial Price')
    plt.legend()
    plt.grid(True)
    plt.savefig(REPORTS / 'pricer_option_prices_vs_initial_price.png')
    plt.show()

    # Test 3 — convergence at extreme spots
    print(f"\nCall Price with S=100000: {model.price(Option(S=100_000, K=100, T=1, r=0.05, sigma=0.2, kind='call')):.4f}")
    print(f"Put Price with S=0.00001: {model.price(Option(S=0.00001, K=100, T=1, r=0.05, sigma=0.2, kind='put')):.4f},"
          f" K*exp(-r*T): {100 * np.exp(-0.05 * 1):.4f}")

    # Test 4 — price vs volatility
    volatilities = np.linspace(0.05, 0.8, 30)
    Strike_price = 100
    Time_to_maturity = 1
    Interest_rate = 0.05

    for Initial_price in [80, 100, 120]:
        call_prices_vol = [model.price(Option(Initial_price, Strike_price, Time_to_maturity, Interest_rate, s, 'call')) for s in volatilities]
        put_prices_vol  = [model.price(Option(Initial_price, Strike_price, Time_to_maturity, Interest_rate, s, 'put'))  for s in volatilities]
        lim_c = max(Initial_price - Strike_price * np.exp(-Interest_rate * Time_to_maturity), 0)
        lim_p = max(Strike_price * np.exp(-Interest_rate * Time_to_maturity) - Initial_price, 0)

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(volatilities, call_prices_vol, label='European Call Option Price', marker='o')
        ax.plot(volatilities, put_prices_vol,  label='European Put Option Price',  marker='o')
        ax.scatter([0], [lim_c], color='blue', zorder=5, s=30, label=r'$\lim_{\sigma\to 0} C = \max(S - Ke^{-rT}, 0)$')
        ax.scatter([0], [lim_p], color='red',  zorder=5, s=30, label=r'$\lim_{\sigma\to 0} P = \max(Ke^{-rT} - S, 0)$')
        stats_text = (
            f"Strike:        {Strike_price:.2f}\n"
            f"Initial Price: {Initial_price:.2f}\n"
            f"Maturity Time: {Time_to_maturity:.2f}\n"
            f"Interest Rate: {Interest_rate:.2f}\n"
            f"Lim C (sigma=0): {lim_c:.2f}\n"
            f"Lim P (sigma=0): {lim_p:.2f}"
        )
        ax.text(0.98, 0.25, stats_text, transform=ax.transAxes,
            fontsize=9, verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray'))
        ax.set_xlabel('Volatility')
        ax.set_ylabel('Option Price')
        ax.set_title('European Option Prices vs Volatility')
        ax.legend()
        ax.grid(True)
        plt.savefig(REPORTS / f'pricer_option_prices_vs_volatility_S_{Initial_price}_K_{Strike_price}.png')
        plt.show()

    # Test 5 — price vs time to maturity
    times_to_maturity = np.linspace(0.01, 3, 30)
    Strike_price = 100
    Interest_rate = 0.05

    for Initial_price in [80, 100, 120]:
        call_prices_time = [model.price(Option(Initial_price, Strike_price, T, Interest_rate, 0.2, 'call')) for T in times_to_maturity]
        put_prices_time  = [model.price(Option(Initial_price, Strike_price, T, Interest_rate, 0.2, 'put'))  for T in times_to_maturity]
        lim_c = max(Initial_price - Strike_price, 0)
        lim_p = max(Strike_price - Initial_price, 0)

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(times_to_maturity, call_prices_time, label='European Call Option Price', marker='o')
        ax.plot(times_to_maturity, put_prices_time,  label='European Put Option Price',  marker='o')
        ax.scatter([0], [lim_c], color='blue', zorder=5, s=30, label=r'$\lim_{T\to 0} C = \max(S - K, 0)$')
        ax.scatter([0], [lim_p], color='red',  zorder=5, s=30, label=r'$\lim_{T\to 0} P = \max(K - S, 0)$')
        stats_text = (
            f"Strike:        {Strike_price:.2f}\n"
            f"Initial Price: {Initial_price:.2f}\n"
            f"Volatility:    {0.2:.2f}\n"
            f"Interest Rate: {Interest_rate:.2f}\n"
            f"Lim C (T=0): {lim_c:.2f}\n"
            f"Lim P (T=0): {lim_p:.2f}"
        )
        ax.text(0.98, 0.25, stats_text, transform=ax.transAxes,
            fontsize=9, verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray'))
        ax.set_xlabel('Time to Maturity (Years)')
        ax.set_ylabel('Option Price')
        ax.set_title('European Option Prices vs Time to Maturity')
        ax.legend()
        ax.grid(True)
        plt.savefig(REPORTS / f'pricer_option_prices_vs_time_to_maturity_S_{Initial_price}_K_{Strike_price}.png')
        plt.show()

    # Test 6 — value table
    print("\nValue Table for European Call Option with initial price S=100, sigma = 0.2:")
    strikes    = [80, 90, 100, 110, 120]
    maturities = {'1M': 1/12, '3M': 3/12, '6M': 6/12, '1Y': 1.0}
    rows = []
    for K in strikes:
        row = {'Strike': K}
        for label, T in maturities.items():
            row[f'Call {label}'] = round(model.price(Option(S=100, K=K, T=T, r=0.05, sigma=0.20, kind='call')), 2)
            row[f'Put {label}']  = round(model.price(Option(S=100, K=K, T=T, r=0.05, sigma=0.20, kind='put')),  2)
        rows.append(row)
    print(pd.DataFrame(rows).set_index('Strike'))
