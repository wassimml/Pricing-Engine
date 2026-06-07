import scipy.stats as stats
import numpy as np

from option import Option
from greeks import BSGreeks
from pricer import BSModel

if __name__ == "__main__":
    # Values for the option and market parameters
    T_weeks = 20
    dt = 1 / 52  # 1 semaine en fraction d'année
    S0 = 100
    r = 0.05
    sigma = 0.2
    K = 100
    nOptions = 100
    nSharesPerOption = 100
    total_Calls = nOptions * nSharesPerOption

    print("Delta Hedging Simulation for a Call Option")
    print(f"Initial Stock Price: {S0}, Strike Price: {K}, "
          f"Time to Maturity: {T_weeks/52:.2f} years ({T_weeks} weeks), "
          f"Risk-free Rate: {r}, Volatility: {sigma}, Number of Calls: {total_Calls}")
    print(f"{'Week':>4} | {'Stock Price':>11} | {'Option Price':>12} | "
          f"{'Delta':>6} | {'Shares Held':>11} | {'Change':>10} | {'Cash':>12}")
    print("-" * 85)

    # Simulate the stock price path using GBM
    Z = np.random.normal(0, 1, T_weeks)
    S = np.zeros(T_weeks + 1)
    S[0] = S0
    for t in range(1, T_weeks + 1):
        S[t] = S[t-1] * np.exp((r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z[t-1])

    # t=0 : initialisation
    initial_bs_price = BSModel().price(Option(S=S0, K=K, T=T_weeks/52, r=r, sigma=sigma, kind='call'))
    first_delta = BSGreeks(Option(S=S0, K=K, T=T_weeks/52, r=r, sigma=sigma, kind='call')).delta()

    shares_held = first_delta * total_Calls
    cash = initial_bs_price * total_Calls - shares_held * S[0]  # prime reçue - coût des actions

    print(f"{0:>4} | {S[0]:>11.2f} | {initial_bs_price:>12.4f} | "
          f"{first_delta:>6.4f} | {shares_held:>11.2f} | {'':>10} | {cash:>12.2f}")

    # t=1 à T-1 : rééquilibrages hebdomadaires
    for t in range(1, T_weeks):
        cash *= np.exp(r * dt)  # intérêts sur le cash

        current_option = Option(S=S[t], K=K, T=(T_weeks - t) / 52, r=r, sigma=sigma, kind='call')
        current_price = BSModel().price(current_option)
        current_delta = BSGreeks(current_option).delta()

        new_shares = current_delta * total_Calls
        shares_change = new_shares - shares_held
        cash -= shares_change * S[t]  # achat/vente d'actions
        shares_held = new_shares

        print(f"{t:>4} | {S[t]:>11.2f} | {current_price:>12.4f} | "
              f"{current_delta:>6.4f} | {shares_held:>11.2f} | {shares_change:>10.2f} | {cash:>12.2f}")

    # t=T : maturité
    cash *= np.exp(r * dt)
    final_payoff_per_call = max(S[-1] - K, 0)
    final_delta = 1.0 if S[-1] > K else 0.0

    print(f"{T_weeks:>4} | {S[-1]:>11.2f} | {final_payoff_per_call:>12.4f} | "
          f"{final_delta:>6.1f} | {shares_held:>11.2f} | {'':>10} | {cash:>12.2f}")
    print("-" * 85)

    # P&L final
    payoff_total = final_payoff_per_call * total_Calls
    portfolio_value = cash + shares_held * S[-1]
    pnl = portfolio_value - payoff_total

    print(f"\nInitial premium received    : {initial_bs_price * total_Calls:>10.2f}€")
    print(f"Portfolio value at maturity : {portfolio_value:>10.2f}€")
    print(f"Payoff to deliver           : {payoff_total:>10.2f}€")
    print(f"P&L                         : {pnl:>10.2f}€")
    if pnl > 0:
        print(f"Payoff to deliver : {payoff_total:>10.2f}€  →  Hedging P&L : +{pnl:>10.2f}€  ✓ Gain")
    else:
        print(f"Payoff to deliver : {payoff_total:>10.2f}€  →  Hedging P&L :  {pnl:>10.2f}€  ✗ Loss")