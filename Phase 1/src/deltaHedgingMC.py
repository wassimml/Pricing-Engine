import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from option import Option
from greeks import BSGreeks
from pricer import BSModel


REPORTS = Path(__file__).parent.parent / "reports"

def simulate_delta_hedging_mc(
    S0: float = 100,
    K: float = 100,
    T_weeks: int = 20,
    r: float = 0.05,
    sigma: float = 0.2,
    nOptions: int = 100,
    nSharesPerOption: int = 100,
    n_simulations: int = 1000,
    rebalancing: str = 'weekly',  # 'weekly' ou 'daily'
    seed: int = 42
) -> pd.DataFrame:

    np.random.seed(seed)

    if rebalancing == 'weekly':
        dt = 1 / 52
        n_steps = T_weeks
    elif rebalancing == 'daily':
        dt = 1 / 252
        n_steps = T_weeks * 5  # 5 jours de trading par semaine

    T = T_weeks / 52
    Initial_Calls = nOptions * nSharesPerOption
    initial_bs_price = BSModel().price(Option(S=S0, K=K, T=T, r=r, sigma=sigma, kind='call'))
    initial_premium = initial_bs_price * Initial_Calls

    results = []

    for sim in range(n_simulations):
        # Simulation GBM
        Z = np.random.normal(0, 1, n_steps)
        S = np.zeros(n_steps + 1)
        S[0] = S0
        for t in range(1, n_steps + 1):
            S[t] = S[t-1] * np.exp((r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z[t-1])

        # t=0
        first_delta = BSGreeks(Option(S=S0, K=K, T=T, r=r, sigma=sigma, kind='call')).delta()
        shares_held = first_delta * Initial_Calls
        cash = initial_premium - shares_held * S[0]

        # t=1 à n_steps-1
        for t in range(1, n_steps):
            cash *= np.exp(r * dt)
            T_remaining = max(T - t * dt, 1e-10)
            current_option = Option(S=S[t], K=K, T=T_remaining, r=r, sigma=sigma, kind='call')
            current_delta = BSGreeks(current_option).delta()
            new_shares = current_delta * Initial_Calls
            cash -= (new_shares - shares_held) * S[t]
            shares_held = new_shares

        # maturité
        cash *= np.exp(r * dt)
        payoff_total = max(S[-1] - K, 0) * Initial_Calls
        portfolio_value = cash + shares_held * S[-1]
        pnl = portfolio_value - payoff_total

        results.append({
            'simulation': sim,
            'S_final': S[-1],
            'payoff': payoff_total,
            'portfolio_value': portfolio_value,
            'pnl': pnl,
            'itm': S[-1] > K
        })

    return pd.DataFrame(results), initial_premium


def plot_comparison(df_weekly: pd.DataFrame, df_daily: pd.DataFrame, 
                    initial_premium: float, K: float = 100):

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # --- Plot 1: Distribution P&L comparée ---
    axes[0].hist(df_weekly['pnl'], bins=50, alpha=0.6, color='steelblue', 
                 edgecolor='white', label=f"Weekly — std={df_weekly['pnl'].std():.0f}€")
    axes[0].hist(df_daily['pnl'], bins=50, alpha=0.6, color='orange',
                 edgecolor='white', label=f"Daily — std={df_daily['pnl'].std():.0f}€")
    axes[0].axvline(0, color='red', lw=1.5, ls='--', label='Break-even')
    axes[0].axvline(df_weekly['pnl'].mean(), color='steelblue', lw=1.5, ls=':',
                    label=f"Mean weekly = {df_weekly['pnl'].mean():.0f}€")
    axes[0].axvline(df_daily['pnl'].mean(), color='orange', lw=1.5, ls=':',
                    label=f"Mean daily = {df_daily['pnl'].mean():.0f}€")
    axes[0].set_title('P&L Distribution — Weekly vs Daily')
    axes[0].set_xlabel('P&L (€)')
    axes[0].set_ylabel('Frequency')
    axes[0].legend(fontsize=8)

    # --- Plot 2: P&L vs S_final weekly ---
    axes[1].scatter(df_weekly['S_final'], df_weekly['pnl'], alpha=0.2, s=5, color='steelblue')
    axes[1].axhline(0, color='red', lw=1, ls='--')
    axes[1].axvline(K, color='black', lw=0.8, ls='--', label=f'Strike K={K}')
    axes[1].set_title('P&L vs $S_T$ — Weekly Rebalancing')
    axes[1].set_xlabel('Final Stock Price $S_T$')
    axes[1].set_ylabel('P&L (€)')
    axes[1].legend()

    # --- Plot 3: P&L vs S_final daily ---
    axes[2].scatter(df_daily['S_final'], df_daily['pnl'], alpha=0.2, s=5, color='orange')
    axes[2].axhline(0, color='red', lw=1, ls='--')
    axes[2].axvline(K, color='black', lw=0.8, ls='--', label=f'Strike K={K}')
    axes[2].set_title('P&L vs $S_T$ — Daily Rebalancing')
    axes[2].set_xlabel('Final Stock Price $S_T$')
    axes[2].set_ylabel('P&L (€)')
    axes[2].legend()

    plt.suptitle('Delta Hedging — Weekly vs Daily Rebalancing (20 weeks, 10 000 paths)', 
                 fontsize=13)
    plt.tight_layout()
    plt.savefig(REPORTS / 'delta_hedging_comparison.png')
    plt.show()

    # Résumé comparatif
    print(f"\n{'='*55}")
    print(f"  {'':30} {'Weekly':>10} {'Daily':>10}")
    print(f"{'='*55}")
    print(f"  {'Initial premium':30} {initial_premium:>10.2f}€")
    print(f"  {'Mean P&L':30} {df_weekly['pnl'].mean():>10.2f}€ {df_daily['pnl'].mean():>10.2f}€")
    print(f"  {'Std P&L':30} {df_weekly['pnl'].std():>10.2f}€ {df_daily['pnl'].std():>10.2f}€")
    print(f"  {'Min P&L':30} {df_weekly['pnl'].min():>10.2f}€ {df_daily['pnl'].min():>10.2f}€")
    print(f"  {'Max P&L':30} {df_weekly['pnl'].max():>10.2f}€ {df_daily['pnl'].max():>10.2f}€")
    print(f"  {'P&L > 0 (%)':30} {100*(df_weekly['pnl']>0).mean():>10.1f}% {100*(df_daily['pnl']>0).mean():>10.1f}%")
    print(f"  {'ITM at maturity (%)':30} {100*df_weekly['itm'].mean():>10.1f}% {100*df_daily['itm'].mean():>10.1f}%")
    print(f"{'='*55}")


if __name__ == "__main__":
    df_weekly, initial_premium = simulate_delta_hedging_mc(
        S0=100, K=100, T_weeks=20, r=0.05, sigma=0.2,
        nOptions=100, nSharesPerOption=100,
        n_simulations=10000, rebalancing='weekly'
    )
    df_daily, _ = simulate_delta_hedging_mc(
        S0=100, K=100, T_weeks=20, r=0.05, sigma=0.2,
        nOptions=100, nSharesPerOption=100,
        n_simulations=10000, rebalancing='daily'
    )
    plot_comparison(df_weekly, df_daily, initial_premium, K=100)