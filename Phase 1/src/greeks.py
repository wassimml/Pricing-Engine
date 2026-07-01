import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
from pathlib import Path

from option import Option

REPORTS = Path(__file__).parent.parent / "reports"

# T is in calendar years → theta is per calendar day (÷365).
# Use ÷252 only if you want theta per trading day.
DAYS_IN_YEAR = 365

class BSGreeks:
    def __init__(self, option: Option):
        self.opt = option

    def _d1_d2(self):
        o = self.opt
        d1 = (np.log(o.S / o.K) + (o.r + 0.5 * o.sigma ** 2) * o.T) / (o.sigma * np.sqrt(o.T))
        d2 = d1 - o.sigma * np.sqrt(o.T)
        return d1, d2

    def delta(self) -> float:
        d1, _ = self._d1_d2()
        if self.opt.kind == 'call':
            return float(norm.cdf(d1))
        return float(norm.cdf(d1) - 1)

    def gamma(self) -> float:
        d1, _ = self._d1_d2()
        return norm.pdf(d1) / (self.opt.S * self.opt.sigma * np.sqrt(self.opt.T))

    def vega(self) -> float:
        # Raw ∂V/∂σ — sigma is in decimal (0.2 = 20%), so this is per unit of σ.
        # Do NOT divide by 100: delta/gamma/theta use the same raw-derivative convention,
        # and dividing here would break hedging calculations by a factor of 100.
        d1, _ = self._d1_d2()
        return self.opt.S * norm.pdf(d1) * np.sqrt(self.opt.T)

    def theta(self) -> float:
        d1, d2 = self._d1_d2()
        if self.opt.kind == 'call':
            return ((-self.opt.S * norm.pdf(d1) * self.opt.sigma) / (2 * np.sqrt(self.opt.T))
                    - self.opt.r * self.opt.K * np.exp(-self.opt.r * self.opt.T) * norm.cdf(d2)) / DAYS_IN_YEAR
        else:
            return (-self.opt.S * norm.pdf(d1) * self.opt.sigma / (2 * np.sqrt(self.opt.T))
                    + self.opt.r * self.opt.K * np.exp(-self.opt.r * self.opt.T) * norm.cdf(-d2)) / DAYS_IN_YEAR

    def rho(self) -> float:
        # Raw ∂V/∂r — r is in decimal (0.05 = 5%), so this is per unit of r.
        # Do NOT divide by 100: same raw-derivative convention as all other Greeks.
        _, d2 = self._d1_d2()
        if self.opt.kind == 'call':
            return self.opt.K * self.opt.T * np.exp(-self.opt.r * self.opt.T) * norm.cdf(d2)
        else:
            return -self.opt.K * self.opt.T * np.exp(-self.opt.r * self.opt.T) * norm.cdf(-d2)

if __name__ == "__main__":
    opt_call = Option(S=100, K=100, T=1, r=0.05, sigma=0.2, kind='call')
    opt_put = Option(S=100, K=100, T=1, r=0.05, sigma=0.2, kind='put')

    # Delta values
    print("Delta (call):", BSGreeks(opt_call).delta())
    print("Delta (put):", BSGreeks(opt_put).delta())

    # Delta evolution plot
    S_range = np.linspace(60, 140, 1000)
    deltas_call_T_1 = [BSGreeks(Option(S=s, K=100, T=1, r=0.05, sigma=0.2, kind='call')).delta() for s in S_range]
    deltas_call_T_0_5 = [BSGreeks(Option(S=s, K=100, T=0.5, r=0.05, sigma=0.2, kind='call')).delta() for s in S_range]
    deltas_call_T_0_01 = [BSGreeks(Option(S=s, K=100, T=0.01, r=0.05, sigma=0.2, kind='call')).delta() for s in S_range]  
    deltas_call_T_0_001 = [BSGreeks(Option(S=s, K=100, T=0.001, r=0.05, sigma=0.2, kind='call')).delta() for s in S_range]
    payoff_call = np.maximum(S_range - opt_call.K, 0)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # --- Plot 1: Delta ---
    step = np.where(S_range >= 100, 1.0, 0.0)
    axes[0].plot(S_range, step,                label='T = 0 (maturity)',  color='black', lw=2, ls='--')
    axes[0].plot(S_range, deltas_call_T_1,     label='T = 1 year')
    axes[0].plot(S_range, deltas_call_T_0_5,   label='T = 0.5 year')
    axes[0].plot(S_range, deltas_call_T_0_01,  label='T = 0.01 year')
    axes[0].plot(S_range, deltas_call_T_0_001, label='T = 0.001 year')
    axes[0].axvline(100, color='red', lw=0.8, ls='--', label='Strike K=100')
    axes[0].axhline(0.5, color='grey', lw=0.5, ls=':')
    axes[0].set_title('Delta vs Spot Price')
    axes[0].set_xlabel('Spot Price S')
    axes[0].set_ylabel('Delta')
    axes[0].set_ylim(-0.1, 1.1)
    axes[0].legend()

    # --- Plot 2: Payoff ---
    payoff_call = np.maximum(S_range - 100, 0)
    axes[1].plot(S_range, payoff_call, color='green', lw=2, label='Call Payoff')
    axes[1].axvline(100, color='red', lw=0.8, ls='--', label='Strike K=100')
    axes[1].axhline(0, color='grey', lw=0.5, ls='--')
    axes[1].set_title('Payoff at Maturity')
    axes[1].set_xlabel('Spot Price $S_T$')
    axes[1].set_ylabel('Payoff (€)')
    axes[1].legend()

    plt.suptitle('Delta vs Payoff (European Call (K=100, T=1, r=0.05, σ=0.2))', fontsize=13)
    plt.tight_layout()
    plt.savefig(REPORTS / 'greeks_delta_payoff.png')
    plt.show()
    

    # Gamma values
    print("Gamma (call):", BSGreeks(opt_call).gamma())
    print("Gamma (put):", BSGreeks(opt_put).gamma())

    # Gamma evolution plot
        # Delta evolution plot
    S_range = np.linspace(60, 140, 1000)
    gamma_call_T_1 = [BSGreeks(Option(S=s, K=100, T=1, r=0.05, sigma=0.2, kind='call')).gamma() for s in S_range]
    gamma_call_T_0_5 = [BSGreeks(Option(S=s, K=100, T=0.5, r=0.05, sigma=0.2, kind='call')).gamma() for s in S_range]
    gamma_call_T_0_01 = [BSGreeks(Option(S=s, K=100, T=0.01, r=0.05, sigma=0.2, kind='call')).gamma() for s in S_range]  
    gamma_call_T_0_001 = [BSGreeks(Option(S=s, K=100, T=0.001, r=0.05, sigma=0.2, kind='call')).gamma() for s in S_range]
    payoff_call = np.maximum(S_range - opt_call.K, 0)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # --- Plot 1: Delta ---
    axes[0].plot(S_range, gamma_call_T_1,     label='T = 1 year')
    axes[0].plot(S_range, gamma_call_T_0_5,   label='T = 0.5 year')
    axes[0].plot(S_range, gamma_call_T_0_01,  label='T = 0.01 year')
    axes[0].plot(S_range, gamma_call_T_0_001, label='T = 0.001 year')
    axes[0].axvline(100, color='red', lw=0.8, ls='--', label='Strike K=100')
    axes[0].axhline(0.5, color='grey', lw=0.5, ls=':')
    axes[0].set_title('Gamma vs Spot Price')
    axes[0].set_xlabel('Spot Price S')
    axes[0].set_ylabel('Gamma')
    axes[0].set_ylim(-0.1, 0.7)
    axes[0].legend()

    # --- Plot 2: Payoff ---
    payoff_call = np.maximum(S_range - 100, 0)
    axes[1].plot(S_range, payoff_call, color='green', lw=2, label='Call Payoff')
    axes[1].axvline(100, color='red', lw=0.8, ls='--', label='Strike K=100')
    axes[1].axhline(0, color='grey', lw=0.5, ls='--')
    axes[1].set_title('Payoff at Maturity')
    axes[1].set_xlabel('Spot Price $S_T$')
    axes[1].set_ylabel('Payoff (€)')
    axes[1].legend()

    plt.suptitle('Gamma vs Payoff (European Call (K=100, T=1, r=0.05, σ=0.2))', fontsize=13)
    plt.tight_layout()
    plt.savefig(REPORTS / 'greeks_gamma_payoff.png')
    plt.show()


    # Vega values
    print("Vega (call):", BSGreeks(opt_call).vega())
    print("Vega (put):", BSGreeks(opt_put).vega())

    # Vega evolution plot
    vegas_call_T_1 = [BSGreeks(Option(S=s, K=100, T=1, r=0.05, sigma=0.2, kind='call')).vega() for s in S_range]
    vegas_call_T_0_5 = [BSGreeks(Option(S=s, K=100, T=0.5, r=0.05, sigma=0.2, kind='call')).vega() for s in S_range]
    vegas_call_T_0_1 = [BSGreeks(Option(S=s, K=100, T=0.1, r=0.05, sigma=0.2, kind='call')).vega() for s in S_range]
    vegas_call_T_0_01 = [BSGreeks(Option(S=s, K=100, T=0.01, r=0.05, sigma=0.2, kind='call')).vega() for s in S_range]

    T_range = np.linspace(0.01, 2, 1000)
    vegas_ATM = [BSGreeks(Option(S=100, K=100, T=t, r=0.05, sigma=0.2, kind='call')).vega() for t in T_range]
    vegas_ITM = [BSGreeks(Option(S=120, K=100, T=t, r=0.05, sigma=0.2, kind='call')).vega() for t in T_range]
    vegas_OTM = [BSGreeks(Option(S=80, K=100, T=t, r=0.05, sigma=0.2, kind='call')).vega() for t in T_range]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # --- Plot 1: Vega vs Spot ---
    axes[0].plot(S_range, vegas_call_T_1,   label='T = 1 year')
    axes[0].plot(S_range, vegas_call_T_0_5, label='T = 0.5 year')
    axes[0].plot(S_range, vegas_call_T_0_1, label='T = 0.1 year')
    axes[0].plot(S_range, vegas_call_T_0_01,label='T = 0.01 year')
    axes[0].axvline(100, color='red', lw=0.8, ls='--', label='Strike K=100')
    axes[0].axhline(0, color='grey', lw=0.5, ls='--')
    axes[0].set_title('Vega vs Spot Price')
    axes[0].set_xlabel('Spot Price S')
    axes[0].set_ylabel('Vega')
    axes[0].legend()

    # --- Plot 2: Vega vs T (ATM) ---
    axes[1].plot(T_range, vegas_ATM, color='purple', lw=2, label='Vega ATM')
    axes[1].plot(T_range, vegas_ITM, color='green', lw=2, label='Vega ITM (S=120)')
    axes[1].plot(T_range, vegas_OTM, color='blue', lw=2, label='Vega OTM (S=80)')
    axes[1].axhline(0, color='grey', lw=0.5, ls='--')
    axes[1].set_title('Vega vs Time to Maturity (ATM)')
    axes[1].set_xlabel('Time to Maturity T (years)')
    axes[1].set_ylabel('Vega')
    axes[1].legend()

    plt.suptitle('Vega (European Call (K=100, r=0.05, σ=0.2))', fontsize=13)
    plt.tight_layout()
    plt.savefig(REPORTS / 'greeks_vega.png')
    plt.show()

    # Theta values
    print("Theta (call):", BSGreeks(opt_call).theta())
    print("Theta (put):", BSGreeks(opt_put).theta())

    # Theta evolution plot
    thetas_call_T_1 = [BSGreeks(Option(S=s, K=100, T=1, r=0.05, sigma=0.2, kind='call')).theta() for s in S_range]
    thetas_call_T_0_5 = [BSGreeks(Option(S=s, K=100, T=0.5, r=0.05, sigma=0.2, kind='call')).theta() for s in S_range]
    thetas_call_T_0_1 = [BSGreeks(Option(S=s, K=100, T=0.1, r=0.05, sigma=0.2, kind='call')).theta() for s in S_range]
    thetas_call_T_0_01 = [BSGreeks(Option(S=s, K=100, T=0.01, r=0.05, sigma=0.2, kind='call')).theta() for s in S_range]

    T_range_theta = np.linspace(0.01, 2, 1000)
    thetas_ATM = [BSGreeks(Option(S=100, K=100, T=t, r=0.05, sigma=0.2, kind='call')).theta() for t in T_range_theta]
    theta_ITM = [BSGreeks(Option(S=120, K=100, T=t, r=0.05, sigma=0.2, kind='call')).theta() for t in T_range_theta]
    theta_OTM = [BSGreeks(Option(S=80, K=100, T=t, r=0.05, sigma=0.2, kind='call')).theta() for t in T_range_theta]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # --- Plot 1: Theta vs Spot ---
    axes[0].plot(S_range, thetas_call_T_1,    label='T = 1 year')
    axes[0].plot(S_range, thetas_call_T_0_5,  label='T = 0.5 year')
    axes[0].plot(S_range, thetas_call_T_0_1,  label='T = 0.1 year')
    axes[0].plot(S_range, thetas_call_T_0_01, label='T = 0.01 year')
    axes[0].axvline(100, color='red', lw=0.8, ls='--', label='Strike K=100')
    axes[0].axhline(0, color='grey', lw=0.5, ls='--')
    axes[0].set_title('Theta vs Spot Price')
    axes[0].set_xlabel('Spot Price S')
    axes[0].set_ylabel('Theta (per day)')
    axes[0].legend()

    # --- Plot 2: Theta vs T (ATM) ---
    axes[1].plot(T_range_theta, thetas_ATM, color='orange', lw=2, label='Theta ATM')
    axes[1].plot(T_range_theta, theta_ITM, color='green', lw=2, label='Theta ITM (S=120)')
    axes[1].plot(T_range_theta, theta_OTM, color='blue', lw=2, label='Theta OTM (S=80)')
    axes[1].axhline(0, color='grey', lw=0.5, ls='--')
    axes[1].set_title('Theta vs Time to Maturity (ATM)')
    axes[1].set_xlabel('Time to Maturity T (years)')
    axes[1].set_ylabel('Theta (per day)')
    axes[1].legend()

    plt.suptitle('Theta (European Call (K=100, r=0.05, σ=0.2))', fontsize=13)
    plt.tight_layout()
    plt.savefig(REPORTS / 'greeks_theta.png')
    plt.show()


    # Rho values
    print("Rho (call):", BSGreeks(opt_call).rho())
    print("Rho (put):", BSGreeks(opt_put).rho())

    # Rho evolution plot
    rhos_call_T_1 = [BSGreeks(Option(S=s, K=100, T=1, r=0.05, sigma=0.2, kind='call')).rho() for s in S_range]
    rhos_call_T_0_5 = [BSGreeks(Option(S=s, K=100, T=0.5, r=0.05, sigma=0.2, kind='call')).rho() for s in S_range]
    rhos_call_T_0_1 = [BSGreeks(Option(S=s, K=100, T=0.1, r=0.05, sigma=0.2, kind='call')).rho() for s in S_range]
    rhos_call_T_0_01 = [BSGreeks(Option(S=s, K=100, T=0.01, r=0.05, sigma=0.2, kind='call')).rho() for s in S_range]

    r_range = np.linspace(0, 0.2, 1000)
    rhos_ATM = [BSGreeks(Option(S=100, K=100, T=1, r=r, sigma=0.2, kind='call')).rho() for r in r_range]
    rhos_ITM = [BSGreeks(Option(S=120, K=100, T=1, r=r, sigma=0.2, kind='call')).rho() for r in r_range]
    rhos_OTM = [BSGreeks(Option(S=80, K=100, T=1, r=r, sigma=0.2, kind='call')).rho() for r in r_range]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # --- Plot 1: Rho vs Spot ---
    axes[0].plot(S_range, rhos_call_T_1,    label='T = 1 year')
    axes[0].plot(S_range, rhos_call_T_0_5,  label='T = 0.5 year')
    axes[0].plot(S_range, rhos_call_T_0_1,  label='T = 0.1 year')
    axes[0].plot(S_range, rhos_call_T_0_01, label='T = 0.01 year')
    axes[0].axvline(100, color='red', lw=0.8, ls='--', label='Strike K=100')
    axes[0].axhline(0, color='grey', lw=0.5, ls='--')
    axes[0].set_title('Rho vs Spot Price')
    axes[0].set_xlabel('Spot Price S')
    axes[0].set_ylabel('Rho')
    axes[0].legend()

    # --- Plot 2: Rho vs r (ATM) ---
    axes[1].plot(r_range * 100, rhos_ATM, color='steelblue', lw=2, label='Rho ATM')
    axes[1].plot(r_range * 100, rhos_ITM, color='green', lw=2, label='Rho ITM (S=120)')
    axes[1].plot(r_range * 100, rhos_OTM, color='blue', lw=2, label='Rho OTM (S=80)')
    axes[1].axhline(0, color='grey', lw=0.5, ls='--')
    axes[1].set_title('Rho vs Risk-Free Rate (ATM)')
    axes[1].set_xlabel('Risk-Free Rate r (%)')
    axes[1].set_ylabel('Rho')
    axes[1].legend()

    plt.suptitle('Rho (European Call (K=100, T=1, σ=0.2))', fontsize=13)
    plt.tight_layout()
    plt.savefig(REPORTS / 'greeks_rho.png')
    plt.show()