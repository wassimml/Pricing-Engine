import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from option import Option
from BSpricer import BSModel

REPORTS = Path(__file__).parent.parent / "reports"

def mc_naive(opt : Option, n_paths : int = 100000, seed : int = 42) -> float:
    """Monte Carlo pricer for European vanilla options."""
    rng = np.random.default_rng(seed)
    s_t = opt.S * np.exp((opt.r - 0.5 * opt.sigma ** 2) * opt.T + opt.sigma * np.sqrt(opt.T) * rng.standard_normal(n_paths))
    if opt.kind == 'call':
        discounted_payoff = np.exp(-opt.r * opt.T) * np.maximum(s_t - opt.K, 0)
    else:
        discounted_payoff = np.exp(-opt.r * opt.T) * np.maximum(opt.K - s_t, 0)
    return float(np.mean(discounted_payoff)), np.std(discounted_payoff) / np.sqrt(n_paths)


def mc_antithetic(opt : Option, n_paths : int = 100000, seed : int = 42) -> float:
    """Monte Carlo pricer with antithetic variates."""
    rng = np.random.default_rng(seed)
    z = rng.standard_normal(n_paths // 2)
    s_t1 = opt.S * np.exp((opt.r - 0.5 * opt.sigma ** 2) * opt.T + opt.sigma * np.sqrt(opt.T) * z)
    s_t2 = opt.S * np.exp((opt.r - 0.5 * opt.sigma ** 2) * opt.T - opt.sigma * np.sqrt(opt.T) * z)
    if opt.kind == 'call':
        payoff1 = np.maximum(s_t1 - opt.K, 0)
        payoff2 = np.maximum(s_t2 - opt.K, 0)
    else:
        payoff1 = np.maximum(opt.K - s_t1, 0)
        payoff2 = np.maximum(opt.K - s_t2, 0)
    discounted_payoff = np.exp(-opt.r * opt.T) * (payoff1 + payoff2) / 2
    # n_paths//2 independent pairs — not n_paths (pairs are correlated)
    return float(np.mean(discounted_payoff)), np.std(discounted_payoff) / np.sqrt(n_paths // 2)


def mc_control(opt : Option, n_paths : int = 100000, seed : int = 42) -> float:
    """Monte Carlo pricer with control variates."""
    rng = np.random.default_rng(seed)
    s_t = opt.S * np.exp((opt.r - 0.5 * opt.sigma ** 2) * opt.T + opt.sigma * np.sqrt(opt.T) * rng.standard_normal(n_paths))
    if opt.kind == 'call':
        payoff = np.maximum(s_t - opt.K, 0)
    else:
        payoff = np.maximum(opt.K - s_t, 0)
    beta = np.cov(payoff, s_t)[0, 1] / np.var(s_t)
    adjusted = payoff - beta * (s_t - opt.S * np.exp(opt.r * opt.T))
    price = np.exp(-opt.r * opt.T) * np.mean(adjusted)
    std_error = np.exp(-opt.r * opt.T) * np.std(adjusted) / np.sqrt(n_paths)
    return float(price), std_error


def mc_control_antithetic(opt : Option, n_paths : int = 100000, seed : int = 42) -> float:
    """Monte Carlo pricer with control variates and antithetic variates."""
    rng = np.random.default_rng(seed)
    z = rng.standard_normal(n_paths // 2)
    s_t1 = opt.S * np.exp((opt.r - 0.5 * opt.sigma ** 2) * opt.T + opt.sigma * np.sqrt(opt.T) * z)
    s_t2 = opt.S * np.exp((opt.r - 0.5 * opt.sigma ** 2) * opt.T - opt.sigma * np.sqrt(opt.T) * z)
    if opt.kind == 'call':
        payoff1 = np.maximum(s_t1 - opt.K, 0)
        payoff2 = np.maximum(s_t2 - opt.K, 0)
    else:
        payoff1 = np.maximum(opt.K - s_t1, 0)
        payoff2 = np.maximum(opt.K - s_t2, 0)
    payoff_combined = (payoff1 + payoff2) / 2
    control_combined = (s_t1 + s_t2) / 2
    control_mean = opt.S
    beta = np.cov(payoff_combined, control_combined)[0][1] / np.var(control_combined)
    discounted_adjusted = np.exp(-opt.r * opt.T) * payoff_combined - beta * (np.exp(-opt.r * opt.T) * control_combined - control_mean)
    # n_paths//2 independent pairs — not n_paths (pairs are correlated)
    return float(np.mean(discounted_adjusted)), np.std(discounted_adjusted) / np.sqrt(n_paths // 2)


def mc_mean_std(pricer, opt, n_paths, n_seeds, base_seed=0):
    """Moyenne et écart-type du prix MC sur n_seeds tirages indépendants.

    Même logique que lsm_mean_std (cf. monteCarloLSM.py) : une seule
    réalisation (seed fixe) ne permet pas de distinguer un vrai comportement
    de convergence d'un artefact du tirage particulier — on répète donc
    chaque n_paths sur plusieurs seeds indépendants.
    """
    prices = np.array([
        pricer(opt, n_paths, seed=base_seed + s)[0]
        for s in range(n_seeds)
    ])
    return prices.mean(), prices.std()


if __name__ == "__main__":
    opt = Option(S=100, K=100, T=1, r=0.05, sigma=0.2, kind='call')
    n_paths = 100000
    N_SEEDS = 25  # répétitions indépendantes par n_paths (cf. monteCarloLSM.py)

    model = BSModel()
    bsm_price = model.price(opt)
    print(f"BSM price: {bsm_price:.4f}")

    METHODS = {
        "naive":              ("Naive", mc_naive),
        "antithetic":         ("Antithetic", mc_antithetic),
        "control":            ("Control Variate", mc_control),
        "control_antithetic": ("Control + Antithetic", mc_control_antithetic),
    }

    path_counts = [100, 500, 1000, 5000, 10000, 50000, 100000, 500000, 1000000, 5000000, 10000000]

    for key, (label, pricer) in METHODS.items():
        print(f"\n----------------- {label} Monte Carlo pricing with {n_paths} paths -----------------")
        price, std_error = pricer(opt, n_paths)
        print(f"Monte Carlo price: {price:.4f}")
        print(f"Standard error (théorique, seed unique): {std_error:.4f}")
        print(f"BSM price: {bsm_price:.4f}")
        print(f"Difference: {abs(price - bsm_price):.4f}, Relative Difference: {abs(price - bsm_price)/bsm_price:.4%}")

        print(f"--- Convergence ({N_SEEDS} seeds indépendants par n_paths) ---")
        rel_diffs, rel_diffs_std = [], []
        for n in path_counts:
            mean_price, std_price = mc_mean_std(pricer, opt, n, N_SEEDS)
            rel_diff     = abs(mean_price - bsm_price) / bsm_price * 100
            rel_diff_std = std_price / bsm_price * 100
            rel_diffs.append(rel_diff)
            rel_diffs_std.append(rel_diff_std)
            print(f"n_paths={n:9d}  price={mean_price:.4f}±{std_price:.4f}  "
                  f"écart BSM={rel_diff:.4f}%±{rel_diff_std:.4f}%")

        plt.figure()
        plt.errorbar(path_counts, rel_diffs, yerr=rel_diffs_std, fmt='o-',
                     capsize=3, alpha=0.85,
                     label=f'{label} (moyenne ± écart-type, {N_SEEDS} seeds)')
        plt.axhline(0, color='red', ls='--', lw=0.8)
        plt.xscale('log')
        plt.xlabel('Number of Paths')
        plt.ylabel('Relative Difference from BSM Price (%)\n(moyenne ± écart-type)')
        plt.title(f'Monte Carlo Price Convergence — {label}')
        plt.legend()
        plt.savefig(REPORTS / f"monte_carlo_convergence_{key}.png")
        plt.show()