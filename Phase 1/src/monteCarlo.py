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

if __name__ == "__main__":
    opt = Option(S=100, K=100, T=1, r=0.05, sigma=0.2, kind='call')
    n_paths = 100000

    print(f"----------------- Naive Monte Carlo pricing with {n_paths} paths -----------------")
    price, std_error = mc_naive(opt, n_paths)
    print(f"Monte Carlo price: {price:.4f}")
    print(f"Standard error: {std_error:.4f}")

    # Compare with BSM price
    model = BSModel()
    bsm_price = model.price(opt)
    print(f"BSM price: {bsm_price:.4f}")

    # Difference
    print(f"Difference: {abs(price - bsm_price):.4f}, Relative Difference: {abs(price - bsm_price)/bsm_price:.4%}")

    # Standard error of the Monte Carlo estimate
    for path in [1000, 10000, 100000, 1000000]:
        price, std_error = mc_naive(opt, path)
        print(f"Standard error of Monte Carlo estimate: {std_error:.4f}, with {path} paths")

    # Plot the difference as we increase the number of paths
    path_counts = [100, 500, 1000, 5000, 10000, 50000, 100000, 500000, 1000000, 5000000, 10000000]
    mc_diffs = [abs(mc_naive(opt, n)[0] - bsm_price)/bsm_price *100 for n in path_counts]
    plt.plot(path_counts, mc_diffs, marker='o', label='Monte Carlo difference')
    plt.xscale('log')
    plt.xlabel('Number of Paths')
    plt.ylabel('Relative Difference from BSM Price (%)')
    plt.title('Monte Carlo Price Convergence')
    plt.legend()
    plt.savefig(REPORTS / "monte_carlo_convergence_naive.png")
    plt.show()


    print(f"----------------- Antithetic Monte Carlo pricing with {n_paths} paths -----------------")
    price, std_error = mc_antithetic(opt, n_paths)
    print(f"Monte Carlo price: {price:.4f}")
    print(f"Standard error: {std_error:.4f}")

    # Compare with BSM price
    print(f"BSM price: {bsm_price:.4f}")

    # Difference
    print(f"Difference: {abs(price - bsm_price):.4f}, Relative Difference: {abs(price - bsm_price)/bsm_price:.4%}")

    # Standard error of the Monte Carlo estimate
    for path in [1000, 10000, 100000, 1000000]:
        price, std_error = mc_antithetic(opt, path)
        print(f"Standard error of Monte Carlo estimate: {std_error:.4f}, with {path} paths")

    # Plot the difference as we increase the number of paths
    path_counts = [100, 500, 1000, 5000, 10000, 50000, 100000, 500000, 1000000, 5000000, 10000000]
    mc_diffs = [abs(mc_antithetic(opt, n)[0] - bsm_price)/bsm_price *100 for n in path_counts]
    plt.plot(path_counts, mc_diffs, marker='o', label='Monte Carlo difference')
    plt.xscale('log')
    plt.xlabel('Number of Paths')
    plt.ylabel('Relative Difference from BSM Price (%)')
    plt.title('Monte Carlo Price Convergence')
    plt.legend()
    plt.savefig(REPORTS / "monte_carlo_convergence_antithetic.png")
    plt.show()


    print(f"----------------- Control Variate Monte Carlo pricing with {n_paths} paths -----------------")
    price, std_error = mc_control(opt, n_paths)
    print(f"Monte Carlo price: {price:.4f}")
    print(f"Standard error: {std_error:.4f}")
    # Compare with BSM price
    print(f"BSM price: {bsm_price:.4f}")

    # Difference
    print(f"Difference: {abs(price - bsm_price):.4f}, Relative Difference: {abs(price - bsm_price)/bsm_price:.4%}")

    # Standard error of the Monte Carlo estimate
    for path in [1000, 10000, 100000, 1000000]:
        price, std_error = mc_control(opt, path)
        print(f"Standard error of Monte Carlo estimate: {std_error:.4f}, with {path} paths")

    # Plot the difference as we increase the number of paths
    path_counts = [100, 500, 1000, 5000, 10000, 50000, 100000, 500000, 1000000, 5000000, 10000000]
    mc_diffs = [abs(mc_control(opt, n)[0] - bsm_price)/bsm_price *100 for n in path_counts]
    plt.plot(path_counts, mc_diffs, marker='o', label='Monte Carlo difference')
    plt.xscale('log')
    plt.xlabel('Number of Paths')
    plt.ylabel('Relative Difference from BSM Price (%)')
    plt.title('Monte Carlo Price Convergence')
    plt.legend()
    plt.savefig(REPORTS / "monte_carlo_convergence_control.png")
    plt.show()

    print(f"----------------- Control and Antithetic Variate Monte Carlo pricing with {n_paths} paths -----------------")
    price, std_error = mc_control_antithetic(opt, n_paths)
    print(f"Monte Carlo price: {price:.4f}")
    print(f"Standard error: {std_error:.4f}")
    # Compare with BSM price
    print(f"BSM price: {bsm_price:.4f}")

    # Difference
    print(f"Difference: {abs(price - bsm_price):.4f}, Relative Difference: {abs(price - bsm_price)/bsm_price:.4%}")

    # Standard error of the Monte Carlo estimate
    for path in [1000, 10000, 100000, 1000000]:
        price, std_error = mc_control_antithetic(opt, path)
        print(f"Standard error of Monte Carlo estimate: {std_error:.4f}, with {path} paths")

    # Plot the difference as we increase the number of paths
    path_counts = [100, 500, 1000, 5000, 10000, 50000, 100000, 500000, 1000000, 5000000, 10000000]
    mc_diffs = [abs(mc_control_antithetic(opt, n)[0] - bsm_price)/bsm_price *100 for n in path_counts]
    plt.plot(path_counts, mc_diffs, marker='o', label='Monte Carlo difference')
    plt.xscale('log')
    plt.xlabel('Number of Paths')
    plt.ylabel('Relative Difference from BSM Price (%)')
    plt.title('Monte Carlo Price Convergence')
    plt.legend()
    plt.savefig(REPORTS / "monte_carlo_convergence_control_antithetic.png")
    plt.show()