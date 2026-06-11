from scipy.stats import norm
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from option import Option
from pricer import BSModel

REPORTS = Path(__file__).parent.parent / "reports"

def mc_naive(opt : Option, n_paths : int = 100000) -> float:
    """Monte Carlo pricer for European vanilla options."""
    # Simulate terminal spot price
    S_T = opt.S * np.exp((opt.r - 0.5 * opt.sigma ** 2) * opt.T + opt.sigma * np.sqrt(opt.T) * norm.rvs(size=n_paths))
    # Compute payoff and discount
    if opt.kind == 'call':
        Discounted_payoff =np.exp(-opt.r * opt.T) * np.maximum(S_T - opt.K, 0)
    else:
        Discounted_payoff = np.exp(-opt.r * opt.T) * np.maximum(opt.K - S_T, 0)
    return float(np.mean(Discounted_payoff)), np.std(Discounted_payoff)/ np.sqrt(n_paths)


def mc_antithetic(opt : Option, n_paths : int = 100000) -> float:
    """Monte Carlo pricer with antithetic variates."""
    # Simulate terminal spot price
    Z = norm.rvs(size=n_paths//2)
    S_T1 = opt.S * np.exp((opt.r - 0.5 * opt.sigma ** 2) * opt.T + opt.sigma * np.sqrt(opt.T) * Z)
    S_T2 = opt.S * np.exp((opt.r - 0.5 * opt.sigma ** 2) * opt.T - opt.sigma * np.sqrt(opt.T) * Z)
    # Compute payoff and discount
    if opt.kind == 'call':
        Payoff1 = np.maximum(S_T1 - opt.K, 0)
        Payoff2 = np.maximum(S_T2 - opt.K, 0)
    else:
        Payoff1 = np.maximum(opt.K - S_T1, 0)
        Payoff2 = np.maximum(opt.K - S_T2, 0)
    Discounted_payoff = np.exp(-opt.r * opt.T) * (Payoff1 + Payoff2) / 2
    return float(np.mean(Discounted_payoff)), np.std(Discounted_payoff)/ np.sqrt(n_paths)

def mc_control(opt : Option, n_paths : int = 100000) -> float:
    """Monte Carlo pricer with control variates."""
    # Simulate terminal spot price
    S_T = opt.S * np.exp((opt.r - 0.5 * opt.sigma ** 2) * opt.T + opt.sigma * np.sqrt(opt.T) * norm.rvs(size=n_paths))
    # Compute payoff and discount
    if opt.kind == 'call':
        Payoff = np.maximum(S_T - opt.K, 0)
        Control = S_T
        Control_mean = opt.S  
    else:
        Payoff = np.maximum(opt.K - S_T, 0)
        Control = S_T
        Control_mean = opt.S
    # Compute control variate coefficient
    cov = np.cov(Payoff, Control)[0][1]
    var_control = np.var(Control)
    beta = cov / var_control if var_control > 0 else 0
    # Adjust payoff with control variate
    Discounted_Adjusted_payoff = np.exp(-opt.r * opt.T) * Payoff - beta * (np.exp(-opt.r * opt.T) * Control - Control_mean)
    return float(np.mean(Discounted_Adjusted_payoff)), np.std(Discounted_Adjusted_payoff)/ np.sqrt(n_paths)


def mc_control_antithetic(opt : Option, n_paths : int = 100000) -> float:
    """Monte Carlo pricer with control variates and antithetic variates."""
    # Simulate terminal spot price
    Z = norm.rvs(size=n_paths//2)
    S_T1 = opt.S * np.exp((opt.r - 0.5 * opt.sigma ** 2) * opt.T + opt.sigma * np.sqrt(opt.T) * Z)
    S_T2 = opt.S * np.exp((opt.r - 0.5 * opt.sigma ** 2) * opt.T - opt.sigma * np.sqrt(opt.T) * Z)
    # Compute payoff and discount
    if opt.kind == 'call':
        Payoff1 = np.maximum(S_T1 - opt.K, 0)
        Payoff2 = np.maximum(S_T2 - opt.K, 0)
        Control1 = S_T1
        Control2 = S_T2
        Control_mean = opt.S  
    else:
        Payoff1 = np.maximum(opt.K - S_T1, 0)
        Payoff2 = np.maximum(opt.K - S_T2, 0)
        Control1 = S_T1
        Control2 = S_T2
        Control_mean = opt.S
    # Compute control variate coefficient
    Payoff_combined = (Payoff1 + Payoff2) / 2
    Control_combined = (Control1 + Control2) / 2
    cov = np.cov(Payoff_combined, Control_combined)[0][1]
    var_control = np.var(Control_combined)
    beta = cov / var_control if var_control > 0 else 0
    # Adjust payoff with control variate
    Discounted_Adjusted_payoff = np.exp(-opt.r * opt.T) * Payoff_combined - beta * (np.exp(-opt.r * opt.T) * Control_combined - Control_mean)
    return float(np.mean(Discounted_Adjusted_payoff)), np.std(Discounted_Adjusted_payoff)/ np.sqrt(n_paths)

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