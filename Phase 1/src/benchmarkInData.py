from pathlib import Path
import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
import time 

# Import all methods 
from option import Option
from binomial import crr_price
from monteCarlo import mc_naive, mc_antithetic, mc_control, mc_control_antithetic
from monteCarloLSM import LSMoptionValue
from pde import pde_crank_nicolson
from BSpricer import BSModel

DATA = Path(__file__).parent.parent / "data"
REPORTS = Path(__file__).parent.parent / "reports"

_bs = BSModel()

PRICERS = {
    "BS":            lambda opt, style: _bs.price(opt) if style != 'American' else None,
    "crr":           lambda opt, style: crr_price(opt, period=1000, american=(style == 'American')),
    "MC Naive":      lambda opt, style: mc_naive(opt, n_paths=100000)[0] if style != 'American' else None,
    "MC Anti":       lambda opt, style: mc_antithetic(opt, n_paths=100000)[0] if style != 'American' else None,
    "MC Contr":      lambda opt, style: mc_control(opt, n_paths=100000)[0] if style != 'American' else None,
    "MC Anti Contr": lambda opt, style: mc_control_antithetic(opt, n_paths=50000)[0] if style != 'American' else None,
    "lsm":           lambda opt, style: LSMoptionValue(opt, n_steps=50, n_paths=20000) if style == 'American' else None,
    "pde":           lambda opt, style: pde_crank_nicolson(opt, style=style.lower(), n_steps=200, n_space=200),
    "BS,pde":        lambda opt, style: _bs.price(opt) if style != 'American' else pde_crank_nicolson(opt, style=style.lower(), n_steps=200, n_space=200),
    "BS,crr":        lambda opt, style: _bs.price(opt) if style != 'American' else crr_price(opt, period=1000, american=True),
    "BS,lsm":        lambda opt, style: _bs.price(opt) if style != 'American' else LSMoptionValue(opt, n_steps=50, n_paths=20000),
}

def calcPriceWithMethod(cleanData: pd.DataFrame, method: str):
    pricer = PRICERS[method]
    times, prices = [], []
    for row in cleanData.itertuples():
        option = Option(S=row[2], K=row[3], T=row[4], r=row[5]/100, sigma=row[6]/100, kind=row[1].lower())
        t0 = time.perf_counter()
        p = pricer(option, row[7])
        times.append(time.perf_counter() - t0)
        prices.append(p)
    return times, prices


def plot_figure(title, methods, data_subset, ref, ref_label, bar_color, filename):
    times, maes = [], []
    for method in methods:
        method_times, prices = calcPriceWithMethod(data_subset, method)
        times.append(sum(method_times))
        p = np.array(prices, dtype=float)
        mask = ~np.isnan(p) & ~np.isnan(ref)
        maes.append(np.mean(np.abs(p[mask] - ref[mask])))
        print(f"  {method}: total={times[-1]:.2f}s  MAE={maes[-1]:.4f}")

    x = np.arange(len(methods))
    _, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

    ax1.bar(x, times, color=bar_color, edgecolor="white")
    ax1.set_ylabel("Total time (s)")
    ax1.set_title(f"{title} — Execution time")
    ax1.set_xticks(x)
    ax1.set_xticklabels(methods, rotation=30, ha="right")

    ax2.bar(x, maes, color="tomato", edgecolor="white")
    ax2.set_ylabel(f"MAE vs {ref_label} (€)")
    ax2.set_title(f"{title} — Mean Absolute Error vs {ref_label}")
    ax2.set_xticks(x)
    ax2.set_xticklabels(methods, rotation=30, ha="right")

    plt.tight_layout()
    plt.savefig(REPORTS / filename, dpi=150)
    plt.show()


if __name__ == "__main__":

    data = pd.read_excel(DATA / "options_benchmark.xlsx", sheet_name=0, header=2, index_col=0)
    cols = ['Kind', 'Spot  S', 'Strike  K', 'T  (years)', 'Rate  r (%)', 'Vol  σ (%)', 'Style']
    cleanData = data[cols]
    cleanData_eu = cleanData[cleanData['Style'] != 'American']
    cleanData_am = cleanData[cleanData['Style'] == 'American']

    # References
    print("Computing references...")
    _, ref_bs_eu   = calcPriceWithMethod(cleanData_eu, "BS")        # exact for European
    _, ref_pde_am  = calcPriceWithMethod(cleanData_am, "pde")       # reference for American
    _, ref_bspde   = calcPriceWithMethod(cleanData,    "BS,pde")    # BS(EU) + PDE(AM) for full book
    ref_bs_eu  = np.array(ref_bs_eu,  dtype=float)
    ref_pde_am = np.array(ref_pde_am, dtype=float)
    ref_bspde  = np.array(ref_bspde,  dtype=float)

    # Figure 1 — American options
    print("\n--- American options ---")
    plot_figure(
        title="American options",
        methods=["lsm", "crr", "pde"],
        data_subset=cleanData_am,
        ref=ref_pde_am,
        ref_label="PDE",
        bar_color="tomato",
        filename="benchmark_american.png",
    )

    # Figure 2 — European options
    print("\n--- European options ---")
    plot_figure(
        title="European options",
        methods=["BS", "MC Naive", "MC Anti", "MC Contr", "MC Anti Contr", "crr", "pde"],
        data_subset=cleanData_eu,
        ref=ref_bs_eu,
        ref_label="BS",
        bar_color="steelblue",
        filename="benchmark_european.png",
    )

    # Figure 3 — Full book (both styles) + combinations
    print("\n--- Full book ---")
    plot_figure(
        title="Full book — both styles",
        methods=["crr", "pde", "BS,pde", "BS,crr", "BS,lsm"],
        data_subset=cleanData,
        ref=ref_bspde,
        ref_label="BS+PDE",
        bar_color="seagreen",
        filename="benchmark_fullbook.png",
    )