from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

import pandas as pd
import time

from option import Option
from binomial import crr_price
from monteCarlo import mc_naive, mc_antithetic, mc_control, mc_control_antithetic
from monteCarloLSM import LSMoptionValue
from pde import pde_crank_nicolson

DATA = Path(__file__).parent.parent / "data"
REPORTS = Path(__file__).parent.parent / "reports"

# Each entry: pricer(opt, param), parameter values, style_filter (None=all, 'European', 'American')
METHODS = {
    "MC Naive":     (lambda opt, p, _: mc_naive(opt, n_paths=p)[0],                               [1_000, 5_000, 10_000, 25_000, 50_000, 100_000, 200_000], "European"),
    "MC Anti":      (lambda opt, p, _: mc_antithetic(opt, n_paths=p)[0],                          [1_000, 5_000, 10_000, 25_000, 50_000, 100_000, 200_000], "European"),
    "MC Control":   (lambda opt, p, _: mc_control(opt, n_paths=p)[0],                             [1_000, 5_000, 10_000, 25_000, 50_000, 100_000, 200_000], "European"),
    "MC Anti+Ctrl": (lambda opt, p, _: mc_control_antithetic(opt, n_paths=p)[0],                  [1_000, 5_000, 10_000, 25_000, 50_000, 100_000, 200_000], "European"),
    "CRR":          (lambda opt, p, style: crr_price(opt, period=p, american=(style == 'American')), [50, 100, 200, 500, 1_000],                              None),
    "LSM":          (lambda opt, p, _: LSMoptionValue(opt, n_steps=50, n_paths=p),                [1_000, 5_000, 10_000, 25_000, 50_000],                    "American"),
}

def run_method(cleanData: pd.DataFrame, pricer, params: list, style_filter):
    rows = [r for r in cleanData.itertuples() if style_filter is None or r[7] == style_filter]
    results = []
    for p in params:
        times, prices = [], []
        for row in rows:
            option = Option(S=row[2], K=row[3], T=row[4], r=row[5]/100, sigma=row[6]/100, kind=row[1].lower())
            t0 = time.perf_counter()
            price = pricer(option, p, row[7])
            times.append(time.perf_counter() - t0)
            prices.append(price)
        results.append((p, sum(times), np.array(prices, dtype=float)))
    return results

def run_pde_filtered(cleanData: pd.DataFrame, style_filter):
    prices = []
    for row in cleanData.itertuples():
        if style_filter is not None and row[7] != style_filter:
            continue
        option = Option(S=row[2], K=row[3], T=row[4], r=row[5]/100, sigma=row[6]/100, kind=row[1].lower())
        prices.append(pde_crank_nicolson(option, style=row[7].lower(), n_steps=200, n_space=200))
    return np.array(prices, dtype=float)


if __name__ == "__main__":

    data = pd.read_excel(DATA / "options_benchmark.xlsx", sheet_name=0, header=2, index_col=0)
    cleanData = data[['Kind', 'Spot  S', 'Strike  K', 'T  (years)', 'Rate  r (%)', 'Vol  σ (%)', 'Style']]

    refs = {}
    for sf in [None, "European", "American"]:
        label = sf or "all"
        print(f"Computing PDE reference ({label})...")
        refs[sf] = run_pde_filtered(cleanData, sf)

    fig = plt.figure(figsize=(13, 8))
    ax = fig.add_subplot(111, projection="3d")
    colors = ["steelblue", "tomato", "seagreen", "darkorange", "mediumpurple", "saddlebrown"]

    for (label, (pricer, params, style_filter)), color in zip(METHODS.items(), colors):
        ref = refs[style_filter]
        print(f"\n--- {label} ---")
        xs, ys, zs = [], [], []
        for param, total_time, prices in run_method(cleanData, pricer, params, style_filter):
            mask = ~np.isnan(prices) & ~np.isnan(ref)
            mae = np.mean(np.abs(prices[mask] - ref[mask]))
            xs.append(param)
            ys.append(total_time)
            zs.append(mae)
            print(f"  param={param:,}  time={total_time:.2f}s  MAE={mae:.4f}")

        ax.plot(xs, ys, zs, marker="o", label=label, color=color)
        ax.scatter(xs, ys, zs, color=color, s=30)

    ax.set_xlabel("Parameter (n_paths / periods)")
    ax.set_ylabel("Total time (s)")
    ax.set_zlabel("MAE vs PDE (€)")
    ax.set_title("Accuracy vs speed — MC, CRR, LSM")
    ax.legend()

    plt.tight_layout()
    plt.savefig(REPORTS / "benchmark_methods_3d.png", dpi=150)
    plt.show()