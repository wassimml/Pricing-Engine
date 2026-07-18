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
from BSpricer import BSModel

DATA = Path(__file__).parent.parent / "data"
REPORTS = Path(__file__).parent.parent / "reports"

_bs = BSModel()

# Each entry: pricer(opt, param), parameter values, style_filter (None=all,
# 'European', 'American'), ref_kind (quelle référence sert au calcul de MAE) :
#   "BS"    : Black-Scholes fermé — pour les méthodes qui ne pricent que du
#             européen (Monte Carlo).
#   "PDE (800,800)"   : PDE Crank-Nicolson au style américain, outil de
#             RÉFÉRENCE (n_steps=800, n_space=800) — pour LSM (intrinsèquement
#             américain, pas de forme fermée disponible). Résolution
#             explicite dans le nom pour ne jamais la confondre avec l'outil
#             de calcul PDE (50,100) utilisé ailleurs dans le projet.
#   "MIXED BS - PDE (800,800)" : BS pour les lignes européennes, PDE (800,800,
#             américain) pour les lignes américaines — pour CRR, qui price
#             les deux styles à la fois.
METHODS = {
    "MC Naive":     (lambda opt, p, _: mc_naive(opt, n_paths=p)[0],                               [1_000, 5_000, 10_000, 25_000, 50_000, 100_000, 200_000], "European", "BS"),
    "MC Anti":      (lambda opt, p, _: mc_antithetic(opt, n_paths=p)[0],                          [1_000, 5_000, 10_000, 25_000, 50_000, 100_000, 200_000], "European", "BS"),
    "MC Control":   (lambda opt, p, _: mc_control(opt, n_paths=p)[0],                             [1_000, 5_000, 10_000, 25_000, 50_000, 100_000, 200_000], "European", "BS"),
    "MC Anti+Ctrl": (lambda opt, p, _: mc_control_antithetic(opt, n_paths=p)[0],                  [1_000, 5_000, 10_000, 25_000, 50_000, 100_000, 200_000], "European", "BS"),
    "CRR":          (lambda opt, p, style: crr_price(opt, period=p, american=(style == 'American')), [50, 100, 200, 500, 1_000],                              None,       "MIXED BS - PDE (800,800)"),
    "LSM":          (lambda opt, p, _: LSMoptionValue(opt, n_steps=50, n_paths=p),                [1_000, 5_000, 10_000, 20_000, 25_000, 50_000],                    "American", "PDE (800,800)"),
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

def compute_reference_prices(cleanData: pd.DataFrame):
    """BS (fermé, toujours calculé) et PDE (au style réel de chaque ligne,
    outil de RÉFÉRENCE n_steps=800/n_space=800 — jamais l'outil de calcul
    50/100), calculés une seule fois pour tout le book — servent ensuite à
    construire la référence de chaque méthode (BS, PDE, ou mixte) sans
    recalcul."""
    bs_prices, pde_prices = [], []
    for row in cleanData.itertuples():
        option = Option(S=row[2], K=row[3], T=row[4], r=row[5]/100, sigma=row[6]/100, kind=row[1].lower())
        bs_prices.append(_bs.price(option))
        pde_prices.append(pde_crank_nicolson(option, style=row[7].lower(), n_steps=800, n_space=800))
    return np.array(bs_prices, dtype=float), np.array(pde_prices, dtype=float)

def select_reference(cleanData: pd.DataFrame, style_filter, ref_kind, bs_all, pde_all):
    """Sélectionne/assemble la référence d'une méthode, alignée sur le même
    filtrage de lignes (style_filter) que run_method."""
    styles = cleanData["Style"].to_numpy()
    mask = np.ones(len(styles), dtype=bool) if style_filter is None else (styles == style_filter)
    if ref_kind == "BS":
        ref = bs_all
    elif ref_kind == "PDE (800,800)":
        ref = pde_all
    elif ref_kind == "MIXED BS - PDE (800,800)":
        ref = np.where(styles == "European", bs_all, pde_all)
    else:
        raise ValueError(f"ref_kind inconnu : {ref_kind}")
    return ref[mask]


if __name__ == "__main__":

    data = pd.read_excel(DATA / "options_benchmark.xlsx", sheet_name=0, header=2, index_col=0)
    cleanData = data[['Kind', 'Spot  S', 'Strike  K', 'T  (years)', 'Rate  r (%)', 'Vol  σ (%)', 'Style']]

    print("Computing BS and PDE (800,800) reference prices (once, per-row true style)...")
    bs_all, pde_all = compute_reference_prices(cleanData)

    fig = plt.figure(figsize=(13, 8))
    ax = fig.add_subplot(111, projection="3d")
    colors = ["steelblue", "tomato", "seagreen", "darkorange", "mediumpurple", "saddlebrown"]

    for (label, (pricer, params, style_filter, ref_kind)), color in zip(METHODS.items(), colors):
        ref = select_reference(cleanData, style_filter, ref_kind, bs_all, pde_all)
        print(f"\n--- {label} (référence : {ref_kind}) ---")
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
    ax.set_zlabel("MAE vs référence (€)")
    ax.set_title("Accuracy vs speed — MC, CRR, LSM\n"
                 "(référence : BS pour MC, PDE (800,800) pour LSM, BS+PDE (800,800) selon le style pour CRR)")
    ax.legend()

    plt.tight_layout()
    plt.savefig(REPORTS / "benchmark_methods_3d.png", dpi=150)
    plt.show()