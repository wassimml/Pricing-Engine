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

N_SEEDS = 10    # répétitions indépendantes par (méthode, paramètre) — cf. run_method
BASE_SEED = 0

# Each entry: pricer(opt, param, style, seed), parameter values, style_filter
# (None=all, 'European', 'American'), ref_kind (quelle référence sert au calcul
# de MAE), stochastic (méthode aléatoire -> répétée sur N_SEEDS seeds
# indépendants pour obtenir moyenne ± écart-type du temps et de la MAE ; CRR
# est déterministe (même arbre binomial à chaque appel) — la répéter ne
# changerait ni le prix ni l'information disponible, un seul passage suffit
# (std = 0 par construction, affiché tel quel plutôt que masqué) :
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
    "MC Naive":     (lambda opt, p, _, seed: mc_naive(opt, n_paths=p, seed=seed)[0],                       [1_000, 5_000, 10_000, 25_000, 50_000, 100_000, 200_000], "European", "BS",                       True),
    "MC Anti":      (lambda opt, p, _, seed: mc_antithetic(opt, n_paths=p, seed=seed)[0],                  [1_000, 5_000, 10_000, 25_000, 50_000, 100_000, 200_000], "European", "BS",                       True),
    "MC Control":   (lambda opt, p, _, seed: mc_control(opt, n_paths=p, seed=seed)[0],                     [1_000, 5_000, 10_000, 25_000, 50_000, 100_000, 200_000], "European", "BS",                       True),
    "MC Anti+Ctrl": (lambda opt, p, _, seed: mc_control_antithetic(opt, n_paths=p, seed=seed)[0],          [1_000, 5_000, 10_000, 25_000, 50_000, 100_000, 200_000], "European", "BS",                       True),
    "CRR":          (lambda opt, p, style, _seed: crr_price(opt, period=p, american=(style == 'American')), [50, 100, 200, 500, 1_000],                              None,       "MIXED BS - PDE (800,800)", False),
    "LSM":          (lambda opt, p, _, seed: LSMoptionValue(opt, n_steps=50, n_paths=p, seed=seed),        [1_000, 5_000, 10_000, 20_000, 25_000, 50_000],          "American", "PDE (800,800)",            True),
}

def run_method(cleanData: pd.DataFrame, pricer, params: list, style_filter, ref: np.ndarray, stochastic: bool,
               n_seeds: int = N_SEEDS, base_seed: int = BASE_SEED):
    """Pour chaque valeur de paramètre, price le book filtré une fois par seed
    (n_seeds seeds indépendants si la méthode est stochastique, 1 seul passage
    sinon — cf. commentaire sur METHODS) et retourne, par paramètre : temps
    total (moyenne ± écart-type sur les seeds) et MAE vs référence (idem).
    Une seule réalisation ne permet pas de distinguer un vrai comportement de
    convergence d'un artefact du tirage particulier (même logique que
    lsm_mean_std / mc_mean_std, appliquée ici au book entier plutôt qu'à une
    option unique)."""
    rows = [r for r in cleanData.itertuples() if style_filter is None or r[7] == style_filter]
    effective_seeds = n_seeds if stochastic else 1
    results = []
    for p in params:
        seed_times, seed_maes = [], []
        for s in range(effective_seeds):
            seed = base_seed + s
            t0 = time.perf_counter()
            prices = np.array([
                pricer(Option(S=row[2], K=row[3], T=row[4], r=row[5]/100, sigma=row[6]/100, kind=row[1].lower()),
                       p, row[7], seed)
                for row in rows
            ], dtype=float)
            elapsed = time.perf_counter() - t0
            mask = ~np.isnan(prices) & ~np.isnan(ref)
            seed_times.append(elapsed)
            seed_maes.append(np.mean(np.abs(prices[mask] - ref[mask])))
        results.append((p, np.mean(seed_times), np.std(seed_times), np.mean(seed_maes), np.std(seed_maes)))
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

    for (label, (pricer, params, style_filter, ref_kind, stochastic)), color in zip(METHODS.items(), colors):
        ref = select_reference(cleanData, style_filter, ref_kind, bs_all, pde_all)
        n_seeds_used = N_SEEDS if stochastic else 1
        print(f"\n--- {label} (référence : {ref_kind}, {n_seeds_used} seed{'s' if n_seeds_used > 1 else ''}) ---")
        xs, ys, zs, y_errs, z_errs = [], [], [], [], []
        for param, t_mean, t_std, mae_mean, mae_std in run_method(cleanData, pricer, params, style_filter, ref, stochastic):
            xs.append(param)
            ys.append(t_mean)
            zs.append(mae_mean)
            y_errs.append(t_std)
            z_errs.append(mae_std)
            print(f"  param={param:,}  time={t_mean:.2f}s±{t_std:.2f}s  MAE={mae_mean:.4f}±{mae_std:.4f}")

        ax.plot(xs, ys, zs, marker="o", label=label, color=color)
        ax.scatter(xs, ys, zs, color=color, s=30)
        # Écart-type sur les seeds -> barres d'erreur en z (MAE) et en y (temps),
        # visibles directement sur le graphe plutôt que seulement dans la console.
        for x, y, z, ye, ze in zip(xs, ys, zs, y_errs, z_errs):
            ax.plot([x, x], [y - ye, y + ye], [z, z], color=color, alpha=0.4, lw=1.2)
            ax.plot([x, x], [y, y], [z - ze, z + ze], color=color, alpha=0.4, lw=1.2)

    ax.set_xlabel("Parameter (n_paths / periods)")
    ax.set_ylabel("Total time (s)")
    ax.set_zlabel("MAE vs référence (€)")
    ax.set_title(f"Accuracy vs speed — MC, CRR, LSM ({N_SEEDS} seeds indépendants, moyenne ± écart-type "
                 "; CRR déterministe -> 1 seed)\n"
                 "(référence : BS pour MC, PDE (800,800) pour LSM, BS+PDE (800,800) selon le style pour CRR)")
    ax.legend()

    plt.tight_layout()
    plt.savefig(REPORTS / "benchmark_methods_3d.png", dpi=150)
    plt.show()