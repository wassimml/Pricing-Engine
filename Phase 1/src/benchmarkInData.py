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


# Distinction importante : "PDE (50,100)" est notre OUTIL DE CALCUL — la
# config rapide utilisée en pratique partout ailleurs dans le projet.
# "PDE (800,800)" est notre OUTIL DE RÉFÉRENCE — construit la vérité terrain
# fiable. Ce sont deux configurations différentes du même schéma numérique :
# comparer "PDE (50,100)" à "PDE (800,800)" (MAE) est donc une comparaison
# valide, pas une tautologie. Les deux apparaissent explicitement, avec leur
# résolution dans le nom, pour ne jamais les confondre dans les prints/graphes
# (contrairement à avant, où le même nom générique "pde" pouvait désigner
# l'une ou l'autre selon l'endroit).
PRICERS = {
    "BS":               lambda opt, style: _bs.price(opt) if style != 'American' else None,
    "crr":              lambda opt, style: crr_price(opt, period=1000, american=(style == 'American')),
    "MC Naive":         lambda opt, style: mc_naive(opt, n_paths=50000)[0] if style != 'American' else None,
    "MC Anti":          lambda opt, style: mc_antithetic(opt, n_paths=100000)[0] if style != 'American' else None,
    "MC Contr":         lambda opt, style: mc_control(opt, n_paths=100000)[0] if style != 'American' else None,
    "MC Anti Contr":    lambda opt, style: mc_control_antithetic(opt, n_paths=50000)[0] if style != 'American' else None,
    "lsm":              lambda opt, style: LSMoptionValue(opt, n_steps=50, n_paths=10000) if style == 'American' else None,
    "PDE (50,100)":     lambda opt, style: pde_crank_nicolson(opt, style=style.lower(), n_steps=50, n_space=100),
    "PDE (800,800)":    lambda opt, style: pde_crank_nicolson(opt, style=style.lower(), n_steps=800, n_space=800),
    "BS,PDE (50,100)":  lambda opt, style: _bs.price(opt) if style != 'American' else pde_crank_nicolson(opt, style=style.lower(), n_steps=50, n_space=100),
    "BS,PDE (800,800)": lambda opt, style: _bs.price(opt) if style != 'American' else pde_crank_nicolson(opt, style=style.lower(), n_steps=800, n_space=800),
    "BS,crr":           lambda opt, style: _bs.price(opt) if style != 'American' else crr_price(opt, period=1000, american=True),
    "BS,lsm":           lambda opt, style: _bs.price(opt) if style != 'American' else LSMoptionValue(opt, n_steps=50, n_paths=20000),
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


def plot_figure(title, methods, data_subset, ref, ref_label, bar_color, filename, mae_methods=None):
    # mae_methods : sous-ensemble de `methods` à afficher sur le graphe de MAE.
    # Par défaut = methods, mais on exclut la méthode qui a servi à calculer
    # `ref` elle-même (sinon sa MAE est trivialement 0, comparaison inutile).
    if mae_methods is None:
        mae_methods = methods

    times = {}
    maes = {}
    for method in methods:
        method_times, prices = calcPriceWithMethod(data_subset, method)
        times[method] = sum(method_times)
        if method in mae_methods:
            p = np.array(prices, dtype=float)
            mask = ~np.isnan(p) & ~np.isnan(ref)
            maes[method] = np.mean(np.abs(p[mask] - ref[mask]))
            print(f"  {method}: total={times[method]:.2f}s  MAE={maes[method]:.4f}")
        else:
            print(f"  {method}: total={times[method]:.2f}s  (référence — MAE non affichée)")

    x_time = np.arange(len(methods))
    x_mae  = np.arange(len(mae_methods))
    _, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

    ax1.bar(x_time, [times[m] for m in methods], color=bar_color, edgecolor="white")
    ax1.set_ylabel("Total time (s)")
    ax1.set_title(f"{title} — Execution time")
    ax1.set_xticks(x_time)
    ax1.set_xticklabels(methods, rotation=30, ha="right")

    ax2.bar(x_mae, [maes[m] for m in mae_methods], color="tomato", edgecolor="white")
    ax2.set_ylabel(f"MAE vs {ref_label} (€)")
    ax2.set_title(f"{title} — Mean Absolute Error vs {ref_label}")
    ax2.set_xticks(x_mae)
    ax2.set_xticklabels(mae_methods, rotation=30, ha="right")

    plt.tight_layout()
    plt.savefig(REPORTS / filename, dpi=150)
    plt.show()


if __name__ == "__main__":

    data = pd.read_excel(DATA / "options_benchmark.xlsx", sheet_name=0, header=2, index_col=0)
    cols = ['Kind', 'Spot  S', 'Strike  K', 'T  (years)', 'Rate  r (%)', 'Vol  σ (%)', 'Style']
    cleanData = data[cols]
    cleanData_eu = cleanData[cleanData['Style'] != 'American']
    cleanData_am = cleanData[cleanData['Style'] == 'American']

    # References — toujours construites avec l'outil de référence (n=800),
    # jamais avec l'outil de calcul (n=50/100) : cf. commentaire au-dessus de
    # PRICERS. Les noms "PDE (50,100)" / "PDE (800,800)" rendent la résolution
    # explicite partout, plus de nom générique ambigu "pde".
    print("Computing references (outil de reference : PDE n_steps=800, n_space=800)...")
    _, ref_bs_eu   = calcPriceWithMethod(cleanData_eu, "BS")                 # exact for European
    _, ref_pde_am  = calcPriceWithMethod(cleanData_am, "PDE (800,800)")      # reference for American
    _, ref_bspde   = calcPriceWithMethod(cleanData,    "BS,PDE (800,800)")   # BS(EU) + PDE ref(AM) for full book
    ref_bs_eu  = np.array(ref_bs_eu,  dtype=float)
    ref_pde_am = np.array(ref_pde_am, dtype=float)
    ref_bspde  = np.array(ref_bspde,  dtype=float)

    # Figure 1 — American options (référence = PDE (800,800)).
    # "PDE (50,100)" (outil de calcul) ET "PDE (800,800)" (outil de référence)
    # sont TOUS LES DEUX affichés dans le graphe de temps, pour qu'on voie
    # explicitement le coût de chacun. "PDE (800,800)" reste exclu du graphe
    # de MAE (comparaison à elle-même, trivialement nulle) ; "PDE (50,100)"
    # y est inclus car ce n'est pas le même calcul que la référence — erreur
    # réelle et informative de la config rapide utilisée en pratique.
    print("\n--- American options ---")
    plot_figure(
        title="American options",
        methods=["lsm", "crr", "PDE (50,100)", "PDE (800,800)"],
        mae_methods=["lsm", "crr", "PDE (50,100)"],
        data_subset=cleanData_am,
        ref=ref_pde_am,
        ref_label="PDE ref (n_steps=800, n_space=800)",
        bar_color="tomato",
        filename="benchmark_american.png",
    )

    # Figure 2 — European options (référence = BS -> exclu du graphe MAE)
    print("\n--- European options ---")
    plot_figure(
        title="European options",
        methods=["BS", "MC Naive", "MC Anti", "MC Contr", "MC Anti Contr", "crr", "PDE (50,100)"],
        mae_methods=["MC Naive", "MC Anti", "MC Contr", "MC Anti Contr", "crr", "PDE (50,100)"],
        data_subset=cleanData_eu,
        ref=ref_bs_eu,
        ref_label="BS",
        bar_color="steelblue",
        filename="benchmark_european.png",
    )

    # Figure 3 — Full book (both styles) + combinations (référence = BS,PDE (800,800)).
    # "BS,PDE (50,100)" (outil de calcul, PDE 50/100 pour le volet américain)
    # ET "BS,PDE (800,800)" (outil de référence) sont tous les deux affichés
    # dans le graphe de temps, pour voir explicitement le coût de chacun.
    # "BS,PDE (800,800)" reste exclu du graphe de MAE (comparaison à
    # elle-même, trivialement nulle). Attention à l'interprétation du temps
    # d'exécution : "BS,PDE (50,100)" doit être rapide ; si un graphe montre
    # une combinaison "BS+PDE" très lente sans préciser (50,100) ou (800,800)
    # en légende, c'est le signe d'une confusion entre les deux configs.
    print("\n--- Full book ---")
    plot_figure(
        title="Full book — both styles",
        methods=["crr", "PDE (50,100)", "BS,PDE (50,100)", "BS,PDE (800,800)", "BS,crr", "BS,lsm"],
        mae_methods=["crr", "PDE (50,100)", "BS,PDE (50,100)", "BS,crr", "BS,lsm"],
        data_subset=cleanData,
        ref=ref_bspde,
        ref_label="BS + PDE ref (n_steps=800, n_space=800)",
        bar_color="seagreen",
        filename="benchmark_fullbook.png",
    )