from pathlib import Path
import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
import time 

# Import all methods
from option import Option
from binomial import crr_price_fast
from monteCarlo import mc_naive_threaded, mc_antithetic_threaded, mc_control_threaded, mc_control_antithetic_threaded
from monteCarloLSM import LSMoptionValue_parallel
from pde import pde_crank_nicolson_auto
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
# FAST_PRICERS : array-based (cf. monteCarlo.py / binomial.py / BSpricer.py)
# ou parallélisé multi-process (LSM, PDE (800,800) — cf. monteCarloLSM.py /
# pde.py) — tout `cleanData` pricé en un seul appel par méthode, jamais de
# boucle Python par ligne. BS et les 4 MC sont toujours appelées ici sur un
# data_subset déjà filtré (cleanData_eu) — jamais de lignes américaines à
# exclure. "crr"/"BS,crr"/"lsm"/"BS,lsm"/"PDE (800,800)"/"BS,PDE (800,800)"
# en revanche sont aussi appelées sur des books mixtes (cleanData_am,
# cleanData complet) : `american` (dérivé de la colonne Style au moment de
# l'appel, cf. calcPriceWithMethod) leur dit, ligne par ligne, quel style
# appliquer.
def _lsm_prices(S, K, T, r, sigma, kind, american):
    """lsm n'a de sens que sur les lignes américaines (early exercise) ; NaN
    ailleurs, comme le faisait l'ancienne version row-based (`else None`).
    n_paths=5000 : paramètre retenu sous contrainte de latence 10s sur un
    book de 2000 options (cf. tableau de sélection des paramètres)."""
    prices = np.full(len(S), np.nan)
    if american.any():
        prices[american] = LSMoptionValue_parallel(
            S[american], K[american], T[american], r[american], sigma[american], kind[american],
            n_steps=50, n_paths=5000, seed=42, n_workers=8)
    return prices

def _bs_lsm_prices(S, K, T, r, sigma, kind, american):
    """BS pour les lignes européennes, LSM (parallélisé) pour les américaines."""
    prices = _bs.price_batch(S, K, T, r, sigma, kind)
    if american.any():
        prices = prices.copy()
        prices[american] = LSMoptionValue_parallel(
            S[american], K[american], T[american], r[american], sigma[american], kind[american],
            n_steps=50, n_paths=5000, seed=42, n_workers=8)
    return prices

# PDE (50,100) [outil de CALCUL] et PDE (800,800) [outil de RÉFÉRENCE] passent
# tous les deux par pde_crank_nicolson_auto, qui MESURE le coût réel et décide
# séquentiel vs parallèle en conséquence (~0.5 ms/option pour (50,100) ->
# reste séquentiel ; ~29 ms/option pour (800,800) -> parallélise, 4.99x à 16
# workers sur le book complet) — pas un cas particulier codé en dur pour ces
# deux résolutions précises : n'importe quel (n_steps, n_space) passerait par
# la même logique de décision.
def _pde_prices(n_steps, n_space):
    def _inner(S, K, T, r, sigma, kind, american):
        return pde_crank_nicolson_auto(S, K, T, r, sigma, kind, american, n_steps=n_steps, n_space=n_space)
    return _inner

def _bs_pde_prices(n_steps, n_space):
    def _inner(S, K, T, r, sigma, kind, american):
        prices = _bs.price_batch(S, K, T, r, sigma, kind)
        if american.any():
            prices = prices.copy()
            prices[american] = pde_crank_nicolson_auto(
                S[american], K[american], T[american], r[american], sigma[american], kind[american],
                american[american], n_steps=n_steps, n_space=n_space)
        return prices
    return _inner

FAST_PRICERS = {
    # Paramètres retenus sous contrainte de latence 10s sur un book de 2000
    # options (cf. tableau de sélection des paramètres, book PARAMS).
    "BS":            lambda S, K, T, r, sigma, kind, american: _bs.price_batch(S, K, T, r, sigma, kind),
    "MC Naive":      lambda S, K, T, r, sigma, kind, american: mc_naive_threaded(S, K, T, r, sigma, kind, n_paths=100000, seed=42, n_workers=8)[0],
    "MC Anti":       lambda S, K, T, r, sigma, kind, american: mc_antithetic_threaded(S, K, T, r, sigma, kind, n_paths=200000, seed=42, n_workers=8)[0],
    "MC Contr":      lambda S, K, T, r, sigma, kind, american: mc_control_threaded(S, K, T, r, sigma, kind, n_paths=200000, seed=42, n_workers=8)[0],
    "MC Anti Contr": lambda S, K, T, r, sigma, kind, american: mc_control_antithetic_threaded(S, K, T, r, sigma, kind, n_paths=200000, seed=42, n_workers=8)[0],
    "crr":           lambda S, K, T, r, sigma, kind, american: crr_price_fast(S, K, T, r, sigma, kind, american, period=500),
    "BS,crr":        lambda S, K, T, r, sigma, kind, american: np.where(american, crr_price_fast(S, K, T, r, sigma, kind, american, period=500), _bs.price_batch(S, K, T, r, sigma, kind)),
    "lsm":           _lsm_prices,
    "BS,lsm":        _bs_lsm_prices,
    "PDE (50,100)":     _pde_prices(50, 100),
    "PDE (800,800)":    _pde_prices(800, 800),
    "BS,PDE (50,100)":  _bs_pde_prices(50, 100),
    "BS,PDE (800,800)": _bs_pde_prices(800, 800),
}

# ROW_PRICERS : Option-based, boucle Python par ligne — vide pour l'instant
# (tout a un équivalent vectorisé/parallélisé/auto), conservé comme point
# d'extension si une future méthode n'en avait pas.
ROW_PRICERS = {}

def calcPriceWithMethod(cleanData: pd.DataFrame, method: str):
    """Retourne (temps_total, prices). Dispatch vers FAST_PRICERS (un seul
    appel vectorisé sur tout data_subset) si possible, sinon ROW_PRICERS
    (boucle Python par ligne)."""
    if method in FAST_PRICERS:
        S = cleanData['Spot  S'].to_numpy(); K = cleanData['Strike  K'].to_numpy(); T = cleanData['T  (years)'].to_numpy()
        r = (cleanData['Rate  r (%)'] / 100).to_numpy(); sigma = (cleanData['Vol  σ (%)'] / 100).to_numpy()
        kind = cleanData['Kind'].str.lower().to_numpy()
        american = (cleanData['Style'] == 'American').to_numpy()
        t0 = time.perf_counter()
        prices = FAST_PRICERS[method](S, K, T, r, sigma, kind, american)
        return time.perf_counter() - t0, prices

    pricer = ROW_PRICERS[method]
    times, prices = [], []
    for row in cleanData.itertuples():
        option = Option(S=row[2], K=row[3], T=row[4], r=row[5]/100, sigma=row[6]/100, kind=row[1].lower())
        t0 = time.perf_counter()
        p = pricer(option, row[7])
        times.append(time.perf_counter() - t0)
        prices.append(p)
    return sum(times), np.array(prices, dtype=float)


def plot_figure(title, methods, data_subset, ref, ref_label, bar_color, filename, mae_methods=None):
    # mae_methods : sous-ensemble de `methods` à afficher sur le graphe de MAE.
    # Par défaut = methods, mais on exclut la méthode qui a servi à calculer
    # `ref` elle-même (sinon sa MAE est trivialement 0, comparaison inutile).
    if mae_methods is None:
        mae_methods = methods

    times = {}
    maes = {}
    for method in methods:
        total_time, prices = calcPriceWithMethod(data_subset, method)
        times[method] = total_time
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

    bars1 = ax1.bar(x_time, [times[m] for m in methods], color=bar_color, edgecolor="white")
    ax1.set_ylabel("Total time (s)")
    ax1.set_title(f"{title} — Execution time")
    ax1.set_xticks(x_time)
    ax1.set_xticklabels(methods, rotation=30, ha="right")
    ax1.margins(y=0.12)
    ax1.bar_label(bars1, fmt="%.2fs", padding=3, fontsize=8)

    bars2 = ax2.bar(x_mae, [maes[m] for m in mae_methods], color="tomato", edgecolor="white")
    ax2.set_ylabel(f"MAE vs {ref_label} (€)")
    ax2.set_title(f"{title} — Mean Absolute Error vs {ref_label}")
    ax2.set_xticks(x_mae)
    ax2.set_xticklabels(mae_methods, rotation=30, ha="right")
    ax2.margins(y=0.12)
    ax2.bar_label(bars2, fmt="%.4f", padding=3, fontsize=8)

    plt.tight_layout()
    plt.savefig(REPORTS / filename, dpi=150)
    plt.show()


if __name__ == "__main__":

    # Book TEST : évalue les méthodes avec des paramètres fixes (choisis à la
    # main à partir du balayage de benchmarkMethods.py, cf. book PARAMS) sur
    # des données jamais vues pendant la recherche de paramètres — condition
    # nécessaire pour que la MAE mesurée ici reflète une vraie généralisation,
    # pas un surapprentissage au book utilisé pour calibrer.
    data = pd.read_excel(DATA / "2_options_book_test_2026-07-19.xlsx", sheet_name=0, header=2, index_col=0)
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