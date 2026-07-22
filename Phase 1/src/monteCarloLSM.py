from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from concurrent.futures import ProcessPoolExecutor
from scipy import stats

from option import Option
from pde import pde_crank_nicolson, pde_crank_nicolson_auto

DATA = Path(__file__).parent.parent / "data"
REPORTS = Path(__file__).parent.parent / "reports"
PARAMS_BOOK = DATA / "1_options_book_params_2026-06-29.xlsx"
PARAM_GRID_CACHE = DATA / "lsm_param_grid.npz"


def LSMoptionValue(option, n_steps=50, n_paths=4096, seed=42):
    rng = np.random.default_rng(seed)
    dt = option.T / n_steps  # time interval
    df = np.exp(-option.r * dt)  # discount factor per time time interval

    # Simulation of Index Levels
    S = np.zeros((n_steps + 1, n_paths), 'd')  # stock price matrix
    S[0, :] = option.S  # initial values for stock price
    for t in range(1, n_steps + 1):
        ran = rng.standard_normal(int(n_paths / 2))
        ran = np.concatenate((ran, -ran))  # antithetic variates
        ran = ran - np.mean(ran)  # correct first moment
        ran = ran / np.std(ran)  # correct second moment
        S[t, :] = S[t - 1, :] * np.exp((option.r - option.sigma ** 2 / 2) * dt
                        + option.sigma * ran * np.sqrt(dt))

    if option.kind == 'call':
        h = np.maximum(S - option.K, 0)  # inner values for call option
    else:
        h = np.maximum(option.K - S, 0)  # inner values for put option

    V = np.zeros_like(h)  # value matrix
    V[-1] = h[-1]

    # Valuation by LSM
    for t in range(n_steps - 1, 0, -1):
        S_t = S[t, :]
        h_t = h[t, :]
        itm = h_t > 0  # uniquement les chemins in-the-money

        V[t, :] = V[t + 1, :] * df  # par défaut : continuation pour tous

        if np.sum(itm) < 10:  # trop peu de chemins ITM pour la régression
            continue

        S_itm = S_t[itm]
        S_mean, S_std = S_itm.mean(), S_itm.std()

        if S_std < 1e-10:  # cas dégénéré : tous les chemins ITM au même niveau
            continue

        S_norm = (S_itm - S_mean) / S_std

        rg = np.polyfit(S_norm, V[t + 1, itm] * df, 3)  # degré 3, standard LSM
        C  = np.polyval(rg, S_norm)

        exercise     = h_t[itm] > C
        idx_itm      = np.where(itm)[0]
        idx_exercise = idx_itm[exercise]
        V[t, idx_exercise] = h_t[idx_exercise]

    V0 = np.sum(V[1, :] * df) / n_paths  # LSM estimator
    return V0


# ---------------------------------------------------------------------------
# Version parallèle : LSM est intrinsèquement séquentiel PAR OPTION (chaque
# pas de temps a besoin de la régression du pas suivant) et doit garder tout
# l'historique des trajectoires en mémoire (n_options x n_steps x n_paths) —
# contrairement à CRR (largeur bornée par `period`, jamais plus de ~16 Mo),
# une vectorisation cross-options exploserait la RAM aux paramètres réalistes
# du sweep (ex. n_steps=200, n_paths=50000, ~900 options américaines ->
# ~160 Go). Le vrai levier ici : chaque option est indépendante des autres,
# donc parallélisable sur plusieurs cœurs CPU plutôt que vectorisée.
#
# Sur Windows, multiprocessing utilise spawn (pas fork) : chaque process
# redémarre avec un coût réel, donc on découpe le book en `n_workers` LOTS
# (un par cœur) plutôt qu'un process par option, pour amortir ce coût.
# ---------------------------------------------------------------------------

def _lsm_chunk(args):
    """Worker : price séquentiellement un lot d'options en LSM (même code que
    LSMoptionValue). Doit être une fonction top-level (picklable) pour
    fonctionner avec ProcessPoolExecutor sous Windows (spawn)."""
    S, K, T, r, sigma, kind, n_steps, n_paths, seed = args
    prices = np.empty(len(S))
    for i in range(len(S)):
        opt = Option(S=S[i], K=K[i], T=T[i], r=r[i], sigma=sigma[i], kind=kind[i])
        prices[i] = LSMoptionValue(opt, n_steps=n_steps, n_paths=n_paths, seed=seed)
    return prices


def LSMoptionValue_parallel(S, K, T, r, sigma, kind, n_steps=50, n_paths=4096, seed=42, n_workers=8):
    """Price tout un book en LSM en parallélisant sur plusieurs process.
    S/K/T/r/sigma/kind : array-like, shape (n_options,). Retourne un array de
    prix, shape (n_options,). À appeler uniquement sous une garde
    `if __name__ == "__main__":` (contrainte de multiprocessing sous
    Windows).

    n_workers=8 par défaut (pas os.cpu_count()) : sous Windows, spawn
    réimporte tout le script __main__ appelant (matplotlib, pandas, QuantLib
    inclus via ce module) dans CHAQUE process — à 16-20 workers simultanés,
    ça peut épuiser le fichier de pagination ("insufficient paging file")
    même sur une machine avec beaucoup de cœurs. 8 workers donne encore
    l'essentiel du gain (3.26x mesuré vs 3.71x à 16) avec une pression
    mémoire deux fois moindre."""
    S, K, T, r, sigma = (np.asarray(x, dtype=float) for x in (S, K, T, r, sigma))
    kind = np.asarray(kind)
    n = len(S)
    n_workers = min(n_workers, n)

    bounds = np.linspace(0, n, n_workers + 1, dtype=int)
    chunks = [
        (S[bounds[i]:bounds[i+1]], K[bounds[i]:bounds[i+1]], T[bounds[i]:bounds[i+1]],
         r[bounds[i]:bounds[i+1]], sigma[bounds[i]:bounds[i+1]], kind[bounds[i]:bounds[i+1]],
         n_steps, n_paths, seed)
        for i in range(n_workers) if bounds[i + 1] > bounds[i]
    ]

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        results = list(executor.map(_lsm_chunk, chunks))

    return np.concatenate(results)


def lsm_mean_std(option, n_steps, n_paths, n_seeds, base_seed=0):
    """Moyenne et écart-type du prix LSM sur n_seeds tirages indépendants.

    Une seule réalisation (seed fixe) ne permet pas de distinguer un vrai
    comportement de convergence d'un artefact du tirage aléatoire particulier
    — on répète donc chaque configuration sur plusieurs seeds indépendants.
    """
    prices = np.array([
        LSMoptionValue(option, n_steps=n_steps, n_paths=n_paths, seed=base_seed + s)
        for s in range(n_seeds)
    ])
    return prices.mean(), prices.std()


# ---------------------------------------------------------------------------
# Recherche jointe des paramètres (n_steps x n_paths) - corrige 3 problèmes de
# l'ancienne exploration (grille croisée sur une option unique, one-at-a-time
# n_paths puis n_steps séparément) :
#
# 1. One-at-a-time, pas une vraie recherche jointe. Fixer n_steps puis
#    balayer n_paths (et inversement) ne peut pas détecter d'interaction
#    entre les deux paramètres. Or une interaction est plausible ici : à
#    chaque étape de temps, LSM régresse la valeur de continuation sur les
#    chemins in-the-money avec le MÊME n_paths (cf. LSMoptionValue ci-dessus)
#    - plus n_steps augmente, plus il y a d'étapes de régression successives
#    sur la même quantité de chemins, donc plus de risque d'accumulation de
#    bruit de régression si n_paths est trop petit pour le nombre d'étapes.
#    Vérifié empiriquement : à n_paths=50000, la MAE descend puis REMONTE
#    quand n_steps dépasse 50 (pas juste un rendement décroissant - une vraie
#    dégradation, à la fois plus lente et moins précise).
# 2. Une seule option (S=K=100, put ATM) - risque d'overfitting : un optimum
#    trouvé sur cette option n'a aucune garantie de généraliser à un book
#    divers. Ici la grille est évaluée sur un ÉCHANTILLON STRATIFIÉ du book
#    PARAMS (options américaines, stratifié par segment de moneyness x
#    tranche de maturité).
# 3. LSMoptionValue séquentielle - trop lente pour un vrai grid 2D x book x
#    seeds. Ici on utilise LSMoptionValue_parallel (parallélisée sur les
#    options, 8 workers).
# ---------------------------------------------------------------------------

def _derive_seed_grid(base_seed: int, n_steps: int, n_paths: int, s: int) -> int:
    """Seed reproductible mais indépendant par (n_steps, n_paths, s) - même
    principe que _derive_seed dans benchmarkMethods.py, étendu à DEUX
    paramètres puisque la grille est jointe (2D) plutôt qu'un seul paramètre
    balayé."""
    return int(np.random.SeedSequence([base_seed, n_steps, n_paths, s]).generate_state(1)[0])


def load_stratified_sample(n_per_stratum: int = 6, seed: int = 42) -> pd.DataFrame:
    """Échantillon stratifié (segment de moneyness x tranche de maturité) du
    book PARAMS, options américaines seulement - évite de calibrer les
    paramètres LSM sur une seule option (cf. point 2 ci-dessus)."""
    data = pd.read_excel(PARAMS_BOOK, sheet_name=0, header=2, index_col=0)
    am = data[data['Style'] == 'American'].copy()

    S = am['Spot  S']; K = am['Strike  K']
    m_ratio = K / S
    kind = am['Kind'].str.lower()
    seg = pd.Series('OTM', index=am.index)
    seg[(m_ratio >= 0.97) & (m_ratio <= 1.03)] = 'ATM'
    seg[(kind == 'call') & (m_ratio < 0.97)] = 'ITM'
    seg[(kind == 'put') & (m_ratio > 1.03)] = 'ITM'
    am['segment'] = seg
    am['T_bucket'] = pd.qcut(am['T  (years)'], q=3, labels=['court', 'moyen', 'long'], duplicates='drop')

    sample = (am.groupby(['segment', 'T_bucket'], observed=True, group_keys=False)
                .apply(lambda g: g.sample(min(len(g), n_per_stratum), random_state=seed)))

    return pd.DataFrame({
        'S': sample['Spot  S'].to_numpy(),
        'K': sample['Strike  K'].to_numpy(),
        'T': sample['T  (years)'].to_numpy(),
        'r': (sample['Rate  r (%)'] / 100).to_numpy(),
        'sigma': (sample['Vol  σ (%)'] / 100).to_numpy(),
        'kind': sample['Kind'].str.lower().to_numpy(),
    })


def pareto_frontier(points):
    """points : itérable de (label, temps, MAE). Retourne les points NON
    dominés (aucun autre point n'est à la fois aussi rapide ET aussi précis,
    strictement meilleur sur au moins un des deux) - ce n'est pas "LE"
    meilleur point, juste la liste des choix qui ne sont jamais objectivement
    mauvais (contrairement à un point dominé, où un autre existe qui fait
    mieux sur les deux axes à la fois)."""
    pts = list(points)
    frontier = [
        p for p in pts
        if not any(q is not p and q[1] <= p[1] and q[2] <= p[2] and (q[1] < p[1] or q[2] < p[2]) for q in pts)
    ]
    return sorted(frontier, key=lambda p: p[1])


def _welch_test(m1, s1, n1, m2, s2, n2):
    """Test de Welch (t-test à variances inégales) à partir de statistiques
    résumées (moyenne, écart-type, n) plutôt que des données brutes - suffit
    ici puisqu'on n'a stocké que mean/std par cellule de la grille."""
    se = np.sqrt(s1 ** 2 / n1 + s2 ** 2 / n2)
    if se == 0:
        return np.inf, 0.0
    t = (m1 - m2) / se
    df = (s1 ** 2 / n1 + s2 ** 2 / n2) ** 2 / ((s1 ** 2 / n1) ** 2 / (n1 - 1) + (s2 ** 2 / n2) ** 2 / (n2 - 1))
    p = 2 * stats.t.sf(abs(t), df)
    return t, p


def indifference_region(cells, n_seeds: int, alpha: float = 0.05):
    """cells : liste de (label, temps, MAE moyenne, écart-type MAE). Avec peu
    de seeds (N_SEEDS_GRID=10), le minimum ponctuel de MAE peut n'être qu'un
    creux de bruit Monte Carlo parmi des cellules statistiquement
    indiscernables (même logique que la power analysis sur la non-
    monotonicité n_paths) - retourne toutes les cellules PAS significativement
    pires que le minimum observé (Welch + correction Holm, moins conservatrice
    que Bonferroni à FWER égal)."""
    best = min(cells, key=lambda c: c[2])
    others = [c for c in cells if c is not best]
    pvals = [_welch_test(best[2], best[3], n_seeds, c[2], c[3], n_seeds)[1] for c in others]
    m = len(pvals)
    order = np.argsort(pvals)
    holm_p = np.zeros(m)
    running_max = 0.0
    for rank, idx in enumerate(order):
        running_max = max(running_max, (m - rank) * pvals[idx])
        holm_p[idx] = min(running_max, 1.0)
    indiff = [best] + [c for c, hp in zip(others, holm_p) if hp >= alpha]
    return sorted(indiff, key=lambda c: c[1])


if __name__ == "__main__":
    # Démo pédagogique sur une option UNIQUE (S=K=100, put ATM) - illustre la
    # convergence LSM (vs n_paths, puis vs n_steps, séparément) mais ne doit
    # plus servir à choisir les paramètres de production : une seule option
    # ne permet pas de généraliser à un book divers (risque d'overfitting),
    # et ces deux balayages 1D séparés ne peuvent pas détecter d'interaction
    # entre n_steps et n_paths. La vraie recherche de paramètres (grille
    # jointe 2D, échantillon stratifié du book PARAMS, LSM parallélisée,
    # seeds indépendants) est dans paramSearchLSM.py.
    kind = 'put'
    option = Option(S=100, K=100, T=1, r=0.05, sigma=0.2, kind=kind)

    N_SEEDS = 25  # répétitions indépendantes par configuration

    # ── Étape 1 : référence fiable via PDE Crank-Nicolson ──
    reference_price = pde_crank_nicolson(option, style='american', n_steps=800, n_space=800)
    print(f"Prix de référence (PDE Crank-Nicolson, N=800x800): {reference_price:.4f}\n")

    # ── Étape 2 : convergence de LSM en fonction de n_paths (n_steps fixé) ──
    n_steps_lsm = 50
    paths_list = [1000, 5000, 10000, 50000, 100000]

    print(f"--- Convergence LSM vs n_paths (n_steps={n_steps_lsm} fixé, {N_SEEDS} seeds) ---")
    results_paths = []
    for n_p in paths_list:
        t0 = time.perf_counter()
        mean_price, std_price = lsm_mean_std(option, n_steps_lsm, n_p, N_SEEDS)
        elapsed = time.perf_counter() - t0
        error     = abs(mean_price - reference_price) / reference_price
        error_std = std_price / reference_price
        results_paths.append((n_p, mean_price, error, error_std, elapsed))
        print(f"n_paths={n_p:7d}  LSM={mean_price:.4f}±{std_price:.4f}  "
              f"erreur={error:.4f}±{error_std:.4f}  ({elapsed:.2f}s)")

    # ── Étape 3 : convergence de LSM en fonction de n_steps (n_paths fixé) ──
    # n_paths_fixed aligné sur la configuration utilisée en production (cf. benchmarkSPY.py)
    n_paths_fixed = 10000
    steps_list = [20, 50, 100, 200]

    print(f"\n--- Convergence LSM vs n_steps (n_paths={n_paths_fixed} fixé, {N_SEEDS} seeds) ---")
    results_steps = []
    for n in steps_list:
        t0 = time.perf_counter()
        mean_price, std_price = lsm_mean_std(option, n, n_paths_fixed, N_SEEDS)
        elapsed = time.perf_counter() - t0
        error     = abs(mean_price - reference_price) / reference_price
        error_std = std_price / reference_price
        results_steps.append((n, mean_price, error, error_std, elapsed))
        print(f"n_steps={n:4d}  LSM={mean_price:.4f}±{std_price:.4f}  "
              f"erreur={error:.4f}±{error_std:.4f}  ({elapsed:.2f}s)")

    # ── Graphes ──
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    axes[0].errorbar([r[0] for r in results_paths], [r[2] for r in results_paths],
                      yerr=[r[3] for r in results_paths], fmt='o-', color='steelblue',
                      ecolor='steelblue', capsize=3, alpha=0.85)
    axes[0].axhline(0, color='red', ls='--', lw=0.8)
    axes[0].set_xscale('log')
    axes[0].set_xlabel('n_paths')
    axes[0].set_ylabel('|LSM - référence PDE|/référence PDE\n(moyenne ± écart-type)')
    axes[0].set_title(f'Convergence vs n_paths (n_steps={n_steps_lsm}, {N_SEEDS} seeds)')

    axes[1].errorbar([r[0] for r in results_steps], [r[2] for r in results_steps],
                      yerr=[r[3] for r in results_steps], fmt='o-', color='orange',
                      ecolor='orange', capsize=3, alpha=0.85)
    axes[1].axhline(0, color='red', ls='--', lw=0.8)
    axes[1].set_xlabel('n_steps')
    axes[1].set_ylabel('|LSM - référence PDE|/référence PDE\n(moyenne ± écart-type)')
    axes[1].set_title(f'Convergence vs n_steps (n_paths={n_paths_fixed}, {N_SEEDS} seeds)')

    plt.suptitle(f'LSM convergence — Put américain (S=K=100, T=1, r=5%, σ=20%)\n'
                 f'Référence PDE Crank-Nicolson = {reference_price:.4f}  |  '
                 f'{N_SEEDS} seeds indépendants par point', fontsize=12)
    plt.tight_layout()
    plt.savefig(REPORTS / 'lsm_convergence.png')
    plt.show()

    # ── Étape 4 : recherche jointe (n_steps x n_paths), multi-options ──
    # cf. docstring des fonctions ci-dessus (load_stratified_sample,
    # pareto_frontier, indifference_region) pour le détail des 3 corrections
    # apportées à l'ancienne grille croisée (one-at-a-time, option unique,
    # séquentielle). Résultat coûteux (~25 min) - mis en cache dans
    # PARAM_GRID_CACHE, recalculé seulement si absent.
    N_STEPS_GRID = [10, 25, 50, 100, 200]
    N_PATHS_GRID = [1_000, 5_000, 10_000, 20_000, 25_000, 50_000]
    N_SEEDS_GRID = 10
    BASE_SEED_GRID = 0
    LSM_BUDGET_S = 10.0  # budget de temps pour pricer un book avec LSM (cf. budget/méthode du benchmark)

    if PARAM_GRID_CACHE.exists():
        print(f"\nGrille (n_steps x n_paths) déjà calculée -> chargement de {PARAM_GRID_CACHE.name} "
              f"(supprimer ce fichier pour forcer un recalcul)")
        cache = np.load(PARAM_GRID_CACHE)
        n_opt = int(cache['n_options'])
        mae_mean, mae_std = cache['mae_mean'], cache['mae_std']
        time_mean, time_std = cache['time_mean'], cache['time_std']
    else:
        sample = load_stratified_sample()
        n_opt = len(sample)
        print(f"\nÉchantillon stratifié : {n_opt} options américaines (book PARAMS)")

        S_arr = sample['S'].to_numpy(); K_arr = sample['K'].to_numpy(); T_arr = sample['T'].to_numpy()
        r_arr = sample['r'].to_numpy(); sigma_arr = sample['sigma'].to_numpy(); kind_arr = sample['kind'].to_numpy()
        american_arr = np.ones(n_opt, dtype=bool)

        print("Référence PDE (800,800)...")
        ref_prices = pde_crank_nicolson_auto(
            S_arr, K_arr, T_arr, r_arr, sigma_arr, kind_arr, american_arr, n_steps=800, n_space=800)

        n_cells = len(N_STEPS_GRID) * len(N_PATHS_GRID)
        print(f"Grille jointe : {len(N_STEPS_GRID)} x {len(N_PATHS_GRID)} = {n_cells} cellules, "
              f"{N_SEEDS_GRID} seeds chacune -> {n_cells * N_SEEDS_GRID} appels LSM parallélisés")

        mae_mean  = np.zeros((len(N_STEPS_GRID), len(N_PATHS_GRID)))
        mae_std   = np.zeros_like(mae_mean)
        time_mean = np.zeros_like(mae_mean)
        time_std  = np.zeros_like(mae_mean)

        for i, n_steps in enumerate(N_STEPS_GRID):
            for j, n_paths in enumerate(N_PATHS_GRID):
                seed_maes, seed_times = [], []
                for s in range(N_SEEDS_GRID):
                    seed = _derive_seed_grid(BASE_SEED_GRID, n_steps, n_paths, s)
                    t0 = time.perf_counter()
                    prices = LSMoptionValue_parallel(
                        S_arr, K_arr, T_arr, r_arr, sigma_arr, kind_arr, n_steps=n_steps, n_paths=n_paths, seed=seed)
                    seed_times.append(time.perf_counter() - t0)
                    seed_maes.append(np.mean(np.abs(prices - ref_prices)))
                mae_mean[i, j], mae_std[i, j]   = np.mean(seed_maes), np.std(seed_maes)
                time_mean[i, j], time_std[i, j] = np.mean(seed_times), np.std(seed_times)
                print(f"  n_steps={n_steps:4d}  n_paths={n_paths:6d} : "
                      f"MAE={mae_mean[i, j]:.4f}+/-{mae_std[i, j]:.4f}$  "
                      f"temps={time_mean[i, j]:.2f}+/-{time_std[i, j]:.2f}s")

        np.savez(PARAM_GRID_CACHE, n_steps_grid=N_STEPS_GRID, n_paths_grid=N_PATHS_GRID,
                 mae_mean=mae_mean, mae_std=mae_std, time_mean=time_mean, time_std=time_std, n_options=n_opt)
        print(f"Données brutes sauvegardées : {PARAM_GRID_CACHE}")

    # Cellules (label=(n_steps,n_paths), temps, MAE moyenne, écart-type MAE)
    cells = [
        ((ns, npth), time_mean[i, j], mae_mean[i, j], mae_std[i, j])
        for i, ns in enumerate(N_STEPS_GRID) for j, npth in enumerate(N_PATHS_GRID)
    ]

    frontier = pareto_frontier([(lbl, t, m) for lbl, t, m, _ in cells])
    print(f"\n--- Frontière de Pareto (temps, MAE) - {len(frontier)}/{len(cells)} cellules non dominées ---")
    for lbl, t, m in frontier:
        print(f"  n_steps={lbl[0]:4d} n_paths={lbl[1]:6d} : temps={t:.2f}s  MAE={m:.4f}")

    # Le test d'indifférence ne doit être appliqué QU'AUX cellules déjà
    # Pareto-efficientes, pas à l'ensemble brut sous budget : sur les 16
    # cellules "sous budget" (dont beaucoup sont dominées - pire MAE ET plus
    # lentes que d'autres), le test devient sous-puissant et déclare "non
    # significativement pire" des cellules objectivement mauvaises (ex.
    # n_steps=25/n_paths=10000, MAE=0.173, ~60% pire que le minimum observé,
    # mais pas "significativement" pire vu le peu de seeds). Restreindre à la
    # frontière élimine déjà ces cas - le test statistique ne sert plus qu'à
    # départager les points déjà objectivement bons.
    frontier_labels = {lbl for lbl, _, _ in frontier}
    frontier_cells = [c for c in cells if c[0] in frontier_labels]
    frontier_under_budget = [c for c in frontier_cells if c[1] <= LSM_BUDGET_S]

    indiff = indifference_region(frontier_under_budget, n_seeds=N_SEEDS_GRID)
    print(f"\n--- Région d'indifférence statistique PARMI la frontière de Pareto, "
          f"sous budget {LSM_BUDGET_S:.0f}s (Welch + Holm, alpha=0.05) ---")
    for lbl, t, m, s in indiff:
        print(f"  n_steps={lbl[0]:4d} n_paths={lbl[1]:6d} : temps={t:.2f}s  MAE={m:.4f}+/-{s:.4f}")
    best = min(indiff, key=lambda c: c[1])  # le moins cher parmi les statistiquement équivalents au minimum
    print(f"\nRecommandation (moins cher parmi les cellules Pareto-efficientes et statistiquement "
          f"indiscernables du minimum, sous budget {LSM_BUDGET_S:.0f}s) : "
          f"n_steps={best[0][0]}, n_paths={best[0][1]} (temps={best[1]:.2f}s, MAE={best[2]:.4f}+/-{best[3]:.4f})")

    # ── Scatter temps vs MAE (pour le rapport - discussion non-monotonicité +
    # méthode de sélection sous budget) ──
    fig_tm, ax_tm = plt.subplots(figsize=(9.5, 7))
    cmap = plt.get_cmap('viridis')
    color_of = {ns: cmap(i / max(len(N_STEPS_GRID) - 1, 1)) for i, ns in enumerate(N_STEPS_GRID)}

    for ns in N_STEPS_GRID:
        row = sorted([c for c in cells if c[0][0] == ns], key=lambda c: c[0][1])
        xs = [c[1] for c in row]; ys = [c[2] for c in row]; yerr = [c[3] for c in row]
        ax_tm.plot(xs, ys, '-', color=color_of[ns], alpha=0.5, lw=1, zorder=1)
        ax_tm.errorbar(xs, ys, yerr=yerr, fmt='o', color=color_of[ns], label=f'n_steps={ns}',
                       capsize=3, ms=5, zorder=2)

    fx = [t for _, t, _ in frontier]; fy = [m for _, _, m in frontier]
    ax_tm.plot(fx, fy, 'k--', lw=1.8, zorder=3, label='Frontière de Pareto')
    ax_tm.axvline(LSM_BUDGET_S, color='red', ls=':', lw=1.5, label=f'Budget LSM = {LSM_BUDGET_S:.0f}s')

    ax_tm.set_xlabel('Temps (s)')
    ax_tm.set_ylabel('MAE vs référence PDE (800,800) ($)')
    ax_tm.set_title(f'LSM — temps vs MAE, grille (n_steps x n_paths)\n'
                     f'{n_opt} options américaines (book PARAMS, échantillon stratifié), '
                     f'{N_SEEDS_GRID} seeds indépendants par cellule')
    ax_tm.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    plt.savefig(REPORTS / 'lsm_param_grid_time_vs_mae.png', dpi=150)
    plt.show()