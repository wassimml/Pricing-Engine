from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import time
from concurrent.futures import ProcessPoolExecutor

from option import Option
from pde import pde_crank_nicolson

REPORTS = Path(__file__).parent.parent / "reports"


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


if __name__ == "__main__":
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

    # ── Étape 4 : grille croisée (n_steps x n_paths) pour le graphe 3D ──
    steps_grid = [10, 25, 50, 100, 200]
    paths_grid = [1000, 5000, 10000, 50000, 100000]

    print(f"\n--- Grille croisée n_steps x n_paths (pour le graphe 3D, {N_SEEDS} seeds) ---")
    error_grid     = np.zeros((len(steps_grid), len(paths_grid)))
    error_std_grid = np.zeros((len(steps_grid), len(paths_grid)))

    for i, n_steps in enumerate(steps_grid):
        for j, n_paths in enumerate(paths_grid):
            t0 = time.perf_counter()
            mean_price, std_price = lsm_mean_std(option, n_steps, n_paths, N_SEEDS)
            elapsed = time.perf_counter() - t0
            error_grid[i, j]     = abs(mean_price - reference_price) / reference_price
            error_std_grid[i, j] = std_price / reference_price
            print(f"n_steps={n_steps:4d}  n_paths={n_paths:7d}  "
                  f"price={mean_price:.4f}±{std_price:.4f}  "
                  f"erreur={error_grid[i, j]:.4f}  ({elapsed:.2f}s)")

    # ── Graphe 3D ──
    Steps, LogPaths = np.meshgrid(steps_grid, np.log10(paths_grid), indexing='ij')

    fig3d = plt.figure(figsize=(12, 9))
    ax = fig3d.add_subplot(111, projection='3d')

    surf = ax.plot_surface(Steps, LogPaths, error_grid,
                            cmap='viridis', edgecolor='k', linewidth=0.3, alpha=0.9)

    ax.set_xlabel('n_steps')
    ax.set_ylabel('log10(n_paths)')
    ax.set_zlabel(f'|LSM - référence PDE| / référence PDE\n(moyenne sur {N_SEEDS} seeds)')
    ax.set_title(f'Convergence LSM — Erreur relative moyenne vs (n_steps, n_paths)\n'
                 f'Put américain (S=K=100, T=1, r=5%, σ=20%) — '
                 f'Référence PDE = {reference_price:.4f}  |  {N_SEEDS} seeds')
    ax.view_init(elev=25, azim=120)
    fig3d.colorbar(surf, ax=ax, shrink=0.6, label='Erreur relative moyenne')

    plt.tight_layout()

    plt.savefig(REPORTS / 'lsm_convergence_3d.png', dpi=150)
    plt.show()