from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

import pandas as pd
import time

from option import Option
from binomial import crr_price_fast
from monteCarlo import mc_naive_threaded, mc_antithetic_threaded, mc_control_threaded, mc_control_antithetic_threaded
from monteCarloLSM import LSMoptionValue_parallel, pareto_frontier, indifference_region, _derive_seed_grid
from pde import pde_crank_nicolson_auto
from BSpricer import BSModel

DATA = Path(__file__).parent.parent / "data"
REPORTS = Path(__file__).parent.parent / "reports"

_bs = BSModel()

N_SEEDS = 10    # répétitions indépendantes par (méthode, paramètre) — cf. run_method
BASE_SEED = 0

def _derive_seed(base_seed: int, p, s: int) -> int:
    """Dérive un seed reproductible mais statistiquement indépendant pour
    chaque (paramètre p, répétition s) — PAS des nombres aléatoires communs.

    Avec `seed = base_seed + s` (ancienne version), tous les p au même s
    démarraient le même flux aléatoire, qui restait aligné exactement le
    temps de consommer autant de tirages que la config la plus courte
    (vérifié empiriquement : identique au premier pas de temps / à la
    première option, brisé dès le suivant) — ni un vrai common-random-numbers
    (qui resterait aligné du début à la fin), ni des tirages réellement
    indépendants. SeedSequence([base_seed, p, s]) mélange les trois valeurs
    pour produire un seed different et non corrélé par (p, s), tout en
    restant déterministe (même triplet -> même seed à chaque run)."""
    return int(np.random.SeedSequence([base_seed, p, s]).generate_state(1)[0])

# Each entry: (kind, pricer, param values, style_filter, ref_kind, stochastic).
# kind = "fast" (pricer array-based : vectorisé pour BS/MC/CRR — cf.
# monteCarlo.py/binomial.py, tout le book filtré traité en un seul appel par
# seed). Toutes les méthodes du sweep sont maintenant "fast" (PDE n'est pas
# swept ici, seulement utilisé comme référence, cf. compute_reference_prices) ;
# "row" (pricer Option-based, boucle Python par ligne) reste supporté par
# run_method au cas où une future méthode n'aurait pas d'équivalent
# vectorisé/parallélisé.
#
# LSM n'est PAS dans ce dict : contrairement aux autres méthodes, sa
# précision dépend de DEUX paramètres (n_steps ET n_paths), pas un seul - un
# balayage 1D à n_steps fixé (comme avant) ne peut pas détecter l'interaction
# entre les deux (vérifié : la MAE décroît puis REMONTE avec n_steps à
# n_paths fixé). LSM a donc sa propre section dédiée (grille jointe 2D, peu
# de valeurs pour borner le temps de calcul) plus bas, plutôt que d'être
# forcée dans ce format à un seul paramètre.
#
# style_filter (None=all, 'European', 'American'), ref_kind (quelle référence
# sert au calcul de MAE), stochastic (méthode aléatoire -> répétée sur
# N_SEEDS seeds indépendants pour obtenir moyenne ± écart-type du temps et de
# la MAE ; CRR est déterministe (même arbre binomial à chaque appel) — la
# répéter ne changerait ni le prix ni l'information disponible, un seul
# passage suffit (std = 0 par construction, affiché tel quel plutôt que
# masqué) :
#   "BS"    : Black-Scholes fermé — pour les méthodes qui ne pricent que du
#             européen (Monte Carlo).
#   "MIXED BS - PDE (800,800)" : BS pour les lignes européennes, PDE (800,800,
#             américain) pour les lignes américaines — pour CRR, qui price
#             les deux styles à la fois.
METHODS = {
    "MC Naive":     ("fast", lambda S,K,T,r,sigma,kind,american,p,seed: mc_naive_threaded(S,K,T,r,sigma,kind, n_paths=p, seed=seed, n_workers=8)[0],              [1_000, 5_000, 10_000, 25_000, 50_000, 100_000, 150_000, 200_000], "European", "BS",                       True),
    "MC Anti":      ("fast", lambda S,K,T,r,sigma,kind,american,p,seed: mc_antithetic_threaded(S,K,T,r,sigma,kind, n_paths=p, seed=seed, n_workers=8)[0],         [1_000, 5_000, 10_000, 25_000, 50_000, 100_000, 150_000, 200_000], "European", "BS",                       True),
    "MC Control":   ("fast", lambda S,K,T,r,sigma,kind,american,p,seed: mc_control_threaded(S,K,T,r,sigma,kind, n_paths=p, seed=seed, n_workers=8)[0],            [1_000, 5_000, 10_000, 25_000, 50_000, 100_000, 150_000, 200_000], "European", "BS",                       True),
    "MC Anti+Ctrl": ("fast", lambda S,K,T,r,sigma,kind,american,p,seed: mc_control_antithetic_threaded(S,K,T,r,sigma,kind, n_paths=p, seed=seed, n_workers=8)[0], [1_000, 5_000, 10_000, 25_000, 50_000, 100_000, 150_000, 200_000], "European", "BS",                       True),
    "CRR":          ("fast", lambda S,K,T,r,sigma,kind,american,p,_seed: crr_price_fast(S,K,T,r,sigma,kind,american, period=p),                  [50, 100, 200, 500, 1_000],                              None,       "MIXED BS - PDE (800,800)", False),
}

# Grille LSM dédiée (n_steps x n_paths) - volontairement PETITE (temps de
# calcul explose vite, cf. mesures réelles : (50,50000) coûte ~45s sur le
# book américain complet). Valeurs choisies pour rester informatives sans
# faire exploser le temps :
#   n_steps=25/50   : n_steps=10 donne une MAE bruitée/médiocre, n_steps=100+
#                     est dominé (plus lent ET moins précis, cf. étude
#                     monteCarloLSM.py) - 25 et 50 encadrent la zone
#                     intéressante.
#   n_paths=5000/10000/20000 : couvre du bon marché au proche du budget de
#                     10s retenu pour LSM (cf. section budget plus bas),
#                     sans aller jusqu'à des configs déjà hors budget.
LSM_N_STEPS_GRID = [25, 50]
LSM_N_PATHS_GRID = [5_000, 10_000, 20_000]

def run_method(cleanData: pd.DataFrame, pricer, params: list, style_filter, ref: np.ndarray, stochastic: bool,
               n_seeds: int = N_SEEDS, base_seed: int = BASE_SEED):
    """Pour chaque valeur de paramètre, price le book filtré une fois par seed
    (n_seeds seeds indépendants si la méthode est stochastique, 1 seul passage
    sinon — cf. commentaire sur METHODS) et retourne, par paramètre : temps
    total (moyenne ± écart-type sur les seeds) et MAE vs référence (idem).
    Une seule réalisation ne permet pas de distinguer un vrai comportement de
    convergence d'un artefact du tirage particulier (même logique que
    lsm_mean_std / mc_mean_std, appliquée ici au book entier plutôt qu'à une
    option unique). Utilisé pour les méthodes "row" (pas encore vectorisées) :
    boucle Python explicite, une option à la fois."""
    rows = [r for r in cleanData.itertuples() if style_filter is None or r[7] == style_filter]
    effective_seeds = n_seeds if stochastic else 1
    results = []
    for p in params:
        seed_times, seed_maes = [], []
        for s in range(effective_seeds):
            seed = _derive_seed(base_seed, p, s)
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

def _eval_param_fast(pricer_fast, S, K, T, r, sigma, kind, american, p, ref: np.ndarray, stochastic: bool,
                      n_seeds: int = N_SEEDS, base_seed: int = BASE_SEED):
    """Évalue UNE seule valeur de paramètre p : temps et MAE moyens +/-
    écart-type sur n_seeds seeds indépendants (1 seul passage si
    stochastic=False). Factorisé hors de run_method_fast pour être réutilisé
    par la recherche par dichotomie (find_time_boundary, plus bas) sur des
    valeurs de p qui ne sont pas forcément dans la liste balayée par
    METHODS."""
    effective_seeds = n_seeds if stochastic else 1
    seed_times, seed_maes = [], []
    for s in range(effective_seeds):
        seed = _derive_seed(base_seed, p, s)
        t0 = time.perf_counter()
        prices = pricer_fast(S, K, T, r, sigma, kind, american, p, seed)
        elapsed = time.perf_counter() - t0
        mask = ~np.isnan(prices) & ~np.isnan(ref)
        seed_times.append(elapsed)
        seed_maes.append(np.mean(np.abs(prices[mask] - ref[mask])))
    return np.mean(seed_times), np.std(seed_times), np.mean(seed_maes), np.std(seed_maes)


def run_method_fast(cleanData: pd.DataFrame, pricer_fast, params: list, style_filter, ref: np.ndarray, stochastic: bool,
                     n_seeds: int = N_SEEDS, base_seed: int = BASE_SEED):
    """Équivalent de run_method pour les méthodes "fast" (array-based, cf.
    monteCarlo.py) : price tout le book filtré en un seul appel vectorisé par
    (paramètre, seed) au lieu d'une boucle Python par option. Même format de
    retour que run_method pour rester interchangeable dans la boucle
    principale ci-dessous."""
    sub = cleanData if style_filter is None else cleanData[cleanData["Style"] == style_filter]
    S = sub['Spot  S'].to_numpy(); K = sub['Strike  K'].to_numpy(); T = sub['T  (years)'].to_numpy()
    r = (sub['Rate  r (%)'] / 100).to_numpy(); sigma = (sub['Vol  σ (%)'] / 100).to_numpy()
    kind = sub['Kind'].str.lower().to_numpy()
    american = (sub['Style'] == 'American').to_numpy()

    results = []
    for p in params:
        t_mean, t_std, mae_mean, mae_std = _eval_param_fast(
            pricer_fast, S, K, T, r, sigma, kind, american, p, ref, stochastic, n_seeds, base_seed)
        results.append((p, t_mean, t_std, mae_mean, mae_std))
    return results


def find_time_boundary(pricer_fast, cleanData: pd.DataFrame, style_filter, ref: np.ndarray, stochastic: bool,
                        budget: float, lo: int, hi: int, n_seeds: int = N_SEEDS, base_seed: int = BASE_SEED):
    """Recherche par DICHOTOMIE le plus grand entier n dans [lo, hi] tel que
    le temps moyen (n_seeds seeds indépendants) reste <= budget - équivalent
    à résoudre temps_moyen(n) = budget, sans se limiter aux quelques valeurs
    déjà balayées par METHODS (celles-ci ne servent qu'à fournir [lo, hi]
    ci-dessous). Suppose temps(n) croissant avec n - vrai par construction
    pour MC/CRR (plus de chemins/périodes coûte toujours plus cher) ; NE PAS
    utiliser pour LSM (MAE - et potentiellement le coût de la régression -
    pas monotone, cf. section LSM dédiée).

    Précondition : temps(lo) <= budget < temps(hi) (lo/hi doivent encadrer
    la frontière - cf. appel ci-dessous, qui les déduit du sweep existant) ;
    léve une ValueError explicite sinon plutôt que de retourner un résultat
    trompeur silencieusement."""
    sub = cleanData if style_filter is None else cleanData[cleanData["Style"] == style_filter]
    S = sub['Spot  S'].to_numpy(); K = sub['Strike  K'].to_numpy(); T = sub['T  (years)'].to_numpy()
    r = (sub['Rate  r (%)'] / 100).to_numpy(); sigma = (sub['Vol  σ (%)'] / 100).to_numpy()
    kind = sub['Kind'].str.lower().to_numpy()
    american = (sub['Style'] == 'American').to_numpy()

    cache = {}

    def ev(p):
        if p not in cache:
            cache[p] = _eval_param_fast(pricer_fast, S, K, T, r, sigma, kind, american, p, ref, stochastic, n_seeds, base_seed)
        return cache[p]

    t_lo = ev(lo)[0]
    t_hi = ev(hi)[0]
    if t_lo > budget:
        raise ValueError(f"borne basse lo={lo} déjà au-dessus du budget ({t_lo:.2f}s > {budget}s) - élargir le bracket")
    if t_hi <= budget:
        raise ValueError(f"borne haute hi={hi} encore sous le budget ({t_hi:.2f}s <= {budget}s) - élargir le bracket")

    while hi - lo > 1:
        mid = (lo + hi) // 2
        if ev(mid)[0] <= budget:
            lo = mid
        else:
            hi = mid

    t_mean, t_std, mae_mean, mae_std = ev(lo)
    return lo, t_mean, t_std, mae_mean, mae_std

def compute_reference_prices(cleanData: pd.DataFrame):
    """BS (fermé, toujours calculé — vectorisé sur tout le book en un seul
    appel, cf. BSModel.price_batch) et PDE (au style réel de chaque ligne,
    outil de RÉFÉRENCE n_steps=800/n_space=800 — jamais l'outil de calcul
    50/100 ; séquentiel ou parallèle choisi automatiquement selon le coût
    réel mesuré, cf. pde_crank_nicolson_auto dans pde.py — pas un cas
    particulier sur cette résolution précise), calculés une seule fois pour
    tout le book — servent ensuite à construire la référence de chaque
    méthode (BS, PDE, ou mixte) sans recalcul."""
    S = cleanData['Spot  S'].to_numpy(); K = cleanData['Strike  K'].to_numpy(); T = cleanData['T  (years)'].to_numpy()
    r = (cleanData['Rate  r (%)'] / 100).to_numpy(); sigma = (cleanData['Vol  σ (%)'] / 100).to_numpy()
    kind = cleanData['Kind'].str.lower().to_numpy()
    american = (cleanData['Style'] == 'American').to_numpy()
    bs_prices = _bs.price_batch(S, K, T, r, sigma, kind)
    pde_prices = pde_crank_nicolson_auto(S, K, T, r, sigma, kind, american, n_steps=800, n_space=800)
    return bs_prices, pde_prices

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

    # Book PARAMS : sert à trouver les paramètres (période/n_paths/...) via le
    # balayage MAE/temps ci-dessous. À ne jamais évaluer sur ce même book une
    # fois les paramètres choisis (cf. book TEST dans benchmarkInData.py),
    # sinon les paramètres "optimaux" ne font que surapprendre ce book précis.
    data = pd.read_excel(DATA / "1_options_book_params_2026-06-29.xlsx", sheet_name=0, header=2, index_col=0)
    cleanData = data[['Kind', 'Spot  S', 'Strike  K', 'T  (years)', 'Rate  r (%)', 'Vol  σ (%)', 'Style']]

    print("Computing BS and PDE (800,800) reference prices (once, per-row true style)...")
    bs_all, pde_all = compute_reference_prices(cleanData)

    fig = plt.figure(figsize=(13, 8))
    ax = fig.add_subplot(111, projection="3d")
    colors = ["steelblue", "tomato", "seagreen", "darkorange", "saddlebrown"]

    # Résultats du balayage par méthode, capturés au passage (sans rien
    # changer au calcul/affichage existant ci-dessous) pour être réutilisés
    # par la section "sélection sous budget de temps" plus bas, sans refaire
    # tourner le sweep une deuxième fois.
    results_by_method = {}

    for (label, (kind_tag, pricer, params, style_filter, ref_kind, stochastic)), color in zip(METHODS.items(), colors):
        ref = select_reference(cleanData, style_filter, ref_kind, bs_all, pde_all)
        n_seeds_used = N_SEEDS if stochastic else 1
        print(f"\n--- {label} (référence : {ref_kind}, {n_seeds_used} seed{'s' if n_seeds_used > 1 else ''}) ---")
        xs, ys, zs, y_errs, z_errs = [], [], [], [], []
        run = run_method_fast if kind_tag == "fast" else run_method
        for param, t_mean, t_std, mae_mean, mae_std in run(cleanData, pricer, params, style_filter, ref, stochastic):
            xs.append(param)
            ys.append(t_mean)
            zs.append(mae_mean)
            y_errs.append(t_std)
            z_errs.append(mae_std)
            print(f"  param={param:,}  time={t_mean:.2f}s±{t_std:.2f}s  MAE={mae_mean:.4f}±{mae_std:.4f}")
        results_by_method[label] = list(zip(xs, ys, zs, y_errs, z_errs))

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
    ax.set_title(f"Accuracy vs speed — MC, CRR ({N_SEEDS} seeds indépendants, moyenne ± écart-type "
                 "; CRR déterministe -> 1 seed)\n"
                 "(référence : BS pour MC, BS+PDE (800,800) selon le style pour CRR ; "
                 "LSM traité séparément - 2 paramètres, cf. section dédiée)")
    ax.legend()

    plt.tight_layout()
    plt.savefig(REPORTS / "benchmark_methods_3d.png", dpi=150)
    plt.show()

    # Budget = 2s pour toutes les méthodes, SAUF LSM (10s) : LSM est
    # nettement moins efficace par unité de temps que MC/CRR pour pricer un
    # book entier (cf. mesures sur le vrai book américain, pas un
    # échantillon) - lui laisser plus de budget est un choix assumé, pas une
    # comparaison qui l'exclurait d'office.
    BUDGET_S = 2.0
    LSM_BUDGET_S = 10.0

    # ── Recherche du point-frontière par dichotomie, AVANT les graphes Pareto ──
    # Le sweep ci-dessus ne teste que quelques valeurs choisies à l'avance
    # (juste pour avoir une vision globale des méthodes) - rien ne garantit
    # qu'une d'elles tombe près du budget. Ici, pour MC/CRR (temps garanti
    # croissant avec le paramètre - PAS pour LSM, cf. section dédiée), on
    # résout PAR DICHOTOMIE le plus grand entier n tel que
    # temps_moyen(n) <= budget < temps_moyen(n+1) - une vraie recherche dans
    # l'espace des paramètres, pas une sélection parmi les valeurs déjà
    # testées. [lo, hi] (bornes de départ de la dichotomie) sont déduits du
    # sweep existant : le plus grand paramètre déjà sous budget et le plus
    # petit déjà au-dessus ; élargis si le sweep ne les fournit pas.
    print("\n\n=== Recherche du point-frontière (dichotomie, budget = 2s) ===")
    boundary_by_method = {}
    for label, (kind_tag, pricer, params, style_filter, ref_kind, stochastic) in METHODS.items():
        ref = select_reference(cleanData, style_filter, ref_kind, bs_all, pde_all)
        n_seeds_used = N_SEEDS if stochastic else 1
        swept = results_by_method[label]  # [(param, t_mean, mae_mean, t_std, mae_std), ...]

        under = [c for c in swept if c[1] <= BUDGET_S]
        over = [c for c in swept if c[1] > BUDGET_S]
        lo = max((c[0] for c in under), default=min(params))
        hi = min((c[0] for c in over), default=max(params) * 4)

        try:
            n_star, t_mean, t_std, mae_mean, mae_std = find_time_boundary(
                pricer, cleanData, style_filter, ref, stochastic, BUDGET_S, lo, hi, n_seeds_used)
            boundary_by_method[label] = (n_star, t_mean, mae_mean, t_std, mae_std)
            print(f"  {label:14} : param={n_star:,} (temps={t_mean:.2f}s+/-{t_std:.2f}s, "
                  f"MAE={mae_mean:.4f}+/-{mae_std:.4f})  [bracket initial lo={lo:,}, hi={hi:,}]")
        except ValueError as e:
            print(f"  {label:14} : dichotomie impossible ({e})")
            boundary_by_method[label] = None

    # ── Sélection sous budget de temps, par méthode ──────────────────────
    # Ne remplace rien ci-dessus (sweep + plot 3D + dichotomie inchangés) -
    # ajoute une réponse directe à "quels paramètres utiliser en
    # production" : le point-frontière trouvé ci-dessus est ajouté à
    # l'ensemble des points déjà balayés, puis la frontière de Pareto
    # (temps, MAE) et la région d'indifférence statistique (Welch + Holm)
    # décident lequel retenir (cf. monteCarloLSM.py pour le détail de ces
    # deux fonctions, réutilisées ici telles quelles, pas dupliquées).
    def select_under_budget(cells, budget, n_seeds, label):
        """cells : liste de (param, temps, MAE moyenne, écart-type MAE).
        Retourne (frontier, indiff, best) - best=None si aucune config ne
        respecte le budget."""
        frontier = pareto_frontier([(p, t, m) for p, t, m, _ in cells])
        frontier_params = {p for p, _, _ in frontier}
        # Le test d'indifférence ne s'applique QU'AUX cellules déjà Pareto-
        # efficientes (cf. monteCarloLSM.py) : sur l'ensemble brut, il
        # deviendrait sous-puissant et déclarerait "non significativement
        # pire" des configs objectivement dominées.
        frontier_cells = [c for c in cells if c[0] in frontier_params]
        under_budget = [c for c in frontier_cells if c[1] <= budget]
        if not under_budget:
            cheapest = min(frontier_cells, key=lambda c: c[1])
            print(f"  ATTENTION ({label}) : aucune config testée ne respecte le budget de {budget:.0f}s "
                  f"- la moins chère coûte {cheapest[1]:.2f}s (param={cheapest[0]:,}).")
            return frontier, [], None
        indiff = indifference_region(under_budget, n_seeds=n_seeds) if len(under_budget) > 1 else under_budget
        best = min(indiff, key=lambda c: c[1])
        return frontier, indiff, best

    print("\n\n=== Sélection des paramètres sous budget de temps (par méthode) ===")
    print(f"Budget = {BUDGET_S:.0f}s pour toutes les méthodes, sauf LSM ({LSM_BUDGET_S:.0f}s).")

    fig_b, axes_b = plt.subplots(2, 3, figsize=(18, 10))
    axes_b = axes_b.flatten()
    lsm_ax = axes_b[-1]  # 6e slot : LSM, traité séparément ci-dessous (grille 2D dédiée)

    for ax, (label, color) in zip(axes_b, zip(METHODS.keys(), colors)):
        _, _, _, _, _, stochastic = METHODS[label]
        n_seeds_used = N_SEEDS if stochastic else 1
        cells = [(p, t, m, ms) for p, t, m, _, ms in results_by_method[label]]

        # Point trouvé par dichotomie (ci-dessus) ajouté aux points balayés -
        # participe à la frontière de Pareto/indifférence comme n'importe
        # quel autre point, pas un choix imposé à côté du calcul.
        boundary = boundary_by_method.get(label)
        if boundary is not None and boundary[0] not in {c[0] for c in cells}:
            n_star, t_mean, mae_mean, t_std, mae_std = boundary
            cells = cells + [(n_star, t_mean, mae_mean, mae_std)]

        frontier, indiff, best = select_under_budget(cells, BUDGET_S, n_seeds_used, label)

        print(f"\n--- {label} ---")
        for p, t, m, ms in cells:
            print(f"  param={p:,}  temps={t:.2f}s  MAE={m:.4f}+/-{ms:.4f}")
        if best is not None:
            print(f"  -> Retenu (Pareto + indifférence) sous budget {BUDGET_S:.0f}s : param={best[0]:,} "
                  f"(temps={best[1]:.2f}s, MAE={best[2]:.4f}+/-{best[3]:.4f})")

        ax.errorbar([c[1] for c in cells], [c[2] for c in cells], yerr=[c[3] for c in cells],
                    fmt='o', color=color, ms=5, capsize=3, alpha=0.7, label='configs testées')
        ax.plot([t for _, t, _ in frontier], [m for _, _, m in frontier], 'k--', lw=1.3,
                label='Frontière de Pareto')
        ax.axvline(BUDGET_S, color='red', ls=':', lw=1.3, label=f'Budget = {BUDGET_S:.0f}s')
        if boundary is not None and (best is None or boundary[0] != best[0]):
            ax.scatter([boundary[1]], [boundary[2]], marker='D', s=100, color='navy', edgecolor='white',
                       zorder=4, label=f'Point-frontière (dichotomie) : {boundary[0]:,}')
        if best is not None:
            ax.scatter([best[1]], [best[2]], marker='*', s=220, color='gold', edgecolor='black',
                       zorder=5, label=f'Retenu : {best[0]:,}')
        ax.set_xlabel('Temps (s)'); ax.set_ylabel('MAE')
        ax.set_title(label)
        ax.legend(fontsize=7)

    # ── LSM : traité à part, grille jointe (n_steps x n_paths), PETITE ───────
    # LSM n'est pas dans METHODS (cf. commentaire plus haut) : sa précision
    # dépend de DEUX paramètres, pas un seul - un balayage 1D à n_steps fixé
    # ne peut pas détecter l'interaction entre les deux (vérifié dans
    # monteCarloLSM.py : la MAE décroît puis REMONTE avec n_steps à n_paths
    # fixé). Grille volontairement petite (LSM_N_STEPS_GRID x
    # LSM_N_PATHS_GRID, cf. justification plus haut) - calculée ICI, sur LE
    # MÊME book américain (cleanData filtré "American") que les autres
    # méthodes, pour que les temps restent directement comparables (measuré :
    # les temps sur un échantillon réduit ne sont PAS transposables
    # linéairement à ce book - l'overhead fixe de démarrage des workers pèse
    # proportionnellement moins sur un plus gros book).
    print(f"\n--- LSM (grille {len(LSM_N_STEPS_GRID)}x{len(LSM_N_PATHS_GRID)} = "
          f"{len(LSM_N_STEPS_GRID) * len(LSM_N_PATHS_GRID)} cellules, {N_SEEDS} seeds, book américain) ---")
    lsm_ref = select_reference(cleanData, "American", "PDE (800,800)", bs_all, pde_all)
    sub_am = cleanData[cleanData["Style"] == "American"]
    S_lsm = sub_am['Spot  S'].to_numpy(); K_lsm = sub_am['Strike  K'].to_numpy(); T_lsm = sub_am['T  (years)'].to_numpy()
    r_lsm = (sub_am['Rate  r (%)'] / 100).to_numpy(); sigma_lsm = (sub_am['Vol  σ (%)'] / 100).to_numpy()
    kind_lsm = sub_am['Kind'].str.lower().to_numpy()

    lsm_cells = []
    for ns in LSM_N_STEPS_GRID:
        for npth in LSM_N_PATHS_GRID:
            seed_times, seed_maes = [], []
            for s in range(N_SEEDS):
                seed = _derive_seed_grid(BASE_SEED, ns, npth, s)
                t0 = time.perf_counter()
                prices = LSMoptionValue_parallel(S_lsm, K_lsm, T_lsm, r_lsm, sigma_lsm, kind_lsm,
                                                  n_steps=ns, n_paths=npth, seed=seed, n_workers=8)
                seed_times.append(time.perf_counter() - t0)
                seed_maes.append(np.mean(np.abs(prices - lsm_ref)))
            t_mean, t_std = np.mean(seed_times), np.std(seed_times)
            m_mean, m_std = np.mean(seed_maes), np.std(seed_maes)
            lsm_cells.append(((ns, npth), t_mean, m_mean, m_std))
            print(f"  n_steps={ns:3d} n_paths={npth:6d} : temps={t_mean:.2f}+/-{t_std:.2f}s  "
                  f"MAE={m_mean:.4f}+/-{m_std:.4f}")

    frontier, indiff, best = select_under_budget(lsm_cells, LSM_BUDGET_S, N_SEEDS, "LSM")
    if best is not None:
        print(f"  -> Retenu sous budget {LSM_BUDGET_S:.0f}s : n_steps={best[0][0]}, n_paths={best[0][1]} "
              f"(temps={best[1]:.2f}s, MAE={best[2]:.4f}+/-{best[3]:.4f})")

    ax = lsm_ax
    ax.errorbar([c[1] for c in lsm_cells], [c[2] for c in lsm_cells], yerr=[c[3] for c in lsm_cells],
                fmt='o', color="mediumpurple", ms=5, capsize=3, alpha=0.7, label='configs testées')
    ax.plot([t for _, t, _ in frontier], [m for _, _, m in frontier], 'k--', lw=1.3,
            label='Frontière de Pareto')
    ax.axvline(LSM_BUDGET_S, color='red', ls=':', lw=1.3, label=f'Budget = {LSM_BUDGET_S:.0f}s')
    if best is not None:
        ax.scatter([best[1]], [best[2]], marker='*', s=220, color='gold', edgecolor='black',
                   zorder=5, label=f"Retenu : ({best[0][0]},{best[0][1]})")
    ax.set_xlabel('Temps (s)'); ax.set_ylabel('MAE')
    ax.set_title('LSM (n_steps, n_paths)')
    ax.legend(fontsize=7)

    fig_b.suptitle("Sélection des paramètres sous budget de temps (Pareto + région d'indifférence "
                    "statistique)\nBudget = 2s (MC/CRR), 10s (LSM) - remplace le choix à l'œil sur "
                    "le sweep/plot 3D ci-dessus", fontsize=12)
    plt.tight_layout()
    plt.savefig(REPORTS / "benchmark_methods_budget_selection.png", dpi=150)
    plt.show()