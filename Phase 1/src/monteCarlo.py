import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

from option import Option
from BSpricer import BSModel

REPORTS = Path(__file__).parent.parent / "reports"

def mc_naive(opt : Option, n_paths : int = 100000, seed : int = 42) -> float:
    """Monte Carlo pricer for European vanilla options."""
    rng = np.random.default_rng(seed)
    s_t = opt.S * np.exp((opt.r - 0.5 * opt.sigma ** 2) * opt.T + opt.sigma * np.sqrt(opt.T) * rng.standard_normal(n_paths))
    if opt.kind == 'call':
        discounted_payoff = np.exp(-opt.r * opt.T) * np.maximum(s_t - opt.K, 0)
    else:
        discounted_payoff = np.exp(-opt.r * opt.T) * np.maximum(opt.K - s_t, 0)
    return float(np.mean(discounted_payoff)), np.std(discounted_payoff) / np.sqrt(n_paths)


def mc_antithetic(opt : Option, n_paths : int = 100000, seed : int = 42) -> float:
    """Monte Carlo pricer with antithetic variates."""
    rng = np.random.default_rng(seed)
    z = rng.standard_normal(n_paths // 2)
    s_t1 = opt.S * np.exp((opt.r - 0.5 * opt.sigma ** 2) * opt.T + opt.sigma * np.sqrt(opt.T) * z)
    s_t2 = opt.S * np.exp((opt.r - 0.5 * opt.sigma ** 2) * opt.T - opt.sigma * np.sqrt(opt.T) * z)
    if opt.kind == 'call':
        payoff1 = np.maximum(s_t1 - opt.K, 0)
        payoff2 = np.maximum(s_t2 - opt.K, 0)
    else:
        payoff1 = np.maximum(opt.K - s_t1, 0)
        payoff2 = np.maximum(opt.K - s_t2, 0)
    discounted_payoff = np.exp(-opt.r * opt.T) * (payoff1 + payoff2) / 2
    # n_paths//2 independent pairs — not n_paths (pairs are correlated)
    return float(np.mean(discounted_payoff)), np.std(discounted_payoff) / np.sqrt(n_paths // 2)


def mc_control(opt : Option, n_paths : int = 100000, seed : int = 42) -> float:
    """Monte Carlo pricer with control variates."""
    rng = np.random.default_rng(seed)
    s_t = opt.S * np.exp((opt.r - 0.5 * opt.sigma ** 2) * opt.T + opt.sigma * np.sqrt(opt.T) * rng.standard_normal(n_paths))
    if opt.kind == 'call':
        payoff = np.maximum(s_t - opt.K, 0)
    else:
        payoff = np.maximum(opt.K - s_t, 0)
    beta = np.cov(payoff, s_t)[0, 1] / np.var(s_t)
    adjusted = payoff - beta * (s_t - opt.S * np.exp(opt.r * opt.T))
    price = np.exp(-opt.r * opt.T) * np.mean(adjusted)
    std_error = np.exp(-opt.r * opt.T) * np.std(adjusted) / np.sqrt(n_paths)
    return float(price), std_error


def mc_control_antithetic(opt : Option, n_paths : int = 100000, seed : int = 42) -> float:
    """Monte Carlo pricer with control variates and antithetic variates."""
    rng = np.random.default_rng(seed)
    z = rng.standard_normal(n_paths // 2)
    s_t1 = opt.S * np.exp((opt.r - 0.5 * opt.sigma ** 2) * opt.T + opt.sigma * np.sqrt(opt.T) * z)
    s_t2 = opt.S * np.exp((opt.r - 0.5 * opt.sigma ** 2) * opt.T - opt.sigma * np.sqrt(opt.T) * z)
    if opt.kind == 'call':
        payoff1 = np.maximum(s_t1 - opt.K, 0)
        payoff2 = np.maximum(s_t2 - opt.K, 0)
    else:
        payoff1 = np.maximum(opt.K - s_t1, 0)
        payoff2 = np.maximum(opt.K - s_t2, 0)
    payoff_combined = (payoff1 + payoff2) / 2
    control_combined = (s_t1 + s_t2) / 2
    control_mean = opt.S
    beta = np.cov(payoff_combined, control_combined)[0][1] / np.var(control_combined)
    discounted_adjusted = np.exp(-opt.r * opt.T) * payoff_combined - beta * (np.exp(-opt.r * opt.T) * control_combined - control_mean)
    # n_paths//2 independent pairs — not n_paths (pairs are correlated)
    return float(np.mean(discounted_adjusted)), np.std(discounted_adjusted) / np.sqrt(n_paths // 2)


# ---------------------------------------------------------------------------
# Versions "fast" (et non "batch" - nom trompeur évité, cf. plus bas) :
# pricent tout un book (n_options) en réutilisant un seul rng et en éliminant
# le travail redondant, plutôt qu'en rappelant la version scalaire ci-dessus
# une fois par option.
#
# Résultat mesuré CONTRAIRE à l'intuition de départ ("empiler les options
# dans une grosse matrice 2D ira plus vite") : profilé sur le book PARAMS
# (1109 options européennes, n_paths=100000), batch_size=1 tourne en 1.7s
# (vs 2.5s pour la boucle scalaire d'origine) alors que batch_size=2 retombe
# déjà à 3.2s, et ça ne s'améliore plus au-delà (200 -> 3.5s, 1109 -> 3.9s).
# Pas de dégradation progressive : une vraie falaise entre 1 et 2. Cause
# identifiée par profilage ligne à ligne : une matrice (batch_size, n_paths)
# dépasse le cache L3 dès batch_size=2 (n_paths=100000 -> 800 Ko par ligne),
# et chaque étape (exp, payoff, discount, mean) doit alors relire/réécrire
# toute la matrice en RAM (memory-bound) au lieu de rester en cache — alors
# que le gain visé (moins d'overhead Python) ne dépend PAS de la taille du
# lot. Le vrai gain vient de deux choses qui, elles, restent valables même à
# batch_size=1 :
#  1. rng créé UNE FOIS et réutilisé, au lieu de np.random.default_rng(seed)
#     recréé à chaque option (le reseeding a un coût réel, mesuré ~0.15s sur
#     1109 recréations).
#  2. Un seul np.maximum(sign*(S_T-K), 0) au lieu de calculer les payoffs
#     call ET put avant de choisir (cf. version np.where du premier jet).
# batch_size reste un paramètre ajustable (utile si n_paths est plus petit,
# où le seuil de cache serait différent) mais vaut 1 par défaut ici.
#
# Ce n'est PAS numériquement identique à la boucle scalaire : la version
# scalaire recrée np.random.default_rng(seed) à CHAQUE appel, donc réutilise
# exactement les mêmes tirages Z pour toutes les options du book. Ici, un
# seul rng est partagé sur tout le book -> chaque option reçoit des tirages
# indépendants (statistiquement plus correct, mais les prix individuels ne
# matcheront pas ceux de la boucle scalaire terme à terme).
# ---------------------------------------------------------------------------

def mc_naive_fast(S, K, T, r, sigma, kind, n_paths=100000, seed=42, batch_size=1, compute_std=False):
    """Version rapide (rng persistant + un seul np.maximum) de mc_naive. S/K/T/r/sigma/kind : array-like, shape (n_options,).
    Retourne (prices, std_errors) ; std_errors est un tableau de NaN si
    compute_std=False (évite le coût de .std(axis=1), cf. commentaire ci-dessus)."""
    S, K, T, r, sigma = (np.asarray(x, dtype=float) for x in (S, K, T, r, sigma))
    sign = np.where(np.asarray(kind) == 'call', 1.0, -1.0)
    n_options = len(S)
    rng = np.random.default_rng(seed)

    prices = np.empty(n_options)
    std_errors = np.full(n_options, np.nan)

    for start in range(0, n_options, batch_size):
        end = min(start + batch_size, n_options)
        Sb, Kb, Tb, rb, sigmab, signb = S[start:end], K[start:end], T[start:end], r[start:end], sigma[start:end], sign[start:end]

        Z = rng.standard_normal((end - start, n_paths))
        drift, vol = (rb - 0.5 * sigmab ** 2) * Tb, sigmab * np.sqrt(Tb)
        S_T = Sb[:, None] * np.exp(drift[:, None] + vol[:, None] * Z)

        payoff = np.maximum(signb[:, None] * (S_T - Kb[:, None]), 0)
        discounted = np.exp(-rb * Tb)[:, None] * payoff

        prices[start:end] = discounted.mean(axis=1)
        if compute_std:
            std_errors[start:end] = discounted.std(axis=1) / np.sqrt(n_paths)

    return prices, std_errors


def mc_antithetic_fast(S, K, T, r, sigma, kind, n_paths=100000, seed=42, batch_size=1, compute_std=False):
    """Version rapide (rng persistant + un seul np.maximum) de mc_antithetic."""
    S, K, T, r, sigma = (np.asarray(x, dtype=float) for x in (S, K, T, r, sigma))
    sign = np.where(np.asarray(kind) == 'call', 1.0, -1.0)
    n_options = len(S)
    rng = np.random.default_rng(seed)
    half = n_paths // 2

    prices = np.empty(n_options)
    std_errors = np.full(n_options, np.nan)

    for start in range(0, n_options, batch_size):
        end = min(start + batch_size, n_options)
        Sb, Kb, Tb, rb, sigmab, signb = S[start:end], K[start:end], T[start:end], r[start:end], sigma[start:end], sign[start:end]

        Z = rng.standard_normal((end - start, half))
        drift, vol = (rb - 0.5 * sigmab ** 2) * Tb, sigmab * np.sqrt(Tb)
        S_T1 = Sb[:, None] * np.exp(drift[:, None] + vol[:, None] * Z)
        S_T2 = Sb[:, None] * np.exp(drift[:, None] - vol[:, None] * Z)

        payoff1 = np.maximum(signb[:, None] * (S_T1 - Kb[:, None]), 0)
        payoff2 = np.maximum(signb[:, None] * (S_T2 - Kb[:, None]), 0)
        discounted = np.exp(-rb * Tb)[:, None] * (payoff1 + payoff2) / 2

        prices[start:end] = discounted.mean(axis=1)
        if compute_std:
            std_errors[start:end] = discounted.std(axis=1) / np.sqrt(half)

    return prices, std_errors


def mc_control_fast(S, K, T, r, sigma, kind, n_paths=100000, seed=42, batch_size=1, compute_std=False):
    """Version rapide (rng persistant + un seul np.maximum) de mc_control. Le beta (cov/var) de la variable de
    contrôle est calculé ligne par ligne (par option) : équivalent batché de
    np.cov(payoff, s_t)[0,1]/np.var(s_t), qui ne supporte pas nativement le
    calcul par lot sur des lignes indépendantes."""
    S, K, T, r, sigma = (np.asarray(x, dtype=float) for x in (S, K, T, r, sigma))
    sign = np.where(np.asarray(kind) == 'call', 1.0, -1.0)
    n_options = len(S)
    rng = np.random.default_rng(seed)

    prices = np.empty(n_options)
    std_errors = np.full(n_options, np.nan)

    for start in range(0, n_options, batch_size):
        end = min(start + batch_size, n_options)
        Sb, Kb, Tb, rb, sigmab, signb = S[start:end], K[start:end], T[start:end], r[start:end], sigma[start:end], sign[start:end]

        Z = rng.standard_normal((end - start, n_paths))
        drift, vol = (rb - 0.5 * sigmab ** 2) * Tb, sigmab * np.sqrt(Tb)
        S_T = Sb[:, None] * np.exp(drift[:, None] + vol[:, None] * Z)
        payoff = np.maximum(signb[:, None] * (S_T - Kb[:, None]), 0)

        payoff_c = payoff - payoff.mean(axis=1, keepdims=True)
        S_T_c    = S_T - S_T.mean(axis=1, keepdims=True)
        beta = (payoff_c * S_T_c).mean(axis=1) / S_T.var(axis=1)

        forward = Sb * np.exp(rb * Tb)
        adjusted = payoff - beta[:, None] * (S_T - forward[:, None])
        discounted = np.exp(-rb * Tb)[:, None] * adjusted

        prices[start:end] = discounted.mean(axis=1)
        if compute_std:
            std_errors[start:end] = discounted.std(axis=1) / np.sqrt(n_paths)

    return prices, std_errors


def mc_control_antithetic_fast(S, K, T, r, sigma, kind, n_paths=100000, seed=42, batch_size=1, compute_std=False):
    """Version rapide (rng persistant + un seul np.maximum) de mc_control_antithetic."""
    S, K, T, r, sigma = (np.asarray(x, dtype=float) for x in (S, K, T, r, sigma))
    sign = np.where(np.asarray(kind) == 'call', 1.0, -1.0)
    n_options = len(S)
    rng = np.random.default_rng(seed)
    half = n_paths // 2

    prices = np.empty(n_options)
    std_errors = np.full(n_options, np.nan)

    for start in range(0, n_options, batch_size):
        end = min(start + batch_size, n_options)
        Sb, Kb, Tb, rb, sigmab, signb = S[start:end], K[start:end], T[start:end], r[start:end], sigma[start:end], sign[start:end]

        Z = rng.standard_normal((end - start, half))
        drift, vol = (rb - 0.5 * sigmab ** 2) * Tb, sigmab * np.sqrt(Tb)
        S_T1 = Sb[:, None] * np.exp(drift[:, None] + vol[:, None] * Z)
        S_T2 = Sb[:, None] * np.exp(drift[:, None] - vol[:, None] * Z)

        payoff1 = np.maximum(signb[:, None] * (S_T1 - Kb[:, None]), 0)
        payoff2 = np.maximum(signb[:, None] * (S_T2 - Kb[:, None]), 0)
        payoff_combined  = (payoff1 + payoff2) / 2
        control_combined = (S_T1 + S_T2) / 2

        pc = payoff_combined - payoff_combined.mean(axis=1, keepdims=True)
        cc = control_combined - control_combined.mean(axis=1, keepdims=True)
        beta = (pc * cc).mean(axis=1) / control_combined.var(axis=1)

        disc = np.exp(-rb * Tb)
        discounted_adjusted = disc[:, None] * payoff_combined - beta[:, None] * (disc[:, None] * control_combined - Sb[:, None])

        prices[start:end] = discounted_adjusted.mean(axis=1)
        if compute_std:
            std_errors[start:end] = discounted_adjusted.std(axis=1) / np.sqrt(half)

    return prices, std_errors


# ---------------------------------------------------------------------------
# Versions "threaded" : parallélisent les fonctions "_fast" ci-dessus sur
# plusieurs THREADS (pas des process, contrairement à LSM/PDE — cf.
# monteCarloLSM.py / pde.py). Différence mesurée sur le book complet
# (1109 options, n_paths=100000) : paralléliser en process est contre-
# productif ici (0.39x à 0.87x, coût de spawn Windows >> ~1.5 ms/option de
# travail réel — même diagnostic que PDE (50,100)). Les threads, eux,
# donnent un vrai gain (jusqu'à 3.77x à 8 threads) SANS ce coût de démarrage,
# parce que NumPy relâche le GIL pendant ses calculs C (exp, maximum, mean,
# ...) — le principal goulot d'étranglement d'une fonction "_fast" n'est
# quasiment que du calcul NumPy, donc plusieurs threads peuvent réellement
# tourner en parallèle dessus malgré le GIL Python.
#
# Piège rencontré et corrigé : donner le MÊME seed entier à chaque chunk/
# thread reproduit exactement le problème qu'on avait réglé en créant les
# fonctions "_fast" (tous les chunks retirent alors les mêmes nombres
# aléatoires en interne, biais mesuré : écart de 2.7 vs la version séquentielle
# sur ce book). Chaque thread doit recevoir un enfant de seed VRAIMENT
# indépendant, via np.random.SeedSequence(seed).spawn(n_workers) — pas
# seed+i, qui n'offre aucune garantie d'indépendance statistique.
# ---------------------------------------------------------------------------

def _threaded_mc(fast_fn, S, K, T, r, sigma, kind, n_paths, seed, n_workers):
    S, K, T, r, sigma = (np.asarray(x, dtype=float) for x in (S, K, T, r, sigma))
    kind = np.asarray(kind)
    n = len(S)
    n_workers = min(n_workers, n)
    bounds = np.linspace(0, n, n_workers + 1, dtype=int)
    child_seeds = np.random.SeedSequence(seed).spawn(n_workers)

    def _chunk(i):
        sl = slice(bounds[i], bounds[i + 1])
        return fast_fn(S[sl], K[sl], T[sl], r[sl], sigma[sl], kind[sl], n_paths=n_paths, seed=child_seeds[i])

    idx = [i for i in range(n_workers) if bounds[i + 1] > bounds[i]]
    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        results = list(executor.map(_chunk, idx))

    prices = np.concatenate([res[0] for res in results])
    std_errors = np.concatenate([res[1] for res in results])
    return prices, std_errors


def mc_naive_threaded(S, K, T, r, sigma, kind, n_paths=100000, seed=42, n_workers=8):
    return _threaded_mc(mc_naive_fast, S, K, T, r, sigma, kind, n_paths, seed, n_workers)

def mc_antithetic_threaded(S, K, T, r, sigma, kind, n_paths=100000, seed=42, n_workers=8):
    return _threaded_mc(mc_antithetic_fast, S, K, T, r, sigma, kind, n_paths, seed, n_workers)

def mc_control_threaded(S, K, T, r, sigma, kind, n_paths=100000, seed=42, n_workers=8):
    return _threaded_mc(mc_control_fast, S, K, T, r, sigma, kind, n_paths, seed, n_workers)

def mc_control_antithetic_threaded(S, K, T, r, sigma, kind, n_paths=100000, seed=42, n_workers=8):
    return _threaded_mc(mc_control_antithetic_fast, S, K, T, r, sigma, kind, n_paths, seed, n_workers)


def mc_mean_std(pricer, opt, n_paths, n_seeds, base_seed=0):
    """Moyenne et écart-type du prix MC sur n_seeds tirages indépendants.

    Même logique que lsm_mean_std (cf. monteCarloLSM.py) : une seule
    réalisation (seed fixe) ne permet pas de distinguer un vrai comportement
    de convergence d'un artefact du tirage particulier — on répète donc
    chaque n_paths sur plusieurs seeds indépendants.
    """
    prices = np.array([
        pricer(opt, n_paths, seed=base_seed + s)[0]
        for s in range(n_seeds)
    ])
    return prices.mean(), prices.std()


if __name__ == "__main__":
    opt = Option(S=100, K=100, T=1, r=0.05, sigma=0.2, kind='call')
    n_paths = 100000
    N_SEEDS = 25  # répétitions indépendantes par n_paths (cf. monteCarloLSM.py)

    model = BSModel()
    bsm_price = model.price(opt)
    print(f"BSM price: {bsm_price:.4f}")

    METHODS = {
        "naive":              ("Naive", mc_naive),
        "antithetic":         ("Antithetic", mc_antithetic),
        "control":            ("Control Variate", mc_control),
        "control_antithetic": ("Control + Antithetic", mc_control_antithetic),
    }

    path_counts = [100, 500, 1000, 5000, 10000, 50000, 100000, 500000, 1000000, 5000000, 10000000]

    for key, (label, pricer) in METHODS.items():
        print(f"\n----------------- {label} Monte Carlo pricing with {n_paths} paths -----------------")
        price, std_error = pricer(opt, n_paths)
        print(f"Monte Carlo price: {price:.4f}")
        print(f"Standard error (théorique, seed unique): {std_error:.4f}")
        print(f"BSM price: {bsm_price:.4f}")
        print(f"Difference: {abs(price - bsm_price):.4f}, Relative Difference: {abs(price - bsm_price)/bsm_price:.4%}")

        print(f"--- Convergence ({N_SEEDS} seeds indépendants par n_paths) ---")
        rel_diffs, rel_diffs_std = [], []
        for n in path_counts:
            mean_price, std_price = mc_mean_std(pricer, opt, n, N_SEEDS)
            rel_diff     = abs(mean_price - bsm_price) / bsm_price * 100
            rel_diff_std = std_price / bsm_price * 100
            rel_diffs.append(rel_diff)
            rel_diffs_std.append(rel_diff_std)
            print(f"n_paths={n:9d}  price={mean_price:.4f}±{std_price:.4f}  "
                  f"écart BSM={rel_diff:.4f}%±{rel_diff_std:.4f}%")

        plt.figure()
        plt.errorbar(path_counts, rel_diffs, yerr=rel_diffs_std, fmt='o-',
                     capsize=3, alpha=0.85,
                     label=f'{label} (moyenne ± écart-type, {N_SEEDS} seeds)')
        plt.axhline(0, color='red', ls='--', lw=0.8)
        plt.xscale('log')
        plt.xlabel('Number of Paths')
        plt.ylabel('Relative Difference from BSM Price (%)\n(moyenne ± écart-type)')
        plt.title(f'Monte Carlo Price Convergence — {label}')
        plt.legend()
        plt.savefig(REPORTS / f"monte_carlo_convergence_{key}.png")
        plt.show()