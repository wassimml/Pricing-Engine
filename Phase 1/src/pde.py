from pathlib import Path
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import QuantLib as ql

from option import Option
import BSpricer

DATA = Path(__file__).parent.parent / "data"
REPORTS = Path(__file__).parent.parent / "reports"

def pde_crank_nicolson(option: Option, style: str = 'american', n_steps: int = 200, n_space: int = 200) -> float:
    """Price an option using the Crank-Nicolson finite difference method via QuantLib.

    Parameters:
        option: Option dataclass with S, K, T, r, sigma, kind.
        style:  'american' or 'european'.
        n_steps: Number of time steps for the finite difference grid.
        n_space: Number of space steps for the finite difference grid.
    Returns:
        The option price.
    """
    today = ql.Date.todaysDate()
    maturity = today + ql.Period(int(option.T * 365), ql.Days)
    ql.Settings.instance().evaluationDate = today

    dc = ql.Actual365Fixed()
    process = ql.BlackScholesProcess(
        ql.QuoteHandle(ql.SimpleQuote(option.S)),
        ql.YieldTermStructureHandle(ql.FlatForward(today, ql.QuoteHandle(ql.SimpleQuote(option.r)), dc)),
        ql.BlackVolTermStructureHandle(ql.BlackConstantVol(today, ql.NullCalendar(), ql.QuoteHandle(ql.SimpleQuote(option.sigma)), dc))
    )

    payoff_type = ql.Option.Call if option.kind == 'call' else ql.Option.Put
    payoff = ql.PlainVanillaPayoff(payoff_type, option.K)

    if style == 'american':
        exercise = ql.AmericanExercise(today, maturity)
    else:
        exercise = ql.EuropeanExercise(maturity)

    vanilla = ql.VanillaOption(payoff, exercise)
    # FdBlackScholesVanillaEngine uses ql.FdmSchemeDesc.Douglas() per default. 
    vanilla.setPricingEngine(ql.FdBlackScholesVanillaEngine(process, n_steps, n_space, 2, ql.FdmSchemeDesc.CrankNicolson()))
    return vanilla.NPV()


if __name__ == "__main__":
    # Étude de convergence PDE vs BS — options européennes uniquement : BS
    # donne un prix exact, donc une vraie référence pour mesurer l'erreur.
    # Pour l'américain il n'existe pas de forme fermée, impossible de faire
    # la même mesure.
    opt = Option(S=100, K=100, T=1, r=0.05, sigma=0.2, kind='put')

    bs_price = BSpricer.BSModel().price(opt)
    print(f"Prix de référence (BS, exact) : {bs_price:.6f}")

    pde_default = pde_crank_nicolson(opt, style='european', n_steps=200, n_space=200)
    print(f"Prix via PDE Crank-Nicolson (n_steps=200, n_space=200) : {pde_default:.6f}  "
          f"(erreur={abs(pde_default - bs_price) / bs_price:.4%})")

    ERROR_THRESHOLD = 1e-4  # 0.01% d'erreur relative — seuil "raisonnable" retenu ici
    PARAM_GRID = [10, 20, 50, 100, 200, 400, 800]
    FIXED = 200  # doit être dans PARAM_GRID : sert de coupe fixe pour les 2 courbes 1D

    # - Grille n_steps x n_space (une seule passe, réutilisée pour tout) -------
    print(f"\n--- Grille n_steps x n_space (Put européen, réf. BS = {bs_price:.6f}) ---")
    combos = {}
    for n_steps in PARAM_GRID:
        for n_space in PARAM_GRID:
            t0 = time.perf_counter()
            price = pde_crank_nicolson(opt, style='european', n_steps=n_steps, n_space=n_space)
            elapsed = time.perf_counter() - t0
            error = abs(price - bs_price) / bs_price
            combos[(n_steps, n_space)] = (price, error, elapsed)
            print(f"  n_steps={n_steps:4d}  n_space={n_space:4d}  PDE={price:.6f}  "
                  f"erreur={error:.4%}  ({elapsed*1000:.3f}ms)")

    # Coupes 1D extraites de la grille (pas de recalcul)
    results_steps = [(n, *combos[(n, FIXED)]) for n in PARAM_GRID]
    results_space = [(n, *combos[(FIXED, n)]) for n in PARAM_GRID]

    # - Graphes de convergence 1D ------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    axes[0].plot([r[0] for r in results_steps], [r[2] for r in results_steps], 'o-', color='steelblue')
    axes[0].axhline(ERROR_THRESHOLD, color='red', ls='--', lw=1, label=f'Seuil {ERROR_THRESHOLD:.2%}')
    axes[0].set_xscale('log')
    axes[0].set_yscale('log')
    axes[0].set_xlabel('n_steps')
    axes[0].set_ylabel('Erreur relative vs BS')
    axes[0].set_title(f'Convergence vs n_steps (n_space={FIXED} fixé)')
    axes[0].legend()

    axes[1].plot([r[0] for r in results_space], [r[2] for r in results_space], 'o-', color='orange')
    axes[1].axhline(ERROR_THRESHOLD, color='red', ls='--', lw=1, label=f'Seuil {ERROR_THRESHOLD:.2%}')
    axes[1].set_xscale('log')
    axes[1].set_yscale('log')
    axes[1].set_xlabel('n_space')
    axes[1].set_ylabel('Erreur relative vs BS')
    axes[1].set_title(f'Convergence vs n_space (n_steps={FIXED} fixé)')
    axes[1].legend()

    plt.suptitle('Convergence PDE Crank-Nicolson vs BS — Put européen (S=K=100, T=1, r=5%, σ=20%)\n'
                 f'Référence BS = {bs_price:.6f}', fontsize=12)
    plt.tight_layout()
    plt.savefig(REPORTS / 'pde_convergence.png', dpi=150)
    plt.show()

    # - Paramètres optimaux par balayage exhaustif -------------------------------
    # Parmi toutes les combinaisons (n_steps, n_space) de la grille dont
    # l'erreur passe sous le seuil, on choisit la moins chère en temps
    # d'exécution — c'est la config à utiliser en pratique (ex. pour pricer
    # un book entier d'options, cf. benchmarkSPY.py) sous contrainte de temps.
    all_combos = [(ns, nx, err, el) for (ns, nx), (_, err, el) in combos.items()]
    valid = [c for c in all_combos if c[2] < ERROR_THRESHOLD]

    print(f"\n--- Paramètres optimaux, balayage exhaustif (erreur < {ERROR_THRESHOLD:.2%}, temps minimal) ---")
    best_ns = best_nx = best_err = best_t = None
    if valid:
        best_ns, best_nx, best_err, best_t = min(valid, key=lambda c: c[3])
        print(f"  n_steps={best_ns}  n_space={best_nx}  erreur={best_err:.4%}  "
              f"temps={best_t*1000:.3f}ms/option")
        print(f"  soit ~{best_t*1000:.2f}s pour pricer 1000 options à ces paramètres.")
    else:
        print(f"  Aucune configuration testée n'atteint le seuil de {ERROR_THRESHOLD:.2%} "
              f"— élargir PARAM_GRID.")

    # - Recherche par bissection (mécanisme à la brentq, pour 2 entiers) --------
    # brentq resserre un bracket [a, b] par bissection jusqu'à isoler le zéro
    # d'une fonction monotone. Ici il n'y a pas de zéro mais un seuil : on
    # cherche, sur chaque dimension, le plus petit entier tel que l'erreur
    # passe sous ERROR_THRESHOLD et y reste. Le mécanisme est le même
    # (bracket qui se resserre de moitié à chaque étape) — la condition
    # nécessaire n'est pas que l'erreur soit strictement monotone (elle ne
    # l'est pas tout à fait ici, cf. la grille : à n_space=200 fixé l'erreur
    # remonte légèrement de n_steps=100 à 800), seulement que le passage sous
    # le seuil soit définitif, ce qui est bien le cas dans la grille ci-dessus.
    # On alterne la bissection sur n_steps et n_space (l'un fixé pendant que
    # l'autre est cherché) jusqu'à stabilité, ce qui gère leur interaction.
    def bisect_min_int(f, lo, hi, threshold):
        """Plus petit entier n dans [lo, hi] tel que f(n) <= threshold, en
        supposant seulement que ce passage sous le seuil est définitif
        (pas besoin que f soit elle-même strictement monotone)."""
        if f(hi) > threshold:
            return None  # même la borne haute ne suffit pas
        if f(lo) <= threshold:
            return lo
        while hi - lo > 1:
            mid = (lo + hi) // 2
            if f(mid) <= threshold:
                hi = mid
            else:
                lo = mid
        return hi

    n_evals = [0]

    def counted_error_at(n_steps, n_space):
        n_evals[0] += 1
        price = pde_crank_nicolson(opt, style='european', n_steps=n_steps, n_space=n_space)
        return abs(price - bs_price) / bs_price

    print("\n--- Recherche par bissection (n_steps, n_space) alternée ---")
    lo, hi = PARAM_GRID[0], PARAM_GRID[-1]
    n_steps_b, n_space_b = hi, hi  # départ pessimiste (le plus précis du domaine)

    for cycle in range(3):
        prev = (n_steps_b, n_space_b)

        # Si aucun n_space ne suffit pour le n_steps courant (surface pas
        # parfaitement monotone : l'erreur peut redescendre sous le seuil
        # puis y remonter plus loin, cf. la grille à n_space=100 fixé), on
        # garde la dernière paire valide plutôt que de planter sur un None.
        candidate_space = bisect_min_int(lambda ns: counted_error_at(n_steps_b, ns), lo, hi, ERROR_THRESHOLD)
        if candidate_space is None:
            print(f"  cycle {cycle + 1} : aucun n_space ne passe le seuil pour n_steps={n_steps_b} - arrêt")
            break
        n_space_b = candidate_space

        candidate_steps = bisect_min_int(lambda ns: counted_error_at(ns, n_space_b), lo, hi, ERROR_THRESHOLD)
        if candidate_steps is None:
            print(f"  cycle {cycle + 1} : aucun n_steps ne passe le seuil pour n_space={n_space_b} - arrêt")
            break
        n_steps_b = candidate_steps

        print(f"  cycle {cycle + 1} : n_steps={n_steps_b}  n_space={n_space_b}  "
              f"({n_evals[0]} évaluations PDE cumulées)")
        if (n_steps_b, n_space_b) == prev:
            break

    err_b = counted_error_at(n_steps_b, n_space_b)
    t0 = time.perf_counter()
    pde_crank_nicolson(opt, style='european', n_steps=n_steps_b, n_space=n_space_b)
    t_b = time.perf_counter() - t0

    print(f"\nRésultat bissection : n_steps={n_steps_b}  n_space={n_space_b}  "
          f"erreur={err_b:.4%}  temps={t_b * 1000:.3f}ms/option  "
          f"({n_evals[0]} évaluations PDE au total)")
    print(f"Comparaison au balayage exhaustif ({len(combos)} évaluations) : "
          f"n_steps={best_ns}  n_space={best_nx}  erreur={best_err:.4%}  temps={best_t * 1000:.3f}ms/option")

    # - Compromis précision / temps : les deux optimaux sur le même graphe ------
    fig2, ax2 = plt.subplots(figsize=(9, 7))
    invalid = [c for c in all_combos if c[2] >= ERROR_THRESHOLD]
    ax2.scatter([c[3]*1000 for c in invalid], [c[2] for c in invalid],
                color='lightgray', s=25, label='Sous le seuil requis')
    ax2.scatter([c[3]*1000 for c in valid], [c[2] for c in valid],
                color='seagreen', s=25, label='Atteint le seuil')
    ax2.axhline(ERROR_THRESHOLD, color='red', ls='--', lw=1, label=f'Seuil {ERROR_THRESHOLD:.2%}')

    if best_ns is not None:
        ax2.scatter([best_t*1000], [best_err], color='steelblue', s=110, zorder=5,
                    edgecolor='black', marker='D',
                    label=f'Optimal grille : n_steps={best_ns}, n_space={best_nx}')

    ax2.scatter([t_b*1000], [err_b], color='gold', s=220, zorder=6,
                edgecolor='black', marker='*',
                label=f'Optimal bissection (gagnant) : n_steps={n_steps_b}, n_space={n_space_b}')

    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.set_xlabel('Temps de calcul (ms)')
    ax2.set_ylabel('Erreur relative vs BS')
    ax2.set_title('Compromis précision / temps — PDE Crank-Nicolson (grille n_steps x n_space)')
    ax2.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(REPORTS / 'pde_tradeoff.png', dpi=150)
    plt.show()

    # =============================================================================
    # - Validation multi-cas : book européen complet (options_benchmark.xlsx) ---
    # =============================================================================
    # Tout ce qui précède ne teste qu'un seul cas (put ATM, T=1). Les paramètres
    # "optimaux" trouvés ne sont valables que pour ce cas précis : un schéma aux
    # différences finies converge différemment selon la géométrie du problème
    # (maturité courte, strike loin du spot, volatilité extrême...). On répète
    # donc l'étude sur le book européen déjà utilisé ailleurs dans le projet
    # (benchmarkMethods.py, benchmarkInData.py), et on choisit les paramètres qui
    # minimisent à la fois l'erreur MOYENNE et l'erreur MAXIMALE (pire cas) sur
    # TOUT le book, sans filtre de vol : chaque option doit recevoir un prix, y
    # compris celles à forte vol, donc les paramètres retenus doivent être ceux
    # qui marchent le mieux sur l'ensemble réel, pas sur un sous-échantillon.
    #
    # Limite structurelle observée : sur les options à très forte vol (σ>50%
    # environ, ex. σ=83% -> erreur ~$0.34 même à n_steps=n_space=2000), l'erreur
    # ne vient PAS d'un manque de résolution — elle reste identique à toute
    # finesse de grille testée. Le constructeur simplifié
    # ql.FdBlackScholesVanillaEngine(process, n_steps, n_space, ...) ne permet
    # pas de régler l'étendue spatiale de la grille (nombre d'écarts-types
    # couverts) — elle est fixée en interne par QuantLib, indépendamment de
    # n_steps/n_space. Conséquence concrète : le critère "erreur max" peut ne
    # jamais être satisfaisable sur le book complet, quel que soit le budget de
    # temps — ce n'est pas un bug de la recherche, c'est documenté et rapporté
    # explicitement ci-dessous si aucune config ne passe le seuil.
    VOL_CUTOFF = 0.50  # seuil indicatif utilisé seulement pour colorer le graphe erreur-vs-vol
    MEAN_THRESHOLD_BOOK = 0.02   # $0.02 d'erreur absolue moyenne sur le book
    MAX_THRESHOLD_BOOK  = 0.22   # $0.22 d'erreur absolue max — peut être inatteignable, cf. ci-dessus
    MAX_THRESHOLD_BOOK_RELAXED = 0.40  # $0.40 — seuil relâché : un regroupement de configs passe
                                        # sous ce niveau alors qu'aucune ne passe sous $0.22

    book_raw = pd.read_excel(DATA / "options_benchmark.xlsx", sheet_name=0, header=2, index_col=0)
    book = book_raw[book_raw["Style"] != "American"]
    book = book[["Kind", "Spot  S", "Strike  K", "T  (years)", "Rate  r (%)", "Vol  σ (%)"]]
    print(f"\n--- Validation sur book europeen complet ({len(book)} options, toutes vols) ---")

    book_opts = [Option(S=r[1], K=r[2], T=r[3], r=r[4] / 100, sigma=r[5] / 100, kind=r[0].lower())
                 for r in book.itertuples(index=False)]
    book_bs = np.array([BSpricer.BSModel().price(o) for o in book_opts])
    book_vols = book["Vol  σ (%)"].to_numpy()

    book_combos = {}
    for n_steps in PARAM_GRID:
        for n_space in PARAM_GRID:
            t0 = time.perf_counter()
            prices = np.array([pde_crank_nicolson(o, style='european', n_steps=n_steps, n_space=n_space)
                                for o in book_opts])
            elapsed = time.perf_counter() - t0
            abs_err = np.abs(prices - book_bs)
            mean_err = abs_err.mean()
            max_idx = int(np.argmax(abs_err))
            max_err = abs_err[max_idx]
            book_combos[(n_steps, n_space)] = (mean_err, max_err, elapsed, max_idx, abs_err)
            print(f"  n_steps={n_steps:4d}  n_space={n_space:4d}  mean_err=${mean_err:.4f}  "
                  f"max_err=${max_err:.4f}  ({elapsed:.2f}s, {elapsed/len(book_opts)*1000:.3f}ms/option)")

    all_book = [(ns, nx, me, mx, el) for (ns, nx), (me, mx, el, _, _) in book_combos.items()]

    def report_book_optimal(criterion_idx, threshold, label):
        valid_c = [c for c in all_book if c[criterion_idx] < threshold]
        print(f"\n--- Paramètres optimaux, book européen complet, critère {label} < {threshold} (temps minimal) ---")
        if not valid_c:
            print(f"  Aucune configuration testée ne passe ce seuil sur le book complet — attendu si le "
                  f"plancher structurel (options à forte vol) dépasse le seuil quelle que soit la résolution.")
            return None
        best = min(valid_c, key=lambda c: c[4])
        ns, nx, me, mx, el = best
        _, _, _, worst_idx, _ = book_combos[(ns, nx)]
        worst_row = book.iloc[worst_idx]
        print(f"  n_steps={ns}  n_space={nx}  mean_err=${me:.4f}  max_err=${mx:.4f}  "
              f"temps={el/len(book_opts)*1000:.3f}ms/option")
        print(f"  Pire contrat a cette config : {worst_row['Kind']} S={worst_row['Spot  S']:.1f} "
              f"K={worst_row['Strike  K']:.1f} T={worst_row['T  (years)']:.2f} "
              f"vol={worst_row['Vol  σ (%)']:.1f}%")
        return best

    best_mean = report_book_optimal(2, MEAN_THRESHOLD_BOOK, "erreur moyenne")
    best_max = report_book_optimal(3, MAX_THRESHOLD_BOOK, "erreur max (pire cas)")
    best_max_relaxed = report_book_optimal(3, MAX_THRESHOLD_BOOK_RELAXED, "erreur max (pire cas, seuil relâché)")

    # - Graphes tradeoff book : moyenne et max, côte à côte ----------------------
    fig3, (ax3, ax4) = plt.subplots(1, 2, figsize=(15, 7))

    for ax, crit_idx, threshold, best, label, ylabel, threshold2, best2 in [
        (ax3, 2, MEAN_THRESHOLD_BOOK, best_mean, "moyenne", "Erreur absolue moyenne ($)", None, None),
        (ax4, 3, MAX_THRESHOLD_BOOK, best_max, "max (pire cas)", "Erreur absolue max ($)",
         MAX_THRESHOLD_BOOK_RELAXED, best_max_relaxed),
    ]:
        times_ms = [c[4] / len(book_opts) * 1000 for c in all_book]
        errs = [c[crit_idx] for c in all_book]
        ok = [c[crit_idx] < threshold for c in all_book]
        ax.scatter([t for t, o in zip(times_ms, ok) if not o], [e for e, o in zip(errs, ok) if not o],
                   color='lightgray', s=25, label='Sous le seuil requis')
        ax.scatter([t for t, o in zip(times_ms, ok) if o], [e for e, o in zip(errs, ok) if o],
                   color='seagreen', s=25, label='Atteint le seuil')
        ax.axhline(threshold, color='red', ls='--', lw=1, label=f'Seuil ${threshold}')
        if best is not None:
            ns, nx, me, mx, el = best
            ax.scatter([el / len(book_opts) * 1000], [best[crit_idx]], color='gold', s=200, zorder=5,
                       edgecolor='black', marker='*', label=f'Optimal : n_steps={ns}, n_space={nx}')

        # Deuxième seuil (relâché), uniquement pour le critère max : dessiné
        # par-dessus les points gris pour les recolorer sans dupliquer le nuage.
        if threshold2 is not None:
            ok2 = [c[crit_idx] < threshold2 for c in all_book]
            ax.scatter([t for t, o in zip(times_ms, ok2) if o], [e for e, o in zip(errs, ok2) if o],
                       color='darkorange', s=25, zorder=3, label=f'Atteint le seuil relâché (${threshold2})')
            ax.axhline(threshold2, color='purple', ls='--', lw=1, label=f'Seuil relâché ${threshold2}')
            if best2 is not None:
                ns2, nx2, me2, mx2, el2 = best2
                ax.scatter([el2 / len(book_opts) * 1000], [best2[crit_idx]], color='deepskyblue', s=200, zorder=6,
                           edgecolor='black', marker='*',
                           label=f'Optimal (seuil relâché) : n_steps={ns2}, n_space={nx2}')

        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel('Temps de calcul (ms/option)')
        ax.set_ylabel(ylabel)
        ax.set_title(f'Critère : erreur {label}')
        ax.legend(fontsize=7)

    plt.suptitle(f'Validation book européen complet (n={len(book_opts)}, toutes vols) — '
                 'compromis précision / temps : moyenne vs pire cas', fontsize=12)
    plt.tight_layout()
    plt.savefig(REPORTS / 'pde_book_tradeoff.png', dpi=150)
    plt.show()

    # - Erreur par option en fonction de σ, aux paramètres retenus (erreur MOYENNE)
    # Réutilise l'erreur déjà calculée pendant la grille ci-dessus (pas de
    # recalcul) : sert à visualiser où se concentre l'erreur résiduelle une
    # fois les paramètres choisis sur le book entier, et confirme visuellement
    # le plancher structurel décrit plus haut pour les options à forte vol.
    if best_mean is not None:
        ns_final, nx_final = best_mean[0], best_mean[1]
    else:
        ns_final, nx_final = PARAM_GRID[-1], PARAM_GRID[-1]  # repli : config la plus fine testée
    final_abs_err = book_combos[(ns_final, nx_final)][4]
    below_cutoff = book_vols <= VOL_CUTOFF * 100

    print(f"\n--- Répartition de l'erreur (n_steps={ns_final}, n_space={nx_final}) ---")
    print(f"  vol <= {VOL_CUTOFF:.0%} ({below_cutoff.sum()} options)  : "
          f"moyenne=${final_abs_err[below_cutoff].mean():.4f}  max=${final_abs_err[below_cutoff].max():.4f}")
    print(f"  vol >  {VOL_CUTOFF:.0%} ({(~below_cutoff).sum()} options) : "
          f"moyenne=${final_abs_err[~below_cutoff].mean():.4f}  max=${final_abs_err[~below_cutoff].max():.4f}")

    fig4, ax5 = plt.subplots(figsize=(10, 6))
    ax5.scatter(book_vols[below_cutoff], final_abs_err[below_cutoff], color='seagreen', s=20, alpha=0.6,
                label=f'σ ≤ {VOL_CUTOFF:.0%}')
    ax5.scatter(book_vols[~below_cutoff], final_abs_err[~below_cutoff], color='tomato', s=20, alpha=0.6,
                label=f'σ > {VOL_CUTOFF:.0%}')
    ax5.axvline(VOL_CUTOFF * 100, color='black', ls=':', lw=1, label=f'σ = {VOL_CUTOFF:.0%} (repère)')
    ax5.axhline(MAX_THRESHOLD_BOOK, color='red', ls='--', lw=1, label=f'Seuil max visé (${MAX_THRESHOLD_BOOK})')
    ax5.set_xlabel('Volatilité σ (%)')
    ax5.set_ylabel('Erreur absolue vs BS ($)')
    ax5.set_title(f'Book complet (n={len(book_opts)}) — n_steps={ns_final}, n_space={nx_final} '
                  '(optimal erreur moyenne, book entier)\n'
                  'Erreur par option en fonction de σ — illustre le plancher structurel au-delà de 50%')
    ax5.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(REPORTS / 'pde_full_book_error_vs_vol.png', dpi=150)
    plt.show()