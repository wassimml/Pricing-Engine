import numpy as np
import time

from option import Option

def crr_price(option: Option, period: int, american: bool = False, verbose: bool = False) -> float:
    dt = option.T / period
    u = np.exp(option.sigma * np.sqrt(dt))
    d = 1 / u
    p = (np.exp(option.r * dt) - d) / (u - d)
    disc = np.exp(-option.r * dt)

    # Tableau triangulaire des prix S(i,j) pour i=0..period, j=0..i
    # On stocke tout dans des listes de tableaux NumPy (une ligne par étape i)
    # u^j * d^(i-j) = u^j * u^-(i-j) = u^(2j-i) puisque d=1/u : une seule
    # opération de puissance au lieu de deux (+ une multiplication évitée) —
    # c'est cette opération qui domine le coût quand period grandit.
    S_tree = [option.S * (u ** (2 * np.arange(i + 1) - i))
              for i in range(period + 1)]

    # Valeurs terminales (vectorisé)
    S_T = S_tree[period]
    if option.kind == "call":
        values = np.maximum(S_T - option.K, 0.0)
    else:
        values = np.maximum(option.K - S_T, 0.0)

    tree = [None] * (period + 1)
    tree[period] = values.copy()

    calc_tree = [None] * (period + 1)
    calc_tree[period] = None

    # Backward induction vectorisée
    for i in range(period - 1, -1, -1):
        v_up = values[1:i + 2]
        v_down = values[0:i + 1]
        continuation = disc * (p * v_up + (1 - p) * v_down)
        S_i = S_tree[i]
        if option.kind == "call":
            intrinsic = np.maximum(S_i - option.K, 0.0)
        else:
            intrinsic = np.maximum(option.K - S_i, 0.0)

        if american:
            values = np.maximum(continuation, intrinsic)
            exercised = intrinsic > continuation
        else:
            values = continuation
            exercised = np.zeros_like(continuation, dtype=bool)

        tree[i] = values.copy()
        calc_tree[i] = (v_up.copy(), v_down.copy(), continuation.copy(),
                         intrinsic.copy(), exercised.copy())

    if verbose:
        print_tree_visual(tree, calc_tree, S_tree, p, disc, period, option, american)

    return float(tree[0][0])


def crr_price_fast(S, K, T, r, sigma, kind, american, period: int) -> np.ndarray:
    """Prix CRR (binomial) vectorisé sur tout un book, à `period` fixe partagé.

    S/K/T/r/sigma : array-like, shape (n_options,).
    kind : array-like de 'call'/'put'. american : array-like de bool (True =
    style américain pour cette ligne, mélange possible dans le même appel).
    Retourne un array de prix, shape (n_options,).

    Contrairement aux versions "fast" de monteCarlo.py, ici le batching à
    travers les options aide réellement (vérifié empiriquement, pas supposé) :
    la largeur des matrices est `period` (<=1000 dans nos balayages), pas
    n_paths (100000) — même à 2000 options, une matrice (2000, 1000) ne pèse
    que ~16 Mo, largement dans la zone où NumPy reste efficace, contrairement
    au cas Monte Carlo où (n_options, n_paths) explosait le cache L3 dès 2
    options groupées. La boucle sur `period` reste séquentielle (induction
    rétrograde), mais chaque étape traite maintenant TOUTES les options d'un
    coup au lieu d'une boucle Python par option — le nombre d'itérations
    Python passe de (n_options x period) à seulement `period`.
    """
    S, K, T, r, sigma = (np.asarray(x, dtype=float) for x in (S, K, T, r, sigma))
    sign = np.where(np.asarray(kind) == 'call', 1.0, -1.0)
    american = np.asarray(american, dtype=bool)[:, None]

    dt = T / period
    u = np.exp(sigma * np.sqrt(dt))
    d = 1.0 / u
    p = (np.exp(r * dt) - d) / (u - d)
    disc = np.exp(-r * dt)

    # u^j * d^(i-j) = u^(2j-i) puisque d=1/u : une seule puissance au lieu de
    # deux (+ une multiplication évitée), à chaque étape — c'est cette
    # opération qui domine le coût quand period grandit, cf. crr_price.
    j = np.arange(period + 1)
    S_T = S[:, None] * (u[:, None] ** (2 * j - period)[None, :])
    values = np.maximum(sign[:, None] * (S_T - K[:, None]), 0.0)

    for i in range(period - 1, -1, -1):
        v_up, v_down = values[:, 1:i + 2], values[:, 0:i + 1]
        continuation = disc[:, None] * (p[:, None] * v_up + (1 - p[:, None]) * v_down)

        j = np.arange(i + 1)
        S_i = S[:, None] * (u[:, None] ** (2 * j - i)[None, :])
        intrinsic = np.maximum(sign[:, None] * (S_i - K[:, None]), 0.0)

        values = np.where(american, np.maximum(continuation, intrinsic), continuation)

    return values[:, 0]


def print_tree_visual(tree, calc_tree, S_tree, p, disc, period, option, american):
    style = "AMERICAN" if american else "EUROPEAN"
    print(f"\nCRR Binomial Tree — {style} {option.kind.upper()} K={option.K}  "
          f"S0={option.S}  T={option.T}  σ={option.sigma}  r={option.r}  N={period}\n")
    print(f"p* = {p:.4f}\n")

    for i in range(period + 1):
        header = f"┌─ t={i} ({i}/{period}) " + "─" * 6
        print(header)
        for j in range(i, -1, -1):
            S_val = S_tree[i][j]
            V_val = tree[i][j]
            marker = "●" if i == period else "○"

            if calc_tree[i] is None:
                # Nœud terminal : juste le payoff
                if option.kind == "call":
                    calc_str = f"max({S_val:.2f}-{option.K}, 0)"
                else:
                    calc_str = f"max({option.K}-{S_val:.2f}, 0)"
                print(f"│  {marker} S={S_val:7.2f}  V = {calc_str} = {V_val:.3f}")

            else:
                v_up, v_down, continuation, intrinsic, exercised = calc_tree[i]
                cont_str = (f"e^(-rΔt)·[{p:.3f}·{v_up[j]:.3f} + "
                            f"{1-p:.3f}·{v_down[j]:.3f}] = {continuation[j]:.3f}")

                if american:
                    flag = " ⚡ EXERCICE" if exercised[j] else ""
                    calc_str = (f"max(cont={continuation[j]:.3f}, "
                                f"intrinsèque={intrinsic[j]:.3f}){flag}")
                    print(f"│  {marker} S={S_val:7.2f}  cont = {cont_str}")
                    print(f"│  {' '*len(marker)} {' '*len(f'S={S_val:7.2f}')}  "
                          f"V = {calc_str} = {V_val:.3f}")
                else:
                    print(f"│  {marker} S={S_val:7.2f}  V = {cont_str}")

        print("└" + "─" * (len(header) - 1) + "\n")

    print(f"Prix de l'option (t=0) : {tree[0][0]:.4f}\n")


if __name__ == "__main__":
    option = Option(S=100, K=100, T=1, r=0.05, sigma=0.2, kind="put")
    period = 3
    
    start = time.perf_counter()
    price_eu = crr_price(option, period, american=False, verbose=True)
    price_us = crr_price(option, period, american=True, verbose=True)
    elapsed = time.perf_counter() - start

    print(f"European: {price_eu:.4f}  |  American: {price_us:.4f}  |  "
          f"Early exercise premium: {price_us - price_eu:.4f}")
    print(f"Calculé en {elapsed*1000:.1f} ms pour N={period}")