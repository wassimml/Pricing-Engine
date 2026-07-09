import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.optimize import brentq
import matplotlib.pyplot as plt
from pathlib import Path

from BSpricer import BSModel
from binomial import crr_price
from monteCarloLSM import LSMoptionValue
from pde import pde_crank_nicolson
from option import Option

DATA = Path(__file__).parent.parent / "data"
REPORTS = Path(__file__).parent.parent / "reports"

# Snapshot figé de la chaîne calls AAPL (cf. makeSnapshotAAPL.py), rejoué ici
# pour recalculer la smile / la surface de vol implicite sur des données
# identiques d'un run à l'autre (reproductibilité des tests).
SNAPSHOT_PATH = DATA / "AAPL_options_2026-07-09.csv"


def BS_price(S: float, K: float, T: float, r: float, sigma: float, option_type: str = "call"):
    """
    Calcule le prix d'une option européenne selon le modèle de Black-Scholes-Merton.

    S             : prix spot de l'actif
    K             : strike
    T             : temps jusqu'à maturité (en années)
    r             : taux sans risque
    sigma         : volatilité
    option_type   : "call" ou "put"
    """
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if option_type == "call":
        price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    elif option_type == "put":
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    else:
        raise ValueError("option_type must be 'call' or 'put'")

    return price

def implied_volatility_CRR(market_price: float, S: float, K: float, T: float, r: float, option_type: str = "call"):
    """
    Calcule la volatilité implicite par résolution numérique.

    market_price : prix observé de l'option
    S             : prix spot de l'actif
    K             : strike
    T             : temps jusqu'à maturité (en années)
    r             : taux sans risque

    Retourne np.nan si aucune solution n'existe dans l'intervalle de recherche.
    """
    def objective(sigma):
        opt = Option(S=S, K=K, T=T, r=r, sigma=sigma, kind=option_type)
        return crr_price(opt, 500, True) - market_price

    # Borne basse à 1% (pas 1e-6) : en dessous, u ≈ d ≈ 1 dans l'arbre CRR,
    # la probabilité risque-neutre p=(e^{rΔt}-d)/(u-d) explose (u-d ≈ 0) et
    # l'induction rétrograde diverge (overflow / NaN) — pas une vol réaliste
    # de toute façon.
    sigma_min = 1e-2

    # Vérifier que la solution existe dans l'intervalle avant d'appeler brentq
    f_low = objective(sigma_min)
    f_high = objective(5)
    if np.isnan(f_low) or np.isnan(f_high) or f_low * f_high > 0:
        return np.nan

    # Recherche entre 1% et 500% de volatilité
    iv = brentq(objective, sigma_min, 5)
    return iv


def implied_volatility_LSM(market_price: float, S: float, K: float, T: float, r: float, option_type: str = "call"):
    """
    Calcule la volatilité implicite par résolution numérique (pricer Longstaff-Schwartz).

    market_price : prix observé de l'option
    S             : prix spot de l'actif
    K             : strike
    T             : temps jusqu'à maturité (en années)
    r             : taux sans risque

    Retourne np.nan si aucune solution n'existe dans l'intervalle de recherche.
    """
    # n_paths/n_steps réduits par rapport à un pricing ponctuel (20000/50) :
    # brentq évalue l'objectif une dizaine de fois par option, et le seed fixe
    # (cf. LSMoptionValue) rend la fonction déterministe donc toujours
    # inversible, même avec moins de trajectoires — nécessaire pour que
    # l'inversion reste rapide sur toute la smile/surface.
    def objective(sigma):
        opt = Option(S=S, K=K, T=T, r=r, sigma=sigma, kind=option_type)
        return LSMoptionValue(opt, n_steps=30, n_paths=4000) - market_price

    sigma_min = 1e-2
    f_low = objective(sigma_min)
    f_high = objective(5)
    if np.isnan(f_low) or np.isnan(f_high) or f_low * f_high > 0:
        return np.nan

    iv = brentq(objective, sigma_min, 5)
    return iv


def implied_volatility_PDE(market_price: float, S: float, K: float, T: float, r: float, option_type: str = "call"):
    """
    Calcule la volatilité implicite par résolution numérique (PDE Crank-Nicolson).

    market_price : prix observé de l'option
    S             : prix spot de l'actif
    K             : strike
    T             : temps jusqu'à maturité (en années)
    r             : taux sans risque

    Retourne np.nan si aucune solution n'existe dans l'intervalle de recherche.
    """
    def objective(sigma):
        opt = Option(S=S, K=K, T=T, r=r, sigma=sigma, kind=option_type)
        return pde_crank_nicolson(opt, style="american", n_steps=200, n_space=200) - market_price

    sigma_min = 1e-2
    f_low = objective(sigma_min)
    f_high = objective(5)
    if np.isnan(f_low) or np.isnan(f_high) or f_low * f_high > 0:
        return np.nan

    iv = brentq(objective, sigma_min, 5)
    return iv


def implied_volatility_BS(market_price: float, S: float, K: float, T: float, r: float, option_type: str = "call"):
    """
    Calcule la volatilité implicite par résolution numérique.

    market_price : prix observé de l'option
    S             : prix spot de l'actif
    K             : strike
    T             : temps jusqu'à maturité (en années)
    r             : taux sans risque

    Retourne np.nan si aucune solution n'existe dans l'intervalle de recherche.
    """
    def objective(sigma):
        return BS_price(S, K, T, r, sigma, option_type) - market_price

    # Vérifier que la solution existe dans l'intervalle avant d'appeler brentq
    f_low = objective(1e-6)
    f_high = objective(5)
    if f_low * f_high > 0:
        return np.nan

    # Recherche entre 0.0001% et 500% de volatilité
    iv = brentq(objective, 1e-6, 5)
    return iv


if __name__ == "__main__":
    # ── Exemple d'utilisation simple ──
    market_price = 10.0
    S = 100.0
    K = 100.0
    T = 1.0
    r = 0.05
    option_type = "call"

    iv = implied_volatility_CRR(market_price, S, K, T, r, option_type)
    print(f"Volatilité implicite : {iv:.4f}")

    # ── Chargement du snapshot d'options AAPL (figé, cf. SNAPSHOT_PATH) ──────
    print(f"\nChargement du snapshot : {SNAPSHOT_PATH.name}")
    df_snap = pd.read_csv(SNAPSHOT_PATH)
    S = float(df_snap["S"].iloc[0])
    r = float(df_snap["r"].iloc[0])
    valid_expirations = sorted(df_snap["expiry"].unique())
    expiry = valid_expirations[0]
    T = float(df_snap.loc[df_snap["expiry"] == expiry, "T"].iloc[0])

    print(f"Spot AAPL (snapshot) : {S:.2f}")
    print(f"Maturité choisie : {expiry}, T = {T:.4f} an(s)")

    # ── Smile + surface, répétées pour chaque méthode d'inversion ────────────
    METHODS = {
        "BS":  implied_volatility_BS,
        "CRR": implied_volatility_CRR,
        "LSM": implied_volatility_LSM,
        "PDE": implied_volatility_PDE,
    }

    for name, iv_func in METHODS.items():
        print(f"\n=== Méthode : {name} ===")

        # -- Smile pour l'échéance la plus proche (déjà filtrée bid/ask > 0) --
        calls_clean = df_snap[df_snap["expiry"] == expiry].copy()

        strikes = calls_clean['strike'].values
        market_prices = calls_clean['mid'].values

        implied_vols = np.array([iv_func(mp, S, K, T, r, option_type)
                                  for mp, K in zip(market_prices, strikes)])

        valid_smile = ~np.isnan(implied_vols)

        diff = np.abs(implied_vols[valid_smile] - calls_clean['impliedVolatility'].values[valid_smile])
        print(f"  Smile — Erreur moyenne absolue : {diff.mean():.4f}")
        print(f"  Smile — Erreur max : {diff.max():.4f}")

        plt.figure(figsize=(10, 6))
        plt.plot(strikes[valid_smile], implied_vols[valid_smile], 'o-', label=f'Implied Volatility ({name} Model)')
        plt.plot(strikes[valid_smile], calls_clean['impliedVolatility'].values[valid_smile], 'x-',
                  label='Implied Volatility (from Yahoo Finance)')
        plt.axvline(S, color='grey', lw=0.8, ls=':', label='Spot')
        plt.title(f'Implied Volatility vs Strike Price — AAPL ({expiry}) — {name}')
        plt.xlabel('Strike Price')
        plt.ylabel('Implied Volatility')
        plt.legend()
        plt.grid()
        plt.savefig(REPORTS / f'implied_volatility_vs_strike_{name}.png')
        plt.show()

        # -- Construction de la surface de volatilité --
        strikes_surface = []
        maturities_surface = []
        implied_vols_surface = []

        for exp in valid_expirations:
            sub = df_snap[df_snap["expiry"] == exp]
            T_exp = float(sub["T"].iloc[0])

            # Filtre moneyness : garde uniquement les strikes raisonnablement proches du spot
            sub = sub[(sub['strike'] / S > 0.7) & (sub['strike'] / S < 1.3)]

            strikes_t = sub['strike'].values
            market_prices_t = sub['mid'].values

            implied_vols_t = np.array([iv_func(mp, S, K, T_exp, r, option_type)
                                        for mp, K in zip(market_prices_t, strikes_t)])

            valid = ~np.isnan(implied_vols_t) & (implied_vols_t > 0.05) & (implied_vols_t < 1.5)

            strikes_surface.extend(strikes_t[valid])
            maturities_surface.extend([T_exp] * sum(valid))
            implied_vols_surface.extend(implied_vols_t[valid])

        strikes_surface = np.array(strikes_surface)
        maturities_surface = np.array(maturities_surface)
        implied_vols_surface = np.array(implied_vols_surface)

        # -- Affichage surface 3D --
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        surface = ax.plot_trisurf(strikes_surface, maturities_surface, implied_vols_surface,
                                   cmap='viridis', linewidth=0.2)
        ax.set_title(f'Surface de volatilité implicite — AAPL — {name}')
        ax.set_xlabel('Strike')
        ax.set_ylabel('Maturité (années)')
        ax.set_zlabel('Volatilité implicite')
        ax.view_init(elev=25, azim=45)
        fig.colorbar(surface, shrink=0.5, aspect=10, label='IV')
        plt.savefig(REPORTS / f'volatility_surface_{name}.png')
        plt.show()