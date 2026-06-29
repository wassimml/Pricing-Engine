import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq
import matplotlib.pyplot as plt
from pathlib import Path
import yfinance as yf
from datetime import datetime

from BSpricer import BSModel
from option import Option

REPORTS = Path(__file__).parent.parent / "reports"


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


def implied_volatility(market_price: float, S: float, K: float, T: float, r: float, option_type: str = "call"):
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

    iv = implied_volatility(market_price, S, K, T, r, option_type)
    print(f"Volatilité implicite : {iv:.4f}")

    # ── Récupération de la chaîne d'options réelle ──
    ticker = yf.Ticker("AAPL")
    expirations = ticker.options

    today = datetime.today()
    valid_expirations = [
        e for e in expirations
        if (datetime.strptime(e, "%Y-%m-%d") - today).days > 7
    ]
    expiry = valid_expirations[0]

    chain = ticker.option_chain(expiry)
    calls = chain.calls

    # ── Vrais paramètres de marché ──
    S = ticker.history(period="1d")["Close"].iloc[-1]
    expiry_date = datetime.strptime(expiry, "%Y-%m-%d")
    T = (expiry_date - today).days / 365
    r = 0.05

    print(f"Spot AAPL : {S:.2f}")
    print(f"Maturité choisie : {expiry}, T = {T:.4f} an(s)")

    # ── Filtrage : ne garder que les options avec cotation active ──
    calls_clean = calls[(calls['bid'] > 0) & (calls['ask'] > 0)].copy()
    calls_clean['mid_price'] = (calls_clean['bid'] + calls_clean['ask']) / 2

    strikes = calls_clean['strike'].values
    market_prices = calls_clean['mid_price'].values

    implied_vols = np.array([implied_volatility(mp, S, K, T, r, option_type)
                              for mp, K in zip(market_prices, strikes)])

    valid_smile = ~np.isnan(implied_vols)

    diff = np.abs(implied_vols[valid_smile] - calls_clean['impliedVolatility'].values[valid_smile])
    print(f"Erreur moyenne absolue : {diff.mean():.4f}")
    print(f"Erreur max : {diff.max():.4f}")

    plt.figure(figsize=(10, 6))
    plt.plot(strikes[valid_smile], implied_vols[valid_smile], 'o-', label='Implied Volatility (BS Model)')
    plt.plot(strikes[valid_smile], calls_clean['impliedVolatility'].values[valid_smile], 'x-',
              label='Implied Volatility (from Yahoo Finance)')
    plt.axvline(S, color='grey', lw=0.8, ls=':', label='Spot')
    plt.title(f'Implied Volatility vs Strike Price — AAPL ({expiry})')
    plt.xlabel('Strike Price')
    plt.ylabel('Implied Volatility')
    plt.legend()
    plt.grid()
    plt.savefig(REPORTS / 'implied_volatility_vs_strike.png')
    plt.show()

    # ── Construction de la surface de volatilité ──
    strikes_surface = []
    maturities_surface = []
    implied_vols_surface = []

    for expiry in valid_expirations:
        expiry_date = datetime.strptime(expiry, "%Y-%m-%d")
        T = (expiry_date - today).days / 365

        chain = ticker.option_chain(expiry)
        calls = chain.calls
        calls_clean = calls[(calls['bid'] > 0) & (calls['ask'] > 0)].copy()
        calls_clean['mid_price'] = (calls_clean['bid'] + calls_clean['ask']) / 2

        # Filtre moneyness : garde uniquement les strikes raisonnablement proches du spot
        calls_clean = calls_clean[
            (calls_clean['strike'] / S > 0.7) & (calls_clean['strike'] / S < 1.3)
        ]

        strikes_t = calls_clean['strike'].values
        market_prices_t = calls_clean['mid_price'].values

        implied_vols_t = np.array([implied_volatility(mp, S, K, T, r, option_type)
                                    for mp, K in zip(market_prices_t, strikes_t)])

        valid = ~np.isnan(implied_vols_t) & (implied_vols_t > 0.05) & (implied_vols_t < 1.5)

        strikes_surface.extend(strikes_t[valid])
        maturities_surface.extend([T] * sum(valid))
        implied_vols_surface.extend(implied_vols_t[valid])

    strikes_surface = np.array(strikes_surface)
    maturities_surface = np.array(maturities_surface)
    implied_vols_surface = np.array(implied_vols_surface)

    # ── Affichage surface 3D ──
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    surface = ax.plot_trisurf(strikes_surface, maturities_surface, implied_vols_surface,
                               cmap='viridis', linewidth=0.2)
    ax.set_title('Surface de volatilité implicite — AAPL')
    ax.set_xlabel('Strike')
    ax.set_ylabel('Maturité (années)')
    ax.set_zlabel('Volatilité implicite')
    ax.view_init(elev=25, azim=45)
    fig.colorbar(surface, shrink=0.5, aspect=10, label='IV')
    plt.savefig(REPORTS / 'volatility_surface.png')
    plt.show()