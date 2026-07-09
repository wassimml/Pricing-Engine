from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import yfinance as yf

DATA = Path(__file__).parent.parent / "data"

# Régénère un snapshot figé de la chaîne d'options calls AAPL (toutes échéances
# > 7 jours, filtrées bid/ask > 0), à rejouer ensuite dans impliedVol.py via
# SNAPSHOT_PATH pour recalculer la smile / la surface de vol implicite sur des
# données identiques d'un run à l'autre (reproductibilité des tests).
if __name__ == "__main__":
    today = datetime.today()
    print(f"Date : {today.date()}")

    ticker = yf.Ticker("AAPL")
    S = float(ticker.history(period="1d")["Close"].iloc[-1])
    r = 0.05  # taux sans risque (cohérent avec impliedVol.py)
    print(f"Spot AAPL : ${S:.2f}")

    # - Échéances valides (T > 7 jours, comme dans impliedVol.py) -------------
    expirations = ticker.options
    valid_expirations = [
        e for e in expirations
        if (datetime.strptime(e, "%Y-%m-%d") - today).days > 7
    ]
    print(f"Échéances valides : {len(valid_expirations)}")

    frames = []
    for exp in valid_expirations:
        calls = ticker.option_chain(exp).calls.copy()
        calls["kind"]   = "call"
        calls["expiry"] = exp
        frames.append(calls)
    raw = pd.concat(frames, ignore_index=True)
    print("Initial :", len(raw))

    # - Maturité en années, spot / taux figés, mid-price ----------------------
    expiry_dates = pd.to_datetime(raw["expiry"])
    raw["T"]   = (expiry_dates - today).dt.days / 365.0
    raw["S"]   = S
    raw["r"]   = r
    raw["mid"] = (raw["bid"] + raw["ask"]) / 2

    # - Filtre : cotation active (même filtre que impliedVol.py) --------------
    df = raw[(raw["bid"] > 0) & (raw["ask"] > 0)].copy()
    print("bid/ask >", len(df))

    # - Filtre : borne d'arbitrage (C >= max(S - K*e^-rT, 0) pour un call) -----
    # Sans ce test, un mid resté sous la borne théorique (cotation stale/spread
    # trop large) peut rendre l'objectif de brentq mal posé : aucune racine
    # réelle n'existe, mais l'inversion peut quand même "converger" vers une
    # racine parasite au lieu de renvoyer NaN proprement.
    lb = np.maximum(df["S"] - df["strike"] * np.exp(-df["r"] * df["T"]), 0)
    n_before = len(df)
    df = df[df["mid"] >= lb]
    print(f"No-arbitrage > {len(df)}  ({n_before - len(df)} écartées)")

    # - Filtre : liquidité minimale (open interest) ----------------------------
    # bid/ask > 0 n'exclut pas les contrats au carnet d'ordre figé (cotation
    # affichée mais jamais rafraîchie faute d'activité) : openInterest est un
    # meilleur proxy de liquidité réelle que le volume du jour (qui peut être
    # nul même sur un contrat activement quoté par les market makers).
    n_before = len(df)
    df = df[df["openInterest"].fillna(0) >= 10]
    print(f"Open interest >= 10 > {len(df)}  ({n_before - len(df)} écartées)")

    # - Sauvegarde --------------------------------------------------------------
    snapshot_path = DATA / f"AAPL_options_{today.date()}.csv"
    df.to_csv(snapshot_path, index=False)
    print(f"Snapshot sauvegardé ({len(df)} lignes) : {snapshot_path}")
    print("Pense à mettre à jour SNAPSHOT_PATH dans impliedVol.py.")
