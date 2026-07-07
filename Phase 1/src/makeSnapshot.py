from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

DATA = Path(__file__).parent.parent / "data"

# Régénère un snapshot figé de la chaîne d'options SPY (filtrée), à rejouer
# ensuite dans benchmarkSPY.py via SNAPSHOT_PATH pour comparer les méthodes
# de pricing sur des données identiques d'un run à l'autre.
if __name__ == "__main__":
    today = pd.Timestamp.today().normalize()
    print(f"Date : {today.date()}")

    ticker = yf.Ticker("SPY")
    hist = ticker.history(period="1y", auto_adjust=True)
    S = float(hist["Close"].iloc[-1])
    r = 0.045  # taux sans risque (T-bill 3 mois approx)
    print(f"Spot SPY : ${S:.2f}")

    # - Récupération des chains (T >= 7 jours) --------------------------------
    expirations = [
        exp for exp in ticker.options
        if (pd.Timestamp(exp) - today).days >= 7
    ]
    frames = []
    for exp in expirations:
        chain = ticker.option_chain(exp)
        for df_leg, kind in [(chain.calls, "call"), (chain.puts, "put")]:
            leg = df_leg.copy()
            leg["kind"]   = kind
            leg["expiry"] = exp
            frames.append(leg)
    raw = pd.concat(frames, ignore_index=True)

    # - Maturité en années -----------------------------------------------------
    raw["T"]   = (pd.to_datetime(raw["expiry"]) - today).dt.days / 365.0
    raw["S"]   = S
    raw["r"]   = r
    raw["mid"] = (raw["bid"] + raw["ask"]) / 2

    # - Filtres -------------------------------------------------------------
    df = raw.copy()
    print("Initial :", len(df))

    df = df[(df["bid"] > 0) & (df["ask"] > 0)]
    print("bid/ask >", len(df))

    df = df[df["T"] > 7/365]
    print("T >", len(df))

    df = df[(df["impliedVolatility"] > 0) & df["impliedVolatility"].notna()]
    print("IV >", len(df))

    disc_K = df["strike"] * np.exp(-r * df["T"])
    lb = np.where(
        df["kind"] == "call",
        np.maximum(S - disc_K, 0),
        np.maximum(disc_K - S, 0),
    )
    df = df[df["mid"] >= lb]
    print("No-arbitrage >", len(df))

    df = df[
        (df.impliedVolatility > 0.05) &
        (df.impliedVolatility < 2.0)
    ]
    print("IV bounds >", len(df))

    # - Moneyness -------------------------------------------------------------
    m_ratio = df["strike"] / S
    seg = pd.Series("OTM", index=df.index)
    seg[(m_ratio >= 0.97) & (m_ratio <= 1.03)] = "ATM"
    seg[(df["kind"] == "call") & (m_ratio < 0.97)] = "ITM"
    seg[(df["kind"] == "put")  & (m_ratio > 1.03)] = "ITM"
    df["segment"] = seg

    # - Sauvegarde --------------------------------------------------------------
    snapshot_path = DATA / f"SPY_options_{today.date()}.csv"
    df.to_csv(snapshot_path, index=False)
    print(f"Snapshot sauvegardé ({len(df)} lignes) : {snapshot_path}")
    print("Pense à mettre à jour SNAPSHOT_PATH dans benchmarkSPY.py.")
