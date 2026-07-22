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
    # Chaque étape imprime combien d'options elle écarte (pas seulement le
    # compte cumulé restant) - un "funnel" récapitulatif est aussi affiché à
    # la fin (cf. section suivante), utile pour quantifier l'effet de chaque
    # filtre pris isolément dans le rapport.
    df = raw.copy()
    print(f"Initial : {len(df)}")
    funnel = [("Initial", len(df), len(df))]

    def apply_filter(df, mask, label):
        n_before = len(df)
        df = df[mask]
        n_after = len(df)
        print(f"{label} : {n_after}  (-{n_before - n_after} ecartees)")
        funnel.append((label, n_before, n_after))
        return df

    df = apply_filter(df, (df["bid"] > 0) & (df["ask"] > 0), "bid/ask > 0")
    df = apply_filter(df, df["T"] > 7/365, "T > 7j")
    df = apply_filter(df, (df["impliedVolatility"] > 0) & df["impliedVolatility"].notna(), "IV renseignee")

    disc_K = df["strike"] * np.exp(-r * df["T"])
    lb = np.where(
        df["kind"] == "call",
        np.maximum(S - disc_K, 0),
        np.maximum(disc_K - S, 0),
    )
    df = apply_filter(df, df["mid"] >= lb, "No-arbitrage (mid >= borne inf.)")

    df = apply_filter(
        df, (df.impliedVolatility > 0.05) & (df.impliedVolatility < 2.0), "IV bounds (0.05-2.0)")

    # - Filtre : liquidité minimale (open interest) ----------------------------
    # Sans ce filtre, des contrats à openInterest=0 (jamais tradés) se glissent
    # dans le book : leur bid/ask n'a plus vraiment de rapport avec un prix de
    # marché réel, et ce sont eux qui dominent le MAE/RMSE des méthodes de
    # pricing (cf. review du rapport Benchmark, section 2.7.2 — le point isolé
    # de la Figure 7 était un put K=1350, T=2.44 ans, openInterest=0).
    df = apply_filter(df, df["openInterest"].fillna(0) >= 10, "Open interest >= 10")

    # - Recap (funnel) ----------------------------------------------------------
    # Note : print() reste ASCII-safe (pas d'accent) - la console Windows par
    # defaut (cp1252) affiche du charabia sinon (cf. meme convention deja
    # etablie dans benchmarkSPY.py).
    print("\n- Recapitulatif des filtres (funnel) -")
    for label, n_before, n_after in funnel:
        if label == "Initial":
            print(f"  {label:32}: {n_after:5d}")
        else:
            pct = (n_before - n_after) / n_before * 100 if n_before else 0.0
            print(f"  {label:32}: {n_after:5d}  (-{n_before - n_after:4d}, -{pct:.1f}%)")
    total_before, total_after = funnel[0][2], funnel[-1][2]
    print(f"  {'Total':32}: {total_after:5d}  (-{total_before - total_after:4d}, "
          f"-{(total_before - total_after) / total_before * 100:.1f}% du book initial)")

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
    print(f"Snapshot sauvegarde ({len(df)} lignes) : {snapshot_path}")
    print("Pense a mettre a jour SNAPSHOT_PATH dans benchmarkSPY.py.")
