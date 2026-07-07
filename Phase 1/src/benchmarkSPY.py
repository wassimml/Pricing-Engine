from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import yfinance as yf

# Import all methods
from option import Option
from binomial import crr_price
from monteCarloLSM import LSMoptionValue
from pde import pde_crank_nicolson
from BSpricer import BSModel

DATA = Path(__file__).parent.parent / "data"
REPORTS = Path(__file__).parent.parent / "reports"

# Snapshot figé de la chaîne d'options SPY (filtrée), généré une fois puis
# rejoué à chaque run pour comparer les méthodes sur des données identiques.
SNAPSHOT_PATH = DATA / "SPY_options_2026-07-07.csv"

_bs = BSModel()

# SPY options are American-style (early exercise allowed) - LSM, CRR and PDE
# below are all run with american=True / style="american" so they price the
# same contract the market quotes. BS has no early-exercise premium and is
# kept as a separate European baseline (see the dedicated BS window).
AMERICANMETHODS = ["LSM", "CRR", "PDE"]


if __name__ == "__main__":
    # - 1. Chargement du snapshot d'options (figé, cf. SNAPSHOT_PATH) --------
    print(f"Chargement du snapshot : {SNAPSHOT_PATH.name}")
    df = pd.read_csv(SNAPSHOT_PATH)
    S  = float(df["S"].iloc[0])
    r  = float(df["r"].iloc[0])
    print(f"Spot SPY (snapshot) : ${S:.2f}   options : {len(df)}")

    # - 2. Historique SPY (contexte visuel uniquement, pas de pricing dessus) -
    # On ne sauvegarde pas cette série (juste l'image du graphique) - on garde
    # seulement le spot live à titre de référence, pour le comparer au spot
    # figé dans le snapshot utilisé pour le pricing.
    ticker  = yf.Ticker("SPY")
    hist    = ticker.history(period="1y", auto_adjust=True)
    S_live  = float(hist["Close"].iloc[-1])
    print(f"Spot SPY live (référence) : ${S_live:.2f}   "
          f"(snapshot : ${S:.2f}, écart {S_live - S:+.2f}$)")

    hist["Close"].plot(title="SPY - Prix de clôture ajusté (1 an)",
                       xlabel="Date", ylabel="Prix ($)")
    plt.tight_layout()
    plt.savefig(REPORTS / "SPY_historical_closing_price.png", dpi=150)
    plt.show()

    # - 6. Pricing ------------------------------─
    # Les options SPY sont de style AMÉRICAIN (exercice anticipé possible) :
    # LSM, CRR et PDE sont donc évaluées en mode américain pour être comparables
    # au prix marché. BS n'a pas de forme fermée américaine - elle sert de
    # référence européenne, étudiée séparément (section 9).
    def price_row(row, method):
        try:
            opt = Option(S=row["S"], K=row["strike"], T=row["T"],
                         r=row["r"], sigma=row["impliedVolatility"], kind=row["kind"])
            if method == "BS":
                return _bs.price(opt)
            elif method == "CRR":
                return crr_price(opt, period=200, american=True)
            elif method == "LSM":
                return LSMoptionValue(opt, n_steps=50, n_paths=20000)
            elif method == "PDE":
                return pde_crank_nicolson(opt, style="american", n_steps=200, n_space=200)
        except Exception:
            return np.nan

    for method in ["BS", *AMERICANMETHODS]:
        print(f"Pricing {method}...")
        df[f"price_{method}"] = df.apply(lambda row: price_row(row, method), axis=1)

    # - 7. Métriques ----------------------------─
    def metrics(sub, price_col):
        v = sub[[price_col, "mid", "bid", "ask"]].dropna()
        diff = v[price_col] - v["mid"]
        rel  = diff / v["mid"] * 100
        in_spread = (v["bid"] <= v[price_col]) & (v[price_col] <= v["ask"])
        return {
            "N":           len(v),
            "MAE":         np.mean(np.abs(diff)),
            "RMSE":        np.sqrt(np.mean(diff**2)),
            "MPE (%)":     np.mean(rel),
            "MAE rel (%)": np.mean(np.abs(rel)),
            "% in spread": in_spread.mean() * 100,
        }

    SEGMENTS = ["OTM", "ATM", "ITM"]
    KINDS    = ["call", "put"]

    # - 8. Affichage résultats ------------------------
    print("\n- Métriques globales (méthodes américaines : LSM, CRR, PDE) -─")
    for method in AMERICANMETHODS:
        m = metrics(df, f"price_{method}")
        print(f"  {method:3}: MAE={m['MAE']:.4f}$  RMSE={m['RMSE']:.4f}$  "
              f"MPE={m['MPE (%)']:+.2f}%  in-spread={m['% in spread']:.1f}%")

    print("\n- Métriques globales BS (référence européenne) --------─")
    m = metrics(df, "price_BS")
    print(f"  BS      : MAE={m['MAE']:.4f}$  RMSE={m['RMSE']:.4f}$  "
          f"MPE={m['MPE (%)']:+.2f}%  in-spread={m['% in spread']:.1f}%")

    # - 9a. Fenêtre BS seule : pourquoi BS ≈ marché suit presque y = x ----
    # On price toujours avec sigma = impliedVolatility de yfinance (jamais de
    # vol historique). Or cette IV est elle-même obtenue en INVERSANT BS sur le
    # prix marché : sigma_IV est défini par BS(S,K,T,r,sigma_IV) = prix_marché.
    # BS(sigma_IV) est donc quasi-circulaire vis-à-vis du prix marché, ce qui
    # explique le nuage proche de y = x et la pente de régression proche de 1.
    # "Quasi" et non "exactement" : le nuage ci-dessous s'écarte réellement de
    # y = x (pente ≠ 1, résidus non nuls, corrélés au delta de l'option) car
    # les S/r utilisés ici pour re-pricer ne sont pas rigoureusement identiques
    # à ceux utilisés par yfinance au moment du calcul de l'IV (spot pris à un
    # instant différent, dividende non modélisé, taux approximatif). C'est
    # précisément ce résidu - et non l'alignement global - qu'il est
    # intéressant d'étudier ici.
    fig_bs, ax_bs = plt.subplots(figsize=(7.5, 6.5))

    sample = df.dropna(subset=["price_BS"]).sample(min(500, len(df)), random_state=42)
    x, y = sample["mid"], sample["price_BS"]
    ax_bs.scatter(x, y, alpha=0.3, s=8, color="steelblue")
    lim = [0, sample[["mid", "price_BS"]].max().max()]
    ax_bs.plot(lim, lim, "r--", lw=1, label="y = x")
    slope, intercept = np.polyfit(x, y, 1)
    r2 = np.corrcoef(x, y)[0, 1] ** 2
    ax_bs.plot(lim, [slope * v + intercept for v in lim], color="black", lw=1,
               label=f"régression : y={slope:.2f}x+{intercept:.2f}  (R²={r2:.3f})")
    ax_bs.set_xlabel("Prix marché (mid $)")
    ax_bs.set_ylabel("Prix BS ($, σ = IV marché)")
    ax_bs.legend(fontsize=8)

    fig_bs.suptitle(
        "BS (σ = IV marché) vs marché - alignement quasi y = x\n"
        "IV marché = σ tel que BS(σ) = prix marché ⇒ quasi-circulaire, mais pas exact :\n"
        "le résidu restant vient du décalage S/r/timing entre notre repricing et le calcul de l'IV",
        fontsize=10)
    plt.subplots_adjust(top=0.85)
    plt.savefig(REPORTS / "SPY_BS_vs_market.png", dpi=150)
    plt.show()

    # - 9b. Fenêtre méthodes américaines : LSM, CRR, PDE -----------
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    colors_m = {"LSM": "mediumpurple", "CRR": "tomato", "PDE": "seagreen"}
    w = 0.25

    ax = axes[0, 0]
    ax.bar(AMERICANMETHODS, [metrics(df, f"price_{m}")["MAE"] for m in AMERICANMETHODS],
           color=[colors_m[m] for m in AMERICANMETHODS], edgecolor="white")
    ax.set_title("MAE globale vs marché ($)")
    ax.set_ylabel("MAE ($)")

    ax = axes[0, 1]
    ax.bar(AMERICANMETHODS, [metrics(df, f"price_{m}")["% in spread"] for m in AMERICANMETHODS],
           color=[colors_m[m] for m in AMERICANMETHODS], edgecolor="white")
    ax.set_title("% prix dans le spread bid-ask")
    ax.set_ylabel("%")
    ax.set_ylim(0, 100)

    ax = axes[1, 0]
    x = np.arange(len(SEGMENTS))
    for i, method in enumerate(AMERICANMETHODS):
        mae_seg = [metrics(df[df["segment"] == s], f"price_{method}")["MAE"] for s in SEGMENTS]
        ax.bar(x + i * w, mae_seg, w, label=method,
               color=colors_m[method], edgecolor="white")
    ax.set_xticks(x + w)
    ax.set_xticklabels(SEGMENTS)
    ax.set_title("MAE par moneyness")
    ax.set_ylabel("MAE ($)")
    ax.legend()

    ax = axes[1, 1]
    x = np.arange(len(KINDS))
    for i, method in enumerate(AMERICANMETHODS):
        rel_kind = [metrics(df[df["kind"] == k], f"price_{method}")["MAE rel (%)"] for k in KINDS]
        ax.bar(x + i * w, rel_kind, w, label=method,
               color=colors_m[method], edgecolor="white")
    ax.set_xticks(x + w)
    ax.set_xticklabels(["Call", "Put"])
    ax.set_title("MAE relative par type (%)")
    ax.set_ylabel("MAE relative (%)")
    ax.legend()

    plt.suptitle(f"SPY Options Benchmark (LSM / CRR / PDE, style américain) - "
                 f"Spot ${S:.2f}  |  r={r:.1%}",
                 fontsize=13)
    plt.subplots_adjust(top=0.85)
    plt.savefig(REPORTS / "SPY_benchmark_american.png", dpi=150)
    plt.show()

