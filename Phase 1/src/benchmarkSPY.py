from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import yfinance as yf

# Import all methods (versions optimisées - cf. binomial.py/monteCarloLSM.py/
# pde.py/BSpricer.py pour le détail de chaque levier et sa justification
# empirique : vectorisation pour BS/CRR, multiprocess pour LSM, décision
# auto séquentiel/parallèle selon le coût réel pour PDE)
from binomial import crr_price_fast
from monteCarloLSM import LSMoptionValue_parallel
from pde import pde_crank_nicolson_auto
from BSpricer import BSModel

DATA = Path(__file__).parent.parent / "data"
REPORTS = Path(__file__).parent.parent / "reports"

# Snapshot figé de la chaîne d'options SPY (filtrée), généré une fois puis
# rejoué à chaque run pour comparer les méthodes sur des données identiques.
SNAPSHOT_PATH = DATA / "SPY_options_2026-07-16.csv"

_bs = BSModel()

# SPY options are American-style (early exercise allowed) - LSM, CRR and PDE
# below are all run with american=True / style="american" so they price the
# same contract the market quotes. BS has no early-exercise premium and is
# kept as a separate European baseline (see the dedicated BS window).
#
# PDE apparaît ici deux fois, à dessein : "PDE (50,100)" est l'outil de calcul
# (config rapide utilisée en pratique partout ailleurs dans le projet quand
# PDE est une méthode candidate), "PDE (800,800)" est l'outil de référence
# (config précise, plafonnée par la limite structurelle de QuantLib à forte
# vol, cf. pde.py). Les deux sont ici jugées de la même façon, directement
# contre le prix marché - contrairement à benchmarkMethods.py où PDE ne joue
# jamais que le rôle de référence pour CRR/LSM.
#
# LSM apparaît ici aussi deux fois : "LSM (50,10000)" (n_paths=10000, config
# utilisée par défaut ailleurs dans le projet, ex. benchmarkInData.py,
# monteCarloLSM.py) et "LSM (50,20000)" (n_paths=20000, plus de paths donc en
# principe plus stable). Contrairement à PDE (50,100)/(800,800), ce n'est pas
# une hiérarchie calcul/référence formelle — juste deux budgets de calcul
# comparés directement contre le marché pour voir si doubler n_paths change
# quelque chose d'observable sur ce book précis.
AMERICANMETHODS = ["LSM (50,10000)", "LSM (50,20000)", "CRR", "PDE (50,100)", "PDE (800,800)"]


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

    # - 6. Pricing (vectorisé/parallélisé - cf. imports) -----------------------
    # Les options SPY sont de style AMÉRICAIN (exercice anticipé possible) :
    # LSM, CRR et PDE sont donc évaluées en mode américain pour être comparables
    # au prix marché. BS n'a pas de forme fermée américaine - elle sert de
    # référence européenne, étudiée séparément (section 9). PDE et LSM sont
    # chacune calculées deux fois, à deux configs (cf. commentaire sur
    # AMERICANMETHODS).
    #
    # Toutes les fonctions ci-dessous pricent le book ENTIER en un seul appel
    # (BS/CRR vectorisés NumPy, LSM/PDE(800,800) parallélisés multi-process,
    # PDE(50,100) reste séquentiel via la décision auto - trop rapide par
    # option pour que la parallélisation soit rentable, cf. pde.py) au lieu
    # d'une boucle Python par ligne (df.apply) - déterminant sur un book de
    # cette taille (6950+ lignes).
    S_arr     = df["S"].to_numpy()
    K_arr     = df["strike"].to_numpy()
    T_arr     = df["T"].to_numpy()
    r_arr     = df["r"].to_numpy()
    sigma_arr = df["impliedVolatility"].to_numpy()
    kind_arr  = df["kind"].to_numpy()
    american_arr = np.ones(len(df), dtype=bool)  # toutes les options SPY sont américaines

    print("Pricing BS...")
    df["price_BS"] = _bs.price_batch(S_arr, K_arr, T_arr, r_arr, sigma_arr, kind_arr)

    print("Pricing CRR...")
    df["price_CRR"] = crr_price_fast(S_arr, K_arr, T_arr, r_arr, sigma_arr, kind_arr, american_arr, period=200)

    print("Pricing LSM (50,10000)...")
    df["price_LSM (50,10000)"] = LSMoptionValue_parallel(
        S_arr, K_arr, T_arr, r_arr, sigma_arr, kind_arr, n_steps=50, n_paths=10000, seed=42)

    print("Pricing LSM (50,20000)...")
    df["price_LSM (50,20000)"] = LSMoptionValue_parallel(
        S_arr, K_arr, T_arr, r_arr, sigma_arr, kind_arr, n_steps=50, n_paths=20000, seed=42)

    print("Pricing PDE (50,100)...")
    df["price_PDE (50,100)"] = pde_crank_nicolson_auto(
        S_arr, K_arr, T_arr, r_arr, sigma_arr, kind_arr, american_arr, n_steps=50, n_space=100)

    print("Pricing PDE (800,800)...")
    df["price_PDE (800,800)"] = pde_crank_nicolson_auto(
        S_arr, K_arr, T_arr, r_arr, sigma_arr, kind_arr, american_arr, n_steps=800, n_space=800)

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

    # - 7c. Score de cohérence bid-ask normalisé par la largeur du spread -----
    # Le critère binaire "% dans le spread" est mécaniquement biaisé par la
    # liquidité : sur une option Deep OTM à spread large ($2,45), presque
    # n'importe quel prix "raisonnable" tombe dedans, alors que sur une option
    # ATM liquide à spread serré ($0,04) même un excellent modèle peut le
    # rater de peu. On normalise donc par la largeur du spread pour rendre le
    # score comparable entre buckets de liquidité (cf. review section 2.6).
    #
    # d_i = (P̂_i - mid_i) / (spread_i / 2)  — déviation normalisée par le
    # demi-spread, partagée par les trois métriques ci-dessous.
    def bidask_deviation(sub, price_col):
        v = sub[[price_col, "mid", "bid", "ask"]].dropna()
        spread = v["ask"] - v["bid"]
        v, spread = v[spread > 0], spread[spread > 0]
        return (v[price_col] - v["mid"]) / (spread / 2)

    # s_i = 1 - |d_i| : 1 si au mid, 0 sur la limite du spread, <0 si hors
    # spread. La moyenne mean(s_i) reste bornée par le haut (≤1) mais pas par
    # le bas : un unique contrat catastrophique (spread quasi nul) peut tirer
    # la moyenne très loin dans le négatif sans que ce soit visible comme tel.
    def bidask_score(sub, price_col):
        return 1 - bidask_deviation(sub, price_col).abs()

    # MAE normalisée = écart moyen au mid, en unités de demi-spread. Toujours
    # ≥0 (pas de compensation entre contrats bien et mal pricés) :
    # 0 = parfait, 1 = sur le bord du spread, >1 = hors spread.
    def bidask_mae_norm(sub, price_col):
        return bidask_deviation(sub, price_col).abs().mean()

    # RMSE normalisé = même idée au carré : pénalise davantage les gros écarts
    # (effet quadratique), toujours ≥0, cohérent avec le MAE/RMSE déjà utilisés
    # ailleurs dans ce script.
    def bidask_rmse_norm(sub, price_col):
        d = bidask_deviation(sub, price_col)
        return np.sqrt((d ** 2).mean())

    # - 7b. Pires contrats (plus grande erreur absolue) -----------------------
    # RMSE >> MAE indique une distribution d'erreurs à queue lourde : quelques
    # contrats à erreur énorme tirent le RMSE vers le haut. On les identifie et
    # on les affiche explicitement plutôt que de les laisser invisibles dans
    # une moyenne (cf. review du rapport Benchmark, section 2.7.2 / Figure 7).
    def report_worst(sub, price_col, label, n=5):
        v = sub[[price_col, "mid", "bid", "ask", "strike", "T", "kind",
                 "openInterest", "volume"]].dropna(subset=[price_col, "mid"])
        err = (v[price_col] - v["mid"]).abs()
        worst = v.assign(abs_err=err).sort_values("abs_err", ascending=False).head(n)
        print(f"  Pires contrats ({label}) :")
        for _, row in worst.iterrows():
            print(f"    {row['kind']:4} K={row['strike']:7.1f}  T={row['T']:.2f}y  "
                  f"mid={row['mid']:8.2f}  {label}={row[price_col]:8.2f}  "
                  f"err={row['abs_err']:7.2f}  OI={row['openInterest']:.0f}  vol={row['volume']:.0f}")

    # - 8. Affichage résultats ------------------------
    # Note : les prints console restent ASCII-safe (pas de macron/box-drawing)
    # car la console Windows par défaut (cp1252) plante sinon sur certains
    # caractères Unicode (— et les labels des graphes, eux, les supportent).
    print(f"\n- Métriques globales (méthodes américaines : {', '.join(AMERICANMETHODS)}) -")
    for method in AMERICANMETHODS:
        m = metrics(df, f"price_{method}")
        mae_n  = bidask_mae_norm(df, f"price_{method}")
        rmse_n = bidask_rmse_norm(df, f"price_{method}")
        print(f"  {method:14}: MAE={m['MAE']:.4f}$  RMSE={m['RMSE']:.4f}$  "
              f"MPE={m['MPE (%)']:+.2f}%  in-spread={m['% in spread']:.1f}%  "
              f"RMSE/MAE={m['RMSE']/m['MAE']:.2f}  "
              f"score bid-ask normalise (moy s)={bidask_score(df, f'price_{method}').mean():.3f}  "
              f"MAE normalisee={mae_n:.3f}  RMSE normalise={rmse_n:.3f}")
        report_worst(df, f"price_{method}", method)

    print("\n- Métriques globales BS (référence européenne) --------")
    m = metrics(df, "price_BS")
    mae_n  = bidask_mae_norm(df, "price_BS")
    rmse_n = bidask_rmse_norm(df, "price_BS")
    print(f"  BS      : MAE={m['MAE']:.4f}$  RMSE={m['RMSE']:.4f}$  "
          f"MPE={m['MPE (%)']:+.2f}%  in-spread={m['% in spread']:.1f}%  "
          f"RMSE/MAE={m['RMSE']/m['MAE']:.2f}  "
          f"score bid-ask normalise (moy s)={bidask_score(df, 'price_BS').mean():.3f}  "
          f"MAE normalisee={mae_n:.3f}  RMSE normalise={rmse_n:.3f}")
    report_worst(df, "price_BS", "BS")

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

    # Pire contrat (plus grande erreur absolue), toujours affiché même s'il
    # n'est pas tombé dans l'échantillon aléatoire — pour ne jamais laisser un
    # point aberrant non commenté sur ce graphe (cf. review section 2.7.2).
    full = df.dropna(subset=["price_BS", "mid"])
    worst_idx = (full["price_BS"] - full["mid"]).abs().idxmax()
    wr = full.loc[worst_idx]

    ax_bs.scatter(x, y, alpha=0.3, s=8, color="steelblue")
    lim = [0, max(sample[["mid", "price_BS"]].max().max(), wr["mid"], wr["price_BS"])]
    ax_bs.plot(lim, lim, "r--", lw=1, label="y = x")
    slope, intercept = np.polyfit(x, y, 1)
    r2 = np.corrcoef(x, y)[0, 1] ** 2
    ax_bs.plot(lim, [slope * v + intercept for v in lim], color="black", lw=1,
               label=f"régression : y={slope:.2f}x+{intercept:.2f}  (R²={r2:.3f})")

    ax_bs.scatter([wr["mid"]], [wr["price_BS"]], color="red", s=60, zorder=5,
                  marker="o", facecolors="none", linewidths=1.5,
                  label=f"pire écart : {wr['kind']} K={wr['strike']:.0f} T={wr['T']:.2f}y "
                        f"(err={abs(wr['price_BS']-wr['mid']):.0f}$, OI={wr['openInterest']:.0f})")

    ax_bs.set_xlabel("Prix marché (mid $)")
    ax_bs.set_ylabel("Prix BS ($, σ = IV marché)")
    ax_bs.legend(fontsize=7.5)

    fig_bs.suptitle(
        "BS (σ = IV marché) vs marché - alignement quasi y = x\n"
        "IV marché = σ tel que BS(σ) = prix marché ⇒ quasi-circulaire, mais pas exact :\n"
        "le résidu restant vient du décalage S/r/timing entre notre repricing et le calcul de l'IV",
        fontsize=10)
    plt.subplots_adjust(top=0.85)
    plt.savefig(REPORTS / "SPY_BS_vs_market.png", dpi=150)
    plt.show()

    # - 9b. Fenêtre méthodes américaines
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    colors_m = {"LSM (50,10000)": "plum", "LSM (50,20000)": "mediumpurple",
                "CRR": "tomato",
                "PDE (50,100)": "seagreen", "PDE (800,800)": "darkgreen"}
    w = 0.8 / len(AMERICANMETHODS)  # largeur générique, quel que soit le nombre de méthodes

    ax = axes[0, 0]
    ax.bar(AMERICANMETHODS, [metrics(df, f"price_{m}")["MAE"] for m in AMERICANMETHODS],
           color=[colors_m[m] for m in AMERICANMETHODS], edgecolor="white")
    ax.set_title("MAE globale vs marché ($)")
    ax.set_ylabel("MAE ($)")
    ax.tick_params(axis='x', labelrotation=15)

    ax = axes[0, 1]
    ax.bar(AMERICANMETHODS, [metrics(df, f"price_{m}")["% in spread"] for m in AMERICANMETHODS],
           color=[colors_m[m] for m in AMERICANMETHODS], edgecolor="white")
    ax.set_title("% prix dans le spread bid-ask")
    ax.set_ylabel("%")
    ax.set_ylim(0, 100)
    ax.tick_params(axis='x', labelrotation=15)

    ax = axes[1, 0]
    x = np.arange(len(SEGMENTS))
    for i, method in enumerate(AMERICANMETHODS):
        mae_seg = [metrics(df[df["segment"] == s], f"price_{method}")["MAE"] for s in SEGMENTS]
        ax.bar(x + i * w, mae_seg, w, label=method,
               color=colors_m[method], edgecolor="white")
    ax.set_xticks(x + w * (len(AMERICANMETHODS) - 1) / 2)
    ax.set_xticklabels(SEGMENTS)
    ax.set_title("MAE par moneyness")
    ax.set_ylabel("MAE ($)")
    ax.legend(fontsize=8)

    ax = axes[1, 1]
    x = np.arange(len(KINDS))
    for i, method in enumerate(AMERICANMETHODS):
        rel_kind = [metrics(df[df["kind"] == k], f"price_{method}")["MAE rel (%)"] for k in KINDS]
        ax.bar(x + i * w, rel_kind, w, label=method,
               color=colors_m[method], edgecolor="white")
    ax.set_xticks(x + w * (len(AMERICANMETHODS) - 1) / 2)
    ax.set_xticklabels(["Call", "Put"])
    ax.set_title("MAE relative par type (%)")
    ax.set_ylabel("MAE relative (%)")
    ax.legend(fontsize=8)

    plt.suptitle(f"SPY Options Benchmark ({' / '.join(AMERICANMETHODS)}, style américain) - "
                 f"Spot ${S:.2f}  |  r={r:.1%}",
                 fontsize=13)
    plt.subplots_adjust(top=0.85)
    plt.savefig(REPORTS / "SPY_benchmark_american.png", dpi=150)
    plt.show()

    # - 9c. Score de cohérence bid-ask normalisé (BS + toutes les méthodes américaines)
    # Nouveau graphe, distinct du "% dans le spread" ci-dessus (inchangé) :
    # le score s̄ = 1 - 2|P̂-mid|/spread pondère chaque option par la largeur
    # de son propre spread, donc n'est plus artificiellement gonflé par les
    # options illiquides à spread large (cf. review section 2.6).
    ALLMETHODS  = ["BS", *AMERICANMETHODS]
    colors_all  = {"BS": "steelblue", **colors_m}
    w4 = 0.8 / len(ALLMETHODS)  # largeur générique, quel que soit le nombre de méthodes

    fig_score, axes_s = plt.subplots(2, 2, figsize=(13, 10))

    # Panneau 1 : les 3 métriques normalisées côte à côte. mean(s_i) est
    # borné par le haut mais pas par le bas — un seul contrat catastrophique
    # (spread quasi nul) peut l'entraîner très loin dans le négatif. MAE_norm
    # et RMSE_norm (toujours ≥0, jamais de compensation entre bons et
    # mauvais contrats) sont algébriquement liées à s̄ (MAE_norm = 1 - s̄)
    # mais RMSE_norm > MAE_norm révèle en plus la queue lourde de la
    # distribution des erreurs, comme le ratio RMSE/MAE utilisé plus haut.
    w3 = 0.25
    xm = np.arange(len(ALLMETHODS))
    ax = axes_s[0, 0]
    scores_global   = [bidask_score(df, f"price_{m}").mean() for m in ALLMETHODS]
    mae_norm_global = [bidask_mae_norm(df, f"price_{m}") for m in ALLMETHODS]
    rmse_norm_global = [bidask_rmse_norm(df, f"price_{m}") for m in ALLMETHODS]
    ax.bar(xm - w3, scores_global, w3, label="s̄ (moyenne signée)", color="steelblue", edgecolor="white")
    ax.bar(xm, mae_norm_global, w3, label="MAE normalisée (≥0)", color="tomato", edgecolor="white")
    ax.bar(xm + w3, rmse_norm_global, w3, label="RMSE normalisé (≥0)", color="seagreen", edgecolor="white")
    ax.axhline(0, color="black", lw=0.8)
    ax.axhline(1, color="grey", lw=0.6, ls=":")
    ax.set_xticks(xm)
    ax.set_xticklabels(ALLMETHODS, rotation=15, ha="right")
    ax.set_title("Cohérence bid-ask normalisée (global)")
    ax.set_ylabel("unités de demi-spread")
    ax.legend(fontsize=7.5)

    ax = axes_s[0, 1]
    xs = np.arange(len(SEGMENTS))
    for i, method in enumerate(ALLMETHODS):
        seg_scores = [bidask_score(df[df["segment"] == s], f"price_{method}").mean() for s in SEGMENTS]
        ax.bar(xs + i * w4, seg_scores, w4, label=method,
               color=colors_all[method], edgecolor="white")
    ax.set_xticks(xs + w4 * (len(ALLMETHODS) - 1) / 2)
    ax.set_xticklabels(SEGMENTS)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_title("Score bid-ask normalisé par moneyness")
    ax.set_ylabel("s̄")
    ax.legend(fontsize=7)

    ax = axes_s[1, 0]
    spread_abs = [(df[df["segment"] == s]["ask"] - df[df["segment"] == s]["bid"]).mean() for s in SEGMENTS]
    ax.bar(SEGMENTS, spread_abs, color="goldenrod", edgecolor="white")
    ax.set_title("Largeur moyenne du spread bid-ask ($)")
    ax.set_ylabel("Spread ($)")

    ax = axes_s[1, 1]
    spread_rel = [((df[df["segment"] == s]["ask"] - df[df["segment"] == s]["bid"])
                   / df[df["segment"] == s]["mid"] * 100).mean() for s in SEGMENTS]
    ax.bar(SEGMENTS, spread_rel, color="darkorange", edgecolor="white")
    ax.set_title("Largeur moyenne du spread bid-ask (% du mid)")
    ax.set_ylabel("Spread relatif (%)")

    plt.suptitle("Cohérence bid-ask normalisée par la largeur du spread\n"
                 "s̄ = 1 - 2|prix modèle - mid| / spread — comparable entre buckets de liquidité, "
                 "contrairement au % brut dans le spread",
                 fontsize=12)
    plt.subplots_adjust(top=0.86)
    plt.savefig(REPORTS / "SPY_bidask_score.png", dpi=150)
    plt.show()

    # - 9d/9e. Évolution de l'erreur (MAE) en fonction de la volatilité et de
    # la maturité, un subplot par méthode américaine (cf. AMERICANMETHODS) ---
    # Complète les vues par segment de moneyness (9b) avec deux axes continus.
    # Un subplot par méthode (plutôt que toutes les courbes superposées) pour
    # que chacune ait sa propre échelle d'axe Y et reste lisible même quand
    # deux méthodes sont proches (cf. CRR/PDE quasi confondus sur le graphe
    # superposé précédent).
    # Binning en quantiles (nombre égal d'options par bin) plutôt qu'en pas
    # fixe de vol/T : la chaîne d'options réelle est très inégalement répartie
    # (beaucoup d'options courtes et peu volatiles, peu d'options longues et
    # très volatiles), un pas fixe donnerait des bins avec parfois 2-3 options
    # et un MAE bruité au possible. Le centre de chaque bin (abscisse) est la
    # moyenne réelle de vol/T des options qu'il contient, pas le milieu de
    # l'intervalle - donc légèrement irrégulier sur l'axe mais fidèle aux
    # données.
    N_BINS = 100

    def mae_by_bin(sub, price_col, bin_col, n_bins=N_BINS):
        v = sub[[price_col, "mid", bin_col]].dropna()
        bins = pd.qcut(v[bin_col], q=n_bins, duplicates="drop")
        grouped = v.assign(bin=bins, abs_err=(v[price_col] - v["mid"]).abs()).groupby("bin", observed=True)
        centers = grouped[bin_col].mean().to_numpy()
        mae = grouped["abs_err"].mean().to_numpy()
        counts = grouped.size().to_numpy()
        return centers, mae, counts

    def plot_mae_vs(bin_col, xlabel, suptitle, filename):
        # Grille dynamique (pas figée à 2x2) : s'adapte au nombre de méthodes
        # américaines (5 désormais, LSM comptant double - cf. AMERICANMETHODS).
        n_methods = len(AMERICANMETHODS)
        ncols = 3 if n_methods > 4 else 2
        nrows = -(-n_methods // ncols)  # division entière arrondie au sup.
        _, axes_e = plt.subplots(nrows, ncols, figsize=(6.5 * ncols, 5 * nrows))
        axes_flat = np.atleast_1d(axes_e).reshape(-1)
        for ax, method in zip(axes_flat, AMERICANMETHODS):
            # counts (options par bin) non annoté sur le graphe à ce nombre de
            # bins (100) — illisible en superposition, cf. décision prise à
            # N_BINS=10->100 : chaque bin garde ~70 options (6950/100), encore
            # un échantillon correct pour une MAE stable, juste plus bruitée
            # bin à bin qu'à 10 bins.
            centers, mae, _ = mae_by_bin(df, f"price_{method}", bin_col)
            ax.plot(centers, mae, "-", lw=1, color=colors_m[method])
            ax.set_xlabel(xlabel)
            ax.set_ylabel("MAE vs marché ($)")
            ax.set_title(method)
        for ax in axes_flat[n_methods:]:  # sous-graphes en trop (grille non pile remplie)
            ax.axis("off")
        plt.suptitle(f"{suptitle}\n"
                     f"{N_BINS} bins par quantiles (échantillon égal par bin, pas un pas fixe)",
                     fontsize=12)
        plt.tight_layout()
        plt.subplots_adjust(top=0.86)
        plt.savefig(REPORTS / filename, dpi=150)
        plt.show()

    plot_mae_vs("impliedVolatility", "Volatilité implicite (σ)",
                "Évolution de l'erreur (MAE vs marché) en fonction de la volatilité implicite",
                "SPY_error_vs_vol.png")

    plot_mae_vs("T", "Maturité T (années)",
                "Évolution de l'erreur (MAE vs marché) en fonction de la maturité",
                "SPY_error_vs_maturity.png")

