from pathlib import Path
from datetime import date
import sys

import numpy as np
import pandas as pd

DATA = Path(__file__).parent.parent / "data"

# Régénère un book synthétique d'options (~2000 lignes), dans le même format
# que 1_options_book_params_2026-06-29.xlsx (colonnes Kind/Style/Spot/Strike/
# K-S/Moneyness/Maturity/T/Rate/Vol, 2 lignes d'en-tête + titre, lu ailleurs
# via pd.read_excel(..., header=2, index_col=0)).
#
# Les 2 lignes d'en-tête du book #1 documentent déjà sa propre méthodologie
# ("Seed 42 · European 55% / American 45% · Calls 50% / Puts 50%") — on
# reprend ces proportions catégorielles à l'identique. Le seed, lui, dépend
# du numéro de book (seed = 42 * book_number) pour que chaque nouveau book
# tire des données réellement indépendantes tout en restant reproductible —
# sinon relancer ce script produit un doublon byte-pour-byte du book #1
# (c'est ce qui s'est produit la première fois : seed=42 fixe pour tous).
# Les distributions de S, K/S, σ et r ne sont pas documentées dans le book
# #1 : elles sont reconstruites statistiquement à partir de ses valeurs
# mesurées (min/max/moyenne/écart-type), donc approximatives, pas la formule
# d'origine.
#
# Usage : python generateOptionsBook.py [purpose]
#   purpose (optionnel) : suffixe libre dans le nom de fichier, ex. "params"
#   ou "test", pour distinguer le book utilisé pour calibrer les paramètres
#   de pricing (benchmarkMethods.py) de celui utilisé pour les évaluer hors
#   échantillon (benchmarkInData.py) — cf. next_book_number.
N = 2000

MATURITY_GRID = {
    "1M": 1 / 12, "2M": 2 / 12, "3M": 0.25, "6M": 0.5,
    "9M": 0.75, "1Y": 1.0, "18M": 1.5, "2Y": 2.0,
}
MATURITY_LABELS = list(MATURITY_GRID.keys())
MATURITY_PROBS  = [0.0825, 0.0785, 0.1605, 0.1970, 0.1195, 0.1820, 0.1025, 0.0775]

RATE_GRID = [1.0, 2.0, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 7.0, 8.0, 10.0]


def classify_moneyness(kind: str, k_over_s: float) -> str:
    """Reproduit les bornes K/S observées dans le book #1 (params)."""
    if kind == "Call":
        if k_over_s < 0.85:  return "Deep ITM"
        if k_over_s < 0.97:  return "ITM"
        if k_over_s <= 1.03: return "ATM"
        if k_over_s <= 1.15: return "OTM"
        return "Deep OTM"
    else:
        if k_over_s > 1.15:  return "Deep ITM"
        if k_over_s > 1.03:  return "ITM"
        if k_over_s >= 0.97: return "ATM"
        if k_over_s >= 0.85: return "OTM"
        return "Deep OTM"


def next_book_number(data_dir: Path) -> int:
    """Cherche le plus grand numéro parmi les books déjà générés (N_options_book*.xlsx)
    et renvoie N+1. Si aucun book numéroté n'existe encore, démarre à 1."""
    numbers = [0]
    for f in data_dir.glob("*_options_book*.xlsx"):
        prefix = f.stem.split("_", 1)[0]
        if prefix.isdigit():
            numbers.append(int(prefix))
    return max(numbers) + 1


if __name__ == "__main__":
    purpose = sys.argv[1] if len(sys.argv) > 1 else None

    book_number = next_book_number(DATA)
    SEED = 42 * book_number

    rng = np.random.default_rng(SEED)

    kind  = rng.choice(["Call", "Put"], size=N, p=[0.5, 0.5])
    style = rng.choice(["European", "American"], size=N, p=[0.55, 0.45])

    S = np.clip(rng.lognormal(mean=5.146, sigma=0.683, size=N), 48.0, 502.0)

    k_over_s = np.clip(rng.normal(1.023, 0.140, size=N), 0.70, 1.30)
    K = S * k_over_s

    maturity_label = rng.choice(MATURITY_LABELS, size=N, p=MATURITY_PROBS)
    T = np.array([MATURITY_GRID[m] for m in maturity_label])

    r_raw = np.clip(rng.normal(5.0, 1.5, size=N), 1.0, 10.0)
    r = np.array([min(RATE_GRID, key=lambda x: abs(x - v)) for v in r_raw])

    sigma = np.clip(rng.normal(32.0, 16.3, size=N), 8.0, 89.0)

    moneyness = [classify_moneyness(k, ks) for k, ks in zip(kind, k_over_s)]

    df = pd.DataFrame({
        "Kind":         kind,
        "Style":        style,
        "Spot  S":      S.round(2),
        "Strike  K":    K.round(2),
        "K / S":        k_over_s.round(4),
        "Moneyness":    moneyness,
        "Maturity":     maturity_label,
        "T  (years)":   T,
        "Rate  r (%)":  r,
        "Vol  σ (%)":   sigma.round(2),
    })
    df.index = pd.RangeIndex(1, N + 1, name="#")

    today = date.today()
    suffix = f"_{purpose}" if purpose else ""
    out_path = DATA / f"{book_number}_options_book{suffix}_{today}.xlsx"

    title   = f"Options Pricing Engine  ·  Benchmark Dataset #{book_number}{' (' + purpose + ')' if purpose else ''}  ·  {N:,} Options".replace(",", " ")
    subtitle = (f"Generated {today.isoformat()}  ·  Seed {SEED}  ·  "
                f"European 55 % / American 45 %  ·  Calls 50 % / Puts 50 %")

    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        pd.DataFrame([[title], [subtitle]]).to_excel(
            writer, sheet_name="Sheet1", header=False, index=False, startrow=0)
        df.to_excel(writer, sheet_name="Sheet1", startrow=2)

    print(f"Book #{book_number} généré ({N} lignes) : {out_path}")
    print(f"Kind: {pd.Series(kind).value_counts(normalize=True).to_dict()}")
    print(f"Style: {pd.Series(style).value_counts(normalize=True).to_dict()}")
