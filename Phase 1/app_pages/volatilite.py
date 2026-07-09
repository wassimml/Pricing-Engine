import streamlit as st

from app_theme import setup_page, page_header, REPORTS_DIR

setup_page("Volatilité implicite")

page_header("Volatilité implicite - AAPL", "Inversion numérique (Brent) sur 4 méthodes de pricing")

st.markdown(
    "Pour chaque méthode, la volatilité implicite est obtenue en inversant le prix de marché "
    "(snapshot figé, options calls AAPL filtrées bid/ask, borne d'arbitrage et liquidité minimale) "
    "par la méthode de Brent. La smile correspond à l'échéance la plus proche, la surface agrège "
    "toutes les échéances filtrées par moneyness (0.7–1.3)."
)

METHODS = ["BS", "CRR", "LSM", "PDE"]
tabs = st.tabs(METHODS)

for tab, m in zip(tabs, METHODS):
    with tab:
        c1, c2 = st.columns(2)
        with c1:
            st.image(str(REPORTS_DIR / f"implied_volatility_vs_strike_{m}.png"),
                      caption=f"Smile - {m}", use_container_width=True)
        with c2:
            st.image(str(REPORTS_DIR / f"volatility_surface_{m}.png"),
                      caption=f"Surface - {m}", use_container_width=True)

st.warning(
    "**Limite connue de LSM sur ce jeu de données** : sur les strikes très ITM à maturité très "
    "courte, le vega réel est quasi nul (le prix ≈ intrinsèque, insensible à σ) - l'inversion "
    "prix→vol y est mal conditionnée. Le bruit Monte Carlo de LSM (même avec seed fixe) y domine "
    "largement le signal, produisant des IV implicites peu fiables sur ce segment précis. "
    "BS, CRR et PDE (déterministes) n'ont pas ce problème.",
    icon=":material/warning:",
)
