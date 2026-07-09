import streamlit as st

from app_theme import setup_page, page_header, REPORTS_DIR

setup_page("Benchmarks")

page_header("Benchmarks", "Précision et vitesse des méthodes de pricing")

tab_book, tab_spy = st.tabs(["Book synthétique (~2000 options)", "Marché réel - SPY"])

with tab_book:
    st.markdown(
        "Toutes les méthodes confrontées entre elles sur un book de ~2000 options synthétiques "
        "(calls/puts, européennes/américaines, moneyness et maturités variées). Les méthodes de "
        "référence (PDE pour l'américain, BS pour l'européen) sont exclues du graphe de MAE "
        "correspondant - se comparer à soi-même donnerait trivialement zéro."
    )
    c1, c2 = st.columns(2)
    with c1:
        st.image(str(REPORTS_DIR / "benchmark_american.png"), caption="Options américaines (LSM / CRR / PDE)", use_container_width=True)
        st.image(str(REPORTS_DIR / "benchmark_fullbook.png"), caption="Book complet - combinaisons BS+CRR/PDE/LSM", use_container_width=True)
    with c2:
        st.image(str(REPORTS_DIR / "benchmark_european.png"), caption="Options européennes (BS / MC / CRR / PDE)", use_container_width=True)
        st.image(str(REPORTS_DIR / "benchmark_methods_3d.png"), caption="Précision vs vitesse vs paramètre (3D)", use_container_width=True)

with tab_spy:
    st.markdown(
        "Confrontation au marché réel : options SPY (américaines) pricées avec **LSM, CRR et PDE en "
        "mode américain** - Black-Scholes est isolé dans sa propre fenêtre car il n'a pas de forme "
        "fermée américaine et sert de référence européenne à part."
    )
    st.image(str(REPORTS_DIR / "SPY_historical_closing_price.png"), caption="SPY - historique de prix (1 an)",
              width=640)
    c1, c2 = st.columns(2)
    with c1:
        st.image(str(REPORTS_DIR / "SPY_benchmark_american.png"), caption="LSM / CRR / PDE vs marché (style américain)", use_container_width=True)
    with c2:
        st.image(str(REPORTS_DIR / "SPY_BS_vs_market.png"), caption="BS (σ = IV marché) vs marché", use_container_width=True)
    st.info(
        "L'alignement quasi y = x entre BS et le marché est **quasi-circulaire** : l'IV utilisée est "
        "elle-même obtenue en inversant BS sur le prix marché. Le résidu réel qui subsiste (pente ≠ 1) "
        "vient du décalage S/r/timing entre le repricing et le calcul de l'IV par le fournisseur de "
        "données - ce n'est pas une validation du modèle.",
        icon=":material/info:",
    )
