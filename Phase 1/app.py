import streamlit as st

from app_theme import setup_page

st.set_page_config(
    page_title="Pricer Engine",
    page_icon="♟",
    layout="wide",
    initial_sidebar_state="expanded",
)
setup_page("Pricer Engine")

with st.sidebar:
    st.markdown("# ♟ Pricer Engine")
    st.markdown(
        "<p style='color:#DFCE94; font-size:0.82rem; margin-top:-0.6rem;'>"
        "Moteur de valorisation d'options - Phase 1</p>",
        unsafe_allow_html=True,
    )
    st.markdown("<hr style='border-color:#3A4A5C;'>", unsafe_allow_html=True)

pages = {
    "Aperçu": [
        st.Page("app_pages/accueil.py", title="Accueil", icon=":material/home:", default=True),
        st.Page("app_pages/documentation.py", title="Documentation", icon=":material/menu_book:"),
    ],
    "Outils interactifs": [
        st.Page("app_pages/pricer.py", title="Pricer", icon=":material/calculate:"),
        st.Page("app_pages/greeks.py", title="Greeks", icon=":material/show_chart:"),
    ],
    "Résultats & études": [
        st.Page("app_pages/dynamique.py", title="Dynamique (GBM & Hedging)", icon=":material/timeline:"),
        st.Page("app_pages/convergence.py", title="Convergence MC / LSM", icon=":material/insights:"),
        st.Page("app_pages/benchmarks.py", title="Benchmarks", icon=":material/bar_chart:"),
        st.Page("app_pages/volatilite.py", title="Volatilité implicite", icon=":material/blur_on:"),
    ],
}

pg = st.navigation(pages)

with st.sidebar:
    st.markdown("<hr style='border-color:#3A4A5C;'>", unsafe_allow_html=True)
    st.markdown(
        "<p style='color:#9AA7B2; font-size:0.75rem;'>Ensimag 2A - préparation césure<br>"
        "Wassim Mlaouhia</p>",
        unsafe_allow_html=True,
    )

pg.run()
