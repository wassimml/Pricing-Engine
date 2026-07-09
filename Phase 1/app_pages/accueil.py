import streamlit as st

from app_theme import setup_page, page_header, OR

setup_page("Accueil")

st.markdown(
    """
    <div style="text-align:center; padding: 1.2rem 0 0.4rem 0;">
        <h1 style="font-size:2.3rem; margin-bottom:0.1rem;">Pricer Engine</h1>
        <p style="color:#2D3E50; font-size:1.05rem;">
            Moteur de valorisation d'options en Python - développé progressivement en quatre phases
        </p>
    </div>
    <div class="bandeau-or"></div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    Projet personnel conduit en marge d'un cours de produits dérivés (Ensimag 2A), en appui sur
    *Options, Futures and Other Derivatives* - John Hull (2024). Développé dans le cadre d'une
    préparation à une césure.
    """
)

page_header("Progression")

PHASES = [
    ("Phase 1", "Black-Scholes, Greeks, CRR, Monte Carlo & PDE", "En cours", True),
    ("Phase 2", "Volatilité stochastique - Heston & SABR", "À venir", False),
    ("Phase 3", "Taux d'intérêt stochastique - Vasicek, CIR & SVSI", "À venir", False),
    ("Phase 4", "Vol Surface Arbitrage Lab", "À venir", False),
]

cols = st.columns(4)
for col, (phase, content, status, active) in zip(cols, PHASES):
    badge_class = "badge-encours" if active else "badge-avenir"
    with col:
        st.markdown(
            f"""
            <div class="carte">
                <p style="color:{OR}; font-weight:700; font-size:0.95rem; margin-bottom:0.2rem;">{phase}</p>
                <p style="min-height:4.4em;">{content}</p>
                <span class="badge {badge_class}">{status}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

page_header("Phase 1 - Black-Scholes, Greeks & Monte Carlo")

st.markdown(
    """
    Implémentation des fondations de la valorisation d'options vanilles :

    - Modélisation du mouvement brownien géométrique (GBM) et dynamique du sous-jacent
    - Formule analytique Black-Scholes-Merton - calls et puts européens
    - Greeks analytiques : Delta, Gamma, Theta, Vega, Rho
    - Simulation Monte Carlo avec réduction de variance (antithétique, variable de contrôle)
    - Arbre binomial CRR (européen / américain)
    - Longstaff-Schwartz LSM - régression polynomiale d'ordre 3 (américain)
    - PDE Crank-Nicolson via QuantLib (européen / américain)
    - Volatilité implicite par inversion numérique (Brent) - BS, CRR, LSM, PDE
    - Benchmark BS / MC / LSM / CRR / PDE sur ~2000 options synthétiques et sur données réelles SPY / AAPL
    """
)

page_header("Parcourir le projet")

nav_cols = st.columns(2)
with nav_cols[0]:
    st.page_link("app_pages/pricer.py", label="**Pricer** - tarifer une option interactivement", icon=":material/calculate:")
    st.page_link("app_pages/greeks.py", label="**Greeks** - sensibilités Black-Scholes en direct", icon=":material/show_chart:")
    st.page_link("app_pages/dynamique.py", label="**Dynamique** - simulations GBM & delta hedging", icon=":material/timeline:")
with nav_cols[1]:
    st.page_link("app_pages/convergence.py", label="**Convergence** - Monte Carlo & LSM, moyenne sur 25 seeds", icon=":material/insights:")
    st.page_link("app_pages/benchmarks.py", label="**Benchmarks** - book synthétique (~2000) & marché réel SPY", icon=":material/bar_chart:")
    st.page_link("app_pages/volatilite.py", label="**Volatilité implicite** - smile & surface AAPL, 4 méthodes", icon=":material/blur_on:")

st.markdown("<div class='bandeau-or'></div>", unsafe_allow_html=True)
st.caption(
    "Documentation complète : *Pricing d'Options - Théorie, Méthodes et Démonstrations* et "
    "*Pricing d'Options - Benchmark des Méthodes* (rapports PDF, dossier Phase 1/)."
)
