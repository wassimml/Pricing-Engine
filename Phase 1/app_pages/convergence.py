import streamlit as st

from app_theme import setup_page, page_header, REPORTS_DIR

setup_page("Convergence")

page_header("Convergence Monte Carlo & LSM", "Moyenne ± écart-type sur 25 seeds indépendants")

st.markdown(
    """
    Une seule réalisation (seed fixe) ne permet pas de distinguer un vrai comportement de
    convergence d'un artefact du tirage aléatoire particulier. Chaque configuration ci-dessous
    est donc répétée sur **25 seeds indépendants** ; les graphes affichent la moyenne et
    l'écart-type des prix obtenus (barre d'erreur = dispersion d'une réalisation unique autour
    de la moyenne, pas l'incertitude sur la moyenne elle-même).
    """
)

tab_mc, tab_lsm = st.tabs(["Monte Carlo (variance reduction)", "Longstaff-Schwartz (LSM)"])

with tab_mc:
    st.caption("Référence : prix Black-Scholes exact - Put ATM (S=K=100, T=1, r=5%, σ=20%).")
    c1, c2 = st.columns(2)
    with c1:
        st.image(str(REPORTS_DIR / "monte_carlo_convergence_naive.png"), caption="MC naïf", use_container_width=True)
        st.image(str(REPORTS_DIR / "monte_carlo_convergence_control.png"), caption="Variable de contrôle", use_container_width=True)
    with c2:
        st.image(str(REPORTS_DIR / "monte_carlo_convergence_antithetic.png"), caption="Variables antithétiques", use_container_width=True)
        st.image(str(REPORTS_DIR / "monte_carlo_convergence_control_antithetic.png"),
                  caption="Antithétique + contrôle", use_container_width=True)
    st.info(
        "**Hiérarchie observée à n=100 000 trajectoires** (écart-type des 25 seeds, en $) : "
        "naïf ≈ 0.043 → antithétique ≈ 0.026 → contrôle ≈ 0.013 → antithétique+contrôle ≈ 0.010 - "
        "conforme à la réduction de variance attendue de chaque technique.",
        icon=":material/insights:",
    )

with tab_lsm:
    st.caption("Référence : PDE Crank-Nicolson (N=500×500) - Put américain (S=K=100, T=1, r=5%, σ=20%).")
    c1, c2 = st.columns(2)
    with c1:
        st.image(str(REPORTS_DIR / "lsm_convergence.png"), caption="Convergence vs n_paths et n_steps", use_container_width=True)
    with c2:
        st.image(str(REPORTS_DIR / "lsm_convergence_3d.png"), caption="Surface d'erreur (n_steps × n_paths)", use_container_width=True)
    st.info(
        "**Le \"rebond\" à n=100 000 observé sur une réalisation unique n'est pas un vrai "
        "comportement** : l'erreur moyenne à n_paths=50 000 (0.16 % ± 0.27 %) et à n_paths=100 000 "
        "(0.20 % ± 0.27 %) sont statistiquement indiscernables - l'écart tient entièrement dans "
        "l'écart-type. C'est du bruit de tirage, pas une divergence du LSM.",
        icon=":material/insights:",
    )
