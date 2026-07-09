import streamlit as st

from app_theme import setup_page, page_header, REPORTS_DIR

setup_page("Dynamique")

page_header("Dynamique du sous-jacent", "Simulation GBM & couverture en delta")

tab_gbm, tab_hedge = st.tabs(["Simulation GBM", "Delta hedging"])

with tab_gbm:
    st.markdown(
        "Simulation du mouvement brownien géométrique $dS_t = rS_t\\,dt + \\sigma S_t\\,dW_t$ "
        "sous-jacent à toutes les méthodes Monte Carlo du projet - vérification empirique de la "
        "log-normalité des prix simulés."
    )
    c1, c2 = st.columns(2)
    with c1:
        st.image(str(REPORTS_DIR / "gbm_simulation.png"), caption="Trajectoires GBM simulées", use_container_width=True)
        st.image(str(REPORTS_DIR / "gbm_final_prices_histogram.png"),
                  caption="Distribution des prix terminaux (log-normale)", use_container_width=True)
    with c2:
        st.image(str(REPORTS_DIR / "gbm_simulation_50_paths.png"), caption="50 trajectoires", use_container_width=True)
        st.image(str(REPORTS_DIR / "gbm_log_returns_histogram.png"),
                  caption="Distribution des rendements log (normale)", use_container_width=True)
    st.image(str(REPORTS_DIR / "gbm_log_returns_qq_plot.png"),
              caption="QQ-plot des rendements log vs loi normale", width=560)

with tab_hedge:
    st.markdown(
        "Réplication d'une position vendeuse d'option par couverture en delta discrète - "
        "impact de la fréquence de rebalancement sur l'erreur de couverture résiduelle."
    )
    st.image(str(REPORTS_DIR / "delta_hedging_comparison.png"),
              caption="Comparaison des stratégies de rebalancement", use_container_width=True)
    c1, c2 = st.columns(2)
    with c1:
        st.image(str(REPORTS_DIR / "delta_hedging_comparison_Weekly_vs_Daily_Hedging.png"),
                  caption="Rebalancement hebdomadaire vs quotidien", use_container_width=True)
    with c2:
        st.image(str(REPORTS_DIR / "delta_hedging_mc_20_Weeks_Hedging_each_week.png"),
                  caption="Delta hedging Monte Carlo - 20 semaines", use_container_width=True)
