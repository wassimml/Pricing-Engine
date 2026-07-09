import time

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from app_theme import setup_page, page_header, ARDOISE, OR, ARDOISE_CLAIR, ROUGE, GRIS_NEUTRE

setup_page("Pricer")

from option import Option
from BSpricer import BSModel
from binomial import crr_price
from monteCarlo import mc_naive, mc_antithetic, mc_control, mc_control_antithetic
from monteCarloLSM import LSMoptionValue
from pde import pde_crank_nicolson

page_header("Pricer", "Tarification interactive - 8 méthodes disponibles")

METHOD_LABELS = {
    "bs": "Black-Scholes-Merton (analytique)",
    "binomial": "Arbre binomial CRR",
    "mc-naive": "Monte Carlo naïf",
    "mc-antithetic": "Monte Carlo - variables antithétiques",
    "mc-control": "Monte Carlo - variable de contrôle",
    "mc-control-antithetic": "Monte Carlo - antithétique + contrôle",
    "mc-lsm": "Monte Carlo Longstaff-Schwartz (LSM)",
    "pde": "PDE Crank-Nicolson (QuantLib)",
}
STYLE_METHODS = {"binomial", "mc-lsm", "pde"}
MC_METHODS = {"mc-naive", "mc-antithetic", "mc-control", "mc-control-antithetic"}

left, right = st.columns([1, 1.55], gap="large")

with left:
    method = st.selectbox("Méthode de pricing", list(METHOD_LABELS.keys()),
                           format_func=lambda m: METHOD_LABELS[m], key="pricer_method")

    with st.form("pricer_form"):
        c1, c2 = st.columns(2)
        with c1:
            S = st.number_input("Spot S", min_value=0.01, value=100.0, step=1.0)
            T = st.number_input("Maturité T (années)", min_value=0.001, value=1.0, step=0.05, format="%.3f")
            sigma = st.number_input("Volatilité σ", min_value=0.001, value=0.20, step=0.01, format="%.3f")
        with c2:
            K = st.number_input("Strike K", min_value=0.01, value=100.0, step=1.0)
            r = st.number_input("Taux sans risque r", min_value=-0.5, value=0.05, step=0.005, format="%.3f")
            kind = st.selectbox("Type", ["call", "put"])

        if method in STYLE_METHODS:
            style = st.radio("Style d'exercice", ["european", "american"], horizontal=True,
                              index=1 if method == "mc-lsm" else 0,
                              disabled=(method == "mc-lsm"))
            if method == "mc-lsm":
                st.caption("LSM est intrinsèquement américain (early exercise via régression).")
        else:
            style = "european"

        if method in ("binomial", "mc-lsm", "pde"):
            steps = st.slider("Pas de temps (steps)", 10, 1000, 200 if method == "pde" else 100)
        else:
            steps = 100

        if method in MC_METHODS | {"mc-lsm"}:
            n_paths = st.select_slider("Nombre de trajectoires MC",
                                        options=[1_000, 5_000, 10_000, 25_000, 50_000, 100_000, 200_000],
                                        value=10_000 if method == "mc-lsm" else 100_000)
            seed = st.number_input("Seed aléatoire", min_value=0, value=42, step=1)
        else:
            n_paths, seed = 100_000, 42

        n_space = st.slider("Pas d'espace (PDE)", 50, 500, 200) if method == "pde" else 200

        submitted = st.form_submit_button("Calculer le prix", use_container_width=True)

if submitted or "last_result" not in st.session_state:
    opt = Option(S=S, K=K, T=T, r=r, sigma=sigma, kind=kind)
    t0 = time.perf_counter()
    std_error = None

    if method == "bs":
        price = BSModel().price(opt)
    elif method == "binomial":
        price = crr_price(opt, period=steps, american=(style == "american"))
    elif method == "mc-naive":
        price, std_error = mc_naive(opt, n_paths=n_paths, seed=seed)
    elif method == "mc-antithetic":
        price, std_error = mc_antithetic(opt, n_paths=n_paths, seed=seed)
    elif method == "mc-control":
        price, std_error = mc_control(opt, n_paths=n_paths, seed=seed)
    elif method == "mc-control-antithetic":
        price, std_error = mc_control_antithetic(opt, n_paths=n_paths, seed=seed)
    elif method == "mc-lsm":
        price = LSMoptionValue(opt, n_steps=steps, n_paths=n_paths, seed=seed)
    elif method == "pde":
        price = pde_crank_nicolson(opt, style=style, n_steps=steps, n_space=n_space)
    elapsed = time.perf_counter() - t0

    # Cross-check rapide (toujours calculé, coût négligeable)
    bs_ref = BSModel().price(Option(S=S, K=K, T=T, r=r, sigma=sigma, kind=kind))
    crr_ref = crr_price(opt, period=300, american=(style == "american" if method in STYLE_METHODS else False))

    st.session_state["last_result"] = dict(
        opt=opt, method=method, price=price, std_error=std_error, elapsed=elapsed,
        bs_ref=bs_ref, crr_ref=crr_ref, kind=kind, S=S, K=K, T=T, r=r, sigma=sigma,
    )
    st.session_state["shared_option"] = dict(S=S, K=K, T=T, r=r, sigma=sigma, kind=kind)

res = st.session_state["last_result"]

with right:
    m1, m2, m3 = st.columns(3)
    with m1:
        label = "Prix (± SE)" if res["std_error"] is not None else "Prix"
        value = f"{res['price']:.4f}"
        delta = f"± {res['std_error']:.4f}" if res["std_error"] is not None else None
        st.metric(label, value, delta=delta, delta_color="off")
    with m2:
        st.metric("Écart vs BS", f"{res['price'] - res['bs_ref']:+.4f}",
                   f"{(res['price'] - res['bs_ref']) / res['bs_ref']:+.2%}")
    with m3:
        st.metric("Temps de calcul", f"{res['elapsed']*1000:.1f} ms")

    st.markdown("###### Comparaison rapide")
    st.dataframe(
        {
            "Méthode": [METHOD_LABELS[res["method"]], "Black-Scholes (référence)", "CRR N=300 (référence)"],
            "Prix": [round(res["price"], 4), round(res["bs_ref"], 4), round(res["crr_ref"], 4)],
        },
        hide_index=True, use_container_width=True,
    )

    opt = res["opt"]
    S_lo, S_hi = max(0.01, opt.S * 0.5), opt.S * 1.6
    S_range = np.linspace(S_lo, S_hi, 200)
    bs_curve = BSModel().price_grid(S_range, opt)
    intrinsic = np.maximum(S_range - opt.K, 0) if opt.kind == "call" else np.maximum(opt.K - S_range, 0)

    tab1, tab2 = st.tabs(["Sensibilité au spot", "Payoff à maturité"])

    with tab1:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=S_range, y=bs_curve, mode="lines", name="Prix BS (courbe lisse)",
                                  line=dict(color=ARDOISE, width=2.4)))
        fig.add_trace(go.Scatter(x=S_range, y=intrinsic, mode="lines", name="Valeur intrinsèque",
                                  line=dict(color=GRIS_NEUTRE, width=1.2, dash="dot")))
        fig.add_trace(go.Scatter(x=[opt.S], y=[res["price"]], mode="markers", name=f"Prix {METHOD_LABELS[res['method']]}",
                                  marker=dict(color=OR, size=13, line=dict(color=ARDOISE, width=1.5), symbol="diamond")))
        fig.add_vline(x=opt.K, line=dict(color=ROUGE, dash="dash", width=1), annotation_text="Strike K")
        fig.update_layout(title="Prix en fonction du spot", xaxis_title="Spot S", yaxis_title="Prix",
                           height=380, legend=dict(orientation="h", y=-0.22))
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        payoff = np.maximum(S_range - opt.K, 0) if opt.kind == "call" else np.maximum(opt.K - S_range, 0)
        pnl = payoff - res["price"]
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=S_range, y=payoff, mode="lines", name="Payoff brut",
                                   line=dict(color=ARDOISE_CLAIR, width=2)))
        fig2.add_trace(go.Scatter(x=S_range, y=pnl, mode="lines", name="P&L net (payoff − prime)",
                                   line=dict(color=OR, width=2.4)))
        fig2.add_hline(y=0, line=dict(color=GRIS_NEUTRE, width=1))
        fig2.add_vline(x=opt.K, line=dict(color=ROUGE, dash="dash", width=1), annotation_text="Strike K")
        fig2.update_layout(title="Payoff & P&L à maturité", xaxis_title="Spot à maturité $S_T$", yaxis_title="€",
                            height=380, legend=dict(orientation="h", y=-0.22))
        st.plotly_chart(fig2, use_container_width=True)

st.caption(
    "Le prix Black-Scholes de référence et l'arbre CRR (N=300) sont systématiquement recalculés à "
    "titre de recoupement, quelle que soit la méthode sélectionnée."
)
