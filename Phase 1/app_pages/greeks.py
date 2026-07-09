import numpy as np
import plotly.graph_objects as go
import streamlit as st

from app_theme import setup_page, page_header, ARDOISE, OR, ARDOISE_CLAIR, ROUGE, COLORWAY

setup_page("Greeks")

from option import Option
from greeks import BSGreeks

page_header("Greeks", "Sensibilités analytiques Black-Scholes-Merton")

shared = st.session_state.get("shared_option", dict(S=100.0, K=100.0, T=1.0, r=0.05, sigma=0.20, kind="call"))

c1, c2, c3, c4, c5, c6 = st.columns(6)
S = c1.number_input("Spot S", min_value=0.01, value=float(shared["S"]), step=1.0)
K = c2.number_input("Strike K", min_value=0.01, value=float(shared["K"]), step=1.0)
T = c3.number_input("Maturité T", min_value=0.001, value=float(shared["T"]), step=0.05, format="%.3f")
r = c4.number_input("Taux r", min_value=-0.5, value=float(shared["r"]), step=0.005, format="%.3f")
sigma = c5.number_input("Vol σ", min_value=0.001, value=float(shared["sigma"]), step=0.01, format="%.3f")
kind = c6.selectbox("Type", ["call", "put"], index=0 if shared["kind"] == "call" else 1)

opt = Option(S=S, K=K, T=T, r=r, sigma=sigma, kind=kind)
g = BSGreeks(opt)

m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Delta Δ", f"{g.delta():.4f}")
m2.metric("Gamma Γ", f"{g.gamma():.5f}")
m3.metric("Vega ν", f"{g.vega():.4f}")
m4.metric("Theta Θ (/jour)", f"{g.theta():.4f}")
m5.metric("Rho ρ", f"{g.rho():.4f}")

page_header("Profil vs spot", "")

GREEK_FUNCS = {
    "Delta": lambda o: BSGreeks(o).delta(),
    "Gamma": lambda o: BSGreeks(o).gamma(),
    "Vega":  lambda o: BSGreeks(o).vega(),
    "Theta": lambda o: BSGreeks(o).theta(),
    "Rho":   lambda o: BSGreeks(o).rho(),
}

selected = st.radio("Greek à tracer", list(GREEK_FUNCS.keys()), horizontal=True)

S_range = np.linspace(max(0.01, S * 0.4), S * 1.8, 250)
T_curves = sorted({round(T, 4), round(T / 2, 4), round(max(T / 10, 0.005), 4), round(max(T / 100, 0.002), 4)}, reverse=True)

fig = go.Figure()
for i, t_val in enumerate(T_curves):
    ys = [GREEK_FUNCS[selected](Option(S=s, K=K, T=t_val, r=r, sigma=sigma, kind=kind)) for s in S_range]
    fig.add_trace(go.Scatter(x=S_range, y=ys, mode="lines", name=f"T = {t_val:g} an(s)",
                              line=dict(width=2.2, color=COLORWAY[i % len(COLORWAY)])))

fig.add_vline(x=K, line=dict(color=ROUGE, dash="dash", width=1), annotation_text="Strike K")
fig.add_vline(x=S, line=dict(color=OR, dash="dot", width=1.3), annotation_text="Spot actuel")
fig.update_layout(
    title=f"{selected} vs Spot - {kind} (K={K:g}, r={r:.2%}, σ={sigma:.2%})",
    xaxis_title="Spot S", yaxis_title=selected,
    height=440, legend=dict(orientation="h", y=-0.2),
)
st.plotly_chart(fig, use_container_width=True)

st.caption(
    "Toutes les Greeks sont calculées analytiquement via Black-Scholes-Merton "
    "(convention dérivée brute : σ et r en décimal, Theta en perte de valeur par jour calendaire)."
)
