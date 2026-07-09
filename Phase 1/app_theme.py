"""Identité visuelle partagée de l'application - palette "Ardoise & Or"
reprise des rapports LaTeX du projet (cf. préambule des PDF Phase 1)."""

import sys
from pathlib import Path

import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

# ── Palette (identique aux \definecolor du rapport LaTeX) ──────────────────
ARDOISE       = "#1C2B3A"
OR            = "#C9A84C"
ARDOISE_CLAIR = "#2D3E50"
FOND_DOUX     = "#F0EDE6"
ROUGE         = "#C0392B"
GRIS_NEUTRE   = "#6B7A8A"
BLANC         = "#FFFFFF"
OR_CLAIR      = "#DFCE94"   # dérivé de OR, pour zones remplies / bandes de confiance

COLORWAY = [OR, ARDOISE, ARDOISE_CLAIR, ROUGE, GRIS_NEUTRE, OR_CLAIR]

SRC_DIR = Path(__file__).parent / "src"
REPORTS_DIR = Path(__file__).parent / "reports"
DATA_DIR = Path(__file__).parent / "data"


def ensure_src_on_path() -> None:
    """Permet d'importer les modules de src/ (option.py, binomial.py, ...)."""
    p = str(SRC_DIR)
    if p not in sys.path:
        sys.path.insert(0, p)


def register_plotly_template() -> None:
    """Enregistre un template Plotly 'ardoise_or' repris comme défaut."""
    template = go.layout.Template()
    template.layout = go.Layout(
        colorway=COLORWAY,
        paper_bgcolor=FOND_DOUX,
        plot_bgcolor=BLANC,
        font=dict(family="Source Sans 3, Segoe UI, sans-serif", color=ARDOISE, size=13),
        title=dict(font=dict(family="Source Serif 4, Georgia, serif", color=ARDOISE, size=18)),
        xaxis=dict(gridcolor="#E2DCCB", zerolinecolor="#D6CFB8", linecolor=GRIS_NEUTRE,
                   title=dict(font=dict(color=ARDOISE_CLAIR))),
        yaxis=dict(gridcolor="#E2DCCB", zerolinecolor="#D6CFB8", linecolor=GRIS_NEUTRE,
                   title=dict(font=dict(color=ARDOISE_CLAIR))),
        legend=dict(bgcolor="rgba(255,255,255,0.6)", bordercolor="#E2DCCB", borderwidth=1),
        hoverlabel=dict(bgcolor=ARDOISE, font=dict(color=BLANC, family="Source Sans 3, sans-serif")),
        margin=dict(t=60, r=30, b=50, l=60),
    )
    pio.templates["ardoise_or"] = template
    pio.templates.default = "ardoise_or"


def inject_css() -> None:
    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Source+Serif+4:wght@500;600;700&family=Source+Sans+3:wght@400;500;600&display=swap');

        html, body, [class*="css"] {{
            font-family: 'Source Sans 3', 'Segoe UI', sans-serif;
        }}

        h1, h2, h3, [data-testid="stMarkdownContainer"] h1,
        [data-testid="stMarkdownContainer"] h2, [data-testid="stMarkdownContainer"] h3 {{
            font-family: 'Source Serif 4', Georgia, serif !important;
            color: {ARDOISE} !important;
            font-weight: 600 !important;
        }}

        [data-testid="stMarkdownContainer"] h2 {{
            margin-top: 1.2rem;
        }}

        [data-testid="stAppViewContainer"] {{
            background-color: {FOND_DOUX};
        }}

        [data-testid="stSidebar"] {{
            background-color: {ARDOISE};
            border-right: 2px solid {OR};
        }}
        [data-testid="stSidebar"] * {{
            color: {FOND_DOUX} !important;
        }}
        [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h1,
        [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h2,
        [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h3 {{
            color: {OR} !important;
            border-bottom: none;
        }}

        /* Navigation Streamlit native (st.navigation) dans la sidebar */
        [data-testid="stSidebarNav"] a, [data-testid="stSidebar"] a {{
            color: {FOND_DOUX} !important;
        }}
        [data-testid="stSidebar"] [aria-current="page"] {{
            background-color: rgba(201, 168, 76, 0.18) !important;
            border-left: 3px solid {OR};
        }}

        div[data-testid="stMetric"] {{
            background-color: {BLANC};
            border: 1px solid #E2DCCB;
            border-left: 4px solid {OR};
            border-radius: 6px;
            padding: 0.9rem 1rem 0.6rem 1rem;
        }}
        div[data-testid="stMetricLabel"] {{
            color: {ARDOISE_CLAIR} !important;
            font-weight: 600;
        }}
        div[data-testid="stMetricValue"] {{
            color: {ARDOISE} !important;
            font-family: 'Source Serif 4', Georgia, serif !important;
        }}

        .stTabs [data-baseweb="tab-list"] {{
            border-bottom: 1.5px solid #E2DCCB;
        }}
        .stTabs [aria-selected="true"] {{
            color: {ARDOISE} !important;
            border-bottom-color: {OR} !important;
        }}

        .stButton > button, .stDownloadButton > button {{
            background-color: {ARDOISE};
            color: {FOND_DOUX};
            border: 1px solid {OR};
            border-radius: 4px;
        }}
        .stButton > button:hover, .stDownloadButton > button:hover {{
            background-color: {OR};
            color: {ARDOISE};
            border: 1px solid {ARDOISE};
        }}

        .bandeau-or {{
            height: 3px;
            background: linear-gradient(90deg, {OR}, {FOND_DOUX});
            margin: 0.2rem 0 1.4rem 0;
            border-radius: 2px;
        }}

        .badge {{
            display: inline-block;
            padding: 0.15rem 0.6rem;
            border-radius: 999px;
            font-size: 0.75rem;
            font-weight: 600;
            letter-spacing: 0.02em;
        }}
        .badge-encours {{ background-color: {OR}; color: {ARDOISE}; }}
        .badge-avenir  {{ background-color: #E2DCCB; color: {ARDOISE_CLAIR}; }}

        .carte {{
            background-color: {BLANC};
            border: 1px solid #E2DCCB;
            border-top: 3px solid {OR};
            border-radius: 6px;
            padding: 1rem 1.2rem;
            height: 100%;
        }}
        .carte p {{ color: {ARDOISE_CLAIR}; font-size: 0.92rem; }}

        footer {{visibility: hidden;}}
        </style>
        """,
        unsafe_allow_html=True,
    )


def page_header(title: str, subtitle: str = "") -> None:
    st.markdown(f"## {title}")
    if subtitle:
        st.markdown(f"<p style='color:{ARDOISE_CLAIR}; margin-top:-0.6rem;'>{subtitle}</p>",
                    unsafe_allow_html=True)
    st.markdown("<div class='bandeau-or'></div>", unsafe_allow_html=True)


def setup_page(page_title: str) -> None:
    """A appeler en tête de chaque page : chemins, template plotly, CSS."""
    ensure_src_on_path()
    register_plotly_template()
    inject_css()
