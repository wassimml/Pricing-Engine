import base64

import streamlit as st

from app_theme import setup_page, page_header, REPORTS_DIR

setup_page("Documentation")

page_header("Documentation", "Les deux rapports complets du projet")

BOOKS = [
    {
        "title": "Théorie, Méthodes et Démonstrations",
        "subtitle": "Fondements mathématiques de chaque méthode, démonstrations et exemples numériques",
        "image": REPORTS_DIR / "PDG_Theo.png",
        "url": "https://github.com/wassimml/Pricing-Engine/blob/main/Phase%201/Pricing%20d'Options%20-%20Th%C3%A9orie%2C%20M%C3%A9thodes%20et%20D%C3%A9monstrations.pdf",
    },
    {
        "title": "Benchmark des Méthodes",
        "subtitle": "Confrontation de toutes les méthodes entre elles et aux données réelles (SPY)",
        "image": REPORTS_DIR / "PDG_Bench.png",
        "url": "https://github.com/wassimml/Pricing-Engine/blob/main/Phase%201/Pricing%20d'Options%20-%20Benchmark%20des%20M%C3%A9thodes.pdf",
    },
]


def _b64(path):
    return base64.b64encode(path.read_bytes()).decode()


cols = st.columns(2, gap="large")
for col, book in zip(cols, BOOKS):
    with col:
        st.markdown(
            f"""
            <a href="{book['url']}" target="_blank" rel="noopener" style="text-decoration:none;">
                <div class="carte-livre">
                    <img src="data:image/png;base64,{_b64(book['image'])}" style="width:100%; max-width:340px; border-radius:4px;">
                    <p class="titre-livre">{book['title']}</p>
                    <p class="soustitre-livre">{book['subtitle']}</p>
                </div>
            </a>
            """,
            unsafe_allow_html=True,
        )

st.caption("Cliquer sur une couverture ou son titre ouvre le PDF correspondant sur GitHub.")
