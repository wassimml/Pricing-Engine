# Pricing Engine

Moteur de valorisation d'options en Python, développé progressivement en trois phases.

Projet personnel conduit en marge d'un cours de produits dérivés (Ensimag 2A), en appui sur *Options, Futures and Other Derivatives* - John Hull (2024). Développé dans le cadre d'une préparation à une césure.

---

## Progression

| Phase | Contenu | Statut |
|-------|---------|--------|
| **Phase 1** | Black-Scholes, Greeks & Monte Carlo | 🔄 En cours |
| **Phase 2** | Volatilité stochastique - Heston & SABR | 🔜 À venir |
| **Phase 3** | Vol Surface Arbitrage Lab | 🔜 À venir |

---

## Phase 1 - Black-Scholes, Greeks & Monte Carlo

Implémentation des fondations de la valorisation d'options vanilles.

- Modélisation du mouvement brownien géométrique (GBM) et dynamique du sous-jacent
- Formule analytique Black-Scholes-Merton - calls et puts européens
- Greeks analytiques : Delta, Gamma, Theta, Vega, Rho
- Simulation Monte Carlo avec réduction de variance (variables antithétiques)
- Arbre binomial CRR - options américaines avec exercice anticipé
- Volatilité implicite par Newton-Raphson
- Benchmark BS / MC / CRR sur données réelles SPY

## Phase 2 - Volatilité stochastique *(à venir)*

Extension vers des modèles à volatilité non constante.

- Construction de la surface de volatilité implicite
- Modèle de Heston - pricing par transformée de Fourier (Carr-Madan)
- Calibration Heston par Differential Evolution
- Modèle SABR - formule approchée d'Hagan

## Phase 3 - Vol Surface Arbitrage Lab *(à venir)*

Analyse de la surface et détection d'incohérences de pricing.

- Détection d'arbitrage : calendar spread & butterfly
- Volatilité locale de Dupire
- Greeks de second ordre : Vanna, Volga, Charm
- P&L Explain
- Stress tests et dashboard de synthèse

---

## Structure

```
Options Pricing Engine/
├── Phase 1/
│   ├── src/
│   │   ├── gbm.py          # Simulation GBM
│   │   └── pricer.py       # Black-Scholes, Greeks, MC, CRR
│   └── reports/            # Graphiques générés
├── Phase 2/                # À venir
└── Phase 3/                # À venir
```

---

*Projet en cours de développement - les phases 2 et 3 seront implémentées progressivement.*
