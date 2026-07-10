# Pricing Engine

Moteur de valorisation d'options en Python, développé progressivement en quatre phases.

Projet personnel conduit en marge d'un cours de produits dérivés (Ensimag 2A), en appui sur *Options, Futures and Other Derivatives* - John Hull (2024). Développé dans le cadre d'une préparation à une césure.

---

## Progression

| Phase | Contenu | Statut |
|-------|---------|--------|
| **Phase 1** | Black-Scholes, Greeks, CRR, Monte Carlo & PDE | En cours |
| **Phase 2** | Volatilité stochastique - Heston & SABR | À venir |
| **Phase 3** | Taux d'intérêt stochastique - Vasicek, CIR & SVSI | À venir |
| **Phase 4** | Vol Surface Arbitrage Lab | À venir |

---

## Phase 1 - Black-Scholes, Greeks & Monte Carlo

Implémentation des fondations de la valorisation d'options vanilles.

- Modélisation du mouvement brownien géométrique (GBM) et dynamique du sous-jacent
- Formule analytique Black-Scholes-Merton - calls et puts européens
- Greeks analytiques : Delta, Gamma, Theta, Vega, Rho 
- Simulation Monte Carlo avec réduction de variance 
- Arbre binomial CRR
- Longstaff-Schwartz LSM - régression polynomiale d'ordre 3
- PDE - Crank-Nicolson
- Volatilité implicite par Newton-Raphson 
- Benchmark BS / MC / LSM / CRR / PDE sur données réelles SPY **(En cours)**

## Phase 2 - Volatilité stochastique *(à venir)*

Extension vers des modèles à volatilité non constante.

- 2.1 Surface de volatilité implicite
- 2.2 Modèle de Heston - Carr-Madan pricing
- 2.3 Calibration Heston - Differential evolution
- 2.4 Modèle SABR - Formule d'Hagan
- 2.5 Confrontation au marché réel (SPY) : réduction de l'écart vs Phase 1

## Phase 3 - Taux d'intérêt stochastique *(à venir)*

Modélisation de la structure par termes des taux et couplage avec le sous-jacent.

- 3.1 Modèle de Vasicek : taux stochastique mean-reverting
- 3.2 Modèle Cox-Ingersoll-Ross (CIR)
- 3.3 Couplage taux/sous-jacent : modèle SVSI
- 3.4 Confrontation au marché réel : options longue maturité

## Phase 4 - Vol Surface Arbitrage Lab *(à venir)*

Analyse de la surface et détection d'incohérences de pricing.

- 4.1 Absence d'arbitrage - Détection calendar & butterfly
- 4.2 Volatilité locale de Dupire
- 4.3 Greeks de second ordre - Vanna, Volga, Charm
- 4.4 P&L Explain
- 4.5 Stress tests + dashboard final

---

## Application web

Une app Streamlit réunit l'ensemble du projet : pricer interactif, Greeks, et une galerie de
toutes les études (convergence, benchmarks, volatilité implicite).

```bash
cd "Phase 1"
streamlit run app.py
```
![Interface](Phase%201/reports/streamlit.png)
![Documentation](Phase%201/reports/documentationInterface.png)
L'interface de l'app (mise en page, pages, identité visuelle) a été réalisée avec Claude. Construire
un dashboard Streamlit n'est pas l'objet de ce projet, les méthodes de pricing le sont.

---

## Documentation

La documentation de la Phase 1 est actuellement scindée en deux documents distincts :

- **[Pricing d'Options - Théorie, Méthodes et Démonstrations](Phase%201/Pricing%20d'Options%20-%20Théorie%2C%20Méthodes%20et%20Démonstrations.pdf)** : le rapport théorique, qui détaille les fondements mathématiques de chaque méthode (Black-Scholes, Greeks, Monte Carlo et réduction de variance, arbre binomial CRR, Longstaff-Schwartz, PDE/Crank-Nicolson, volatilité implicite et surface de volatilité), avec démonstrations et exemples numériques.
- **[Pricing d'Options - Benchmark des Méthodes](Phase%201/Pricing%20d'Options%20-%20Benchmark%20des%20Méthodes.pdf)** : le document pratique, qui confronte l'ensemble des méthodes entre elles, sur un volume important d'options (~2000), puis face aux données réelles d'un marché liquide (SPY).

---

## Structure

```
Options Pricing Engine/
├── Phase 1/
│   ├── app.py                    # Point d'entrée de l'app Streamlit (navigation)
│   ├── app_theme.py              # Identité visuelle "ardoise & or" partagée (palette, CSS, template Plotly)
│   ├── app_pages/                # Pages Streamlit (pricer, greeks, benchmarks, ...)
│   ├── src/
│   │   ├── option.py            # Contrat option (dataclass)
│   │   ├── BSpricer.py          # Black-Scholes-Merton
│   │   ├── greeks.py            # Greeks (Delta, Gamma, Theta, Vega, Rho)
│   │   ├── gbm.py               # Simulation GBM (MCEngine)
│   │   ├── deltaHedging.py      # Delta hedging discret
│   │   ├── deltaHedgingMC.py    # Delta hedging Monte Carlo
│   │   ├── monteCarlo.py        # Méthodes Monte Carlo (naïf, antithétique, contrôle) + convergence moyennée sur seeds
│   │   ├── monteCarloLSM.py     # Monte Carlo Longstaff-Schwartz (américain) + convergence moyennée sur seeds
│   │   ├── binomial.py          # Arbre binomial CRR (européen / américain)
│   │   ├── pde.py               # PDE Crank-Nicolson via QuantLib
│   │   ├── impliedVol.py        # Inversion de vol implicite (BS/CRR/LSM/PDE) + smile & surface (AAPL)
│   │   ├── timeValue.py         # Étude de la valeur temps (BS, surface 3D spot x maturité)
│   │   ├── benchmarkMethods.py  # Précision/vitesse vs balayage de paramètres (3D)
│   │   ├── benchmarkInData.py   # Benchmark de toutes les méthodes sur ~2000 options synthétiques
│   │   ├── benchmarkSPY.py      # Benchmark LSM/CRR/PDE face aux données réelles SPY
│   │   ├── makeSnapshotSPY.py   # Génère un snapshot reproductible d'options SPY
│   │   └── makeSnapshotAAPL.py  # Génère un snapshot reproductible d'options AAPL
│   ├── data/                    # Book synthétique + snapshots marché figés
│   ├── tests/                   # Tests
│   └── reports/                 # Graphiques générés
├── Phase 2/                     # À venir
├── Phase 3/                     # À venir
└── Phase 4/                     # À venir
```

---

*Projet en cours de développement - les phases 2, 3 et 4 seront implémentées progressivement.*
