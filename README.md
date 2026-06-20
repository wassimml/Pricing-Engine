# Pricing Engine

Moteur de valorisation d'options en Python, développé progressivement en trois phases.

Projet personnel conduit en marge d'un cours de produits dérivés (Ensimag 2A), en appui sur *Options, Futures and Other Derivatives* - John Hull (2024). Développé dans le cadre d'une préparation à une césure.

---

## Progression

| Phase | Contenu | Statut |
|-------|---------|--------|
| **Phase 1** | Black-Scholes, Greeks & Monte Carlo | En cours |
| **Phase 2** | Volatilité stochastique - Heston & SABR | À venir |
| **Phase 3** | Vol Surface Arbitrage Lab | À venir |

---

## Phase 1 - Black-Scholes, Greeks & Monte Carlo

Implémentation des fondations de la valorisation d'options vanilles.

- Modélisation du mouvement brownien géométrique (GBM) et dynamique du sous-jacent
- Formule analytique Black-Scholes-Merton - calls et puts européens
- Greeks analytiques : Delta, Gamma, Theta, Vega, Rho 
- Simulation Monte Carlo avec réduction de variance **(En cours)**
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

## Utilisation

Depuis le dossier `Phase 1/` :

```bash
python src/pricerTerminal.py --S <spot> --K <strike> --T <maturité> --r <taux> --sigma <vol> --kind <call|put> [--method <méthode>] [--style <european|american>] [--steps N] [--n-paths N] [--seed N]
```

**Méthodes disponibles**

| `--method` | Description | Style |
|---|---|---|
| `bs` *(défaut)* | Formule analytique Black-Scholes-Merton | européen |
| `binomial` | Arbre binomial CRR | européen / américain |
| `mc-naive` | Monte Carlo naïf | européen |
| `mc-antithetic` | Monte Carlo - variables antithétiques | européen |
| `mc-control` | Monte Carlo - variable de contrôle | européen |
| `mc-control-antithetic` | Monte Carlo - antithétique + variable de contrôle | européen |

**Flags spécifiques**

| Flag | Défaut | Usage |
|---|---|---|
| `--style` | `european` | Style d'exercice pour `binomial` |
| `--steps` | `100` | Nombre de pas de l'arbre binomial |
| `--n-paths` | `100000` | Nombre de trajectoires MC |
| `--seed` | `42` | Seed aléatoire MC |

**Exemples**

```bash
# Black-Scholes
python src/pricerTerminal.py --S 100 --K 100 --T 1 --r 0.05 --sigma 0.2 --kind call

# Put américain — arbre binomial, 500 pas
python src/pricerTerminal.py --S 100 --K 100 --T 1 --r 0.05 --sigma 0.2 --kind put --method binomial --style american --steps 500

# Monte Carlo antithétique, 200 000 trajectoires
python src/pricerTerminal.py --S 100 --K 100 --T 1 --r 0.05 --sigma 0.2 --kind put --method mc-antithetic --n-paths 200000
```

```bash
python src/pricerTerminal.py -h   # aide complète
```

---

## Documentation

La documentation du projet est actuellement relativement dense en raison des développements mathématiques nécessaires à la compréhension des modèles de valorisation, des méthodes numériques et des techniques de réduction de variance. Elle sera progressivement restructurée au fur et à mesure de l'avancement du projet afin d'améliorer sa lisibilité.

À terme, la documentation sera scindée en plusieurs parties distinctes : une section théorique détaillant les fondements mathématiques et une section plus pratique centrée sur l'implémentation et l'analyse des données du moteur de pricing.

---

## Structure

```
Options Pricing Engine/
├── Phase 1/
│   ├── src/
│   │   ├── option.py           # Contrat option (dataclass)
│   │   ├── pricer.py           # Black-Scholes-Merton
│   │   ├── greeks.py           # Greeks (Delta, Gamma, Theta, Vega, Rho)
│   │   ├── gbm.py              # Simulation GBM (MCEngine)
│   │   ├── deltaHedging.py     # Delta hedging discret
│   │   ├── deltaHedgingMC.py   # Delta hedging Monte Carlo
│   │   ├── monteCarlo.py       # Méthodes Monte Carlo (naïf, antithétique, contrôle)
│   │   ├── binomial.py         # Arbre binomial CRR (européen / américain)
│   │   └── pricerTerminal.py   # Interface CLI
│   ├── tests/                  # Tests
│   └── reports/                # Graphiques générés
├── Phase 2/                    # À venir
└── Phase 3/                    # À venir
```

---

*Projet en cours de développement - les phases 2 et 3 seront implémentées progressivement.*
