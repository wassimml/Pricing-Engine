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
- Simulation Monte Carlo avec réduction de variance 
- Arbre binomial CRR / Longstaff-Schwartz LSM / PDE - options américaines avec exercice anticipé
- Volatilité implicite par Newton-Raphson 
- Benchmark BS / MC / CRR sur données réelles SPY **(En cours)**

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
python src/pricerTerminal.py --S <spot> --K <strike> --T <maturité> --r <taux> --sigma <vol> --kind <call|put> [--method <méthode>] [--style <european|american>] [--steps N] [--n-paths N] [--seed N] [--n-space N]
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
| `mc-lsm` | Monte Carlo Longstaff-Schwartz | américain |
| `pde` | PDE Crank-Nicolson via QuantLib | européen / américain |

**Flags spécifiques**

| Flag | Défaut | Usage |
|---|---|---|
| `--style` | `european` | Style d'exercice pour `binomial`, `mc-lsm` et `pde` |
| `--steps` | `100` | Nombre de pas de temps — binomial, mc-lsm, pde |
| `--n-paths` | `100000` | Nombre de trajectoires MC |
| `--seed` | `42` | Seed aléatoire MC |
| `--n-space` | `200` | Nombre de pas en espace pour `pde` |

**Exemples**

```bash
# Black-Scholes
python src/pricerTerminal.py --S 100 --K 100 --T 1 --r 0.05 --sigma 0.2 --kind call

# Put américain — arbre binomial, 500 pas
python src/pricerTerminal.py --S 100 --K 100 --T 1 --r 0.05 --sigma 0.2 --kind put --method binomial --style american --steps 500

# Monte Carlo antithétique, 200 000 trajectoires
python src/pricerTerminal.py --S 100 --K 100 --T 1 --r 0.05 --sigma 0.2 --kind put --method mc-antithetic --n-paths 200000

# Put américain — Longstaff-Schwartz
python src/pricerTerminal.py --S 100 --K 100 --T 1 --r 0.05 --sigma 0.2 --kind put --method mc-lsm --style american --steps 50 --n-paths 50000

# Put américain — PDE Crank-Nicolson
python src/pricerTerminal.py --S 100 --K 100 --T 1 --r 0.05 --sigma 0.2 --kind put --method pde --style american --steps 200 --n-space 200
```

```bash
python src/pricerTerminal.py -h   # aide complète
```

---

## Documentation

La documentation de la Phase 1 est actuellement scindée en deux documents distincts :

- **[Pricing d'Options - Théorie, Méthodes et Démonstrations](Phase%201/Pricing%20d'Options%20-%20Théorie%2C%20Méthodes%20et%20Démonstrations.pdf)** - le rapport théorique, qui détaille les fondements mathématiques de chaque méthode (Black-Scholes, Greeks, Monte Carlo et réduction de variance, arbre binomial CRR, Longstaff-Schwartz, PDE/Crank-Nicolson, volatilité implicite et surface de volatilité), avec démonstrations et exemples numériques.
- **[Pricing d'Options - Benchmark des Méthodes](Phase%201/Pricing%20d'Options%20-%20Benchmark%20des%20Méthodes.pdf)** - le document pratique, qui confronte l'ensemble des méthodes entre elles, sur un volume important d'options (~2000), puis face aux données réelles d'un marché liquide (SPY).

---

## Structure

```
Options Pricing Engine/
├── Phase 1/
│   ├── src/
│   │   ├── option.py           # Contrat option (dataclass)
│   │   ├── BSpricer.py         # Black-Scholes-Merton
│   │   ├── greeks.py           # Greeks (Delta, Gamma, Theta, Vega, Rho)
│   │   ├── gbm.py              # Simulation GBM (MCEngine)
│   │   ├── deltaHedging.py     # Delta hedging discret
│   │   ├── deltaHedgingMC.py   # Delta hedging Monte Carlo
│   │   ├── monteCarlo.py       # Méthodes Monte Carlo (naïf, antithétique, contrôle)
│   │   ├── monteCarloLSM.py    # Monte Carlo Longstaff-Schwartz (américain)
│   │   ├── binomial.py         # Arbre binomial CRR (européen / américain)
│   │   ├── pde.py              # PDE Crank-Nicolson via QuantLib
│   │   ├── pricerTerminal.py   # Interface CLI
│   │   └── benchmark.py        # Benchmark de toutes les méthodes sur ~2000 options réelles
│   ├── tests/                  # Tests
│   └── reports/                # Graphiques générés
├── Phase 2/                    # À venir
└── Phase 3/                    # À venir
```

---

*Projet en cours de développement - les phases 2 et 3 seront implémentées progressivement.*
