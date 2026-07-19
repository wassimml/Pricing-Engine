# Pricing Engine

Python-based options pricing engine, developed incrementally across four phases.

Personal project built alongside a derivatives course (Ensimag 2A), drawing on *Options, Futures and Other Derivatives* - John Hull (2024). Developed as part of a gap year preparation.

---

## Progress

| Phase | Content | Status |
|-------|---------|--------|
| **Phase 1** | Black-Scholes, Greeks, CRR, Monte Carlo & PDE | In progress |
| **Phase 2** | Stochastic Volatility - Heston & SABR | In progress |
| **Phase 3** | Stochastic Interest Rates - Vasicek, CIR & SVSI | Coming soon |
| **Phase 4** | Vol Surface Arbitrage Lab | Coming soon |

---

## Phase 1 - Black-Scholes, Greeks & Monte Carlo

Implementation of the foundations of vanilla option pricing.

- Geometric Brownian motion (GBM) modelling and underlying dynamics
- Black-Scholes-Merton analytical formula - European calls and puts
- Analytical Greeks: Delta, Gamma, Theta, Vega, Rho
- Monte Carlo simulation with variance reduction
- CRR Binomial tree
- Longstaff-Schwartz LSM - order-3 polynomial regression
- PDE - Crank-Nicolson
- Implied volatility inversion via Brent's method - BS, CRR, LSM & PDE, with smile and 3D surface on real AAPL data
- LSM / CRR / PDE (American) vs BS baseline, benchmarked on real SPY market data

## Phase 2 - Stochastic Volatility

Extension towards non-constant volatility models.

- Implied volatility surface construction
- Heston model - Carr-Madan pricing
- Heston calibration - Differential Evolution
- SABR model - Hagan approximation formula
- Confrontation with real market data (SPY): closing the gap vs Phase 1

## Phase 3 - Stochastic Interest Rates *(coming soon)*

Term-structure modelling and coupling with the underlying.

- Vasicek model: mean-reverting stochastic rate
- Cox-Ingersoll-Ross (CIR) model
- Rate/underlying coupling: SVSI model
- Confrontation with real market data: long-maturity options

## Phase 4 - Vol Surface Arbitrage Lab *(coming soon)*

Surface analysis and detection of pricing inconsistencies.

- No-arbitrage - calendar spread & butterfly detection
- Dupire local volatility
- Second-order Greeks: Vanna, Volga, Charm
- P&L Explain
- Stress tests + final dashboard

---

## Web app

A Streamlit app ties the whole project together: interactive pricer, Greeks, and a browsable
gallery of every study (convergence, benchmarks, implied volatility).

```bash
cd "Phase 1"
streamlit run app.py
```
![Interface](Phase%201/reports/Streamlit.png)
![Documentation](Phase%201/reports/DocumentationInterface.png)
The app's interface (layout, pages, visual identity) was built with Claude. Building a
Streamlit dashboard isn't the point of this project, pricing methods are.

---

## Documentation

The Phase 1 documentation is split into two distinct documents:

- **[Pricing d'Options - Théorie, Méthodes et Démonstrations](Phase%201/Pricing%20d'Options%20-%20Théorie%2C%20Méthodes%20et%20Démonstrations.pdf)** : theoretical report covering the mathematical foundations of each method (Black-Scholes, Greeks, Monte Carlo and variance reduction, CRR binomial tree, Longstaff-Schwartz, PDE/Crank-Nicolson, implied volatility and volatility surface), with proofs and numerical examples.
- **[Pricing d'Options - Benchmark des Méthodes](Phase%201/Pricing%20d'Options%20-%20Benchmark%20des%20Méthodes.pdf)** : practical document benchmarking all methods against each other on a large set of options (~2000), then against real market data from a liquid underlying (SPY).

---

## Structure

```
Options Pricing Engine/
├── Phase 1/
│   ├── app.py                    # Streamlit app entry point (navigation)
│   ├── app_theme.py              # Shared "slate & gold" visual identity (palette, CSS, Plotly template)
│   ├── app_pages/                # Streamlit pages (pricer, greeks, benchmarks, ...)
│   ├── src/
│   │   ├── option.py            # Option contract (dataclass)
│   │   ├── BSpricer.py          # Black-Scholes-Merton
│   │   ├── greeks.py            # Greeks (Delta, Gamma, Theta, Vega, Rho)
│   │   ├── gbm.py               # GBM simulation (MCEngine)
│   │   ├── deltaHedging.py      # Discrete delta hedging
│   │   ├── deltaHedgingMC.py    # Monte Carlo delta hedging
│   │   ├── monteCarlo.py        # Monte Carlo methods (naive, antithetic, control) + seed-averaged convergence
│   │   ├── monteCarloLSM.py     # Longstaff-Schwartz Monte Carlo (american) + seed-averaged convergence
│   │   ├── binomial.py          # CRR binomial tree (european / american)
│   │   ├── pde.py               # Crank-Nicolson PDE via QuantLib
│   │   ├── impliedVol.py        # Implied vol inversion (BS/CRR/LSM/PDE) + smile & surface (AAPL)
│   │   ├── timeValue.py         # Time value study (BS, spot x maturity 3D surface)
│   │   ├── benchmarkMethods.py  # Accuracy/speed vs parameter sweep (3D)
│   │   ├── generateOptionsBook.py # Generate the reproducible ~2000-option synthetic book
│   │   ├── benchmarkInData.py   # Benchmark all methods on ~2000 synthetic options
│   │   ├── benchmarkSPY.py      # Benchmark LSM/CRR/PDE vs real SPY market data
│   │   ├── makeSnapshotSPY.py   # Generate a reproducible SPY options snapshot
│   │   └── makeSnapshotAAPL.py  # Generate a reproducible AAPL options snapshot
│   ├── data/                    # Synthetic book + frozen market snapshots
│   ├── tests/                   # Tests
│   └── reports/                 # Generated charts
├── Phase 2/                     # Coming soon
├── Phase 3/                     # Coming soon
└── Phase 4/                     # Coming soon
```

---

*Project under active development - phases 2, 3 and 4 will be implemented incrementally.*
