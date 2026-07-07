# Pricing Engine

Python-based options pricing engine, developed incrementally across four phases.

Personal project built alongside a derivatives course (Ensimag 2A), drawing on *Options, Futures and Other Derivatives* - John Hull (2024). Developed as part of a gap year preparation.

---

## Progress

| Phase | Content | Status |
|-------|---------|--------|
| **Phase 1** | Black-Scholes, Greeks & Monte Carlo | In progress |
| **Phase 2** | Stochastic Volatility - Heston & SABR | Coming soon |
| **Phase 3** | Stochastic Interest Rates - Vasicek, CIR & SVSI | Coming soon |
| **Phase 4** | Vol Surface Arbitrage Lab | Coming soon |

---

## Phase 1 - Black-Scholes, Greeks & Monte Carlo

Implementation of the foundations of vanilla option pricing.

- Geometric Brownian motion (GBM) modelling and underlying dynamics
- Black-Scholes-Merton analytical formula - European calls and puts
- Analytical Greeks: Delta, Gamma, Theta, Vega, Rho
- Monte Carlo simulation with variance reduction
- CRR Binomial tree / Longstaff-Schwartz LSM / PDE - American options with early exercise
- Implied volatility via Newton-Raphson 
- BS / MC / CRR benchmark on real SPY data **(In progress)**

## Phase 2 - Stochastic Volatility *(coming soon)*

Extension towards non-constant volatility models.

- 2.1 Implied volatility surface construction
- 2.2 Heston model - Carr-Madan pricing
- 2.3 Heston calibration - Differential Evolution
- 2.4 SABR model - Hagan approximation formula
- 2.5 Confrontation with real market data (SPY): closing the gap vs Phase 1

## Phase 3 - Stochastic Interest Rates *(coming soon)*

Term-structure modelling and coupling with the underlying.

- 3.1 Vasicek model: mean-reverting stochastic rate
- 3.2 Cox-Ingersoll-Ross (CIR) model
- 3.3 Rate/underlying coupling: SVSI model
- 3.4 Confrontation with real market data: long-maturity options

## Phase 4 - Vol Surface Arbitrage Lab *(coming soon)*

Surface analysis and detection of pricing inconsistencies.

- 4.1 No-arbitrage - calendar spread & butterfly detection
- 4.2 Dupire local volatility
- 4.3 Second-order Greeks: Vanna, Volga, Charm
- 4.4 P&L Explain
- 4.5 Stress tests + final dashboard

---

## Usage

From the `Phase 1/` directory:

```bash
python src/pricerTerminal.py --S <spot> --K <strike> --T <maturity> --r <rate> --sigma <vol> --kind <call|put> [--method <method>] [--style <european|american>] [--steps N] [--n-paths N] [--seed N] [--n-space N]
```

**Available methods**

| `--method` | Description | Style |
|---|---|---|
| `bs` *(default)* | Black-Scholes-Merton analytical formula | european |
| `binomial` | CRR binomial tree | european / american |
| `mc-naive` | Naive Monte Carlo | european |
| `mc-antithetic` | Monte Carlo - antithetic variates | european |
| `mc-control` | Monte Carlo - control variate | european |
| `mc-control-antithetic` | Monte Carlo - antithetic + control variate | european |
| `mc-lsm` | Longstaff-Schwartz Monte Carlo | american |
| `pde` | Crank-Nicolson PDE via QuantLib | european / american |

**Specific flags**

| Flag | Default | Usage |
|---|---|---|
| `--style` | `european` | Exercise style for `binomial`, `mc-lsm` and `pde` |
| `--steps` | `100` | Number of time steps — binomial, mc-lsm, pde |
| `--n-paths` | `100000` | Number of MC paths |
| `--seed` | `42` | MC random seed |
| `--n-space` | `200` | Number of space steps for `pde` |

**Examples**

```bash
# Black-Scholes
python src/pricerTerminal.py --S 100 --K 100 --T 1 --r 0.05 --sigma 0.2 --kind call

# American put — binomial tree, 500 steps
python src/pricerTerminal.py --S 100 --K 100 --T 1 --r 0.05 --sigma 0.2 --kind put --method binomial --style american --steps 500

# Antithetic Monte Carlo, 200,000 paths
python src/pricerTerminal.py --S 100 --K 100 --T 1 --r 0.05 --sigma 0.2 --kind put --method mc-antithetic --n-paths 200000

# American put — Longstaff-Schwartz
python src/pricerTerminal.py --S 100 --K 100 --T 1 --r 0.05 --sigma 0.2 --kind put --method mc-lsm --style american --steps 50 --n-paths 50000

# American put — Crank-Nicolson PDE
python src/pricerTerminal.py --S 100 --K 100 --T 1 --r 0.05 --sigma 0.2 --kind put --method pde --style american --steps 200 --n-space 200
```

```bash
python src/pricerTerminal.py -h   # full help
```

---

## Documentation

The Phase 1 documentation is split into two distinct documents:

- **[Pricing d'Options - Théorie, Méthodes et Démonstrations](Phase%201/Pricing%20d'Options%20-%20Théorie%2C%20Méthodes%20et%20Démonstrations.pdf)** — theoretical report covering the mathematical foundations of each method (Black-Scholes, Greeks, Monte Carlo and variance reduction, CRR binomial tree, Longstaff-Schwartz, PDE/Crank-Nicolson, implied volatility and volatility surface), with proofs and numerical examples.
- **[Pricing d'Options - Benchmark des Méthodes](Phase%201/Pricing%20d'Options%20-%20Benchmark%20des%20Méthodes.pdf)** — practical document benchmarking all methods against each other on a large set of options (~2000), then against real market data from a liquid underlying (SPY).

---

## Structure

```
Options Pricing Engine/
├── Phase 1/
│   ├── src/
│   │   ├── option.py           # Option contract (dataclass)
│   │   ├── BSpricer.py         # Black-Scholes-Merton
│   │   ├── greeks.py           # Greeks (Delta, Gamma, Theta, Vega, Rho)
│   │   ├── gbm.py              # GBM simulation (MCEngine)
│   │   ├── deltaHedging.py     # Discrete delta hedging
│   │   ├── deltaHedgingMC.py   # Monte Carlo delta hedging
│   │   ├── monteCarlo.py       # Monte Carlo methods (naive, antithetic, control)
│   │   ├── monteCarloLSM.py    # Longstaff-Schwartz Monte Carlo (american)
│   │   ├── binomial.py         # CRR binomial tree (european / american)
│   │   ├── pde.py              # Crank-Nicolson PDE via QuantLib
│   │   ├── pricerTerminal.py   # CLI interface
│   │   └── benchmark.py        # Benchmark all methods on ~2000 real options
│   ├── tests/                  # Tests
│   └── reports/                # Generated charts
├── Phase 2/                    # Coming soon
├── Phase 3/                    # Coming soon
└── Phase 4/                    # Coming soon
```

---

*Project under active development - phases 2, 3 and 4 will be implemented incrementally.*
