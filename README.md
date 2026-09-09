# Market Regime Detection

Python research project for identifying and monitoring financial-market regimes using unsupervised learning on market time-series data.

The current implementation combines a Gaussian Mixture Model with weekly market features to classify observations into **Bear Market**, **Transition**, and **Bull Market** states, estimate regime probabilities, study regime persistence, and produce an indicative one-month regime projection.

## Methodology

Market data are downloaded with `yfinance`, transformed into daily features, and then aggregated to a weekly frequency. The model is trained on the available history from 1990 onward when data are available for the selected asset.

The Gaussian Mixture Model is estimated on seven standardized variables:

- weekly return;
- realized volatility;
- relative volume;
- RSI;
- momentum;
- intraperiod price range;
- drawdown.

The underlying Gaussian Mixture Model estimates two primary latent states. These components are ordered economically using their historical return, volatility, momentum, and drawdown characteristics and mapped to Bear and Bull regimes. A separate uncertainty and regime-switching rule identifies Transition observations.

## Outputs

The application provides:

- current market regime and associated probability;
- historical regime classification;
- regime probabilities through time;
- historical performance and volatility conditional on each regime;
- regime persistence and historical frequency;
- weekly transition matrix;
- indicative one-month regime probabilities;
- current values and standardized deviations of the seven model inputs;
- graphical visualization of market prices and historical regime zones.

## Supported markets

The current interface includes:

- S&P 500;
- Nasdaq;
- Gold;
- EURO STOXX 50;
- MSCI World;
- SMI;
- CAC 40.

## Installation

```bash
python3 -m pip install numpy pandas yfinance matplotlib scikit-learn
```

## Run

```bash
python3 market_regime_detection.py
```

The application uses a Tkinter desktop interface.

## Current status

The project is functional and remains under development. The current version focuses on interpretable unsupervised regime classification, historical regime statistics, and transition analysis rather than predictive trading signals.

## Limitations

Regime labels are inferred statistically and are not directly observed market states. Results depend on the selected features, model specification, historical sample, and heuristic definition of the Transition regime. Transition probabilities are estimated from historical classifications and implicitly assume that past transition behavior remains informative. The one-month projection is therefore indicative rather than a validated forecast of future returns or market direction.

Further work may include alternative clustering methods, hidden Markov models, stability testing, walk-forward validation, sensitivity analysis, and comparison across different feature sets and sampling frequencies.