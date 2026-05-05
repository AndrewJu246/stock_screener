# Stock Screener v1.0

A signal-based stock screening engine that scans ~1,500 US stocks nightly for breakout candidates, using 9 independent signal families, a quality gate to filter lottery stocks, and regime-aware composite scoring.

## Quick start

```bash
# Install dependencies
pip install yfinance pandas numpy

# Run a quick scan on specific tickers
python screener.py --quick NVDA AAPL TSLA AMD CRWD PLTR

# Run a full scan (S&P 1500, takes ~15-20 minutes)
python screener.py

# Set your portfolio capital (affects position sizing)
python screener.py --capital 5000

# Verbose mode (debug logging)
python screener.py -v
```

Results are saved to `output/screener_results.json`.

## Architecture

```
Layer 1 — Data sources (all free)
  ├── yfinance: price, volume, financials, analyst data
  ├── SEC EDGAR: insider transactions (Form 4)
  └── SEC EDGAR: institutional holdings (13F)

Layer 2 — Signal engine (9 families, each 0-100)
  ├── Volume anomaly      — unusual volume vs history
  ├── Momentum             — multi-timeframe trend strength
  ├── Fundamentals         — revenue/earnings acceleration
  ├── Smart money          — insider buying + institutional
  ├── Sector rotation      — money flowing into sector
  ├── Earnings quality     — cash flow vs reported earnings
  ├── Analyst revisions    — estimate upgrades
  ├── Tech megatrend       — exposure to accelerating trends
  └── Relative strength    — outperforming sector peers

Layer 3 — Composite scoring
  ├── Regime detection (bull / bear / recovery / balanced)
  ├── Regime-aware signal weighting
  └── Benchmark against balanced weights

Layer 4 — Strategy classification
  ├── Long-term hold (strong fundamentals, early growth)
  └── Short-term momentum (technical breakout)

Layer 5 — Portfolio management
  ├── Position sizing (capital-tier-aware)
  ├── Soft landing exit rules (tiered stop loss)
  └── Confidence ladder (time, agreement, backtest)
```

## Quality gate (lottery stock filter)

Every stock must pass ALL gates:
- Market cap ≥ $300M
- Average volume ≥ 100K shares/day
- Must have revenue (no pre-revenue hype)
- Fundamental score ≥ 25/100
- Cash flow quality not terrible
- Volatility not excessive (< 2.5x sector peers)
- At least 2 signal families agree (score ≥ 50)

## Files

| File | Purpose |
|------|---------|
| `config.py` | All thresholds, weights, regime definitions |
| `data_pipeline.py` | Data fetching and caching |
| `signals.py` | 9 signal calculators |
| `quality_gate.py` | Lottery stock filter + confidence ladder |
| `scorer.py` | Composite scoring + regime detection |
| `screener.py` | Main orchestrator |
| `sample_data.py` | Generate sample data for testing |

## Regime detection

The model auto-detects market regime from S&P 500 trend and VIX:

| Regime | Conditions | Weight emphasis |
|--------|-----------|----------------|
| Bull | Above 50 & 200 DMA, VIX < 20 | Momentum, sector rotation |
| Bear | Below both DMAs or VIX > 28 | Fundamentals, smart money, earnings quality |
| Early recovery | Above 50 DMA, below 200 | Balanced with fundamental lean |
| Balanced | Default / fallback | Even distribution |

## Exit rules (soft landing)

**Stop loss tiers:**
- Sell 1/3 at -7%
- Sell 1/3 at -12%
- Sell remaining at -18%

**Profit taking:**
- Take 1/3 at +20%
- Rest rides with 12% trailing stop

**Thesis break:** Exit when the signal that triggered the buy deteriorates.

## Coming in Phase 3: Backtesting

- Historical signal validation
- Sector-aware win rate tracking
- Signal combination optimization
- Paper trading mode

## Disclaimer

This is a decision-support tool, not financial advice. All data comes from free public sources. Past patterns don't guarantee future results. Always do your own research.
