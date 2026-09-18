# Tested Assets

This directory records the markets on which Adaptive-DDQN-MT5 has been tested during historical Strategy Tester experiments.

The purpose of this section is to document:

- which instruments have been evaluated,
- the historical periods used,
- testing configuration,
- observed behaviour,
- risk characteristics,
- and notable differences between assets.

The results shown here should be interpreted as **research observations rather than performance guarantees**.

---

## Why Test Across Multiple Assets?

A reinforcement-learning trading system should not be evaluated only on a single instrument.

Different markets can behave very differently in terms of:

```text
Volatility
Trend Persistence
Mean Reversion
Spread
Session Structure
Liquidity
Gap Risk
Basket Behaviour
```

Testing the same architecture across multiple assets helps identify whether observed behaviour is:

```text
Asset-Specific
        or
Potentially Generalisable
```

The objective is therefore not simply to find the asset with the highest historical return.

The more important question is:

> **How does the learned policy behave under different market structures and volatility environments?**

---

## Tested Asset Summary

| Asset | Asset Class | Timeframe | Test Period | Status | Notes |
| --- | --- | --- | --- | --- | --- |
| XAUUSD | Precious Metals | TBD | TBD | Tested | Detailed results to be added |
| EURUSD | FX | TBD | TBD | Tested | Detailed results to be added |
| GBPUSD | FX | TBD | TBD | Tested | Detailed results to be added |

Additional instruments will be added as testing progresses.

---

## What Is Evaluated?

Testing is not limited to final P/L.

The research framework also considers:

```text
Net Result
Maximum Drawdown
Basket Depth
Average Basket Depth
Recovery Duration
Maximum Exposure
Reward Behaviour
Action Distribution
Policy Stability
Regime Behaviour
Danger Events
```

This is important because two assets may produce similar returns while exposing the system to very different levels of risk.

---

## Example Interpretation

Suppose two instruments produce the following behaviour:

```text
Asset A
+ profitable
+ shallow baskets
+ moderate drawdown
+ relatively fast recovery

Asset B
+ profitable
+ deep baskets
+ prolonged drawdown
+ high exposure
```

A profit-only comparison might treat both assets as successful.

The research framework would treat these outcomes very differently.

Adaptive-DDQN-MT5 is intended to study the **quality of learned behaviour**, not only the final historical return.

---

## Testing Methodology

Unless otherwise stated, asset tests should record:

```text
Symbol
Asset Class
Broker / Data Source
Testing Period
Base Timeframe
Initial Capital
Model State
Fresh or Previously Trained Model
Training Mode
Exploration Settings
Major Risk Parameters
Reward Configuration
```

This information is important because reinforcement-learning results can change substantially when the model begins from a different saved state or training configuration.

---

## Fresh vs Continued Training

Each test should identify whether it uses:

### Fresh Training

```text
Random / Fresh Model
        ↓
No Previous Learned State
        ↓
Training Begins From Scratch
```

### Continued Training

```text
Previously Trained Model
        ↓
Saved Neural / Memory State Loaded
        ↓
Learning Continues
```

These two test types should not be directly compared without acknowledging the difference.

---

## Cross-Asset Research

Future experiments may investigate whether knowledge developed on one asset can provide useful information when testing another.

For example:

```text
Train on XAUUSD
        ↓
Evaluate on another metal

or

Train on EURUSD
        ↓
Evaluate on another FX pair
```

Such experiments would be treated separately from standard single-asset testing.

---

## Important Limitations

A tested asset should **not** be interpreted as an approved or recommended asset for live trading.

A historical test only demonstrates how the system behaved under:

```text
a specific market
+
a specific historical period
+
a specific configuration
+
a specific learned state
```

Future market behaviour may differ substantially.

For broader research limitations, see:

[Research Scope & Limitations](../doc/limitations.md)

