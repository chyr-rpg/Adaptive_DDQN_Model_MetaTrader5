# Tested Assets

This directory contains selected historical Strategy Tester experiments conducted with **Adaptive-DDQN-MT5**.

The purpose of this section is to document how the research architecture behaves across different markets, execution timeframes and asset-specific configurations.

The focus is not simply on which test produced the highest historical return. The more important objective is to compare:

```text
Return
Drawdown
Basket Behaviour
Exposure
Recovery
Trading Frequency
Policy Behaviour
and Learning Stability
```

> **Important:** These results are research observations, not performance guarantees, live-trading recommendations or evidence that similar results will occur in future markets.

---

## Tested Asset Summary

The current repository contains results for three markets:

| Asset | Asset Class | TF | Test Period | Net P/L | Return* | Equity DD | Profit Factor | Sharpe | Trades |
| --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| [EURUSD](EURUSD_M5/) | FX | M5 | 2026-07-01 → 2026-09-19 | $47,274.20 | 47.27% | 12.36% | 1.91 | 3.67 | 335 |
| [SP500](SP500_M15/) | Equity Index | M15 | 2026-07-01 → 2026-09-19 | $3,455.70 | 3.46% | 3.54% | 2.25 | 1.32 | 52 |
| [XAUUSD](XAUUSD_M1/) | Precious Metals | M1 | 2026-07-01 → 2026-09-19 | $154,987.31 | 154.99% | 17.87% | 2.49 | 5.78 | 1,788 |

\* Return shown relative to the $100,000 initial testing balance.

Unless otherwise stated, these tests used:

```text
Initial Balance:     $100,000
Leverage:            1:100
Reported Data Quality: 100%
Training Mode:       Enabled
```

These results should be interpreted as **adaptive training runs**, not frozen out-of-sample validation.

---

## Why Test Across Multiple Assets?

A reinforcement-learning trading system should not be evaluated using only one instrument.

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
Recovery Characteristics
```

Testing the architecture across multiple assets helps investigate whether observed behaviour is:

```text
Asset-Specific
        or
Potentially Generalisable
```

The underlying research question is therefore:

> **How does the learning system behave when exposed to different market structures and volatility environments?**

---

## These Are Asset-Specific Configurations

The three tests should **not** be interpreted as a controlled ranking of EURUSD, SP500 and XAUUSD.

The research architecture is shared, but each test uses an asset-specific configuration.

Parameters such as:

```text
Position Size
Grid Behaviour
Basket Exit Logic
Volatility Sensitivity
Risk Thresholds
Execution Timeframe
```

may differ between assets.

Contract specifications also differ substantially between FX, precious metals and equity-index CFDs.

For this reason:

> **A higher historical return on one asset does not imply that the asset or configuration is objectively superior.**

The tests are better understood as separate research cases examining how the architecture behaves under different environments.

---

## Why Equity Drawdown Matters

For a basket-based trading system, **equity drawdown is particularly important**.

Balance drawdown records realised account changes, while equity drawdown also captures unrealised losses from open positions.

A basket can therefore appear relatively stable on a balance curve while carrying substantial floating exposure.

Conceptually:

```text
Balance Drawdown
        ↓
Realised account decline

Equity Drawdown
        ↓
Realised decline
        +
Floating basket risk
```

For this reason, the asset summaries in this repository emphasise **equity drawdown** when evaluating risk.

---

## Initial Observations

The three current tests already show materially different behavioural profiles.

### EURUSD — M5

EURUSD produced a moderate-frequency trading profile during the July–September test.

Key characteristics include:

```text
335 trades
47.27% historical return
12.36% relative equity drawdown
1.91 profit factor
3.67 Sharpe ratio
```

Additional EURUSD tests are retained because earlier periods exposed substantially more difficult basket and drawdown behaviour.

These stress cases are useful for understanding why later research focuses on:

```text
Drawdown Memory
Danger Recognition
Risk-Aware Rewards
Basket Expansion Control
```

rather than considering profitable periods alone.

[View EURUSD test files →](EURUSD_M5/)

---

### SP500 — M15

The SP500 test produced a much lower trading frequency:

```text
52 trades
3.46% historical return
3.54% relative equity drawdown
2.25 profit factor
1.32 Sharpe ratio
```

The sample therefore represents a substantially different trading environment from EURUSD and XAUUSD.

Because only 52 trades occurred during the test period, results should be interpreted cautiously.

[View SP500 test files →](SP500_M15/)

---

### XAUUSD — M1

XAUUSD produced the highest trading frequency of the current three tests:

```text
1,788 trades
154.99% historical return
17.87% relative equity drawdown
2.49 profit factor
5.78 Sharpe ratio
```

The historical return was substantially higher, but the system also experienced the largest relative equity drawdown among the three tests.

This illustrates why return alone is insufficient for evaluating adaptive basket behaviour.

[View XAUUSD test files →](XAUUSD_M1/)

---

## What Is Evaluated?

Testing is not limited to final P/L.

The research framework considers areas such as:

```text
Net Result
Maximum Equity Drawdown
Balance Drawdown
Trading Frequency
Profit Factor
Sharpe Ratio
Basket Depth
Maximum Exposure
Recovery Duration
Reward Behaviour
Action Distribution
Policy Stability
Regime Behaviour
Danger Events
```

As the research framework develops, additional learning-specific diagnostics will be added.

Two strategies can produce similar returns while exposing the account to very different levels of risk.

For example:

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

A profit-only comparison might classify both as successful.

Adaptive-DDQN-MT5 instead attempts to study the **quality of the behaviour that produced the result**.

---

## Model & Memory State

Adaptive-DDQN-MT5 can persist neural-network and memory information between testing sessions.

During Strategy Tester operation, the system may generate `.dat` files containing learned state and memory information.

A typical local MetaTrader 5 Strategy Tester location is similar to:

```text
C:\Users\<USER>\AppData\Roaming\MetaQuotes\Tester\
<TESTER-ID>\
Agent-127.0.0.1-<PORT>\
MQL5\Files\
```

The exact folder depends on the local MetaTrader installation and testing agent.

Persistent state is important because:

```text
Previous Training
        ↓
Saved Neural / Memory State
        ↓
Loaded During Later Test
        ↓
Different Starting Knowledge
        ↓
Potentially Different Behaviour
```

A model that starts with saved memory is therefore not directly comparable with a model trained from scratch.

---

## Fresh vs Continued Training

Every experiment should identify its starting model state.

### Fresh Training

```text
Fresh / Random Model
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

This distinction is important for reproducibility.

Two tests using identical market data and parameters may produce different behaviour if they begin from different learned states.

---

## Current Tests Are Adaptive Training Runs

The tests currently published in this directory were conducted with learning enabled.

Conceptually:

```text
Historical Market Data
        ↓
Agent Observes State
        ↓
Agent Trades
        ↓
Reward Generated
        ↓
Model / Memory Updated
        ↓
Later Decisions Use
Accumulated Experience
```

This means the results demonstrate how the system behaved **while adapting to the historical period**.

They should not yet be interpreted as strict out-of-sample evidence.

---

## Next Research Stage: Frozen Evaluation

A stronger validation framework separates training data from later unseen evaluation data.

The planned workflow is:

```text
TRAINING PERIOD
        ↓
TrainingMode = true
        ↓
Agent learns and updates
model / memory
        ↓
Save learned state
        ↓
────────────────────────
        ↓
UNSEEN TEST PERIOD
        ↓
TrainingMode = false
        ↓
Freeze learning
        ↓
Evaluate learned policy
on later market data
```

This creates a clearer distinction between:

```text
Learning Performance
```

and:

```text
Out-of-Sample Policy Performance
```

Future asset experiments will increasingly use this structure.

---

## Stress Cases Are Also Valuable Results

This repository does not aim to show only favourable equity curves.

Poor-performing periods can provide some of the most useful information for reinforcement-learning research.

A stress episode can reveal:

```text
Excessive Basket Expansion
Poor Recovery Behaviour
Inadequate Drawdown Penalties
Regime Failure
Memory Failure
Risk-Control Weaknesses
```

Where useful, these cases will be retained alongside stronger historical periods.

The objective is to understand:

> **Why did the system behave poorly, and what should the learning architecture remember from that behaviour?**

rather than simply removing unsuccessful tests.

---

## Testing Methodology

Each asset experiment should record enough information to understand the testing environment.

Important fields include:

```text
Symbol
Asset Class
Testing Period
Execution Timeframe
Initial Capital
Leverage

Fresh or Previously Trained Model
Training Mode
Exploration Configuration

Major Execution Parameters
Risk Configuration
Reward Configuration

Net Result
Equity Drawdown
Trade Count
Profit Factor
Sharpe Ratio

Notable Behaviour
Stress Events
Research Observations
```

This information allows later tests to be compared more meaningfully.

---

## Raw Settings and Reports

MetaTrader Strategy Tester reports and `.set` files can contain detailed implementation parameters.

The public repository may therefore use **selected result summaries and non-sensitive configuration information**, while full research settings and proprietary implementation parameters may remain in the private development repository.

This separation is consistent with the project's public/private source model:

```text
Public Repository
        ↓
Research Methodology
Selected Results
Learning Edition
Reproducible Public Examples

Private Repository
        ↓
Current Full Implementation
Full Research Configuration
Proprietary Tuning
Complete Internal Test Data
```

---

## Cross-Asset Research

Future experiments may investigate whether behaviour learned on one instrument transfers to related markets.

For example:

```text
Train on XAUUSD
        ↓
Evaluate on another metal
```

or:

```text
Train on EURUSD
        ↓
Evaluate on another FX pair
```

Such experiments would be documented separately from standard single-asset tests because transfer learning introduces additional methodological questions.

---

## Directory Structure

The current structure is organised by asset and timeframe:

```text
tested_assets/
│
├── README.md
│
├── EURUSD_M5/
│   ├── performance images
│   ├── Strategy Tester report
│   └── test configuration
│
├── SP500_M15/
│   ├── performance images
│   ├── Strategy Tester report
│   └── test configuration
│
└── XAUUSD_M1/
    ├── performance images
    ├── Strategy Tester report
    └── test configuration
```

Additional assets and validation periods may be added as research progresses.

---

## Important Limitations

A tested asset should **not** be interpreted as an approved or recommended asset for live trading.

A historical result only demonstrates how the system behaved under:

```text
a specific market
+
a specific historical period
+
a specific configuration
+
a specific model state
+
a specific execution environment
```

Future behaviour may differ materially.

Historical Strategy Tester performance does not account perfectly for all live-market conditions, including:

```text
Slippage
Liquidity Changes
Execution Latency
Spread Changes
Market Gaps
Broker Differences
Unexpected Tail Events
```

For broader research limitations, see:

[**Research Scope & Limitations →**](../doc/limitations.md)

---

## Research Objective

The purpose of this directory is not to answer:

> **Which asset produced the largest backtest profit?**

The more useful research question is:

> **How does the adaptive system change its behaviour across different markets, and can that behaviour become more efficient, stable and risk-aware as experience accumulates?**

Future experiments will therefore increasingly focus on:

```text
Policy Evolution
Drawdown Behaviour
Basket Efficiency
Memory Utilisation
Regime Adaptation
Risk-Adjusted Performance
and Out-of-Sample Generalisation
```
