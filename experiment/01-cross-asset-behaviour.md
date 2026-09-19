# Experiment 01 — Cross-Asset Behaviour

## Objective

This experiment examines how the Adaptive-DDQN-MT5 research architecture behaves across three different markets:

- EURUSD
- SP500
- XAUUSD

The objective is **not** to determine which asset is "best" or which configuration produces the highest historical return.

Instead, the experiment asks:

> **How does the behaviour of the adaptive trading system change across different market structures, execution timeframes and asset-specific configurations?**

Particular attention is given to:

```text
Trading Frequency
Historical Return
Equity Drawdown
Profit Factor
Holding Duration
Risk / Recovery Behaviour
and Differences in Exposure Characteristics
```

---

## 1. Experimental Context

The three tests were conducted using the private Adaptive-DDQN-MT5 research implementation.

Each configuration uses the same broader reinforcement-learning architecture, but parameters are adapted to the characteristics of the individual market.

Therefore, this is **not a controlled asset-ranking experiment**.

The experiment is better interpreted as:

```text
Same Research Architecture
          +
Different Assets
          +
Different Execution Timeframes
          +
Asset-Specific Configuration
          ↓
Compare Behavioural Outcomes
```

---

## 2. Test Environment

The currently documented tests share the following broad conditions:

```text
Initial Deposit:       $100,000
Leverage:              1:100
Test Start:            2026-07-01
Test End:              2026-09-19
Training Mode:         Enabled
Reported Data Quality: 100%
```

The execution timeframes differ:

```text
EURUSD  → M5
SP500   → M15
XAUUSD  → M1
```

Because learning remained enabled during these tests, the results represent **adaptive training runs**.

The model was able to update its behaviour as the historical simulation progressed.

These results should therefore not yet be interpreted as frozen out-of-sample validation.

---

## 3. Results Snapshot

| Market | Asset Class | TF | Net P/L | Return* | Equity DD | Profit Factor | Sharpe | Trades | Avg. Holding |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| EURUSD | FX | M5 | $47,274.20 | 47.27% | 12.36% | 1.91 | 3.67 | 335 | 8:54:42 |
| SP500 | Equity Index | M15 | $3,455.70 | 3.46% | 3.54% | 2.25 | 1.32 | 52 | 73:24:34 |
| XAUUSD | Precious Metals | M1 | $154,987.31 | 154.99% | 17.87% | 2.49 | 5.78 | 1,788 | 1:26:35 |

\* Return calculated relative to the $100,000 initial testing balance.

Individual reports are available under:

- [EURUSD — M5](../tested_assets/eurusd/)
- [SP500 — M15](../tested_assets/SP500/)
- [XAUUSD — M1](../tested_assets/xauusd/)

---

## 4. Trading Frequency

The most immediate difference between the three tests is trading frequency.

```text
SP500
52 trades
        ↓
EURUSD
335 trades
        ↓
XAUUSD
1,788 trades
```

This difference is substantial.

XAUUSD generated more than five times as many trades as EURUSD and more than thirty times as many as SP500 during the same broad testing period.

Several factors may contribute:

```text
Execution Timeframe
Market Volatility
Price Behaviour
Asset-Specific Parameters
Basket Formation
Entry / Exit Frequency
```

The results therefore suggest that the same broader architecture can express very different levels of activity depending on the environment in which it operates.

---

## 5. Holding Duration

The markets also produced very different holding characteristics.

```text
XAUUSD
~1 hour 27 minutes average

EURUSD
~8 hours 55 minutes average

SP500
~73 hours 24 minutes average
```

This indicates three substantially different trading profiles.

### XAUUSD

The M1 implementation generated frequent and comparatively short-lived trading activity.

### EURUSD

The M5 test produced a slower cycle, with positions remaining active for several hours on average.

### SP500

The M15 configuration produced relatively few trades with much longer average holding periods.

This difference matters because holding duration affects:

```text
Exposure Time
Recovery Behaviour
Swap / Financing Sensitivity
Gap Risk
Capital Utilisation
and Basket Persistence
```

The three tests should therefore not be interpreted only through final P/L.

---

## 6. Equity Drawdown

Relative equity drawdown was:

```text
SP500   →  3.54%
EURUSD  → 12.36%
XAUUSD  → 17.87%
```

For this project, equity drawdown is more informative than balance drawdown because the system can carry open basket exposure.

Conceptually:

```text
Balance
   ↓
Realised result

Equity
   ↓
Realised result
+
Unrealised basket exposure
```

XAUUSD generated the largest historical return in this sample, but it also experienced the largest relative equity drawdown.

This is an important example of why the project does not define successful learning purely through return.

---

## 7. Return and Risk Are Not the Same Measurement

The historical returns differ significantly:

```text
EURUSD  →  47.27%
SP500   →   3.46%
XAUUSD  → 154.99%
```

However, these values should not be interpreted independently of the paths used to produce them.

For example:

```text
Higher Return
       ≠
Automatically Better Behaviour
```

A high-return trajectory may also involve:

```text
More Exposure
More Trades
Greater Drawdown
Deeper Baskets
Higher Turnover
Greater Tail Risk
```

The central research question is therefore not:

> Which configuration earned the most?

but:

> **What type of behaviour generated that return, and how much risk was required to sustain it?**

---

## 8. Profit Factor

The reported profit factors were:

```text
EURUSD  → 1.91
SP500   → 2.25
XAUUSD  → 2.49
```

All three historical samples produced gross profit greater than gross loss.

However, profit factor alone does not capture:

```text
Floating Drawdown
Exposure Growth
Basket Depth
Holding Time
Tail Events
or Path Dependence
```

For a basket-based adaptive system, these additional dimensions remain important.

---

## 9. Sharpe Ratio

The reported Strategy Tester Sharpe ratios were:

```text
EURUSD  → 3.67
SP500   → 1.32
XAUUSD  → 5.78
```

These values describe the historical return path generated under the specific Strategy Tester assumptions.

They should not be interpreted as expected future Sharpe ratios.

In particular, the relatively small SP500 trade sample and the adaptive nature of all three training runs limit direct comparison.

---

## 10. Behavioural Profiles

The current results suggest three distinct historical profiles.

### EURUSD — Medium Activity / Medium Holding Horizon

```text
335 trades
M5 execution
~9 hour average holding time
12.36% equity drawdown
47.27% historical return
```

EURUSD produced a middle-ground profile relative to the other two assets.

The system traded considerably more often than SP500 but much less frequently than XAUUSD.

The additional EURUSD stress-period testing is particularly valuable because it demonstrates that favourable behaviour was not consistent across all historical periods.

---

### SP500 — Low Activity / Long Holding Horizon

```text
52 trades
M15 execution
~73 hour average holding time
3.54% equity drawdown
3.46% historical return
```

SP500 generated the smallest sample.

The system remained in positions substantially longer and traded relatively infrequently.

This may indicate that the asset-specific configuration was more selective or that the market generated fewer qualifying state/action opportunities.

Because the sample contains only 52 trades, conclusions should remain tentative.

---

### XAUUSD — High Activity / Short Holding Horizon

```text
1,788 trades
M1 execution
~1.5 hour average holding time
17.87% equity drawdown
154.99% historical return
```

XAUUSD produced a very different operating regime.

The combination of:

```text
higher trading frequency
+
shorter average holding period
+
larger historical return
+
larger equity drawdown
```

suggests a much more active exposure cycle.

This test is particularly useful for studying whether high-frequency learning behaviour creates additional drawdown and basket-management pressure.

---

## 11. The EURUSD Stress Case

An earlier EURUSD test contains a substantially more adverse period than the July–September result.

This case is intentionally retained rather than excluded.

The purpose is to study questions such as:

```text
What happened before the drawdown?

Did basket depth increase?

Was the reward system penalising risk strongly enough?

Did the agent repeatedly enter similar adverse states?

Could historical memory recognise the developing pattern?

Could later architecture changes reduce recurrence?
```

This type of failure case is one of the motivations for later research components including:

```text
Drawdown-Event Memory
Danger Replay
Deep-Basket Replay
Danger Brain
Risk-Aware Rewards
Smart Basket Add Controls
```

A poor historical episode can therefore provide valuable learning-system evidence even when its P/L is undesirable.

---

## 12. Asset-Specific Configuration Matters

The three experiments use different configurations.

For example, parameters related to:

```text
Position Size
Basket Take-Profit
Grid Expansion
Channel Behaviour
Risk Thresholds
and Execution Timeframe
```

are not identical.

In addition, contract specifications differ considerably across:

```text
FX
Gold
Equity-Index CFDs
```

The purpose of the experiment is therefore **not** to claim that asset differences alone caused the observed results.

Observed behaviour reflects the interaction between:

```text
Market
+
Timeframe
+
Configuration
+
Learning State
+
Trading Architecture
```

---

## 13. What Can Be Concluded?

The current evidence supports several descriptive observations.

### Observation 1 — The architecture does not express the same behaviour on every market

Trading frequency and holding duration differ substantially across the three assets.

### Observation 2 — Higher historical return can coexist with higher floating risk

XAUUSD produced both the largest historical return and the largest relative equity drawdown in this sample.

### Observation 3 — Lower activity does not necessarily imply higher historical return

SP500 generated far fewer trades than the other configurations and produced a comparatively modest return during this testing window.

### Observation 4 — Behavioural analysis is more informative than return alone

Metrics such as:

```text
Equity Drawdown
Holding Duration
Trading Frequency
Basket Depth
Recovery Behaviour
```

are necessary to understand what the policy is actually doing.

### Observation 5 — Stress periods are valuable research evidence

The adverse EURUSD test provides information that favourable periods alone cannot provide.

---

## 14. What Cannot Yet Be Concluded?

This experiment does **not** establish that:

```text
one asset is superior,
one configuration is optimal,
the architecture generalises to unseen data,
the model will reproduce these returns,
or continued learning will always improve behaviour.
```

The tests use asset-specific configurations and learning remains active during the historical test.

Therefore:

> **Cross-asset behavioural differences are observable, but causal conclusions and out-of-sample generalisation have not yet been established.**

---

## 15. Main Experimental Limitation

The most important methodological limitation is:

```text
Training
and
Evaluation

occurred during the same historical run.
```

Because:

```text
TrainingMode = true
```

the model can adapt to the historical period while that period is being evaluated.

This makes the current tests useful for analysing:

```text
Learning Behaviour
Adaptive Trading Behaviour
Risk Development
and Cross-Asset Differences
```

but less suitable as evidence of out-of-sample predictive performance.

---

## 16. Next Experiment — Frozen Policy Evaluation

The next stage should separate learning from evaluation.

A proposed design is:

```text
PHASE 1 — TRAINING
Historical Period A
        ↓
TrainingMode = true
        ↓
DDQN + Memory Update
        ↓
Persist Learned State


PHASE 2 — EVALUATION
Later Unseen Period B
        ↓
Load Learned State
        ↓
TrainingMode = false
        ↓
No Neural / Memory Learning
        ↓
Measure Frozen Policy Behaviour
```

This would allow the research to ask a stronger question:

> **Does behaviour learned during one period remain useful when the policy is frozen and exposed to later unseen data?**

---

## 17. Future Cross-Asset Metrics

Later iterations of this experiment should add learning-specific measurements such as:

```text
Maximum Basket Depth

Average Basket Depth

Maximum Exposure

Average Recovery Time

Danger Activations

Replay Composition

Episode Reward

Action Distribution

Q-Value Distribution

HOLD / BUY / SELL Frequency

Regime Distribution

Number of Forced / Protective Exits
```

These metrics would make it possible to compare not only performance but also how the internal policy behaves across markets.

---

## 18. Potential Future Experiment — Normalised Risk

The current asset-specific lot sizing and contract specifications make direct P/L comparison difficult.

A future controlled experiment could normalise risk using:

```text
Same Initial Capital
+
Comparable Risk Budget
+
Comparable Maximum Exposure
+
Comparable Drawdown Limit
```

before comparing assets.

This would help isolate market behaviour from configuration-driven differences.

---

## 19. Potential Future Experiment — Same Architecture, Multiple Seeds

Reinforcement learning contains randomness through:

```text
Network Initialisation
Exploration
Replay Sampling
```

A stronger experiment would repeat each asset test with several random seeds.

Conceptually:

```text
EURUSD
Seed 1
Seed 2
Seed 3
Seed 4
Seed 5

        ↓

Compare Distribution
of Outcomes
```

This would help determine whether the observed result is stable or heavily dependent on one training path.

---

## 20. Research Interpretation

The first cross-asset experiment suggests that Adaptive-DDQN-MT5 can express substantially different trading behaviour across different market environments.

The most visible differences appear in:

```text
Trade Frequency
Holding Duration
Historical Return
and Equity Drawdown
```

However, the current evidence does not yet isolate whether those differences originate primarily from:

```text
Market Structure
Configuration
Timeframe
Learning Path
or their Interaction
```

For this reason, the current experiment should be treated as a **behavioural baseline**.

It establishes the starting point for more controlled experiments.

---

## 21. Conclusion

The purpose of this experiment was not to identify a winning asset.

It was to establish whether the current adaptive architecture exhibits measurably different behaviour across markets.

The answer from the current historical tests is clearly:

> **Yes — the system produced materially different activity, holding duration, drawdown and return profiles across EURUSD, SP500 and XAUUSD.**

The more important research task now is to understand **why** those differences emerge and whether learned behaviour remains useful outside the historical period in which it was trained.

The next major experiment will therefore focus on:

> **Training the model on one historical period and evaluating the frozen learned policy on a later unseen period.**

---

## Related Material

### Tested Assets

- [EURUSD — M5](../tested_assets/EURUSD_M5/)
- [SP500 — M15](../tested_assets/SP500_M15/)
- [XAUUSD — M1](../tested_assets/XAUUSD_M1/)
- [Tested Assets Summary](../tested_assets/README.md)

### Technical Documentation

- [System Architecture](../doc/architecture.md)
- [How the Agent Learns](../doc/learning-system.md)
- [Memory System](../doc/memory-system.md)
- [Research Scope & Limitations](../doc/limitations.md)
