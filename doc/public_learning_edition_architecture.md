---

## Public Learning Edition

A simplified but functional implementation of the project is available in:

[`src/AdaptiveDQN_MT5_LearningEdition.mq5`](src/AdaptiveDQN_MT5_LearningEdition.mq5)

The purpose of the public source is to provide a practical implementation that readers can compile, inspect, modify and study directly inside MetaTrader 5.

The Learning Edition includes:

- a neural network implemented natively in MQL5,
- two hidden layers,
- HOLD / BUY / SELL action selection,
- epsilon-greedy exploration,
- multi-symbol operation,
- market and basket state construction,
- volatility and higher-timeframe features,
- supply/demand-zone context,
- basic risk-aware reward shaping,
- adaptive basket and grid management,
- and persistent neural-network save/load.

The public implementation is intentionally simpler than the current research system.

Its purpose is to expose the **foundational reinforcement-learning workflow** without publishing the complete proprietary architecture used in the latest private research edition.

---

## Private Research Edition

The current full implementation of Adaptive-DDQN-MT5 is maintained separately in a private repository.

The private research edition extends the public learning system with later-generation components including:

```text
Double DQN
Dueling DQN
Feature-specific branch encoders
Shared feature fusion
Regime-specific model banks
Online and target networks

Main Experience Replay
Recent Replay
Danger Replay
Deep-Basket Replay
Efficient Replay

Persistent Q-Memory
Episode Memory
Pattern Memory
Regime-Event Memory
Drawdown-Event Memory

Danger Brain
Delayed Transition Learning
Dense Basket-Health Feedback
Risk-Aware Reward Engineering
Unified Decision Support
Adaptive Basket-Risk Controls
```

These components represent ongoing research and proprietary trading-system development and are therefore not distributed publicly.

Selected access to the full implementation may be considered for legitimate:

- academic research,
- technical review,
- professional evaluation,
- or research collaboration.

Access is provided at the author's discretion.

---

## How the Agent Learns

At its core, the learning process can be simplified as:

```text
Observe Current State
        ↓
Estimate Q-Values
        ↓
Select an Action
        ↓
Observe the Consequence
        ↓
Calculate Reward
        ↓
Store Experience
        ↓
Replay Historical Experience
        ↓
Update the Neural Network
        ↓
Repeat
```

The current research architecture extends this basic process with:

- delayed outcome learning,
- specialised replay memory,
- regime-aware models,
- historical similarity memory,
- risk-sensitive rewards,
- and adaptive decision support.

A more detailed explanation is available here:

[**How the Agent Learns →**](doc/learning-system.md)

---

## Public Source vs Research Documentation

An important distinction in this repository is:

```text
PUBLIC SOURCE
     │
     ▼
Simplified functional implementation
of the foundational learning system


RESEARCH DOCUMENTATION
     │
     ▼
Current broader architecture
under active development
```

Some advanced components described in the architecture and learning documentation are therefore **not included in the public source file**.

This separation is intentional.

The public implementation is designed to demonstrate how a neural reinforcement-learning trading system can be constructed directly in MQL5.

The documentation, meanwhile, records the structure and research direction of the more advanced private system.

---

## Explore the Project

| Section | Description |
| --- | --- |
| [System Architecture](doc/architecture.md) | Technical overview of the current research architecture |
| [How the Agent Learns](doc/learning-system.md) | Reinforcement-learning workflow and training process |
| `doc/memory-system.md` | Memory and specialised replay architecture *(in progress)* |
| `doc/limitations.md` | Research scope, limitations and risk considerations *(planned)* |
| [Public Learning Edition](src/AdaptiveDQN_MT5_LearningEdition.mq5) | Simplified functional MQL5 implementation |

---

## Research Questions

Adaptive-DDQN-MT5 is intended to explore more than whether a historical backtest produces a positive return.

The project investigates questions such as:

> **How does policy behaviour change as trading experience accumulates?**

> **Can an agent distinguish an efficient profitable trade from a profitable recovery that required excessive drawdown and exposure?**

> **Can specialised replay memory improve learning from rare but important adverse events?**

> **Do different volatility regimes produce meaningfully different learned policies?**

> **Can historical memory reduce repetition of previously harmful basket sequences?**

> **How should the agent's existing exposure influence its interpretation of the same market environment?**

> **Can memory-based decision support complement a neural policy without replacing it?**

Future experiments will increasingly focus on **policy evolution, reward behaviour, replay composition, drawdown response and learning stability**, rather than presenting backtest return alone.

---

## Strategy Tester Research

One advantage of implementing the learning system directly in MQL5 is that the agent can be studied inside the MetaTrader 5 Strategy Tester.

The objective is not only to observe:

```text
where the EA buys
where the EA sells
whether a trade makes money
```

but also to observe:

```text
what the agent currently believes
how that belief changes
what it remembers
and how experience affects later decisions
```

Future visual diagnostics are intended to display variables such as:

```text
Current Market Regime

Q(HOLD)
Q(BUY)
Q(SELL)

Selected Action

Exploration Rate

Current / Cumulative Reward

Episode Count

Danger State

Basket Depth

Replay Memory Size

Memory Confidence
```

This will allow Strategy Tester demonstrations to show the **learning process itself**, rather than only the resulting trade history.

---

## Public Learning Workflow

The public Learning Edition can be used as a starting point for studying reinforcement learning inside MetaTrader 5.

A simplified workflow is:

```text
1. Open the source file in MetaEditor
        ↓
2. Compile the Expert Advisor
        ↓
3. Open MetaTrader 5 Strategy Tester
        ↓
4. Select a market and testing period
        ↓
5. Enable Training Mode
        ↓
6. Run the historical simulation
        ↓
7. Observe exploration and trading behaviour
        ↓
8. Allow the neural model to accumulate experience
        ↓
9. Save the resulting network state
        ↓
10. Continue experimentation with different settings
```

The public implementation is intended primarily as a **learning and experimentation baseline**.

It should not be interpreted as the recommended configuration or implementation of the private research system.

---

## What the Public Version Demonstrates

The Learning Edition provides a relatively compact example of several important concepts.

### Neural Network in Native MQL5

Forward inference and neural-network weight updates are implemented directly inside the Expert Advisor.

No external Python process or machine-learning runtime is required.

### State Construction

The agent receives a numerical representation of its environment containing information derived from areas such as:

```text
Momentum
Trend
Volatility
Current Positions
Price vs Average Entry
Virtual Equity
Drawdown
Supply / Demand Zones
Short-Term Returns
Longer-Term Returns
Higher-Timeframe Context
```

### Reinforcement-Learning Action Space

The network estimates values for:

```text
HOLD
BUY
SELL
```

The highest-valued action can be selected, subject to exploration during training.

### Epsilon-Greedy Exploration

During training, the agent sometimes explores actions other than its current highest-valued choice.

As training progresses, the exploration rate can gradually decline.

### Reward Feedback

Trading outcomes and changes in virtual equity provide reinforcement signals that can modify neural-network behaviour.

### Persistent Learning

The learned neural-network state can be written to disk and loaded again during later sessions.

This makes it possible to experiment with continued learning across multiple Strategy Tester or MT5 sessions.

---

## Public and Private Architecture Relationship

The two editions are conceptually related but should not be interpreted as identical implementations.

```text
Public Learning Edition
        │
        │ establishes the foundation
        ▼
Native MQL5 DQN
State Construction
Reward Feedback
Exploration
Persistence
Multi-Asset Support
        │
        │ extended through ongoing research
        ▼
Private Research Edition
        │
        ├── Double DQN
        ├── Dueling Architecture
        ├── Branch Encoders
        ├── Experience Replay
        ├── Specialised Replay Banks
        ├── Historical Memory
        ├── Danger Brain
        ├── Delayed Outcomes
        ├── Advanced Reward Engineering
        └── Adaptive Risk Decision Support
```

The public version therefore provides the **conceptual and implementation foundation**, while the private edition represents the current research frontier of the project.

---

## Repository Structure

```text
Adaptive-DDQN-MT5/
│
├── README.md
├── .gitignore
│
├── src/
│   └── AdaptiveDQN_MT5_LearningEdition.mq5
│
├── doc/
│   ├── architecture.md
│   ├── learning-system.md
│   ├── memory-system.md
│   └── limitations.md
│
├── assets/
│   ├── system-architecture.png
│   ├── memory-architecture.png
│   ├── tester-dashboard.png
│   └── strategy-tester-learning.gif
│
├── presets/
│   └── visual-training-demo.set
│
└── experiments/
    └── 01-policy-evolution.md
```

Some files shown above represent planned additions and will be added as the research documentation and experimental framework develop.

---

## Current Project Status

The project is under active research and development.

### Completed

```text
✓ Native MQL5 neural-learning foundation
✓ Multi-asset DQN Learning Edition
✓ Advanced private DDQN research implementation
✓ System architecture documentation
✓ Reinforcement-learning documentation
✓ System architecture visualisation
```

### In Progress

```text
→ Memory-system documentation
→ Strategy Tester learning dashboard
→ Visual learning diagnostics
→ Public example presets
```

### Planned

```text
○ Memory architecture diagram
○ Strategy Tester demonstration GIF
○ Policy-evolution experiments
○ Replay-memory experiments
○ Reward-engineering experiments
○ Regime-behaviour analysis
```

---

## Research Scope & Limitations

Adaptive-DDQN-MT5 is an experimental reinforcement-learning research project.

It should not be interpreted as evidence that reinforcement learning removes trading risk or guarantees profitable future behaviour.

The underlying trading architecture contains basket and averaging mechanisms, meaning exposure can increase when prices move adversely.

Reinforcement learning, historical memory and adaptive risk controls can influence this behaviour but cannot eliminate:

```text
Market Risk
Model Risk
Execution Risk
Liquidity Risk
Regime-Change Risk
Tail-Event Risk
```

A policy learned from historical data may also behave differently when market structure changes.

For this reason, evaluation should consider more than final return.

Relevant research metrics include:

```text
Maximum Drawdown
Basket Depth
Exposure
Recovery Duration
Reward Quality
Policy Stability
Action Distribution
Regime Behaviour
Replay Composition
and Trading Performance
```

Backtesting and historical learning results do not guarantee future performance.

The current research system is primarily intended for:

- reinforcement-learning research,
- Strategy Tester experimentation,
- MQL5 machine-learning study,
- controlled technical evaluation,
- and supervised trading-system development.

---

## Source Availability

Adaptive-DDQN-MT5 uses a two-level source model.

### Public Source

The Learning Edition is available directly in this repository:

[`src/AdaptiveDQN_MT5_LearningEdition.mq5`](src/AdaptiveDQN_MT5_LearningEdition.mq5)

It is provided to support technical learning, experimentation and study of neural reinforcement learning in MQL5.

### Full Research Source

The current full research implementation is maintained in a separate private repository.

The private source contains proprietary implementation details and ongoing research that are not included in the public Learning Edition.

Access to the private implementation is not automatically granted through access to this repository.

Selected access may be considered for legitimate research, technical review or collaboration.

---

## Ownership

Copyright © 2025–2026 Chen Yurui.

All rights reserved unless otherwise stated.

The public Learning Edition is made available for technical study and research experimentation.

The private research implementation remains proprietary.

No permission is granted to redistribute, sublicense, sell, publish or incorporate proprietary private implementation components into another commercial product without explicit authorisation.

---

## Disclaimer

This repository is provided for research, educational and technical experimentation purposes.

Nothing contained in this repository constitutes:

- investment advice,
- a recommendation to trade,
- a solicitation to enter a financial transaction,
- or a representation of future trading performance.

Any live-market experimentation should use appropriate independent risk controls and human supervision.

---

## Author

**Chen Yurui**

Research interests:

`Reinforcement Learning` · `Algorithmic Trading` · `Quantitative Finance` · `MQL5` · `Adaptive Systems`
