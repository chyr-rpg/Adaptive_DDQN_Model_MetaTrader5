# Adaptive-DDQN-MT5

### A Memory-Augmented Reinforcement Learning Trading System for MetaTrader 5

Adaptive-DDQN-MT5 is a self-contained reinforcement-learning research project implemented natively in MQL5.

The project explores how a trading agent can learn from market experience, retain both successful and adverse trading episodes, recognise changing market regimes, and adapt future decisions through persistent memory.

> **Research focus:** studying adaptive decision-making and reinforcement learning inside the MetaTrader 5 environment with automatic trading system.

---

## Project Editions

Adaptive-DDQN-MT5 is maintained in two editions.

| Edition | Purpose |
| --- | --- |
| **Public Learning Edition** | A simplified but functional MQL5 DQN implementation that can be compiled, inspected, modified and studied. |
| **Private Research Edition** | The current full research system containing the latest DDQN, memory, replay, risk and adaptive decision architecture. |

The documentation in this repository describes the **broader current research architecture**, while the source file under `src/` provides an accessible implementation of its foundational reinforcement-learning concepts.

The public implementation should therefore be treated as a **learning edition**, rather than a line-for-line reproduction of the latest private research system.

---

## Why This Project?

Most Expert Advisors execute predefined trading rules.

This project explores a different question:

> **Can an MT5 trading system learn from what happened after its previous decisions and use those experiences to change future behaviour?**

The current research architecture combines a Double Dueling Deep Q-Network with persistent experience memory, regime-aware learning, risk-sensitive rewards and historical danger recognition.

The objective is not simply to predict the next market move.

The broader research problem is to study whether an agent can learn:

- when a market state is favourable,
- when an apparently profitable behaviour carries excessive risk,
- when current conditions resemble historically dangerous episodes,
- how its own basket exposure should influence the next decision,
- and how accumulated experience should alter future action preferences.

---

## Why I Shared This Project

This project started as a personal experiment in applying reinforcement learning and neural networks to trading decisions inside MetaTrader 5.

I previously ran the system on a demo account across different markets and trading frequencies, but that historical record was lost when the demo-server environment became unavailable. Because of that, I do not use those earlier results as evidence of performance here.

I decided to share the project mainly because I am interested in a broader question:

> **How can neural networks contribute meaningfully to trading decisions, rather than simply being added on top of a conventional strategy?**

The current implementation is still heavily influenced by a **grid / basket-based execution framework**, so its use of reinforcement learning remains constrained by that structure.

I therefore see this repository as an ongoing research project rather than a finished trading system, and I am particularly interested in alternative ideas for state design, reward functions, action spaces, memory systems and ways of combining neural models with conventional risk controls.

---
## Public vs. Research Edition

The source code published in this repository is a **Learning Edition** designed
to demonstrate the core reinforcement-learning workflow in a readable and
reproducible form.

My current private research implementation is substantially more complex and
includes additional components such as:

- Double DQN and target-network learning
- dueling value / advantage architecture
- multiple feature encoders
- regime-aware model selection
- specialized replay and memory systems
- drawdown and danger-state memory
- additional basket and risk-management logic
- model persistence and decision-support mechanisms

The public version is therefore **not a line-for-line release of the latest
research system**.

The full implementation remains private while the architecture continues to
evolve. Selected access may be considered for academic, technical or
professional collaboration.

---

## Core Research Concepts

### 🧠 Native Neural Network

The neural network, forward inference, backpropagation and reinforcement-learning logic are implemented directly in MQL5 without requiring an external Python or machine-learning runtime.

This allows the learning system to operate directly inside MetaTrader 5 and the Strategy Tester environment.

### 🔀 Multi-Branch State Encoding

In the current research architecture, different groups of information are processed separately before being combined into a shared neural representation.

These feature groups include:

- basket and exposure state,
- technical indicator state,
- volatility state,
- market structure,
- and supply/demand and candle context.

This allows fundamentally different forms of information to develop their own internal representations before being fused by the neural policy.

### ♻️ Experience Replay

The agent learns from previous state-action-outcome transitions instead of relying only on the most recent observation.

The current research system extends standard replay with specialised memory banks for different classes of trading experience.

### 🗃️ Memory-Augmented Learning

The broader architecture maintains multiple forms of trading memory, including:

- recent experience,
- dangerous episodes,
- deep-basket sequences,
- efficient periods,
- episode memory,
- pattern memory,
- regime-event memory,
- drawdown-event memory,
- and persistent Q-state memory.

The intention is to combine:

```text
Neural Generalisation
        +
Historical Recall
```

rather than relying on neural-network weights alone.

### 🌡️ Regime-Aware Learning

Different learning behaviour can be maintained for different volatility regimes.

This allows market context to influence both inference and training, rather than assuming that one policy behaves identically under all volatility conditions.

### 🛡️ Risk-Aware Decision Support

Raw neural-network outputs are not treated as the entire decision process.

The current research architecture supplements learned action values with historical memory, drawdown context, basket state and adaptive risk mechanisms before reaching the execution layer.

---

## System Architecture

At a high level, the current research system follows:
<img width="1536" height="1024" alt="image" src="https://github.com/user-attachments/assets/41c7bd77-a75b-496d-85b6-b56dce50be7b" />

```text
Market Environment
        │
        ▼
State Construction
        │
        ├── Basket / Exposure
        ├── Indicators
        ├── Volatility
        ├── Market Structure
        └── Zone / Candle Context
        │
        ▼
Feature Branch Encoders
        │
        ▼
Shared Neural Representation
        │
        ▼
Double Dueling DQN
        │
        ▼
Q(HOLD) / Q(BUY) / Q(SELL)
        │
        ▼
Memory & Decision Support
        │
        ▼
Risk / Execution Layer
        │
        ▼
Trading Outcome
        │
        ▼
Reward + Experience Replay
        │
        └──────────────► Learning
```

A more detailed explanation is available in:

[**System Architecture →**](doc/architecture.md)

---

## Public Learning Edition

A simplified implementation is available in:

[`src/AdaptiveDQN_MT5_LearningEdition.mq5`](src/AdaptiveDQN_MT5_LearningEdition.mq5)

The purpose of this version is to provide a practical and readable example of how reinforcement learning can be implemented directly inside MQL5.

The public learning edition includes:

- a native MQL5 neural network,
- two hidden layers,
- HOLD / BUY / SELL action selection,
- epsilon-greedy exploration,
- multi-symbol support,
- market and basket state construction,
- volatility features,
- higher-timeframe context,
- supply/demand-zone features,
- basic risk-aware reward shaping,
- adaptive grid and basket logic,
- and persistent neural-network save/load.

It is intentionally smaller than the current private research edition.

The goal is to make the foundational learning process accessible without exposing the complete proprietary research architecture.

---

## Private Research Edition

The current full implementation is maintained separately in a private repository.

The private research edition contains later-generation components including:

```text
Double DQN
Dueling DQN
Feature-specific branch encoders
Regime-specific model banks
Online and target networks

Main experience replay
Recent replay
Danger replay
Deep-basket replay
Efficient replay

Persistent Q-memory
Episode memory
Pattern memory
Regime-event memory
Drawdown-event memory

Danger Brain
Delayed transition learning
Dense basket-health feedback
Risk-aware reward engineering
Unified decision support
Adaptive basket-risk controls
```

These components represent ongoing research and proprietary trading-system development and are therefore not distributed publicly.

Selected access may be considered for legitimate:

- academic research,
- technical review,
- professional evaluation,
- or research collaboration.

Access is provided at the author's discretion.

---

## How the Agent Learns

The reinforcement-learning process can be simplified as:

```text
Observe State
      ↓
Estimate Q-Values
      ↓
Select Action
      ↓
Observe Consequence
      ↓
Calculate Reward
      ↓
Store Experience
      ↓
Replay Historical Experience
      ↓
Update Neural Network
      ↓
Repeat
```

The complete research architecture extends this process with delayed outcomes, specialised replay memory, regime-specific learning and risk-aware historical context.

Read more:

[**How the Agent Learns →**](doc/learning-system.md)


---

## Public Source vs Research Documentation

An important distinction in this repository is:

```text
Public Source Code
        ↓
Foundational implementation
of the learning system

Research Documentation
        ↓
Current broader architecture
under active development
```

Some components described in the architecture and learning documentation are therefore **not present in the public source file**.

This separation is intentional.

The public source is designed to show how the core reinforcement-learning workflow can be implemented in MQL5, while the documentation records the direction and structure of the more advanced private research system.

---

## Explore the Project

| Section | Description |
| --- | --- |
| [System Architecture](doc/architecture.md) | Full architectural overview of the current research system |
| [How the Agent Learns](doc/learning-system.md) | Reinforcement-learning and DDQN learning process |
| `doc/memory-system.md` | Memory and specialised replay architecture *(in progress)* |
| `doc/limitations.md` | Research scope, limitations and risk considerations *(planned)* |
| [Public Learning Source](src/AdaptiveDQN_MT5_LearningEdition.mq5) | Simplified functional MQL5 implementation |

---

## Research Direction

The project is intended to investigate questions such as:

> How does policy behaviour change as trading experience accumulates?

> Can the agent distinguish efficient profitable trades from profitable but high-risk recovery sequences?

> Can specialised replay memory help the system learn from rare drawdown events?

> Do different volatility regimes produce meaningfully different learned policies?

> Can historical memory reduce repetition of previously harmful basket sequences?

> How should the agent's own exposure affect its interpretation of the same market environment?

Future repository experiments will focus increasingly on **policy evolution, reward behaviour, replay composition, drawdown response and learning diagnostics**, rather than presenting backtest return alone.

---

## Strategy Tester Research

One advantage of implementing the learning architecture directly in MQL5 is that the agent can be studied inside MetaTrader 5 Strategy Tester.

Future visual diagnostics will expose variables such as:

```text
Current Regime

Q(HOLD)
Q(BUY)
Q(SELL)

Selected Action

Exploration Rate

Episode Reward

Danger State

Basket Depth

Replay Memory

Memory Confidence
```

The intention is to make not only the resulting trades visible, but also the evolution of the agent's internal decision process.

---

## Research Scope & Limitations

Adaptive-DDQN-MT5 is an experimental reinforcement-learning project.

It should not be interpreted as evidence that reinforcement learning removes trading risk or guarantees future profitability.

The system contains basket and averaging behaviour, meaning exposure can increase during adverse market movement.

Learning and historical memory may influence this behaviour but cannot eliminate:

```text
Market Risk
Model Risk
Execution Risk
Liquidity Risk
Regime-Change Risk
Tail-Event Risk
```

Backtesting and historical learning results do not guarantee future performance.

The current research system is better suited to **controlled research, Strategy Tester experimentation and supervised trading-system development** than fully unattended operation.

---

## Source Availability

The public learning edition is distributed for technical study and experimentation.

The complete current research implementation is maintained privately.

Copyright is retained by the author.

No permission is granted to redistribute, sublicense, sell or incorporate proprietary private implementation components into another product without explicit authorisation.

---

## Disclaimer

This repository is provided for research, educational and technical experimentation purposes.

Nothing in this repository constitutes investment advice, a recommendation to trade, or a representation of future trading performance.

Any live-market experimentation should be conducted with appropriate independent risk controls and supervision.

---

## Author

**CYR**

Research interests:

`Reinforcement Learning` · `Algorithmic Trading` · `Quantitative Finance` · `MQL5` · `Adaptive Systems`
