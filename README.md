# Adaptive_DDQN_MetaTrader5
Memory-augmented reinforcement learning trading system built natively in MQL5 for MetaTrader 5 research and experimentation.
# Adaptive-DDQN-MT5

### A Memory-Augmented Reinforcement Learning Trading System for MetaTrader 5

Adaptive-DDQN-MT5 is a self-contained reinforcement-learning research
project implemented natively in MQL5.

The project explores how a trading agent can learn from market experience,
retain both successful and adverse trading episodes, recognise changing
market regimes, and adapt future decisions through persistent memory.

> **Research focus:** studying adaptive decision-making and reinforcement
> learning inside the MetaTrader 5 environment rather than building a
> fully unattended 24/5 trading robot.

---

## Why This Project?

Most Expert Advisors execute predefined trading rules.

This project explores a different question:

> **Can an MT5 trading system learn from what happened after its previous
> decisions and use those experiences to change future behaviour?**

The system combines a Double Dueling Deep Q-Network with persistent
experience memory, regime-aware learning, risk-sensitive rewards and
historical danger recognition.

---

## Core Research Concepts

### 🧠 Native Neural Network

The neural network, forward inference, backpropagation and reinforcement
learning logic are implemented directly in MQL5 without requiring an
external Python or machine-learning runtime.

### 🔀 Multi-Branch State Encoding

Different groups of information are processed separately before being
combined into a shared representation:

- Basket and exposure state
- Technical indicator state
- Volatility state
- Market structure
- Supply/demand and candle context

### ♻️ Experience Replay

The agent learns from previous state-action-outcome transitions instead
of relying only on the most recent observation.

### 🗃️ Memory-Augmented Learning

The architecture maintains multiple forms of trading memory, including:

- Recent experience
- Dangerous episodes
- Deep-basket sequences
- Efficient periods
- Episode memory
- Pattern memory
- Regime-event memory
- Drawdown-event memory
- Persistent Q-state memory

### 🌡️ Regime-Aware Learning

Separate learning behaviour can be maintained for different volatility
regimes, allowing market context to influence both inference and training.

### 🛡️ Risk-Aware Decision Support

Neural-network outputs are supplemented by historical memory, drawdown
context and adaptive risk mechanisms before reaching the execution layer.

---

## System Architecture

Architecture diagram coming next.

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
Q(Hold) / Q(Buy) / Q(Sell)
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
