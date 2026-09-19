# XAUUSD M5 — Public DQN Learning Edition

This experiment demonstrates the publicly available
`AdaptiveDQN_MT5_LearningEdition.mq5` operating on XAUUSD.

Unlike the broader tested-asset results generated with the private
Adaptive-DDQN research architecture, this test uses the simplified public
DQN implementation available under `src/`.

## Purpose

The primary purpose of this run is to verify that the public Learning Edition
can:

- construct market and basket states,
- perform neural-network inference,
- use epsilon-greedy action selection,
- update the network during training,
- generate trading decisions,
- accumulate episode rewards,
- and persist learned DQN state between sessions.

This test should therefore be interpreted primarily as a **functional
reinforcement-learning demonstration**, rather than as performance validation.

## Test Modelling

**MetaTrader 5 modelling mode:** 1 minute OHLC

This mode uses M1 OHLC information rather than the complete historical tick
sequence.

As a result, the experiment is suitable for observing broad learning and
trading behaviour, but is not intended to provide high-precision estimates of:

- intrabar execution,
- exact grid-add timing,
- maximum short-lived drawdown,
- TP/SL sequencing,
- slippage,
- or live-market performance.

A higher-fidelity real-tick test will be used for later performance-oriented
evaluation.

## Adaptive Learning
In the performance png, you can observe that the system is adapting to xauusd price pattern in the second half part of backtesting period, though market condition changes may also be invovled
