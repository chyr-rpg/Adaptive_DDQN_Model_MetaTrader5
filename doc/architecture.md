# System Architecture

Adaptive-DDQN-MT5 is a self-contained reinforcement-learning trading
system implemented natively in MQL5.

Although the implementation is contained within a single Expert Advisor,
the internal design is organised as a set of interacting subsystems:

**Perception → Neural Policy → Memory → Decision Support → Risk & Execution → Learning**

The objective is not simply to predict the next price movement. The agent
also attempts to understand its own exposure, remember previous trading
outcomes, distinguish different market regimes, and adapt future decisions
from accumulated experience.

---

## 1. High-Level Architecture

```text
                         MARKET ENVIRONMENT
                                │
                                ▼
                       STATE CONSTRUCTION
                                │
          ┌─────────────┬───────┼───────┬─────────────┐
          ▼             ▼       ▼       ▼             ▼
       Basket        Indicator  Vol.  Structure    Zone/Candle
       State          State     State    State        State
          │             │       │       │             │
          └─────────────┴───────┴───────┴─────────────┘
                                │
                                ▼
                      BRANCH ENCODER NETWORKS
                                │
                                ▼
                         FEATURE FUSION
                                │
                                ▼
                     DOUBLE DUELING DQN
                          ┌─────┴─────┐
                          ▼           ▼
                     State Value   Advantage
                          └─────┬─────┘
                                ▼
                    Q(HOLD) / Q(BUY) / Q(SELL)
                                │
                                ▼
                       DECISION SUPPORT
                 ┌──────────────┼──────────────┐
                 ▼              ▼              ▼
             Q-Memory      Experience      Danger /
                            Archives       DD Memory
                 └──────────────┼──────────────┘
                                ▼
                       RISK & ACTION GATES
                                │
                                ▼
                        TRADE EXECUTION
                                │
                                ▼
                      MARKET CONSEQUENCE
                                │
                                ▼
                     REWARD / TRANSITION
                                │
                    ┌───────────┴───────────┐
                    ▼                       ▼
              EXPERIENCE REPLAY       LONG-TERM MEMORY
                    │                       │
                    └───────────┬───────────┘
                                ▼
                          NETWORK UPDATE
                                │
                                └──────────────► NEXT DECISION
