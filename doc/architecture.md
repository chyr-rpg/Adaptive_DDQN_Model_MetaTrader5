# System Architecture

Adaptive-DDQN-MT5 is a self-contained reinforcement-learning trading system implemented natively in MQL5 for MetaTrader 5.

Although the implementation is contained within a single Expert Advisor, the internal design is organised as a collection of interacting subsystems:

**Perception → Neural Policy → Memory → Decision Support → Risk & Execution → Learning**

The objective is not simply to predict the next price movement. The agent also attempts to understand its own exposure, retain information from previous trading outcomes, recognise changing market regimes, and adapt future decisions as experience accumulates.

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
             Q-Memory       Historical      Danger /
                              Memory        DD Memory
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
```

---

## 2. Design Philosophy

A conventional Expert Advisor often follows a relatively fixed process:

```text
Indicator condition
        ↓
Trading rule
        ↓
Order
```

Adaptive-DDQN-MT5 is designed around a different sequence of questions:

```text
What is happening now?
        ↓
What happened in similar situations before?
        ↓
What does the learned policy currently prefer?
        ↓
Is that decision acceptable from a risk perspective?
```

The neural network is therefore not the entire trading system.

It is the central policy estimator inside a larger architecture containing:

- market-state construction,
- multi-timeframe feature processing,
- historical memory,
- reinforcement-learning feedback,
- risk controls,
- basket management,
- and trade execution.

The project is better understood as an **adaptive decision architecture** rather than simply an EA containing a neural network.

---

## 3. Perception Layer

The perception layer converts raw market, account, and position information into a numerical state representation that can be processed by the reinforcement-learning system.

The state is divided into several conceptual feature groups.

### 3.1 Basket and Self-Awareness State

The agent observes not only the market but also its own trading condition.

Relevant information can include:

- current exposure,
- number of open positions,
- basket direction,
- average entry price,
- distance from relevant reference prices,
- basket age,
- drawdown,
- virtual budget utilisation,
- current position sequence,
- and the state of an existing recovery process.

This is important because two identical market environments may require very different actions depending on whether the system is:

- flat,
- holding a single position,
- or managing a deeper multi-position basket.

The agent therefore treats its own exposure as part of the environment.

---

### 3.2 Indicator State

Technical indicators provide compact representations of momentum, trend, and relative market behaviour.

The implementation contains features derived from indicators such as:

```text
RSI
CCI
MACD
EMA
RVI
ATR
```

Indicator information can be constructed from the base timeframe and optional higher-timeframe contexts.

These features are not intended to operate only as fixed entry conditions. They become inputs to the learning system.

---

### 3.3 Volatility State

Volatility is represented as its own feature family rather than being left implicit inside directional indicators.

The system compares shorter- and longer-horizon volatility behaviour using ATR-related measures and other volatility context.

Volatility information contributes to both:

```text
state representation
        +
market regime classification
```

This allows a similar price pattern to be interpreted differently depending on whether the market is:

- relatively calm,
- moderately active,
- or experiencing strong volatility expansion.

---

### 3.4 Market Structure State

The system also attempts to describe structural characteristics of price rather than relying only on conventional technical indicators.

Structural information includes concepts such as:

- swing highs and lows,
- local trend tendency,
- distance to recent structure,
- structural amplitude,
- displacement,
- compression and expansion,
- continuation behaviour,
- and reversal behaviour.

Multiple time horizons can be included:

```text
Execution timeframe
Medium timeframe
Long timeframe
Optional structural timeframe
```

The purpose is to expose the neural policy to both immediate price behaviour and broader market context.

---

### 3.5 Supply/Demand and Candle Context

Supply/demand zones and local candle behaviour form another feature family.

The system can track information related to:

- distance to demand,
- distance to supply,
- whether price is inside or near a zone,
- zone relevance,
- breakout state,
- retest behaviour,
- rejection,
- acceptance,
- indecision,
- and impulse characteristics.

This allows price behaviour around structural areas to become part of the learning state rather than functioning only as deterministic trading rules.

---

### 3.6 Multi-Timeframe Context

The architecture can process market information across several time horizons.

Conceptually:

```text
Execution TF
    │
    ├── immediate price behaviour
    │
Medium TF
    │
    ├── local market context
    │
Long TF
    │
    ├── broader directional structure
    │
Structural TF
    │
    └── optional higher-order context
```

The goal is to prevent the agent from making decisions based exclusively on one local timeframe.

---

## 4. Neural Network Architecture

The complete state vector is not passed directly through one undifferentiated dense network.

Adaptive-DDQN-MT5 uses **feature-specific branch encoders**.

Conceptually:

```text
Basket State ───────► Basket Encoder ────────┐
Indicator State ────► Indicator Encoder ─────┤
Volatility State ───► Volatility Encoder ────┤
Structure State ────► Structure Encoder ─────┼──► Fusion
Zone/Candle State ──► Zone Encoder ──────────┘
```

Each enabled branch can pass through two dense encoding layers before being combined with the other feature groups.

This allows different forms of information to develop their own internal representations before interacting inside the shared network.

---

### 4.1 Branch Encoders

Each branch receives only the section of the state vector relevant to that feature family.

A simplified encoder structure is:

```text
Raw branch features
        ↓
Dense Layer 1
        ↓
SiLU activation
        ↓
Dense Layer 2
        ↓
SiLU activation
        ↓
Encoded representation
```

The encoder dimensions are determined dynamically within configured minimum and maximum widths.

This keeps the architecture flexible as the number of active features changes.

---

### 4.2 Feature Fusion

Outputs from the enabled branch encoders are concatenated into a common latent representation.

Conceptually:

```text
Encoded Basket
Encoded Indicators
Encoded Volatility
Encoded Structure
Encoded Zones
        │
        ▼
   Concatenation
        │
        ▼
 Shared Fusion Layer
        │
        ▼
 Deeper Shared Layer
        │
        ▼
   Dueling DQN Heads
```

The fused representation becomes the shared internal description of the current environment.

---

## 5. Dueling Q-Network

Instead of estimating one independent Q-value for each action directly from a single output layer, the architecture separates:

```text
State Value V(s)

and

Action Advantage A(s,a)
```

These components are recombined to estimate the final action values.

Conceptually:

```text
Shared Representation
        │
   ┌────┴────┐
   ▼         ▼
 Value     Advantage
 V(s)      A(s,a)
   │         │
   └────┬────┘
        ▼
     Q(s,a)
```

The current action space contains three actions:

```text
0 = HOLD
1 = BUY
2 = SELL
```

The Dueling architecture is useful when the quality of the current market state itself matters even when multiple actions have similar expected values.

---

## 6. Double DQN

The learning system maintains both:

```text
Online Network
Target Network
```

The online network represents the actively updated policy.

The target network provides a more stable reference when constructing reinforcement-learning targets.

With Double-DQN behaviour enabled, action selection and target evaluation are separated to reduce the tendency of standard Q-learning to systematically overestimate action values.

The implementation also includes stability mechanisms such as:

- Huber loss,
- gradient clipping,
- target-network synchronisation,
- soft target updates,
- and configurable replay warm-up.

These mechanisms are intended to improve training stability within a fully MQL5-native implementation.

---

## 7. Regime-Aware Model Bank

The system can maintain different DQN models for different volatility regimes.

The current architecture supports three regime states.

Conceptually:

```text
             Current Market Context
                     │
               Regime Detection
                     │
        ┌────────────┼────────────┐
        ▼            ▼            ▼
    Low Regime   Mid Regime   High Regime
       DQN          DQN           DQN
```

This reflects the idea that a policy learned under low-volatility conditions may not behave identically under a highly volatile market environment.

Soft regime inference can also be used to reduce abrupt changes when the market is close to a regime boundary.

---

## 8. Reinforcement-Learning Loop

The system learns through state-action-outcome transitions.

At a simplified level:

```text
State S(t)
    │
    ▼
Estimate Q-values
    │
    ▼
Select action A(t)
    │
    ▼
Trading consequence
    │
    ▼
Observe S(t+1)
    │
    ▼
Calculate reward R(t)
    │
    ▼
Store transition
[S(t), A(t), R(t), S(t+1)]
    │
    ▼
Experience replay
    │
    ▼
Network update
```

The important distinction is that the learning signal depends on what happened **after** an action rather than only on the market condition that caused the action.

---

### 8.1 Exploration

During training, the system can use an exploration rate to occasionally take actions other than the currently highest-valued action.

Conceptually:

```text
Early training
high exploration
        ↓
More accumulated experience
        ↓
Lower exploration
        ↓
Increasing reliance on learned policy
```

The exploration rate can decay over time toward a configured minimum.

---

## 9. Delayed Outcome Learning

Trading actions cannot always be evaluated immediately.

An entry may initially look successful but later result in:

- multiple basket additions,
- prolonged drawdown,
- poor recovery efficiency,
- or a forced protective exit.

The architecture therefore supports **pending transitions**.

A decision can retain its original state and action until more meaningful outcome information becomes available.

Later events such as:

```text
position close
basket close
drawdown development
recovery quality
trade duration
```

can therefore contribute to the learning signal associated with the earlier action.

---

## 10. Reward Architecture

The reward system is deliberately risk-aware.

The objective is not simply:

```text
profit = good
loss   = bad
```

A profitable basket achieved through severe drawdown and repeated averaging should not necessarily receive the same learning signal as an efficient low-risk trade.

Conceptually, reward can include:

### Positive components

```text
+ realised profit
+ efficient recovery
+ profit relative to maximum drawdown
+ clean one-round completion
+ historically efficient behaviour
+ favourable basket-health progress
```

### Negative components

```text
- repeated basket additions
- excessive basket depth
- drawdown
- margin stress
- prolonged basket age
- reversal risk
- historically dangerous patterns
- inefficient recovery
- risky profitable behaviour
```

This creates an important distinction between:

> **profitability**

and

> **quality of profitability**

---

## 11. Dense Basket Feedback

Learning does not need to wait until a complete basket has closed.

While a basket remains open, the system can generate intermediate learning feedback from changes in:

- open P/L,
- drawdown,
- number of positions,
- basket age,
- recovery progress,
- and exposure development.

For example, if a newly added position significantly worsens basket drawdown without improving recovery progress, the system can receive negative intermediate feedback before the full basket eventually closes.

This creates a denser learning signal than relying exclusively on terminal outcomes.

---

## 12. Memory Architecture

One of the defining characteristics of Adaptive-DDQN-MT5 is that learning is not stored only inside neural-network weights.

The system maintains several additional forms of memory.

```text
                         EXPERIENCE
                              │
          ┌───────────────────┼───────────────────┐
          ▼                   ▼                   ▼
       Replay             Historical          Similarity
       Memory               Archive             Memory
          │                   │                   │
   ┌──────┼──────┐       ┌────┼────┐             ▼
   ▼      ▼      ▼       ▼    ▼    ▼          Q-Memory
Recent  Danger  Deep   Episode Pattern Regime
              Basket
          │
          ▼
      Efficient
       Replay
```

These memory systems serve different purposes and should not be interpreted as interchangeable.

---

## 13. Main Experience Replay

Standard replay memory stores previous reinforcement-learning transitions.

Instead of learning only from the newest transition, historical state-action-outcome examples can be sampled again during later updates.

This helps:

- reduce correlation between consecutive observations,
- reuse rare but important experiences,
- and stabilise incremental learning.

---

## 14. Recent Replay

Recent replay increases representation of comparatively recent market experience.

Its purpose is to prevent the learning system from relying only on distant historical observations when market behaviour may have changed.

Conceptually:

```text
Longer-term experience
        +
Recent behaviour
        =
more adaptive replay distribution
```

---

## 15. Danger Replay

Danger replay focuses on adverse experiences.

Examples can include states associated with:

- large drawdown,
- danger classification,
- poor recovery,
- excessive basket expansion,
- and stressful market conditions.

These events may be relatively rare compared with normal transitions.

A dedicated replay bank prevents them from becoming diluted by a much larger number of ordinary experiences.

---

## 16. Deep-Basket Replay

Deep-basket replay preserves sequences associated with increasing exposure.

This is especially important for averaging systems because failure may result not from one isolated entry, but from a **sequence of increasingly expensive decisions**.

The architecture therefore attempts to remember the path leading toward deep exposure rather than storing only the final closing outcome.

---

## 17. Efficient Replay

The system also preserves comparatively efficient periods.

This provides a complementary learning objective:

```text
remember what created pain
        +
remember what worked efficiently
```

Efficient experiences provide positive examples alongside danger-oriented replay.

---

## 18. Episode Memory

A complete trading basket can be represented as an episode rather than merely as a collection of individual trades.

Episode memory can retain information such as:

- episode start and end,
- basket direction,
- maximum position count,
- number of additions,
- final P/L,
- total reward,
- reward efficiency,
- maximum drawdown,
- maximum danger state,
- one-round completion,
- recovery efficiency,
- session context,
- regime context,
- and pattern context.

This creates a higher-level historical record of complete trading sequences.

---

## 19. Pattern Memory

Historical outcomes can also be grouped according to recurring market characteristics.

Pattern memory records contextual information including:

- pattern classification,
- strength,
- volatility class,
- liquidity class,
- historical reward efficiency,
- historical drawdown,
- and typical number of basket additions.

This enables the system to ask conceptually:

> Have similar market patterns historically produced efficient or dangerous outcomes?

---

## 20. Regime Event Memory

Regime-event memory stores experience associated with broader market conditions.

Information can include:

- regime classification,
- event type,
- volatility,
- spread behaviour,
- trend-strength measures,
- and reward efficiency.

This allows the system to preserve context about how particular market environments behaved historically.

---

## 21. Drawdown-Event Memory

Significant drawdown events receive their own historical representation.

When drawdown develops, the system can preserve information from before, during, and after the event.

Relevant information can include:

- state at trigger,
- Q-values at trigger,
- basket direction,
- basket depth,
- drawdown severity,
- peak drawdown,
- time under water,
- recovery failure,
- market fingerprints,
- and macro/micro context.

Current conditions can later be compared with these historical drawdown states.

The purpose is not to replace the neural policy.

Instead, drawdown memory provides a **cautionary historical context** when current conditions resemble previously harmful situations.

---

## 22. Persistent Q-Memory

The system also maintains a persistent state/Q memory.

A stored entry can associate:

```text
State representation
        ↓
Historical Q-values
        ↓
Confidence
        ↓
Usage / quality score
```

When a similar state appears again, Q-memory can provide an additional historical reference.

This memory is intentionally subordinate to the central DDQN policy rather than functioning as an unrestricted alternative policy.

---

## 23. Danger Brain

The Danger Brain is a protective memory system focused on identifying situations resembling historically adverse episodes.

It uses compact market fingerprints and internal state representations to estimate similarity with previous danger patterns.

The system can operate conceptually in three modes:

```text
NORMAL
CAUTION
DANGER
```

These modes can influence:

- learning intensity,
- action preference,
- risk tolerance,
- and whether new exposure should be restricted.

The Danger Brain should therefore be understood as a **protective context layer**, not as a second independent trading strategy.

---

## 24. Unified Decision Support

The raw DDQN output is not always sent directly to execution.

A broader decision-support context can incorporate information such as:

- DDQN action values,
- Q-memory agreement,
- Q-memory conflict,
- archive similarity,
- drawdown-event risk,
- danger probability,
- pain recurrence,
- deep-basket risk,
- counter-trend failure risk,
- regime-break risk,
- trend persistence,
- reversal probability,
- spike risk,
- expected basket depth,
- and strategy-mode confidence.

Conceptually:

```text
Base DDQN Q-values
        │
        ├── Q-memory
        ├── DD-event memory
        ├── archive similarity
        ├── Danger Brain
        └── current risk context
        │
        ▼
Bounded Decision Adjustment
        │
        ▼
Final Action Preference
```

A key design principle is **DDQN primacy**.

Historical memory can support, caution, or partially veto a decision, but it is not intended to freely overwrite the learned neural policy.

---

## 25. Adaptive Entry Quality

When no basket is currently open, the system evaluates more than simple directional preference.

The entry layer attempts to distinguish between situations more likely to produce:

```text
a clean one-round trade
```

and situations more likely to produce:

```text
a deep or expensive basket
```

The assessment can incorporate:

- historical trading quality,
- recent basket depth,
- danger probability,
- current regime,
- reversal risk,
- trend persistence,
- expected basket depth,
- and memory-based support.

The objective is therefore not only to improve **direction selection**, but also to improve the quality of situations in which new exposure is initiated.

---

## 26. Smart Basket Add Gate

When a basket already exists, adding another position is treated as a separate risk decision.

The Smart Add Gate evaluates a basket-expansion risk score using factors such as:

- current drawdown,
- basket depth,
- basket age,
- danger probability,
- replay risk,
- historical add risk,
- market direction,
- reversal risk,
- spread conditions,
- recent trading quality,
- and expected basket depth.

Depending on the resulting risk level, the system can:

```text
allow the addition
        ↓
widen required spacing
        ↓
reduce position size
        ↓
block the addition
```

This creates an additional layer of control around averaging behaviour.

---

## 27. Dynamic Grid and Basket Management

The execution engine contains adaptive basket-management mechanisms.

Grid spacing can depend on:

- current price range,
- rolling channel width,
- volatility statistics,
- basket depth,
- and additional risk controls.

The architecture can also widen later basket entries when current volatility or historical risk suggests that tightly spaced averaging may be inappropriate.

Basket exits can be managed through hybrid take-profit logic and recovery-oriented close behaviour.

These mechanisms sit downstream of the RL policy.

---

## 28. Additional Risk Controls

The reinforcement-learning architecture operates alongside deterministic safeguards.

The implementation contains mechanisms covering areas such as:

- equity-based protection,
- fixed floating-loss protection,
- global account monitoring,
- minimum price spacing between additions,
- minimum time between new positions,
- re-entry cooldown,
- adaptive basket spacing,
- basket take-profit behaviour,
- post-profit pauses,
- relative-market z-score monitoring,
- emergency hedge behaviour,
- and recovery controls.

These mechanisms exist because reinforcement learning does not remove the need for explicit trading-risk constraints.

---

## 29. Relative-Market Risk Monitoring

The system can monitor additional markets using z-score-based relative-price logic.

Conceptually:

```text
Primary trading market
        +
Reference market A
        +
Optional reference market B
        ↓
Relative-price stress evaluation
```

When extreme conditions are detected, the risk layer can:

- pause new trading,
- close smaller baskets,
- activate emergency protection,
- or wait for a configured number of quieter bars before resuming activity.

This layer is separate from the core DDQN policy.

---

## 30. Persistence

Adaptive behaviour is only useful if learned information can survive beyond one runtime session.

The architecture can persist selected learning components including:

- DQN model state,
- target-network state,
- Q-memory,
- historical archive memory,
- replay banks,
- danger memory,
- and drawdown-event memory.

Conceptually:

```text
Session 1
   ↓
Learn
   ↓
Save state
   ↓
Restart MT5 / Strategy Tester
   ↓
Load previous state
   ↓
Continue learning
```

Persistence therefore forms part of the learning architecture rather than functioning only as configuration storage.

---

## 31. End-to-End Decision Flow

A simplified complete decision cycle is:

```text
1. Receive new market information
             │
             ▼
2. Update indicators, volatility and structure
             │
             ▼
3. Build market + basket state
             │
             ▼
4. Determine current volatility regime
             │
             ▼
5. Run feature-specific branch encoders
             │
             ▼
6. Fuse encoded representations
             │
             ▼
7. Estimate DDQN action values
             │
             ▼
8. Retrieve relevant historical memory
             │
             ▼
9. Build unified decision-support context
             │
             ▼
10. Evaluate entry / basket risk
             │
             ▼
11. Apply deterministic safety gates
             │
             ▼
12. HOLD / BUY / SELL / manage basket
             │
             ▼
13. Observe post-action state
             │
             ▼
14. Calculate intermediate or final reward
             │
             ▼
15. Store transition and episode information
             │
             ▼
16. Route experience into replay banks
             │
             ▼
17. Sample historical experience
             │
             ▼
18. Update online DQN
             │
             ▼
19. Update target network
             │
             ▼
20. Update long-term memories
             │
             ▼
21. Persist selected learning state
             │
             └──────────────► repeat
```

---

## 32. What the Neural Network Does — and Does Not Do

It is important to distinguish the neural policy from the complete trading system.

The neural network primarily learns relationships of the form:

```text
Current State
      ↓
Expected value of:
HOLD / BUY / SELL
```

It does not independently control every part of the trading system.

The complete architecture combines:

```text
Learned Policy
      +
Historical Memory
      +
Explicit Risk Controls
      +
Basket Management
      +
Execution Logic
```

This makes Adaptive-DDQN-MT5 a hybrid system rather than a purely neural trading agent.

---

## 33. Why the System Uses Multiple Forms of Memory

Neural-network weights are good at generalising patterns but are not always ideal for remembering specific rare events.

A trading system may encounter:

- an unusual volatility spike,
- an extended directional move,
- a deep basket sequence,
- an abnormal spread environment,
- or a rare but severe drawdown.

These events may represent only a very small portion of the total training sample.

Adaptive-DDQN-MT5 therefore combines:

```text
Generalisation
        │
        ▼
Neural Network

with

Specific Historical Recall
        │
        ▼
Memory Systems
```

The intention is to give the agent both:

> a learned general policy

and

> access to selected historical context.

---

## 34. Why Risk Is Part of the Learning Objective

A major design goal is to prevent the system from learning the wrong lesson from profitable recovery trades.

Consider two simplified episodes.

### Episode A

```text
1 position
small drawdown
quick recovery
profit
```

### Episode B

```text
7 positions
large drawdown
long recovery
high margin usage
profit
```

A simple profit-only reward function may treat both episodes as successful.

A risk-aware reward system can distinguish them.

Conceptually:

```text
Episode A
profit + efficiency + low drawdown
        ↓
high-quality positive experience

Episode B
profit - basket depth - drawdown - time - risk
        ↓
lower-quality or even negative experience
```

This distinction is central to the project.

---

## 35. Why the Project Is Particularly Suited to Visual Backtesting

The MetaTrader 5 Strategy Tester provides an opportunity to observe the learning process while the historical market is replayed.

The project is therefore suitable for visualising:

- changing Q-values,
- selected actions,
- volatility regime,
- exploration rate,
- danger state,
- basket depth,
- replay-memory growth,
- episode reward,
- historical-memory confidence,
- and policy changes over time.

The objective of visual backtesting is not only to inspect trades.

It is also to observe how the **internal decision process evolves as experience accumulates**.

---

## 36. Research Scope

Adaptive-DDQN-MT5 should primarily be viewed as an experiment in implementing reinforcement-learning concepts directly inside the MetaTrader 5 environment.

The most interesting questions are not limited to whether a backtest is profitable.

The repository is intended to explore questions such as:

> How does the learned policy change as experience accumulates?

> Does the system behave differently after encountering repeated adverse trading episodes?

> Can specialised replay memory improve learning from rare but important drawdown events?

> How does behaviour change across volatility regimes?

> Can reward design distinguish efficient profitable trading from profitable but high-risk recovery?

> Can historical memory reduce repetition of previously harmful basket sequences?

> Does state self-awareness improve behaviour when the system already carries exposure?

> Can memory-assisted decision support complement a neural policy without replacing it?

These questions form the basis for the experiments, diagnostics, and visualisations that will be added to the repository.

---

## 37. Project Scope and Limitations

This project is intended for:

- reinforcement-learning research,
- MQL5 machine-learning experimentation,
- Strategy Tester analysis,
- educational study,
- and supervised trading-system research.

It should not be interpreted as evidence that reinforcement learning removes trading risk.

The architecture includes averaging and basket-management behaviour, and exposure can increase during adverse price movement.

Adaptive learning, historical memory, and risk controls may influence this behaviour but cannot eliminate:

- model risk,
- market risk,
- execution risk,
- liquidity risk,
- regime-change risk,
- or tail-event risk.

Backtesting and historical learning results do not guarantee future performance.

Human supervision and independent risk controls remain appropriate for any live-market experimentation.

---

## 38. Repository Navigation

The broader repository is organised around the architecture described here.

```text
src/
    Main MQL5 implementation

docs/
    Technical documentation

assets/
    Architecture diagrams, screenshots and visual demonstrations

presets/
    Example Strategy Tester and research configurations

experiments/
    Reproducible studies of learning behaviour
```

Further documentation will cover:

- reinforcement-learning mechanics,
- memory architecture,
- reward engineering,
- state construction,
- risk controls,
- Strategy Tester workflow,
- and experimental results.
