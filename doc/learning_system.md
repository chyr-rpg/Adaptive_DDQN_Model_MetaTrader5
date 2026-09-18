# How the Agent Learns

Adaptive-DDQN-MT5 uses reinforcement learning to modify its action-value estimates as trading experience accumulates.

The learning problem is framed around three possible actions:

```text
HOLD
BUY
SELL
```

Rather than learning a direct rule such as:

```text
RSI < X → BUY
```

the system attempts to learn a value function:

```text
Q(State, Action)
```

which represents the estimated long-term value of taking a particular action under the current market and portfolio state.

At a high level:

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

The learning architecture combines:

- Double DQN,
- Dueling DQN,
- experience replay,
- prioritised replay,
- regime-aware model banks,
- delayed transition resolution,
- dense basket feedback,
- target networks,
- persistent Q-memory,
- and specialised historical memory.

---

## 1. The Learning Problem

At decision time, the agent receives a state representation:

```text
S(t)
```

The state contains information about both:

```text
Market Environment
        +
Agent's Own Trading State
```

Examples include:

- price and indicator information,
- volatility,
- market structure,
- supply/demand context,
- current positions,
- basket direction,
- basket depth,
- basket age,
- drawdown,
- and other risk information.

The neural network then produces three action values:

```text
Q(S, HOLD)
Q(S, BUY)
Q(S, SELL)
```

These values are not probabilities.

They represent the network's current estimate of the relative long-term value of each action.

---

## 2. Why Reinforcement Learning?

Supervised learning normally requires a labelled target:

```text
Input → Correct Answer
```

Trading rarely provides such a simple label.

A BUY decision may initially move into profit, later experience drawdown, require several additional positions, and eventually close profitably.

Was the original BUY decision good?

The answer depends on more than the final P/L.

Adaptive-DDQN-MT5 therefore approaches the problem as:

```text
State
  ↓
Action
  ↓
Sequence of consequences
  ↓
Reward
```

The system learns from the consequences of its actions rather than from predefined labels saying which action was "correct."

---

## 3. State → Q-Values

After the current state has been constructed, it passes through the neural architecture.

Conceptually:

```text
Market + Basket State
          ↓
Feature Branch Encoders
          ↓
Feature Fusion
          ↓
Shared Neural Layers
          ↓
Dueling DQN Heads
          ↓
Q(HOLD), Q(BUY), Q(SELL)
```

The network separates:

```text
State Value V(s)
```

from:

```text
Action Advantage A(s,a)
```

before recombining them into final Q-values.

This allows the model to learn both:

> How favourable is the current state overall?

and:

> Which action is preferable relative to the alternatives?

---

## 4. Regime-Aware Inference

The learning architecture can maintain separate neural networks for different volatility regimes.

Conceptually:

```text
Low Volatility Network
Medium Volatility Network
High Volatility Network
```

The current market regime is estimated using volatility context.

When soft regime inference is enabled, inference does not necessarily switch abruptly from one network to another.

Instead, outputs from nearby regime models can be blended.

Conceptually:

```text
Current volatility
       ↓
Regime weights
       ↓

0.15 × Low-Regime Q
0.70 × Mid-Regime Q
0.15 × High-Regime Q
       ↓
Blended Q-values
```

This reduces instability near regime boundaries.

---

## 5. Action Selection

During training, the system uses an exploration-versus-exploitation process.

Most of the time, the agent uses its learned action values.

Some of the time, it explores.

Conceptually:

```text
                  Current State
                       │
                       ▼
                  Q-Values
                       │
             ┌─────────┴─────────┐
             ▼                   ▼
        Exploration          Exploitation
             │                   │
      Random Action       Highest Adjusted Q
```

The exploration rate is controlled by:

```text
epsilon (ε)
```

At higher epsilon:

```text
more exploration
```

At lower epsilon:

```text
more reliance on learned behaviour
```

---

## 6. Exploration Decay

The system gradually reduces exploration as training progresses.

Conceptually:

```text
Early Training

ε = relatively high
        ↓
many exploratory actions

        ↓
experience accumulates
        ↓

Later Training

ε = lower
        ↓
greater reliance on learned policy
```

The decay is configurable and bounded by a minimum exploration rate.

This prevents exploration from disappearing completely if a non-zero minimum is retained.

When the Danger Brain enters a high-risk state, exploratory behaviour can also be reduced so that random action selection becomes less aggressive during historically dangerous conditions.

---

## 7. DDQN Is the Base Policy — Not the Entire Decision

The highest raw neural-network Q-value does not automatically become the final trading action.

The system first generates:

```text
Base DDQN Q-values
```

and then constructs a broader decision-support context.

Additional information can include:

- Q-memory agreement,
- historical episode similarity,
- drawdown-event similarity,
- danger-memory context,
- expected basket depth,
- trend persistence,
- reversal probability,
- recent trading quality,
- and other risk information.

Conceptually:

```text
        DDQN Q-values
              │
              ▼
     Historical Context
              │
              ▼
      Risk-Aware Biases
              │
              ▼
      Bounded Adjustment
              │
              ▼
    Final Action Preference
```

The design intentionally keeps DDQN as the primary policy.

Historical memory acts as supporting evidence, caution, or bounded opposition rather than freely replacing the neural policy.

---

## 8. From Action to Experience

After an action is taken, the learning system eventually constructs a transition:

```text
(S(t), A(t), R(t), S(t+1))
```

where:

```text
S(t)     = state before the action
A(t)     = selected action
R(t)     = resulting reward
S(t+1)   = state after the consequence
```

A terminal flag can also indicate whether the associated episode or position sequence has ended.

Conceptually:

```text
State Before
     │
     ▼
   Action
     │
     ▼
Market / Basket Changes
     │
     ▼
State After
     │
     ▼
Reward
     │
     ▼
Transition
```

These transitions are the fundamental learning examples used by the DQN.

---

## 9. Why Trading Requires Delayed Transitions

In many reinforcement-learning examples, the consequence of an action appears immediately.

Trading is different.

Consider an entry:

```text
BUY
 ↓
small profit
 ↓
market reverses
 ↓
second position added
 ↓
larger drawdown
 ↓
recovery
 ↓
basket finally closes
```

If the original BUY were rewarded immediately after the first favourable movement, the agent might learn the wrong lesson.

Adaptive-DDQN-MT5 therefore supports **pending transitions**.

A transition can retain information about:

- the original state,
- the selected action,
- entry price,
- entry time,
- position size,
- basket state,
- and accumulated intermediate reward.

The transition can later be resolved when more meaningful information becomes available.

---

## 10. Pending Transition Resolution

When a position or basket reaches a meaningful outcome, the pending transition can be resolved.

Conceptually:

```text
Original Decision
      ↓
Pending Transition
      ↓
Intermediate Basket Behaviour
      ↓
Position / Basket Resolution
      ↓
Final Reward
      ↓
Completed Replay Experience
```

The resolved reward may combine:

```text
terminal reward
+
dense intermediate reward
+
per-position adjustment
```

The resulting transition is then added to replay memory and can influence future network training.

This creates a stronger connection between an earlier decision and its eventual trading consequences.

---

## 11. Dense Basket-Health Learning

Not all learning waits until a position closes.

The system can also evaluate changes in an open basket over time.

Intermediate feedback can consider changes in:

```text
Open P/L
Drawdown
Basket Depth
Basket Age
Recovery Progress
```

For example:

```text
Basket at t

2 positions
2% drawdown

        ↓

Basket at t+1

3 positions
4% drawdown

        ↓

negative intermediate feedback
```

Conversely:

```text
drawdown decreases
+
open return improves
+
basket remains shallow
        ↓
positive intermediate feedback
```

This provides a denser training signal than relying only on the final basket close.

---

## 12. Reward Is Risk-Aware

The reinforcement-learning objective is deliberately different from:

```text
profit → reward
loss   → punishment
```

Trading quality also matters.

Conceptually:

```text
Reward
   =
Trading Outcome
+ Recovery Quality
+ Efficiency
- Drawdown
- Basket Expansion
- Margin Stress
- Time Cost
- Historical Risk
```

This helps the agent distinguish between two profitable outcomes that reached profit through very different paths.

---

## 13. Clean Profit vs Risky Profit

Consider two simplified episodes.

### Episode A

```text
BUY
 ↓
small drawdown
 ↓
profit target
 ↓
close
```

### Episode B

```text
BUY
 ↓
drawdown
 ↓
ADD
 ↓
deeper drawdown
 ↓
ADD
 ↓
large exposure
 ↓
long recovery
 ↓
eventual profit
```

A profit-only reward system could treat both as successful.

Adaptive-DDQN-MT5 attempts to distinguish them.

Conceptually:

```text
Episode A

Profit
+ Low Drawdown
+ Low Basket Depth
+ Efficient Recovery

        ↓

Higher-Quality Reward
```

versus:

```text
Episode B

Profit
- Deep Basket
- Large Drawdown
- Recovery Time
- Increased Exposure

        ↓

Reduced Reward
```

A profitable result can therefore still contain negative learning information.

---

## 14. Experience Replay

Completed transitions are stored in replay memory.

Instead of training only on the newest experience:

```text
Experience 1001
      ↓
immediate training
```

the system can train on older experiences:

```text
Experience 0243
Experience 0751
Experience 0992
Experience 0318
...
```

This is known as **experience replay**.

It helps:

- reduce correlation between consecutive observations,
- reuse important experiences,
- improve data efficiency,
- and prevent learning from being dominated by the latest market sequence.

---

## 15. Replay Warm-Up

Training does not need to begin immediately after the first few transitions.

The system can wait until replay memory contains a minimum amount of experience.

Conceptually:

```text
Replay Size < Warm-Up Threshold
            ↓
Store experience
No replay training yet

            ↓

Replay Size ≥ Warm-Up Threshold
            ↓
Begin replay training
```

This prevents the neural network from being trained repeatedly on an extremely small and unrepresentative early sample.

---

## 16. Specialised Replay Banks

The system does not treat every experience as identical.

Replay experiences can also be routed into specialised banks.

```text
                     Experience
                          │
                          ▼
                      Main Replay
                          │
          ┌───────────────┼───────────────┐
          ▼               ▼               ▼
       Recent          Danger          Deep Basket
       Replay          Replay           Replay
                          │
                          ▼
                     Efficient
                       Replay
```

The main categories are:

### Recent Replay

Maintains representation of comparatively recent behaviour.

### Danger Replay

Emphasises adverse or high-risk situations.

### Deep-Basket Replay

Preserves experiences associated with increasing basket depth and exposure.

### Efficient Replay

Stores examples of comparatively efficient trading behaviour.

These replay banks allow rare but important experiences to remain represented during training.

---

## 17. Mixed Replay Sampling

When a replay batch is trained, samples can be drawn from several replay sources rather than only from the main buffer.

Conceptually:

```text
Training Batch

46% Main Experience
18% Recent Experience
 8% Danger Experience
 8% Deep-Basket Experience
20% Efficient Experience
```

The exact weights are configurable.

The objective is not to enforce one permanent distribution.

It is to prevent the training process from forgetting important experience classes simply because they occur less frequently in the raw transition stream.

---

## 18. Priority-Based Sampling

Experiences can also carry a priority value.

Transitions with stronger learning significance can therefore receive a greater probability of being sampled.

Conceptually:

```text
Small, ordinary transition
priority = low

Large error / unusual outcome
priority = higher
```

Replay priority can be influenced by factors such as:

- reward magnitude,
- learning error,
- danger classification,
- basket depth,
- and efficiency characteristics.

After a transition is replayed, its priority can be recalculated using the network's updated evaluation.

---

## 19. Replay Training Cycle

A simplified replay update looks like:

```text
Replay Memory
      ↓
Sample Experience
      ↓
Read:
S, A, R, S'
      ↓
Estimate Current Q(S,A)
      ↓
Estimate Target
      ↓
Calculate Error
      ↓
Backpropagate
      ↓
Update Network
      ↓
Recalculate Replay Priority
```

Multiple samples and multiple replay iterations can be processed during one replay-training event.

---

## 20. Double DQN Target

The system supports Double DQN to reduce Q-value overestimation.

In ordinary DQN, the same network can effectively participate in both:

```text
choosing the best next action
```

and:

```text
estimating its value
```

Double DQN separates these roles.

Conceptually:

```text
Online Network
      ↓
Choose best action in S(t+1)

Target Network
      ↓
Evaluate that action

      ↓
Construct learning target
```

The target can be understood conceptually as:

```text
Target =
Reward
+
Discounted value of the next state
```

for non-terminal transitions.

For terminal outcomes, future value is not added.

---

## 21. Discount Factor

The discount factor determines how strongly future value contributes to the current learning target.

Conceptually:

```text
Q Target =
Immediate Reward
+
γ × Future Value
```

where:

```text
γ = discount factor
```

A lower value places relatively more importance on immediate outcomes.

A higher value gives greater weight to longer-horizon consequences.

This is particularly relevant for basket trading, where the consequence of one action may develop over many later decisions.

---

## 22. Huber Loss

The implementation can use Huber loss when calculating the neural-network update.

Huber loss behaves approximately like:

```text
squared error
```

for relatively small errors, while becoming more similar to:

```text
absolute error
```

for very large errors.

This can reduce the influence of extreme TD errors compared with unrestricted squared-error loss.

That is useful in trading environments where unusually large rewards or penalties may occasionally occur.

---

## 23. Gradient Clipping

The system can also clip gradients during backpropagation.

Conceptually:

```text
Very large gradient
        ↓
Clip magnitude
        ↓
Apply bounded update
```

This reduces the risk that one extreme experience causes an excessively large neural-network update.

---

## 24. Online and Target Networks

For each supported regime, the architecture can maintain:

```text
Online DQN
Target DQN
```

The online network is actively trained.

The target network changes more slowly.

This stabilises the learning target.

Two update approaches are supported conceptually:

### Periodic Hard Synchronisation

```text
Online Network
      ↓
after N updates
      ↓
copy weights
      ↓
Target Network
```

### Soft Target Update

```text
Target =
(1 - τ) × Old Target
+
τ × Online Network
```

where:

```text
τ
```

is a small update coefficient.

Soft updates cause the target network to track the online network gradually rather than changing abruptly.

---

## 25. Regime-Specific Training

Experience is associated with a market regime.

The corresponding regime model can therefore learn from the transition.

During configured warm-up behaviour, experience may also be used to train multiple regime models before sufficient specialised replay history exists.

Conceptually:

```text
Early learning

One experience
     ↓
broader regime training

Later learning

More accumulated history
     ↓
greater regime specialisation
```

This attempts to balance early sample efficiency with later regime-specific behaviour.

---

## 26. Historical Memory vs Neural Learning

Not every form of learning occurs through backpropagation.

Adaptive-DDQN-MT5 contains two broad learning mechanisms:

```text
                    EXPERIENCE
                         │
              ┌──────────┴──────────┐
              ▼                     ▼
        Neural Learning       Historical Memory
              │                     │
        Update weights        Store episodes /
                              states / events
```

### Neural Learning

Changes the network's general action-value function.

### Historical Memory

Preserves specific prior contexts that may later be retrieved.

This distinction is important.

A neural network attempts to **generalise**.

Memory systems allow the agent to **recall**.

---

## 27. Persistent Q-Memory

Q-memory stores representations of previously encountered states together with historical Q-value information.

Conceptually:

```text
Historical State
      +
Historical Q-Values
      +
Confidence
      ↓
Persistent Q-Memory
```

When a similar state appears later, the system can retrieve this historical preference.

If Q-memory agrees with the DDQN:

```text
confidence may increase
```

If Q-memory conflicts with the DDQN:

```text
the system may become more cautious
```

Q-memory is therefore used as supporting context rather than as an independent unrestricted policy.

---

## 28. Learning from Drawdown

Large drawdown episodes contain information that ordinary replay may not preserve strongly enough.

The system therefore maintains dedicated drawdown-event memory.

When significant drawdown occurs, the system can retain information about:

```text
conditions before drawdown
conditions during drawdown
basket progression
Q-values
market fingerprint
severity
recovery
```

If a similar environment occurs later, this historical experience can influence decision support.

Conceptually:

```text
Current State
      ↓
Compare with Historical DD Events
      ↓
High Similarity?
      ↓
Increase Caution
```

The drawdown archive provides context; it does not replace the DDQN itself.

---

## 29. Learning from Good and Bad Sequences

The learning architecture attempts to preserve both:

```text
What worked efficiently
```

and:

```text
What created unnecessary risk
```

This is particularly important for basket-based strategies.

A deep basket is not simply one bad state.

It is a sequence:

```text
Entry
  ↓
Add
  ↓
Add
  ↓
Drawdown
  ↓
Recovery Attempt
  ↓
Add
  ↓
Exit
```

Deep-basket replay therefore preserves sequence-related experience so that later training can give earlier decisions within problematic sequences meaningful weight.

---

## 30. Learning Frequency

Neural training does not necessarily occur on every market tick.

The architecture includes configurable controls for:

- exploration decay frequency,
- decision-bar training frequency,
- replay warm-up,
- replay batch size,
- replay iterations,
- and target-network update frequency.

This separates:

```text
Market Observation Frequency
```

from:

```text
Neural Update Frequency
```

which is useful because heavy neural training on every tick would be both computationally expensive and potentially noisy.

---

## 31. Fast Strategy Tester Training

Because the entire architecture runs inside MQL5, backtesting efficiency matters.

The project includes mechanisms intended to reduce unnecessary repeated computation during Strategy Tester runs.

These include:

- state caching,
- indicator caching,
- structure/zone caching,
- configurable feature skipping,
- sparse memory refresh,
- and replay scheduling.

The objective is to preserve the learning architecture while making longer historical training experiments practical.

---

## 32. Persistence Across Sessions

The agent can persist selected learned state.

Conceptually:

```text
Training Session A
       ↓
Learn
       ↓
Save
       ↓
MT5 Stops

       ...

Training Session B
       ↓
Load Previous State
       ↓
Continue Learning
```

Persisted components can include:

- neural-network parameters,
- target networks,
- replay memory,
- Q-memory,
- historical archives,
- danger memory,
- and drawdown-event memory.

This means the learning process does not necessarily need to restart from random initialisation every time the EA is launched.

---

## 33. Resuming Exploration

When a trained model is loaded, exploration can also be adjusted.

Instead of returning automatically to the original high exploration level, the system can resume with a reduced epsilon.

Conceptually:

```text
Brand-New Model
      ↓
higher exploration

Previously Trained Model
      ↓
lower restart exploration
```

This helps preserve learned behaviour while still allowing further adaptation.

---

## 34. What "Learning" Means in This Project

The phrase **the system learns** should not be interpreted as:

> The EA continuously discovers a guaranteed profitable strategy.

In this project, learning means that accumulated experience can modify:

- neural-network weights,
- Q-value estimates,
- replay composition,
- state/action memory,
- danger prototypes,
- historical episode statistics,
- and decision-support context.

As a result:

```text
Same market condition
        +
Different accumulated experience
        ↓
Potentially different action preference
```

That adaptive behaviour is the central research feature of the project.

---

## 35. What Learning Does Not Guarantee

Reinforcement learning does not guarantee that later behaviour is always better than earlier behaviour.

The agent can still learn from:

- incomplete historical samples,
- unstable regimes,
- noisy rewards,
- changing market relationships,
- rare tail events,
- and imperfect state representations.

A policy that performs well under one historical environment may degrade when the environment changes.

For this reason, learning should be evaluated through:

```text
behaviour
risk
stability
drawdown
basket depth
reward quality
and performance
```

rather than profit alone.

---

## 36. Visualising the Learning Process

One of the main goals of this repository is to make internal learning behaviour observable through MetaTrader 5 Strategy Tester.

Future visual diagnostics will expose variables such as:

```text
Current Regime

Q(HOLD)
Q(BUY)
Q(SELL)

Selected Action

Exploration Rate

Episode Count

Reward

Danger Probability

Basket Depth

Replay Memory Size

Danger Replay Size

Deep-Basket Replay Size

Efficient Replay Size

Memory Confidence
```

This creates a visual distinction between:

```text
price movement
```

and:

```text
how the agent interprets and learns from that movement
```

---

## 37. Simplified End-to-End Learning Cycle

The complete learning process can be summarised as:

```text
1. Observe market
        ↓
2. Construct market + self-awareness state
        ↓
3. Determine regime context
        ↓
4. Run DDQN inference
        ↓
5. Retrieve relevant memory
        ↓
6. Adjust action preference within bounded limits
        ↓
7. Explore or select preferred action
        ↓
8. Execute / manage position
        ↓
9. Observe resulting market and basket state
        ↓
10. Accumulate intermediate basket feedback
        ↓
11. Resolve transition when sufficient outcome exists
        ↓
12. Calculate risk-aware reward
        ↓
13. Store transition in replay
        ↓
14. Route experience to specialised replay banks
        ↓
15. Sample historical transitions
        ↓
16. Construct Double-DQN targets
        ↓
17. Backpropagate network error
        ↓
18. Update replay priority
        ↓
19. Update target network
        ↓
20. Update historical memory
        ↓
21. Persist selected learned state
        ↓
22. Repeat
```

---

## 38. Research Questions

The learning system makes several research questions possible.

### Policy Evolution

> How does action preference change as replay experience accumulates?

### Exploration

> At what point does the system begin relying primarily on learned behaviour rather than exploration?

### Risk-Aware Rewards

> Does penalising deep profitable recovery alter later basket behaviour?

### Replay Architecture

> Does specialised danger and deep-basket replay change how the network responds to rare adverse situations?

### Regime Learning

> Do regime-specific models develop meaningfully different action preferences?

### Memory

> Does historical similarity reduce repetition of previously harmful trading sequences?

### Persistence

> Does continued training from stored experience behave differently from repeated training from random initialisation?

These questions will form the basis of future experiments in the repository.

---

## 39. Related Documentation

For a broader overview of the complete trading system, see:

- [System Architecture](architecture.md)
- [Memory Architecture](memory-system.md) *(planned)*
- [Limitations & Research Scope](limitations.md) *(planned)*

The main MQL5 implementation is available in:

```text
src/AdaptiveDDQN_MT5.mq5
```
