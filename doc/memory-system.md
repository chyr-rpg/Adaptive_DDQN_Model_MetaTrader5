# Memory System

Adaptive-DDQN-MT5 uses a multi-layer memory architecture rather than relying exclusively on neural-network weights.

The central idea is:

> **Learning and remembering are related, but they are not the same process.**

The DDQN learns general relationships between states and actions through gradient updates.

The memory system preserves specific experiences that may be useful later.

Conceptually:

```text
                         EXPERIENCE
                              │
             ┌────────────────┴────────────────┐
             ▼                                 ▼
       Neural Learning                  Explicit Memory
             │                                 │
      Update DDQN weights               Preserve context
             │                                 │
             └────────────────┬────────────────┘
                              ▼
                    Future Decision-Making
```

The current research architecture combines several forms of memory:

```text
Experience Replay
Specialised Replay Banks
Episode Memory
Pattern Memory
Regime-Event Memory
Drawdown-Event Memory
Persistent Q-Memory
Danger Memory
```

Each system serves a different purpose.

---

## 1. Why Use Multiple Forms of Memory?

A conventional DQN usually stores transitions such as:

```text
State
Action
Reward
Next State
```

inside an experience replay buffer.

That is useful for neural-network training, but trading introduces additional questions:

```text
Was this entire basket efficient?

Did this market pattern previously lead to danger?

Have similar states historically preferred another action?

Did this sequence create a deep basket?

How did the system behave before a major drawdown?

Was the final profit achieved efficiently or through excessive recovery?
```

A single transition buffer does not naturally represent all of these questions.

Adaptive-DDQN-MT5 therefore separates memory according to the type of information being preserved.

---

## 2. Memory Architecture Overview

At a high level:

```text
                           TRADING EXPERIENCE
                                  │
          ┌───────────────────────┼────────────────────────┐
          │                       │                        │
          ▼                       ▼                        ▼
    Replay Memory          Historical Archive      Similarity Memory
          │                       │                        │
   ┌──────┼───────┐        ┌──────┼───────┐               │
   ▼      ▼       ▼        ▼      ▼       ▼               ▼
 Main   Recent  Danger  Episode Pattern Regime         Q-Memory
                 │
                 ├── Deep Basket
                 │
                 └── Efficient
                          │
                          ▼
                    Decision Support
                          │
          ┌───────────────┼────────────────┐
          ▼               ▼                ▼
      DD Memory       Danger Brain     Archive Context
          │               │                │
          └───────────────┼────────────────┘
                          ▼
                   Risk-Aware Decision
```

The purpose is not to make every memory influence every decision equally.

Instead, memories provide different forms of historical context.

---

## 3. Main Experience Replay

The main replay buffer stores reinforcement-learning transitions.

A simplified replay item contains:

```text
S(t)
A(t)
R(t)
S(t+1)
Terminal State
Regime Context
```

Rather than immediately training only on the newest observation, previous experiences can be sampled again.

Conceptually:

```text
New Experience
      │
      ▼
 Replay Buffer
      │
      ├── old transition
      ├── recent transition
      ├── profitable transition
      ├── adverse transition
      └── unusual transition
               │
               ▼
        Training Sample
               │
               ▼
         DDQN Update
```

Replay reduces dependence on the chronological order of market observations and allows important experience to influence the neural policy more than once.

---

## 4. Replay Priority

Not every historical transition necessarily deserves the same probability of being replayed.

Replay items can carry a priority representing their current learning significance.

Conceptually:

```text
Ordinary Experience
priority = lower

Large Prediction Error
priority = higher

Severe Drawdown Experience
priority = higher

Deep-Basket Experience
priority = higher
```

After the network trains on a sampled transition, its priority can be refreshed using updated learning information.

This allows replay to evolve together with the neural policy.

---

## 5. Recent Replay

The Recent Replay Bank gives comparatively recent market experience its own representation.

This addresses a basic problem in financial markets:

> The most historically common behaviour is not always the most currently relevant behaviour.

Conceptually:

```text
Long-Term Replay
        +
Recent Replay
        ↓
Historical Knowledge
        +
Current Market Adaptation
```

Recent replay is especially useful when market behaviour begins changing faster than the long-term replay distribution.

---

## 6. Danger Replay

The Danger Replay Bank preserves adverse experiences.

Examples can include transitions associated with:

- elevated drawdown,
- danger classifications,
- poor recovery behaviour,
- high pain severity,
- problematic basket expansion,
- or historically stressful states.

Without a specialised bank, rare severe events can become diluted by thousands of ordinary transitions.

Conceptually:

```text
10,000 Ordinary Experiences
        +
50 Severe Experiences
```

could cause the severe examples to receive very little training attention.

Danger replay gives those events a dedicated path back into training.

---

## 7. Deep-Basket Replay

Deep-basket experience is treated separately because basket risk often develops as a **sequence**, not as one isolated decision.

For example:

```text
Initial Entry
      ↓
First Add
      ↓
Second Add
      ↓
Third Add
      ↓
Increasing Drawdown
      ↓
Late Recovery Attempt
      ↓
Exit
```

If only the final transition were emphasised, the system could fail to learn which earlier actions contributed to excessive exposure.

The Deep-Basket Replay Bank therefore preserves experiences associated with increasing basket depth and recovery difficulty.

Its purpose is to help the network learn:

> **How did the system arrive at the dangerous state?**

rather than only:

> **What happened at the end?**

---

## 8. Efficient Replay

The memory system does not focus only on failure.

The Efficient Replay Bank preserves comparatively high-quality trading experience.

Examples can include episodes with:

```text
Positive Reward
Low Basket Depth
Low Drawdown
Good Reward Efficiency
Clean Recovery
Few Additions
```

Conceptually:

```text
Danger Memory
      ↓
What should be avoided?

Efficient Memory
      ↓
What should be reinforced?
```

This creates a balance between learning from pain and learning from efficient behaviour.

---

## 9. Bank-Aware Replay Sampling

Training batches can be drawn from multiple replay sources.

The current architecture can sample from:

```text
Main Replay
Recent Replay
Danger Replay
Deep-Basket Replay
Efficient Replay
```

rather than treating all stored experience as one undifferentiated pool.

A simplified conceptual batch might look like:

```text
Main Experience       ─────────────┐
Recent Experience     ───────┐     │
Danger Experience     ───┐   │     │
Deep Basket           ───┼───┼─────┼──► Training Batch
Efficient Experience  ───────┘     │
                                   ▼
                               DDQN Update
```

The relative sampling weights are configurable.

The current implementation can also modify those weights according to the prevailing replay-risk context rather than keeping them permanently fixed. :contentReference[oaicite:1]{index=1}

---

## 10. Adaptive Replay Distribution

Replay-bank weighting can change as the system's risk context changes.

Conceptually:

```text
Normal / Efficient Period
        ↓
more emphasis on
recent + efficient experience
```

while:

```text
Increasing Drawdown
or
Deep-Basket Pressure
        ↓
more emphasis on
danger + deep-basket experience
```

This means replay composition itself can become part of the adaptive learning process.

The objective is not merely to remember adverse experience.

It is to make adverse experience **more available to learning when it becomes relevant again**.

---

## 11. Episode Memory

Individual RL transitions capture local actions.

Trading baskets often need a higher-level representation.

Episode Memory stores information about a complete trading sequence.

A recorded episode can contain context such as:

```text
Episode ID
Symbol
Start Time
End Time

Basket Direction
Maximum Positions
Number of Adds
Number of Actions

Final P/L
Total Reward
Reward Efficiency

Maximum Drawdown
Maximum Danger
Margin Stress

One-Round Trade?
Inefficient Recovery?
Forced-Stop-Like Event?

Session Context
Liquidity Context
Regime Context
Pattern Context
```

This allows the system to reason about an entire trading sequence instead of viewing every entry independently.

The current implementation explicitly constructs episode records with basket depth, add count, reward efficiency, maximum drawdown, risk classifications, session/liquidity/regime/pattern context, and multi-timeframe span references. :contentReference[oaicite:2]{index=2}

---

## 12. Why Episode Memory Matters

Consider two episodes.

### Episode A

```text
1 position
small drawdown
20-minute duration
positive result
```

### Episode B

```text
6 positions
large drawdown
18-hour duration
positive result
```

At the final P/L level:

```text
both = profitable
```

At the behavioural level:

```text
Episode A ≠ Episode B
```

Episode Memory allows these paths to remain distinguishable.

This is particularly important when the research objective is **quality of profitability**, not simply profitability.

---

## 13. Pattern Memory

Pattern Memory groups historical information according to recurring market structure.

Instead of remembering only:

```text
"This exact state occurred before"
```

the system can retain information about broader pattern classes.

Conceptually:

```text
Current Market
      │
      ▼
Pattern Classification
      │
      ▼
Historical Pattern Memory
      │
      ├── typical reward
      ├── typical drawdown
      ├── typical basket depth
      └── historical efficiency
```

This allows previous experience to be reused at a higher level of abstraction than exact state matching.

---

## 14. Regime-Event Memory

Market behaviour is affected by volatility and broader regime context.

Regime-Event Memory preserves historical experience associated with different market environments.

Examples may include relationships involving:

```text
Volatility Regime
Liquidity Environment
Spread Behaviour
Market Event Context
Reward Efficiency
Drawdown Characteristics
```

The purpose is to answer questions such as:

> Did similar behaviour work differently under high volatility?

or:

> Did this pattern historically create more risk in this regime?

This complements the regime-specific neural-network bank.

---

## 15. Persistent Q-Memory

Q-Memory stores historical state representations together with remembered action-value information.

Conceptually:

```text
State Key
   +
Historical Q(HOLD)
Historical Q(BUY)
Historical Q(SELL)
   +
Confidence
   +
Usage / Quality
```

When a sufficiently similar state appears later, Q-Memory can return a historical Q-value perspective.

This creates a second source of information:

```text
Current Neural Policy
        +
Historical State Recall
```

---

## 16. Q-Memory Is Not the Main Policy

The architecture intentionally avoids allowing Q-Memory to replace the DDQN.

Instead:

```text
DDQN
  ↓
Primary Action-Value Estimate

Q-Memory
  ↓
Historical Supporting Evidence
```

If they agree, historical confidence can support the decision.

If they conflict, the system can become more cautious or increase HOLD preference.

Conceptually:

```text
DDQN says BUY
Q-Memory says BUY
        ↓
confirmation
```

versus:

```text
DDQN says BUY
Q-Memory strongly disagrees
        ↓
caution / reduced conviction
```

The adjustment is bounded so that memory remains subordinate to the neural policy.

---

## 17. Q-Memory Merging

Storing every observed state independently would cause memory to grow rapidly.

Similar state entries can therefore be merged.

Conceptually:

```text
New State
   │
   ▼
Compare with Q-Memory
   │
   ├── sufficiently similar
   │         ↓
   │      merge / update
   │
   └── sufficiently different
             ↓
          new entry
```

Feature values and confidence can evolve as more similar observations are encountered.

Memory pruning and quality thresholds can also be used to control long-term growth.

---

## 18. Drawdown-Event Memory

Drawdown is important enough to receive its own event-level memory system.

When drawdown crosses configured conditions, the system can create a historical event containing context from:

```text
BEFORE DRAWDOWN
      │
      ▼
DRAWdown trigger
      │
      ▼
EXPANSION / STRESS
      │
      ▼
RECOVERY OR FAILURE
```

A stored drawdown event can preserve:

```text
trigger time
symbol
regime
basket direction
drawdown at trigger
hard / soft trigger classification

state representation
Q-values at trigger

baskets before the event
baskets after the event

market trace
recovery information
```

The implementation explicitly records trigger-state keys, trigger Q-values, prior basket history, subsequent context, and event traces before marking an event complete. :contentReference[oaicite:3]{index=3}

---

## 19. Why Preserve Pre-Drawdown Context?

A major research question is not simply:

> What does a dangerous state look like?

It is:

> What did the system look like **before** it became dangerous?

For example:

```text
Apparently Normal State
        ↓
First Add
        ↓
Second Add
        ↓
Volatility Expansion
        ↓
Large Drawdown
```

The earliest useful warning may occur several decisions before the final drawdown.

DD-event memory is therefore designed to retain the path surrounding the event.

---

## 20. Drawdown Memory as Decision Context

When current conditions resemble a historical DD event, the memory layer can provide cautionary information.

Conceptually:

```text
Current State
      │
      ▼
Compare Against Historical DD Events
      │
      ▼
Similarity Detected
      │
      ├── historical pre-DD phase
      ├── historical expansion phase
      └── historical recovery phase
      │
      ▼
Contextual Action Bias
```

The intention is not to say:

```text
similar state = guaranteed future drawdown
```

Instead:

```text
similar state = historically relevant warning context
```

This distinction is important because historical similarity does not imply deterministic repetition.

---

## 21. Danger Brain

The Danger Brain is a live protective memory layer.

It combines historical similarity with current exposure/risk context to estimate whether the present environment resembles previously dangerous experience.

Conceptually:

```text
Current Market Fingerprint
        +
Current Mini-State
        +
Historical Danger Prototypes
        +
Exposure State
        ↓
Danger Score
        ↓
NORMAL / CAUTION / DANGER
```

The implementation supports separate caution and danger thresholds, mode cooldown behaviour, learning-rate scaling under elevated danger, stored danger memory, delta similarity and configurable danger action policies. :contentReference[oaicite:4]{index=4}

---

## 22. Market Fingerprints

Danger memory does not need to compare the complete high-dimensional state directly.

Compact fingerprints can summarise selected characteristics of the market environment.

Conceptually:

```text
Recent Price Behaviour
Volatility
Trend / Structure
Selected State Features
        ↓
Compact Fingerprint
```

The current environment can then be compared against previously stored danger prototypes.

This reduces the need to perform unrestricted full-state comparison for every historical event.

---

## 23. Mini-State Memory

The Danger Brain also uses a compact internal state representation.

This provides information beyond raw price movement.

For example, two visually similar price patterns may represent very different risks when:

```text
Basket A = no exposure

Basket B = already deep in drawdown
```

The mini-state helps historical similarity incorporate selected aspects of internal trading state.

---

## 24. Bad-Episode Learning

Historical adverse episodes can be condensed into prototypes.

Instead of remembering every bad event independently:

```text
Bad Episode 1
Bad Episode 2
Bad Episode 3
Bad Episode 4
...
```

similar events can contribute to a more general adverse prototype.

Conceptually:

```text
Similar Bad Episodes
        ↓
Merged Prototype
        ↓
Historical Action Bias
```

This allows repeated harmful patterns to accumulate stronger historical significance.

---

## 25. Safe-Recovery Memory

The purpose of danger-oriented memory is not necessarily to stop all trading whenever conditions become difficult.

Recovery behaviour matters too.

A dangerous state can eventually lead to:

```text
successful recovery
```

or:

```text
continued deterioration
```

Historical memory can therefore be used to distinguish between dangerous sequences that later recovered efficiently and sequences that continued worsening.

This helps prevent memory from becoming a simple permanent "fear" mechanism.

---

## 26. Memory Aging

Financial markets change.

A memory architecture therefore should not assume that an observation from the distant past has the same relevance forever.

Memory aging allows older information to gradually lose influence.

Conceptually:

```text
Recent Relevant Experience
        ↓
higher influence


Very Old Experience
        ↓
lower influence
```

Historical memory can also be pruned when its usefulness or quality falls below configured thresholds.

---

## 27. Why Aging Matters

Without aging, the system could eventually accumulate a large historical archive dominated by obsolete conditions.

For example:

```text
Old volatility regime
Old liquidity structure
Old market correlation
Old execution characteristics
```

may become less representative of the current environment.

Memory aging therefore acts as a counterweight to unlimited persistence.

---

## 28. Archive Similarity

Episode, pattern and regime memories can also support live decision-making.

Conceptually:

```text
Current Situation
      │
      ▼
Retrieve Similar Historical Episodes
      │
      ├── risky outcomes
      ├── clean outcomes
      ├── deep baskets
      └── efficient trades
      │
      ▼
Historical Risk / Support Score
```

This score can become one component of the unified decision-support layer.

The historical archive does not directly dictate the final action.

It supplies context.

---

## 29. Efficient-Period Support

Historical retrieval can also identify situations resembling previously efficient periods.

For example:

```text
Current State
      ↓
Similar to historically clean episodes
      ↓
positive supporting context
```

This complements danger-oriented retrieval.

The overall memory philosophy is therefore not:

```text
Remember only failure
```

but:

```text
Remember inefficient behaviour
        +
Remember efficient behaviour
```

---

## 30. Memory and Reward Interact

Memory is not limited to action selection.

Historical information can also influence reward construction.

Conceptually:

```text
Current Episode Outcome
        +
Historical Similarity
        ↓
Contextual Reward Adjustment
```

For example:

```text
profitable result
+
strong similarity to historically risky recoveries
        ↓
reduced reward quality
```

or:

```text
clean result
+
similarity to historically efficient episodes
        ↓
positive quality reinforcement
```

This allows the agent to learn not only from the current result, but also from the historical meaning of that result.

---

## 31. Replay-Aware Reward

Specialised replay history can also provide context for reward engineering.

If the current state resembles repeated anti-patterns associated with:

```text
danger
deep baskets
inefficient recovery
```

the reward system can become more cautious.

Conversely, historically efficient patterns can provide positive reinforcement.

This creates a feedback loop:

```text
Experience
    ↓
Replay Classification
    ↓
Historical Risk Pattern
    ↓
Future Reward Context
    ↓
Future Learning
```

The objective is to discourage the system from repeatedly relearning the same harmful behaviour.

---

## 32. Memory and DDQN Primacy

A large memory system creates one important design risk:

> Memory could become so influential that the neural policy stops being the real policy.

Adaptive-DDQN-MT5 therefore uses a principle of **DDQN primacy**.

Conceptually:

```text
DDQN
        ↓
Base Policy

Memory Systems
        ↓
Bounded Context / Bias

Risk Layer
        ↓
Final Decision
```

Rather than:

```text
Memory says something
        ↓
completely replace DDQN
```

The architecture limits subordinate memory influence relative to the learned neural preference.

---

## 33. Unified Memory-Assisted Decision Support

At decision time, several memory systems can contribute contextual information.

Conceptually:

```text
                         DDQN Q-Values
                              │
       ┌──────────────────────┼──────────────────────┐
       ▼                      ▼                      ▼
    Q-Memory            Archive Memory         DD-Event Memory
       │                      │                      │
       └──────────────┬───────┴──────────────┬──────┘
                      ▼                      ▼
                 Danger Brain          Replay Context
                      │                      │
                      └──────────┬───────────┘
                                 ▼
                       Unified Decision Support
                                 │
                                 ▼
                       Bounded Action Adjustment
                                 │
                                 ▼
                         Risk / Execution Layer
```

The memory system therefore acts as a collection of historical advisers around the DDQN rather than as one monolithic policy.

---

## 34. Persistence

A memory architecture would have limited value if everything disappeared whenever MetaTrader 5 stopped.

Selected components can therefore be persisted between sessions.

These can include:

```text
Neural Networks
Target Networks

Main Replay
Recent Replay
Danger Replay
Deep-Basket Replay
Efficient Replay

Q-Memory
Episode Archives
Pattern Memory
Regime Memory

Drawdown Events
Danger Memory
```

Conceptually:

```text
Session A
   ↓
Accumulate Experience
   ↓
Save Memory
   ↓
Restart
   ↓
Load Memory
   ↓
Session B
   ↓
Continue From Previous Experience
```

Persistence allows the research process to study long-horizon adaptation rather than isolated training runs.

---

## 35. Persistence Also Creates Experimental Risk

Persistent memory means that two visually identical backtests may not begin from the same learning state.

For research purposes, experiments should clearly distinguish between:

```text
Fresh Run
```

and:

```text
Continuation Run
```

A fresh run starts without previously accumulated learning.

A continuation run intentionally inherits previous neural and memory state.

This distinction is essential for reproducibility.

---

## 36. Memory Capacity

Unbounded memory growth is neither computationally practical nor necessarily desirable.

The architecture therefore applies capacity limits to different memory systems.

Conceptually:

```text
New Experience
      ↓
Memory at Capacity?
      │
      ├── No → Store
      │
      └── Yes
           ↓
     Merge / Replace / Trim / Prune
```

Different memories can use different capacities because their purposes differ.

For example, efficient replay may require a different historical depth from DD-event memory.

---

## 37. Why Not Use Only the Neural Network?

A sufficiently large neural network can theoretically encode a great deal of historical knowledge.

However, explicit memory offers several advantages for this research project.

### Neural weights are distributed

It can be difficult to identify whether one rare adverse event remains strongly represented after thousands of later updates.

### Rare events can be forgotten

Extreme drawdowns may occur too infrequently to dominate ordinary gradient training.

### Sequence-level information matters

A complete basket episode contains information not naturally represented by one local transition.

### Historical retrieval is interpretable

It is easier to observe:

```text
"current state resembles 12 previous risky episodes"
```

than to infer why a neural-network weight changed.

For these reasons, the project explores:

> **Neural generalisation + explicit historical memory**

rather than choosing only one.

---

## 38. Memory Does Not Equal Certainty

Historical memory can provide context, but financial markets are non-stationary.

A state may look similar to a historical event while producing a completely different future outcome.

Therefore:

```text
Historical Similarity
        ≠
Future Certainty
```

Memory outputs should be interpreted as:

```text
evidence
context
risk information
historical precedent
```

rather than prediction guarantees.

---

## 39. Memory System Research Questions

The architecture allows several questions to be studied directly.

### Replay Composition

> Does specialised replay alter the behaviour learned from rare adverse experiences?

### Deep-Basket Memory

> Can preserving early steps in dangerous basket sequences reduce repeated exposure escalation?

### Efficient Memory

> Does deliberately replaying efficient behaviour improve the quality of later episodes?

### Q-Memory

> Does historical state/Q recall improve decision stability when the neural policy is uncertain?

### DD-Event Memory

> Can pre-drawdown historical context provide useful warning before similar exposure develops again?

### Danger Brain

> Does prototype-based danger recognition reduce repetition of historically harmful behaviour?

### Aging

> How quickly should historical experience lose influence when the market environment changes?

### Policy Primacy

> How much historical bias can be introduced before memory begins overpowering the learned DDQN policy?

These questions will form the basis of future repository experiments.

---

## 40. Public vs Private Implementation

The full memory architecture described in this document belongs to the current **private research edition**.

The public Learning Edition intentionally implements a much smaller system focused on foundational concepts such as:

```text
Native MQL5 DQN
State Construction
Action Selection
Reward Feedback
Basic Persistence
```

The following systems remain part of the private research architecture:

```text
Experience Replay
Specialised Replay Banks
Episode Memory
Pattern Memory
Regime-Event Memory
Q-Memory
DD-Event Memory
Danger Brain
Archive-Aware Decision Support
Replay-Aware Reward
```

This distinction is intentional.

The public source provides an accessible implementation for learning, while this document describes the broader memory architecture being researched in the private system.

---

## 41. Simplified Memory Lifecycle

The complete memory process can be summarised as:

```text
1. Agent observes market and basket state
             ↓
2. DDQN selects / evaluates an action
             ↓
3. Trading outcome develops
             ↓
4. Transition is resolved
             ↓
5. Experience enters main replay
             ↓
6. Experience is classified
             ↓
7. Route to relevant specialised replay bank
             ↓
8. Basket completion creates episode memory
             ↓
9. Pattern / regime context is archived
             ↓
10. Significant drawdown may create DD event
             ↓
11. Q-memory is updated
             ↓
12. Danger prototypes may be updated
             ↓
13. Historical memory is persisted
             ↓
14. Future states retrieve relevant context
             ↓
15. Memory contributes bounded decision support
             ↓
16. Replay samples feed future DDQN training
             ↓
17. Aging / pruning manage long-term relevance
             ↓
18. Repeat
```

---

## 42. Design Principle

The memory architecture can be summarised in one principle:

> **The agent should not only learn what happened — it should retain enough context to recognise when something similar begins happening again.**

The neural network provides generalisation.

Replay provides repeated learning.

Episode archives provide sequence context.

Q-Memory provides state/action recall.

DD-event memory preserves historical pain.

Efficient memory preserves good behaviour.

The Danger Brain provides live protective context.

Together, these mechanisms form the memory-augmented component of Adaptive-DDQN-MT5.

---

## Related Documentation

For the broader system:

- [System Architecture](architecture.md)
- [How the Agent Learns](learning-system.md)
- [Research Scope & Limitations](limitations.md)

The public Learning Edition is available in:

```text
src/AdaptiveDQN_MT5_LearningEdition.mq5
```

The complete memory-augmented research implementation is maintained privately.
