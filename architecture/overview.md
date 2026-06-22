# Predictive Active Workspace (PAW)

EIDOS implements the **Predictive Active Workspace** architecture — a biologically-grounded cognitive system combining three major frameworks:

## Component Map

| Biological Framework | Component | File |
|---------------------|-----------|------|
| Predictive Processing (Friston, Rao & Ballard) | `PredictionEngine` | `prediction_engine.py` |
| Global Workspace Theory (Baars, Dehaene) | `WorkspaceBuffer` | `working_memory.py` |
| Hebbian Learning (Hebb) | `AssociationStore` | `association_store.py` |
| Attention (Corbetta & Shulman) | `AttentionGate` | `attention_gate.py` |
| Reward Prediction Error (Schultz) | `RewardSignal` | `reward_signal.py` |
| Dual-Process Theory (Kahneman) | `ReasoningLoop` + `MetaCognitionMonitor` | `reasoning_loop.py`, `meta_cognition.py` |
| Hippocampal Replay (Wilson & McNaughton) | `consolidate_belief()` + `SleepReplay` | `prediction_engine.py`, `sleep_replay.py` |
| Complementary Learning Systems (McClelland) | `EpisodicBuffer` + `BeliefGraph` | `episodic_buffer.py`, `belief_graph.py` |

## Data Flow

```
INPUT → AttentionGate → WorkspaceBuffer → PredictionEngine → prediction_error
                                    ↓                              ↓
                            AssociationStore ← RewardSignal
                                    ↓
                            ReasoningLoop (if error > threshold)
                                    ↓
                              OUTPUT / ACTION
```

## Design Decisions

1. **PredictionEngine is the spine** — all cognition flows through predict-compare-learn
2. **Workspace capacity = 7** — Miller's Law default, configurable
3. **No ML frameworks** — pure numpy for transparency
4. **Intrinsic motivation only** — reward = surprise reduction
5. **System 2 is conditional** — ReasoningLoop activates only above error threshold

See `diagrams/architecture.mermaid` for the full flowchart.
