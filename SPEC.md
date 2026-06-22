# KISAMAPA LABS — EXPERIMENT 06
## PROJECT EIDOS: A Biologically-Grounded Reasoning Architecture
### Cursor Build Specification — v1.0

---

## 0. WHAT THIS IS

**EIDOS** (Emergent Intelligence via Distributed Organisational Systems) is a research-first project to design and implement a small AI model whose architecture is not derived from transformer/LLM paradigms, but instead from verified cognitive and neuroscientific mechanisms — the way biological brains actually produce thought, memory, and reasoning.

This is a two-phase project:

- **Phase 1 — RESEARCH:** Cursor reads papers and structured sources to extract a set of "cognitive primitives" — minimal mechanisms that together produce reasoning-like behaviour. All findings are stored as structured knowledge.
- **Phase 2 — BUILD:** Cursor implements a small experimental agent architecture based on the extracted primitives and runs tests to observe emergent reasoning behaviour.

There is no external API, no Groq, no hosted LLM involved. Cursor is the sole intelligence doing the research and the building.

---

## 1. GOALS

| Goal | Description |
|------|-------------|
| G1 | Build a reasoning agent whose architecture is grounded in real cognitive science, not LLM token prediction |
| G2 | Use Cursor to research papers and extract concrete, implementable cognitive mechanisms |
| G3 | Implement a working prototype — even tiny — that demonstrates non-trivial reasoning behaviour |
| G4 | Produce a structured knowledge base of cognitive primitives as a reusable lab artifact |
| G5 | Keep this entirely self-contained inside this repo — no cloud dependencies |

## NON-GOALS

- This is NOT trying to match GPT-4. Scale is irrelevant.
- This is NOT an LLM fine-tuning project.
- This is NOT a production system. It is a laboratory experiment.
- This is NOT connected to any other project.

---

## 2. FOLDER STRUCTURE

```
eidos/
├── README.md
├── SPEC.md                        ← this file
│
├── research/
│   ├── primitives/                ← extracted cognitive mechanisms (JSON)
│   │   ├── memory.json
│   │   ├── prediction.json
│   │   ├── attention.json
│   │   ├── learning.json
│   │   ├── reasoning.json
│   │   └── emotion_valence.json
│   ├── papers/                    ← paper summaries extracted by Cursor
│   │   └── [paper_slug].md
│   ├── synthesis.md               ← Cursor's synthesised architecture proposal
│   └── knowledge_base.json        ← master index of all primitives + sources
│
├── architecture/
│   ├── overview.md                ← final chosen architecture, annotated
│   ├── components/
│   │   ├── working_memory.py
│   │   ├── prediction_engine.py
│   │   ├── association_store.py
│   │   ├── attention_gate.py
│   │   ├── reward_signal.py
│   │   └── reasoning_loop.py
│   └── diagrams/
│       └── architecture.mermaid
│
├── agent/
│   ├── eidos.py                   ← main agent class
│   ├── config.py                  ← hyperparameters and settings
│   └── run.py                     ← entry point to run the agent
│
├── tests/
│   ├── test_memory.py
│   ├── test_prediction.py
│   ├── test_reasoning.py
│   └── test_full_agent.py
│
├── experiments/
│   ├── exp_01_basic_prediction/
│   ├── exp_02_memory_consolidation/
│   └── exp_03_reasoning_chain/
│
└── requirements.txt
```

---

## 3. PHASE 1 — RESEARCH

### 3.1 OVERVIEW

Cursor will systematically research the following domains and extract **cognitive primitives** — concrete, implementable mechanisms described in the scientific literature. Each primitive gets a structured JSON entry.

### 3.2 COGNITIVE PRIMITIVE FORMAT

Each extracted primitive must be stored in `research/primitives/[category].json` using this schema:

```json
{
  "id": "predictive_processing_v1",
  "name": "Predictive Processing",
  "category": "prediction",
  "source": {
    "author": "Karl Friston",
    "paper": "The free-energy principle: a unified brain theory?",
    "year": 2010,
    "journal": "Nature Reviews Neuroscience",
    "doi": "10.1038/nrn2787"
  },
  "biological_basis": "The brain continuously generates predictions about incoming sensory data and updates its internal model based on prediction error signals propagated upward through cortical hierarchies.",
  "core_claim": "Cognition is not stimulus-response but constant prediction-correction. The brain minimises surprise (free energy) by either updating beliefs or acting on the world.",
  "implementable_mechanism": {
    "description": "A hierarchical system where each layer predicts the output of the layer below. Errors flow upward. Predictions flow downward. Learning = reducing mismatch.",
    "inputs": ["sensory_input", "prior_belief"],
    "outputs": ["prediction_error", "updated_belief", "action_signal"],
    "pseudocode": "error = sensory_input - prediction; belief = belief + learning_rate * error; action = f(residual_error)"
  },
  "relevance_score": 9,
  "notes": "Central to EIDOS architecture. This should be the spine."
}
```

### 3.3 RESEARCH DOMAINS AND TARGETS

Cursor must research across the following domains. For each domain, key papers and mechanisms to extract are listed:

---

#### DOMAIN A: Predictive Processing & Active Inference

**Key Papers to Research:**
- Friston, K. (2010) — *The free-energy principle: a unified brain theory?* — Nature Reviews Neuroscience
- Friston, K. et al. (2017) — *Active inference: a process theory* — Neural Computation
- Clark, A. (2013) — *Whatever next? Predictive brains, situated agents, and the future of cognitive science* — Behavioral and Brain Sciences
- Rao, R. & Ballard, D. (1999) — *Predictive coding in the visual cortex* — Nature Neuroscience

**Mechanisms to Extract:**
- Hierarchical prediction architecture
- Prediction error propagation
- Precision weighting (attention as prediction-error confidence)
- Action as inference (acting to fulfil predictions)
- Free energy minimisation as the unifying objective

---

#### DOMAIN B: Working Memory & Cognitive Architecture

**Key Papers to Research:**
- Baddeley, A. (2003) — *Working memory: looking back and looking forward* — Nature Reviews Neuroscience
- Baars, B. (1988) — *A cognitive theory of consciousness* — Cambridge University Press (Global Workspace Theory)
- Dehaene, S. et al. (1998) — *A neuronal model of a global workspace in effortful cognitive tasks* — PNAS
- Miller, G.A. (1956) — *The magical number seven, plus or minus two* — Psychological Review

**Mechanisms to Extract:**
- Limited-capacity active memory buffer (~4–7 slots)
- Global Workspace broadcasting (information becomes globally available to specialised processors)
- Working memory as a "scratchpad" for reasoning chains
- Phonological loop and visuospatial sketchpad as distinct stores

---

#### DOMAIN C: Learning & Memory Consolidation

**Key Papers to Research:**
- Hebb, D.O. (1949) — *The organisation of behaviour* — Wiley
- McClelland, J. et al. (1995) — *Why there are complementary learning systems in the hippocampus and neocortex* — Psychological Review
- Wilson, M.A. & McNaughton, B.L. (1994) — *Reactivation of hippocampal ensemble memories during sleep* — Science
- Kumaran, D. et al. (2016) — *What learning systems do intelligent agents need?* — Trends in Cognitive Sciences

**Mechanisms to Extract:**
- Hebbian learning ("neurons that fire together wire together")
- Complementary Learning Systems (CLS): fast hippocampal learning vs slow cortical consolidation
- Sleep replay for memory consolidation (offline batch learning pass)
- Episodic vs semantic memory distinction
- Catastrophic forgetting and how biology avoids it

---

#### DOMAIN D: Attention & Gating

**Key Papers to Research:**
- Corbetta, M. & Shulman, G. (2002) — *Control of goal-directed and stimulus-driven attention in the brain* — Nature Reviews Neuroscience
- Desimone, R. & Duncan, J. (1995) — *Neural mechanisms of selective visual attention* — Annual Review of Neuroscience
- Posner, M.I. (1980) — *Orienting of attention* — Quarterly Journal of Experimental Psychology

**Mechanisms to Extract:**
- Top-down vs bottom-up attention
- Biased competition model (attention as suppression of irrelevant signals)
- Attention as a routing mechanism (which signals get global access)
- Salience as a gating function

---

#### DOMAIN E: Reasoning & Abstraction

**Key Papers to Research:**
- Newell, A. & Simon, H.A. (1976) — *Computer science as empirical inquiry: symbols and search* — Communications of the ACM
- Kahneman, D. (2011) — *Thinking, fast and slow* — (System 1 / System 2 dual-process theory)
- Gentner, D. (1983) — *Structure-mapping: a theoretical framework for analogy* — Cognitive Science
- Hofstadter, D. (1979) — *Gödel, Escher, Bach* — concepts on self-reference and analogy

**Mechanisms to Extract:**
- Dual-process theory: fast intuitive (System 1) vs slow deliberate (System 2)
- Analogical reasoning as structure-mapping
- Symbol grounding via sensorimotor patterns
- Recursive self-reference as a component of meta-cognition

---

#### DOMAIN F: Reinforcement & Reward Signals

**Key Papers to Research:**
- Schultz, W. et al. (1997) — *A neural substrate of prediction and reward* — Science
- Dayan, P. & Niv, Y. (2008) — *Reinforcement learning: the good, the bad and the ugly* — Current Opinion in Neurobiology
- Sutton, R. & Barto, A. (2018) — *Reinforcement learning: an introduction* — MIT Press (Chapter 6 on TD learning)

**Mechanisms to Extract:**
- Dopaminergic reward prediction error (RPE)
- Temporal difference (TD) learning as a biological plausible credit assignment
- Intrinsic motivation / curiosity as reward for prediction error reduction
- Value functions as internal maps of future expected reward

---

#### DOMAIN G: Developmental & Evolutionary Constraints

**Key Papers to Research:**
- Piaget, J. — Object permanence and stage-based cognitive development
- Spelke, E.S. (2000) — *Core knowledge* — American Psychologist
- Marcus, G. (2001) — *The algebraic mind* — MIT Press
- Deacon, T. (1997) — *The symbolic species* — W.W. Norton (language and symbol use co-evolving with prefrontal cortex)

**Mechanisms to Extract:**
- Core knowledge systems as innate priors (geometry, numerosity, agency, object tracking)
- Developmental staging as curriculum (easy before hard)
- Symbol use as a late-emerging emergent behaviour, not a primitive

---

### 3.4 KNOWLEDGE BASE SYNTHESIS

After extracting all primitives, Cursor must produce `research/synthesis.md` answering:

1. What is the **minimal set of mechanisms** that, together, produce the appearance of reasoning?
2. Which mechanisms are most implementable in code with current tools?
3. What architecture should EIDOS use, and why?
4. What are the key risks or unknowns?

---

## 4. PHASE 2 — ARCHITECTURE

### 4.1 CHOSEN ARCHITECTURE: PREDICTIVE ACTIVE WORKSPACE (PAW)

Based on the research synthesis, EIDOS will implement the **Predictive Active Workspace** architecture — a combination of three frameworks that map cleanly to each other:

| Biological Framework | Implementation Component |
|---------------------|--------------------------|
| Predictive Processing | `PredictionEngine` — hierarchical prediction-error loop |
| Global Workspace Theory | `WorkspaceBuffer` — limited-slot active memory, globally readable |
| Complementary Learning Systems | `AssociationStore` (fast) + `BeliefGraph` (slow) |
| Attention as gating | `AttentionGate` — routes signals based on salience + top-down priors |
| Dopaminergic RPE | `RewardSignal` — intrinsic reward = surprise reduction |
| System 2 deliberation | `ReasoningLoop` — iterative hypothesis generation and testing |

### 4.2 DATA FLOW

```
[INPUT] → AttentionGate → WorkspaceBuffer
                              ↓
                        PredictionEngine
                        (predict next state)
                              ↓
                     compare → prediction_error
                              ↓
               ┌──────────────┴──────────────┐
               ↓                             ↓
        RewardSignal                  AssociationStore
        (intrinsic reward             (fast, episode-level
        = surprise reduction)          Hebbian update)
               ↓
        ReasoningLoop
        (if error > threshold:
         generate hypotheses,
         simulate, test, select)
               ↓
        BeliefGraph (slow, consolidated knowledge)
               ↓
        [OUTPUT / ACTION]
```

### 4.3 COMPONENT SPECIFICATIONS

---

#### `WorkspaceBuffer` — `/architecture/components/working_memory.py`

```
Purpose: Limited-capacity active memory. Represents the "stage" where information
is made globally available to all other components.

Capacity: configurable, default 7 slots (Miller's Law)
Slot structure: {content, salience, age, source}
Eviction policy: lowest salience + oldest age
Broadcast: any component can read all active slots
```

**Cursor prompt to implement:**
```
Implement WorkspaceBuffer in Python. It is a fixed-capacity data structure with N 
slots (default 7). Each slot holds a dict: {id, content, salience (float 0-1), 
age (int, incremented each cycle), source (str)}. On each cycle, age all slots by 1. 
When inserting a new item, if capacity is full, evict the slot with lowest 
salience * (1/age). Include methods: insert(content, salience, source), 
broadcast() -> list of all active slots, get_by_id(id), clear(). 
Add a __repr__ that shows a clean ASCII table of current slots.
```

---

#### `PredictionEngine` — `/architecture/components/prediction_engine.py`

```
Purpose: Hierarchical system that predicts the next state given current workspace 
contents. Computes prediction error. Updates internal model based on error.

Architecture: Two-layer hierarchy. 
Layer 1 (low): predicts immediate next input from current input.
Layer 2 (high): predicts Layer 1's predictions from abstract context.
Learning: online gradient descent on prediction error.
```

**Cursor prompt to implement:**
```
Implement PredictionEngine in Python using numpy only (no PyTorch). 
It has two layers. Each layer is a simple linear model: y = W @ x + b.
Weights are initialised randomly small. Forward pass: given input vector x, 
Layer 1 predicts x_hat. Layer 2 receives the abstract context (mean-pooled 
workspace content embeddings) and predicts Layer 1's internal representation.
Compute prediction_error = ||x - x_hat||^2. 
Backprop the error and update weights with SGD. 
Return: {prediction, prediction_error, updated_weights_flag}. 
Add a method get_surprise() which returns the rolling mean of recent 
prediction errors over a window of 20 steps.
```

---

#### `AssociationStore` — `/architecture/components/association_store.py`

```
Purpose: Fast episodic memory. Implements Hebbian-style association learning.
When two concepts co-activate in the workspace, their association strengthens.
Implements catastrophic forgetting protection via decay.

Structure: Weighted adjacency graph of concept nodes.
Learning rule: Hebbian — co-activation increases weight, time decreases it.
```

**Cursor prompt to implement:**
```
Implement AssociationStore in Python. Use a defaultdict to store a weighted 
adjacency graph: {concept_a: {concept_b: weight}}. 
Implement hebbian_update(active_concepts: list[str]) — for every pair of 
active concepts, increment their shared weight by a learning_rate (default 0.1). 
Each cycle, apply weight decay: multiply all weights by decay_factor (default 0.99).
Prune edges with weight < 0.01. 
Implement get_associates(concept, top_k=5) -> sorted list of (concept, weight).
Implement get_strongest_path(a, b) using BFS on the graph weighted by association.
Store all data in a plain dict so it can be serialised to JSON.
```

---

#### `AttentionGate` — `/architecture/components/attention_gate.py`

```
Purpose: Routes signals into the workspace. Computes salience score for each 
incoming signal. Uses both bottom-up (surprise) and top-down (goal relevance) 
attention signals.

Salience = α * surprise_score + β * goal_relevance_score
Default: α=0.6, β=0.4
```

**Cursor prompt to implement:**
```
Implement AttentionGate in Python. 
It accepts a list of incoming signals: [{id, content_vector, source}].
It has a current_goal vector (set externally by the agent).
For each signal, compute:
  - surprise_score: cosine distance between signal and current workspace mean vector
  - goal_relevance: cosine similarity between signal and current_goal
  - salience = alpha * surprise_score + beta * goal_relevance
Return a sorted list of signals by salience (highest first), 
with salience scores attached. 
Include set_goal(goal_vector) and get_salience_distribution() methods.
Use numpy for all vector operations.
```

---

#### `RewardSignal` — `/architecture/components/reward_signal.py`

```
Purpose: Intrinsic reward module. Implements reward = f(prediction_error_reduction).
The agent is intrinsically motivated to reduce surprise.
Secondary reward: successful completion of a goal state.

No external reward is defined for the prototype — pure intrinsic curiosity.
```

**Cursor prompt to implement:**
```
Implement RewardSignal in Python.
It tracks a history of prediction_error values (rolling window, size 50).
intrinsic_reward() returns: previous_error - current_error 
  (positive = surprise was reduced = good).
Add extrinsic_reward(achieved: bool) that returns +10.0 if achieved, 0.0 otherwise.
total_reward() = intrinsic_reward() + extrinsic_reward().
Add update(new_error: float, goal_achieved: bool) to step the signal forward.
Add get_reward_history() -> list of total rewards over time.
```

---

#### `ReasoningLoop` — `/architecture/components/reasoning_loop.py`

```
Purpose: Implements deliberate System 2-style reasoning. 
Triggered when prediction_error > threshold (surprise is high, fast system fails).
Mechanism: hypothesis generation → simulation → evaluation → selection.

This is the "slow thinking" module. It uses workspace contents and association store 
to generate candidate explanations, simulate their consequences, 
and select the one that minimises predicted error.
```

**Cursor prompt to implement:**
```
Implement ReasoningLoop in Python.
Constructor takes: workspace (WorkspaceBuffer), associations (AssociationStore), 
prediction_engine (PredictionEngine), threshold=0.5.

Method: run(current_error: float) -> selected_hypothesis or None
  If current_error < threshold: return None (fast system is fine)
  Else:
    1. Read all active workspace slots → active_concepts
    2. For each concept, get top-3 associates from AssociationStore
    3. Generate hypotheses: each hypothesis = a candidate new belief 
       (represented as a string label + a mock vector perturbation)
    4. For each hypothesis, simulate: feed perturbed input into PredictionEngine, 
       get predicted_error_if_hypothesis_true
    5. Select hypothesis with lowest predicted_error_if_hypothesis_true
    6. Return selected hypothesis as {label, confidence, predicted_error}
  
Log each reasoning step to a reasoning_trace list. 
Add get_trace() -> list of dicts describing each reasoning episode.
```

---

### 4.4 MAIN AGENT CLASS — `/agent/eidos.py`

**Cursor prompt:**
```
Implement the EIDOS agent class in /agent/eidos.py.

It composes all five components:
- self.workspace = WorkspaceBuffer(capacity=7)
- self.prediction = PredictionEngine(input_dim=64, hidden_dim=32)
- self.associations = AssociationStore()
- self.attention = AttentionGate(alpha=0.6, beta=0.4)
- self.reward = RewardSignal()
- self.reasoner = ReasoningLoop(workspace, associations, prediction, threshold=0.5)

Core loop method: step(raw_input: np.ndarray, input_label: str, goal: np.ndarray)
  1. AttentionGate scores the input against current goal
  2. If salience > 0.3: insert into workspace
  3. PredictionEngine predicts next state; compute prediction_error
  4. AssociationStore.hebbian_update(active workspace concept labels)
  5. RewardSignal.update(prediction_error, goal_achieved=False)
  6. If prediction_error > 0.5: ReasoningLoop.run(prediction_error)
  7. Return step_result: {salience, prediction_error, reward, hypothesis}

Add run_episode(inputs: list[dict], goal: np.ndarray) that runs N steps 
and returns a full episode log.

Add save_state(path) and load_state(path) that serialise the agent to JSON 
(weights as lists, association graph, reward history).
```

---

## 5. PHASE 3 — EXPERIMENTS

### Experiment 01: Basic Prediction
**File:** `experiments/exp_01_basic_prediction/run.py`

**Cursor prompt:**
```
Write an experiment in experiments/exp_01_basic_prediction/run.py.

Generate a sequence of 200 random input vectors (dim=64) with 
a hidden repeating pattern: every 5th vector is a slight variation 
of vector[0]. Run the EIDOS agent on this sequence with no explicit goal.
Plot prediction_error over time using matplotlib. 
The agent should show decreasing error as it learns the pattern.
Save the plot to experiments/exp_01_basic_prediction/results.png.
Print a summary: mean error first 50 steps vs last 50 steps.
```

---

### Experiment 02: Memory Consolidation
**File:** `experiments/exp_02_memory_consolidation/run.py`

**Cursor prompt:**
```
Write an experiment in experiments/exp_02_memory_consolidation/run.py.

Create 10 concept labels: ["cat", "dog", "tree", "car", "sky", 
"fire", "water", "stone", "leaf", "bird"].
Assign each a random 64-dim vector.
Run 100 steps where certain concepts co-occur frequently 
(cat+dog always appear together; fire+water never appear together).
After training, print the AssociationStore's top associates for "cat" 
and for "fire". Verify that "dog" is cat's top associate 
and "water" is NOT fire's top associate.
Export the full association graph to 
experiments/exp_02_memory_consolidation/association_graph.json.
```

---

### Experiment 03: Reasoning Under Surprise
**File:** `experiments/exp_03_reasoning_chain/run.py`

**Cursor prompt:**
```
Write an experiment in experiments/exp_03_reasoning_chain/run.py.

Set up the agent with a goal vector corresponding to the concept "fire".
Run 50 normal steps to warm up the agent.
Then inject a high-surprise input (random vector far from learned distribution).
Observe whether the ReasoningLoop activates (error > threshold).
Print the full reasoning_trace from the reasoning loop.
Show which hypothesis was selected and what its predicted_error was.
Save the reasoning trace to 
experiments/exp_03_reasoning_chain/reasoning_trace.json.
```

---

## 6. TESTS

**Cursor prompt:**
```
Write pytest tests for all five components in /tests/.

test_memory.py:
- Test WorkspaceBuffer inserts correctly
- Test eviction when at capacity (lowest salience/oldest evicted)
- Test broadcast returns all active slots

test_prediction.py:
- Test PredictionEngine forward pass returns a vector of correct shape
- Test prediction_error decreases after 50 training steps on a fixed input
- Test get_surprise() returns a float

test_association.py:
- Test hebbian_update strengthens co-activated pairs
- Test weight decay reduces weights over time
- Test get_associates returns sorted list

test_reasoning.py:
- Test ReasoningLoop returns None when error < threshold
- Test ReasoningLoop returns a hypothesis dict when error > threshold
- Test reasoning trace is non-empty after a triggered reasoning episode

test_full_agent.py:
- Test agent.step() returns a valid step_result dict
- Test agent.run_episode() returns a list of N step results
- Test agent.save_state() and load_state() round-trip correctly
```

---

## 7. REQUIREMENTS

**Cursor prompt:**
```
Create requirements.txt with:
numpy
matplotlib
pytest
```

*All agent code must use only numpy, the Python standard library, and matplotlib for plotting. No PyTorch, no TensorFlow, no Scikit-learn. This is intentional — EIDOS must be fully transparent and hand-implemented.*

---

## 8. README

**Cursor prompt:**
```
Write README.md for the EIDOS project. Include:
- One-paragraph description of what EIDOS is and why it is different from LLMs
- The biological frameworks it draws from (Predictive Processing, Global Workspace Theory, CLS, Hebbian Learning)
- Folder structure overview
- How to install (pip install -r requirements.txt)
- How to run experiments (python experiments/exp_01_basic_prediction/run.py)
- How to run tests (pytest tests/)
- A section called "Cognitive Primitives" listing all extracted mechanisms and their sources
- A section called "What This Is Not" (not AGI, not an LLM replacement, a laboratory tool for understanding cognition computationally)
- A Kisamapa Labs footer
```

---

## 9. ARCHITECTURE DIAGRAM

**Cursor prompt:**
```
Create architecture/diagrams/architecture.mermaid with a flowchart 
showing the full EIDOS data flow:
- INPUT node
- AttentionGate
- WorkspaceBuffer (with 7 slots shown)
- PredictionEngine (2 layers)
- prediction_error branch: low error → AssociationStore only; 
  high error → ReasoningLoop → then AssociationStore
- RewardSignal receiving from prediction_error
- BeliefGraph (output of slow consolidation)
- OUTPUT/ACTION node
Label all edges with what flows across them.
```

---

## 10. CURSOR MASTER PROMPT — START HERE

When beginning this project, feed Cursor this prompt first:

```
You are building EIDOS — a small, biologically-grounded reasoning AI for Kisamapa Labs.
This is not an LLM. It does not predict tokens. 
It is built from cognitive science primitives: predictive processing, 
global workspace theory, Hebbian learning, and intrinsic curiosity reward.

Your first task is PHASE 1 — RESEARCH.

1. Create the folder structure as specified in SPEC.md.
2. Research the following papers and frameworks. 
   For each, extract the core implementable mechanism and store it 
   as a JSON entry in research/primitives/[category].json 
   using the schema defined in SPEC.md Section 3.2:

   PREDICTIVE PROCESSING:
   - Friston (2010) free-energy principle
   - Friston et al. (2017) active inference process theory
   - Rao & Ballard (1999) predictive coding visual cortex
   
   WORKING MEMORY / GLOBAL WORKSPACE:
   - Baddeley (2003) working memory
   - Baars (1988) global workspace theory
   - Dehaene et al. (1998) neuronal global workspace model
   
   LEARNING / MEMORY CONSOLIDATION:
   - Hebb (1949) organisation of behaviour
   - McClelland et al. (1995) complementary learning systems
   - Wilson & McNaughton (1994) hippocampal sleep replay
   
   ATTENTION:
   - Corbetta & Shulman (2002) goal-directed attention
   - Desimone & Duncan (1995) biased competition model
   
   REASONING:
   - Kahneman (2011) dual-process theory (System 1 / System 2)
   - Gentner (1983) structure-mapping analogy theory
   
   REWARD:
   - Schultz et al. (1997) reward prediction error
   - TD learning from Sutton & Barto (2018) Ch. 6

3. After extracting all primitives, write research/synthesis.md 
   answering these four questions:
   - What is the minimal mechanism set for reasoning-like behaviour?
   - Which mechanisms are most directly implementable?
   - What is your proposed architecture for EIDOS?
   - What are the key risks?

4. When PHASE 1 is complete, proceed to PHASE 2 and implement each 
   component as specified in SPEC.md Sections 4.2–4.4.

5. Then run the three experiments and report results.

Do not use any external LLM APIs. Do not install PyTorch or TensorFlow.
Use only numpy, matplotlib, and the Python standard library.
Build everything from first principles.
```

---

## 11. EVALUATION CRITERIA

The experiment is successful if:

| Criterion | Target |
|-----------|--------|
| Prediction error decreases over Exp 01 | >20% reduction mean error, first 50 vs last 50 steps |
| Association graph correctly captures co-occurrence | "dog" is cat's #1 associate after training |
| ReasoningLoop activates on high-surprise input | Trace is non-empty, hypothesis returned |
| All tests pass | `pytest tests/` → 0 failures |
| Full save/load round-trip works | Agent state survives serialisation |

---

## 12. FUTURE EXTENSIONS (DO NOT BUILD NOW)

- Add a language interface layer: map natural language tokens to concept vectors so the agent can "read" and "think" about text
- Implement offline consolidation pass (sleep replay) as a batch update over AssociationStore → BeliefGraph
- Add meta-cognition: the agent monitors its own reasoning_trace and flags inconsistencies
- Multi-agent extension: two EIDOS agents share a workspace and develop communication conventions (links to Babel Worlds)
- Integrate spiking neural network substrate instead of continuous vectors (closer to biological plausibility)

---

*KISAMAPA LABS — EXPERIMENT 06 — EIDOS v1.0*
*Classification: Research Prototype*
*Status: Specification Complete — Ready for Cursor*
