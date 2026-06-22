"""EIDOS agent configuration."""

INPUT_DIM = 64
HIDDEN_DIM = 32
WORKSPACE_CAPACITY = 7
ATTENTION_ALPHA = 0.6
ATTENTION_BETA = 0.4
SALIENCE_THRESHOLD = 0.3

# v1.1: relative surprise (replaces fixed REASONING_THRESHOLD for triggering)
SURPRISE_WINDOW = 20
SURPRISE_SPIKE_RATIO = 2.0
SURPRISE_STD_MULTIPLIER = 2.0
SURPRISE_MIN_HISTORY = 5

# Legacy absolute floor — reasoning never triggers below this even if relative spike
REASONING_ABSOLUTE_FLOOR = 0.5

HYPOTHESIS_BLEND_WEIGHT = 0.5
HYPOTHESIS_ASSOCIATION_BOOST = 0.3
HYPOTHESIS_PERSISTENCE_STEPS = 3

# v1.2: belief consolidation — reasoning updates prediction weights directly
BELIEF_CONSOLIDATION_STEPS = 25
BELIEF_CONSOLIDATION_LR = 0.02
CONSOLIDATION_PREVIEW_STEPS = 8
SURPRISE_CORRUPTION_LR = 0.08

PREDICTION_LEARNING_RATE = 0.01
HEBBIAN_LEARNING_RATE = 0.1
ASSOCIATION_DECAY = 0.99
RANDOM_SEED = 42

# v1.4: autonomous recovery probe from recent episodic context
RECENT_HISTORY_WINDOW = 20

# v2.0: complementary learning systems — episodic buffer + belief graph + sleep replay
EPISODIC_BUFFER_CAPACITY = 500
SLEEP_REPLAY_COUNT = 50
SLEEP_REPLAY_INTEGRATE_WEIGHT = 1.0
SLEEP_ASSOCIATION_BOOST = 0.05
SLEEP_CONSOLIDATE_TOP_K = 2
SLEEP_CONSOLIDATE_STEPS = 10
SLOW_STORE_CONFLICT_RATIO = 1.5

# v3.0: meta-cognition — misleading context + reasoning quality monitoring
MISLEADING_CONTEXT_RATIO = 2.0
META_LOW_CONFIDENCE = 0.4
CLOSE_HYPOTHESIS_EPSILON = 0.5

# v3.1: consequential meta-cognition — act on flags (defer / auto-sleep)
META_AUTO_SLEEP_REPLAYS = 25
META_AUTO_SLEEP_ON_MISLEADING = True
META_AUTO_SLEEP_ON_AMBIGUITY = True

# v4.0: active inference — action selection via expected free energy (Friston 2017)
EFE_EPISTEMIC_WEIGHT = 1.0
EFE_PRAGMATIC_WEIGHT = 0.6
ACTIVE_PROBE_NOISE_STD = 0.03
ACTIVE_SLEEP_EPISTEMIC_BONUS = 0.15
ACTIVE_BELIEF_STRENGTH_THRESHOLD = 0.05
ACTIVE_SLEEP_REPLAYS = 25

# v5.0: text grounding bridge — hash embeddings (no GPU / no torch)
TEXT_GROUNDING_SEED = 42
TEXT_GROUNDING_BUCKETS = 256
TEXT_ANOMALY_LABEL = "text_anomaly"

# v6.0: unified gate policy — text alignment + cognitive fusion
GATE_MIN_DRAFT_GOAL_ALIGN = 0.82
GATE_CONCEPT_AMBIGUITY_EPS = 0.08
SBERT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# v7.0: hybrid sidecar defaults + metacognitive injection
HYBRID_DEFAULT_EMBEDDING = "sbert"
META_INJECTION_MAX_ROUNDS = 2

# v7.1: live Groq eval + belief-grounded prompts
GROQ_DEFAULT_MODEL = "llama-3.3-70b-versatile"
GROQ_DEFAULT_BASE_URL = "https://api.groq.com/openai/v1"
BELIEF_CONTEXT_TOP_K = 3
GATE_DRAFT_CONCEPT_MISMATCH = True

# Deprecated alias kept for backward compatibility in tests/docs
REASONING_THRESHOLD = REASONING_ABSOLUTE_FLOOR
