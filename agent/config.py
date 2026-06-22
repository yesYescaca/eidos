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

# Deprecated alias kept for backward compatibility in tests/docs
REASONING_THRESHOLD = REASONING_ABSOLUTE_FLOOR
