# config.py
# Central configuration for all datasets and experiment parameters

RANDOM_STATE = 42
TREE_MAX_DEPTH = 4
TEST_SIZE = 0.3

# ── Elbow-method k search range ───────────────────────────────────────────────
# find_optimal_k() tries every k in [K_RANGE[0], K_RANGE[1]) and picks the
# elbow. Override per-dataset with DATASET_K_OVERRIDE (None = auto).
K_RANGE = (2, 11)

# Per-dataset k override: set to an integer to skip elbow, None to auto-detect
DATASET_K_OVERRIDE = {
    "bank":   None,   # auto
    "adult":  None,   # auto
    "compas": None,   # auto
    "german": None,   # auto
}

# Quadtree parameters (Twagner method)
QUADTREE_MAX_LEVELS  = 0       # 0 = full recursive split
QUADTREE_RANDOM_SHIFT = True   # toroidal shift – randomises grid origin
                               # (Theorem 3.2, Chierichetti et al. 2017)
                               # Fixed: now uses modular arithmetic so data
                               # stays within the bounding box at all times.
QUADTREE_EPSILON = 1e-4        # minimum cell size threshold

# Sample sizes per dataset (None = use full dataset)
SAMPLE_SIZES = {
    "bank":    7000,
    "adult":   7000,
    "compas":  None,   # ~7000 rows, use all
    "german":  None,   # ~1000 rows, use all
}

# Dataset file paths
DATASETS = {
    "bank":   "data/bank-full.csv",
    "adult":  "data/adult-clean.csv",
    "compas": "data/compas-scores-two-years_clean.csv",
    "german": "data/german_data_credit.csv",
}

# Sensitive attribute configuration per dataset
# Aligned with thesis proposal (Harish Sharma, Nov 2025)
# Reference: Adult=gender/race, COMPAS=race/sex, Bank=age_group, German=sex
SENSITIVE_CONFIGS = {
    "bank": {
        "column":      "age",
        "type":        "threshold",   # age > threshold -> 1 (older group)
        "threshold":   35,
        "description": "Age group: older (>35) = 1, younger = 0"
    },
    "adult": {
        "column":      "gender",
        "type":        "binary_map",
        "map":         {"Male": 1, "Female": 0},
        "description": "Gender: Male = 1, Female = 0"
    },
    "compas": {
        "column":      "race",
        "type":        "binary_map",
        "map":         {"African-American": 1},  # all others -> 0
        "default":     0,
        "description": "Race: African-American = 1, others = 0"
    },
    "german": {
        "column":      "sex",
        "type":        "binary_map",
        "map":         {"male": 1, "female": 0},
        "description": "Sex: male = 1, female = 0"
    },
}

# Bounded-representation assignment method
# True  → LP-based assignment (scipy HiGHS; exact proportional fairness)
#          n ≤ 15 000: LP every EM iteration   (german, compas)
#          n >  15 000: greedy EM + single LP final pass (adult, bank)
# False → greedy constrained assignment (fast, approximate; may produce
#          degenerate clusters on skewed/large datasets)
BOUNDED_REP_USE_LP = True

# Results output directory
RESULTS_DIR = "results"
