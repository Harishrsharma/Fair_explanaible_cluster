# config.py
# Central configuration for all datasets and experiment parameters

RANDOM_STATE = 42
K_CLUSTERS = 5
TREE_MAX_DEPTH = 4
TEST_SIZE = 0.3

# Fairness balance parameters (p/q ratio)
# Balance condition: min(r/b, b/r) >= p/q
# p=1, q=2 means at least 1 minority for every 2 majority members per cluster
FAIR_P = 1
FAIR_Q = 2

# Quadtree parameters (Twagner method)
QUADTREE_MAX_LEVELS = 0        # 0 = full recursive split
QUADTREE_RANDOM_SHIFT = True   # reduces worst-case quadtree behavior
QUADTREE_EPSILON = 1e-4        # minimum cell size threshold

# Sample sizes per dataset (None = use full dataset)
SAMPLE_SIZES = {
    "bank":    30000,
    "adult":   30000,
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

# Results output directory
RESULTS_DIR = "results"
