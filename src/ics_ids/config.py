# Path setup only — repo was reorganized so config.py now lives at
# src/ics_ids/config.py instead of ics_model_eval/config.py. All paths below
# point to the same relative structure as before, just anchored to the new
# repo root instead of the old ics_model_eval/ folder.
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
REPO_ROOT = BASE_DIR.parent.parent  # src/ics_ids -> src -> repo root

DATA_ROOT_DIR = REPO_ROOT / "data" / "processed"
WITH_TIME_DATA_DIR = DATA_ROOT_DIR / "with-time"
NO_TIME_DATA_DIR = DATA_ROOT_DIR / "no_time"

DATA_DIR = NO_TIME_DATA_DIR
RESULTS_DIR = REPO_ROOT / "results"
MODELS_DIR = RESULTS_DIR / "models"
METRICS_DIR = RESULTS_DIR / "metrics"
FIGURES_DIR = RESULTS_DIR / "figures"
PREDICTIONS_DIR = RESULTS_DIR / "predictions"


TIME_LEAKAGE_COLS = [
    "startOffset",
    "endOffset",
    "raw_start_time",
    "raw_end_time",
    "event_time_seconds",
    "event_end_seconds",
    "start_seconds_of_day",
    "end_seconds_of_day",
    "timestamp_seconds_of_day",
    "Timestamp_seconds_of_day",
    "Time_seconds_of_day",
    "time_seconds_of_day",
]

ABLATION_DIR = RESULTS_DIR / "time_ablation"
HYPERPARAM_DIR = RESULTS_DIR / "hyperparameter_study"
ROBUSTNESS_DIR = RESULTS_DIR / "robustness"
DIAGNOSTICS_DIR = RESULTS_DIR / "dataset_diagnostics"

SEEDS = [42, 7, 21, 100, 2026]

for d in [MODELS_DIR, METRICS_DIR, FIGURES_DIR, PREDICTIONS_DIR, ABLATION_DIR, HYPERPARAM_DIR, ROBUSTNESS_DIR, DIAGNOSTICS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

BINARY_TRAIN_CSV = DATA_DIR / "binary_detection_train.csv"
BINARY_TEST_CSV = DATA_DIR / "binary_detection_test.csv"
MULTICLASS_TRAIN_CSV = DATA_DIR / "multiclass_classification_train.csv"
MULTICLASS_TEST_CSV = DATA_DIR / "multiclass_classification_test.csv"

MULTICLASS_LABEL_MAP_CSV_CANDIDATES = [
    DATA_DIR / "multiclass_label_map.json",
]

BINARY_LABEL_COL = "binary_label"
MULTICLASS_LABEL_COL = "multiclass_label"

RANDOM_STATE = 42
