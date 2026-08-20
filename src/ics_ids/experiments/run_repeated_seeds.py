import argparse
import ast
import json
import time
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import RobustScaler, label_binarize
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_curve, auc,
)
from sklearn.utils.class_weight import compute_sample_weight
try:
    from xgboost import XGBClassifier
except ImportError:
    XGBClassifier = None
from .. import config
from .. import utils

DEFAULT_SEEDS = [42, 7, 21, 100, 2026]
MODEL_DISPLAY = {"random_forest": "Random Forest", "xgboost": "XGBoost", "mlp": "MLP"}
TASK_DISPLAY = {"binary": "Binary", "multiclass": "Multiclass"}

MODEL_ORDER = ["mlp", "random_forest", "xgboost"]
MODEL_COLORS = {
    "mlp": "#1f77b4",            
    "random_forest": "#ff7f0e",  
    "xgboost": "#2ca02c",       
}

METRICS_FOR_SUMMARY = [
    "accuracy", "precision", "recall", "f1_score", "balanced_accuracy", "false_positive_rate", "roc_auc", "pr_auc",
    "macro_precision", "macro_recall", "macro_f1", "weighted_f1", "micro_roc_auc", "macro_roc_auc", "weighted_roc_auc",
    "training_time_seconds", "total_runtime_seconds"
]

METRIC_LABELS = {
    "accuracy": "Accuracy",
    "precision": "Precision",
    "recall": "Recall",
    "f1_score": "F1-score",
    "false_positive_rate": "False positive rate",
    "roc_auc": "ROC-AUC",
    "macro_precision": "Macro precision",
    "macro_recall": "Macro recall",
    "macro_f1": "Macro F1",
    "weighted_f1": "Weighted F1",
    "micro_roc_auc": "Micro ROC-AUC",
    "macro_roc_auc": "Macro ROC-AUC",
    "weighted_roc_auc": "Weighted ROC-AUC",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Repeated-seed robustness study using tuned hyperparameters.")
    parser.add_argument("--models", nargs="+", default=["random_forest", "xgboost", "mlp"], choices=["random_forest", "xgboost", "mlp"])
    parser.add_argument("--tasks", nargs="+", default=["binary", "multiclass"], choices=["binary", "multiclass"])
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--data-dir", default=str(config.TIME_ABLATED_DATA_DIR), help="Dataset directory. Default: time-ablated benchmark.")
    parser.add_argument("--best-params", default=str(config.RESULTS_DIR / "hyperparameter_study" / "best_hyperparameters.json"))
    parser.add_argument("--output-dir", default=str(config.RESULTS_DIR / "robustness"))
    parser.add_argument("--plot-only", action="store_true", help="Regenerate repeated-seed figures from existing CSV files without retraining.")
    return parser.parse_args()


def clean_value(value):
    if value is None:
        return None
    if isinstance(value, float) and np.isnan(value):
        return None
    if isinstance(value, str):
        v = value.strip()
        if v == "" or v.lower() in {"none", "nan", "null"}:
            return None
        if v.lower() == "true":
            return True
        if v.lower() == "false":
            return False
        if (v.startswith("(") and v.endswith(")")) or (v.startswith("[") and v.endswith("]")):
            parsed = ast.literal_eval(v)
            return tuple(parsed) if isinstance(parsed, list) else parsed
        try:
            if any(ch in v for ch in [".", "e", "E"]):
                return float(v)
            return int(v)
        except ValueError:
            return v
    return value


def clean_params(params):
    cleaned = {}
    for k, v in dict(params).items():
        if str(k).startswith("_"):
            continue
        cleaned[k] = clean_value(v)
    return cleaned


def load_best_params(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Missing best hyperparameter file: {path}")
    raw = json.loads(path.read_text())
    best = {}
    for _, item in raw.items():
        model = item.get("model")
        task = item.get("task")
        if model and task:
            best[(model, task)] = clean_params(item.get("params", {}))
    return best


def load_full_dataset(data_dir, task):
    data_dir = Path(data_dir)
    full_path = data_dir / "full_labeled_flow_dataset.csv"
    if not full_path.exists():
        raise FileNotFoundError(f"Missing full dataset file: {full_path}")

    if task == "binary":
        feature_path = data_dir / "binary_detection_feature_list.json"
        label_col = config.BINARY_LABEL_COL
    else:
        feature_path = data_dir / "multiclass_classification_feature_list.json"
        label_col = config.MULTICLASS_LABEL_COL

    features = json.loads(feature_path.read_text())
    df = pd.read_csv(full_path)

    missing = [c for c in features + [label_col] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in {full_path}: {missing}")

    X = df[features].copy()
    y = df[label_col].copy()
    return X, y, features


def split_and_scale(X, y, seed):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=seed
    )
    scaler = RobustScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X.columns, index=X_train.index)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X.columns, index=X_test.index)
    return X_train_scaled, X_test_scaled, y_train, y_test


def build_random_forest(params, task, seed):
    p = clean_params(params)
    p.setdefault("n_estimators", 300)
    p["random_state"] = seed
    p.setdefault("n_jobs", -1)
    p.setdefault("class_weight", "balanced_subsample")
    return RandomForestClassifier(**p)


def build_xgboost(params, task, seed, y_train):
    if XGBClassifier is None:
        raise ImportError("xgboost is not installed.")
    p = clean_params(params)
    p["random_state"] = seed
    p.setdefault("n_jobs", -1)
    p.setdefault("eval_metric", "logloss" if task == "binary" else "mlogloss")
    p.setdefault("tree_method", "hist")

    if task == "binary":
        p.setdefault("objective", "binary:logistic")
        pos = int((y_train == 1).sum())
        neg = int((y_train == 0).sum())
        if pos > 0:
            p["scale_pos_weight"] = neg / pos
    else:
        p.setdefault("objective", "multi:softprob")
        p["num_class"] = int(pd.Series(y_train).nunique())

    return XGBClassifier(**p)


def build_mlp(params, task, seed):
    p = clean_params(params)
    p.setdefault("hidden_layer_sizes", (128, 64))
    p.setdefault("activation", "relu")
    p.setdefault("solver", "adam")
    p.setdefault("max_iter", 100)
    p["random_state"] = seed
    p.setdefault("early_stopping", True)
    p.setdefault("validation_fraction", 0.1)
    p.setdefault("n_iter_no_change", 10)
    p.setdefault("verbose", False)
    return MLPClassifier(**p)


def train_predict(model_key, task, params, X_train, y_train, X_test):
    seed = int(params.get("_seed", 42))
    if model_key == "random_forest":
        model = build_random_forest(params, task, seed=seed)
        model.fit(X_train, y_train)
    elif model_key == "xgboost":
        model = build_xgboost(params, task, seed=seed, y_train=y_train)
        if task == "multiclass":
            sample_weights = compute_sample_weight(class_weight="balanced", y=y_train)
            model.fit(X_train, y_train, sample_weight=sample_weights)
        else:
            model.fit(X_train, y_train)
    elif model_key == "mlp":
        model = build_mlp(params, task, seed=seed)
        model.fit(X_train, y_train)
    else:
        raise ValueError(f"Unknown model: {model_key}")

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test) if hasattr(model, "predict_proba") else None
    return model, y_pred, y_prob


def run_one(model_key, task, base_params, seed, data_dir, cached_data, pred_dir=None):
    params = dict(base_params)
    params["_seed"] = seed
    X, y, features = cached_data[task]
    X_train, X_test, y_train, y_test = split_and_scale(X, y, seed)

    start = time.time()
    model, y_pred, y_prob = train_predict(model_key, task, params, X_train, y_train, X_test)
    training_time = time.time() - start

    if task == "binary":
        y_score = y_prob[:, 1] if y_prob is not None and y_prob.ndim == 2 else None
        metrics = utils.binary_metrics(y_test, y_pred, y_score)
        metrics["balanced_accuracy"] = float((metrics["recall"] + (1.0 - metrics["false_positive_rate"])) / 2.0)
        metrics["pr_auc"] = float(average_precision_score(y_test, y_score)) if y_score is not None else None
        if pred_dir is not None and y_score is not None:
            save_binary_prediction_scores(pred_dir, model_key, seed, y_test, y_pred, y_score)
        objective_metric = "f1_score"
        objective_score = metrics["f1_score"]
    else:
        labels = sorted(pd.unique(y))
        metrics = utils.multiclass_metrics(y_test, y_pred, labels=labels, y_prob=y_prob, data_dir=data_dir)
        objective_metric = "macro_f1"
        objective_score = metrics["macro_f1"]
        # Save the already-computed per-class probabilities so the final
        # multiclass micro-average ROC figure can be reproduced without
        # retraining. This does not change what train_predict computes.
        if pred_dir is not None and y_prob is not None:
            class_labels = list(getattr(model, "classes_", labels))
            save_multiclass_prediction_probabilities(pred_dir, model_key, seed, y_test, y_pred, y_prob, class_labels)

    metrics.update({
        "model": model_key,
        "model_display": MODEL_DISPLAY[model_key],
        "task": task,
        "task_display": TASK_DISPLAY[task],
        "seed": seed,
        "seed_label": f"Seed {seed}",
        "benchmark": "time_ablated",
        "n_features": len(features),
        "objective_metric": objective_metric,
        "objective_score": float(objective_score),
        "training_time_seconds": float(training_time),
        "total_runtime_seconds": float(training_time),
    })
    return metrics


def flatten_metrics(metrics):
    skip = {"confusion_matrix", "confusion_matrix_labels", "label_order"}
    return {k: v for k, v in metrics.items() if k not in skip}



def add_derived_binary_metrics(results_df):
    out = results_df.copy()
    if "task" in out.columns and "balanced_accuracy" not in out.columns:
        out["balanced_accuracy"] = np.nan
    if {"task", "recall", "false_positive_rate"}.issubset(out.columns):
        mask = out["task"].eq("binary") & out["recall"].notna() & out["false_positive_rate"].notna()
        out.loc[mask, "balanced_accuracy"] = (out.loc[mask, "recall"] + (1.0 - out.loc[mask, "false_positive_rate"])) / 2.0
    return out

def summarize_results(results_df):
    rows = []
    for (model, task), group in results_df.groupby(["model", "task"]):
        row = {
            "model": model,
            "model_display": MODEL_DISPLAY.get(model, model),
            "task": task,
            "task_display": TASK_DISPLAY.get(task, task),
            "n_runs": len(group),
            "seeds": ", ".join(str(int(s)) for s in group["seed"].tolist()),
        }
        for metric in METRICS_FOR_SUMMARY + ["objective_score"]:
            if metric in group.columns:
                vals = pd.to_numeric(group[metric], errors="coerce").dropna()
                if len(vals) > 0:
                    row[f"{metric}_mean"] = float(vals.mean())
                    row[f"{metric}_std"] = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
                    row[f"{metric}_min"] = float(vals.min())
                    row[f"{metric}_max"] = float(vals.max())
        rows.append(row)
    return pd.DataFrame(rows)


def make_paper_table(summary, out_path):
    rows = []
    for _, r in summary.iterrows():
        task = r["task"]
        if task == "binary":
            selected = ["accuracy", "precision", "recall", "f1_score", "balanced_accuracy", "pr_auc", "false_positive_rate"]
        else:
            selected = ["accuracy", "macro_precision", "macro_recall", "macro_f1", "weighted_f1", "micro_roc_auc"]
        row = {"Model": r["model_display"], "Task": r["task_display"], "Runs": int(r["n_runs"])}
        for metric in selected:
            mean_col = f"{metric}_mean"
            std_col = f"{metric}_std"
            if mean_col in r and pd.notna(r[mean_col]):
                row[METRIC_LABELS.get(metric, metric)] = f"{r[mean_col] * 100:.2f} ± {r[std_col] * 100:.2f}%"
        rows.append(row)
    paper = pd.DataFrame(rows)
    paper.to_csv(out_path, index=False)
    return paper



def save_binary_prediction_scores(pred_dir, model_key, seed, y_true, y_pred, y_score):
    pred_dir = Path(pred_dir)
    pred_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame({
        "model": model_key,
        "seed": int(seed),
        "y_true": np.asarray(y_true),
        "y_pred": np.asarray(y_pred),
        "y_score": np.asarray(y_score) if y_score is not None else np.nan,
    })
    out_path = pred_dir / f"{model_key}_binary_seed{seed}_predictions.csv"
    df.to_csv(out_path, index=False)


def save_multiclass_prediction_probabilities(pred_dir, model_key, seed, y_true, y_pred, y_prob, class_labels):
    """Save the already-computed y_true/y_pred/per-class probabilities for a
    multiclass repeated-seed run, so the final multiclass micro-average ROC
    figure can be reproduced from disk without retraining. Does not affect
    any prediction, metric, or result calculation.
    """
    pred_dir = Path(pred_dir)
    pred_dir.mkdir(parents=True, exist_ok=True)
    data = {
        "model": model_key,
        "seed": int(seed),
        "y_true": np.asarray(y_true),
        "y_pred": np.asarray(y_pred),
    }
    y_prob = np.asarray(y_prob)
    for idx, label in enumerate(class_labels):
        data[f"class_{label}_probability"] = y_prob[:, idx]
    df = pd.DataFrame(data)
    out_path = pred_dir / f"{model_key}_multiclass_seed{seed}_probabilities.csv"
    df.to_csv(out_path, index=False)


# =========================================================
# Final repeated-seed paper figures
# =========================================================
def mean_roc_curve_from_files(files):
    fpr_grid = np.linspace(0.0, 1.0, 400)
    tpr_rows = []
    auc_scores = []

    for file_path in files:
        df = pd.read_csv(file_path)
        if "y_true" not in df.columns or "y_score" not in df.columns:
            continue

        y_true = df["y_true"].astype(int).to_numpy()
        y_score = df["y_score"].astype(float).to_numpy()

        fpr, tpr, _ = roc_curve(y_true, y_score)
        roc_auc = auc(fpr, tpr)

        tpr_interp = np.interp(fpr_grid, fpr, tpr)
        tpr_interp[0] = 0.0
        tpr_rows.append(tpr_interp)
        auc_scores.append(roc_auc)

    if not tpr_rows:
        return None

    tpr_arr = np.vstack(tpr_rows)
    mean_tpr = tpr_arr.mean(axis=0)
    mean_tpr[-1] = 1.0

    return {
        "fpr_grid": fpr_grid,
        "tpr_mean": mean_tpr,
        "tpr_std": tpr_arr.std(axis=0, ddof=1) if tpr_arr.shape[0] > 1 else np.zeros_like(fpr_grid),
        "auc_mean": float(np.mean(auc_scores)),
        "auc_std": float(np.std(auc_scores, ddof=1)) if len(auc_scores) > 1 else 0.0,
    }


def mean_pr_curve_from_files(files):
    recall_grid = np.linspace(0.0, 1.0, 400)
    precision_rows = []
    ap_scores = []

    for file_path in files:
        df = pd.read_csv(file_path)
        if "y_true" not in df.columns or "y_score" not in df.columns:
            continue

        y_true = df["y_true"].astype(int).to_numpy()
        y_score = df["y_score"].astype(float).to_numpy()

        precision, recall, _ = precision_recall_curve(y_true, y_score)
        pr_auc = average_precision_score(y_true, y_score)

        recall_asc = recall[::-1]
        precision_asc = precision[::-1]

        precision_interp = np.interp(recall_grid, recall_asc, precision_asc)
        precision_rows.append(precision_interp)
        ap_scores.append(pr_auc)

    if not precision_rows:
        return None

    precision_arr = np.vstack(precision_rows)

    return {
        "recall_grid": recall_grid,
        "precision_mean": precision_arr.mean(axis=0),
        "precision_std": precision_arr.std(axis=0, ddof=1) if precision_arr.shape[0] > 1 else np.zeros_like(recall_grid),
        "ap_mean": float(np.mean(ap_scores)),
        "ap_std": float(np.std(ap_scores, ddof=1)) if len(ap_scores) > 1 else 0.0,
    }


def plot_supervised_binary_roc_pr(pred_dir, fig_dir):
    """Final paper ROC/PR figure for binary repeated-seed results.

    Same implementation/style as the former paper_figures.plot_supervised_binary_roc_pr():
    400-point interpolation, mean curves with std bands across seeds, exact
    model colors, ROC-AUC/PR-AUC mean +/- SD, 300 DPI.
    """
    pred_dir = Path(pred_dir)
    fig_dir = Path(fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(8.8, 4.8))
    ax_roc, ax_pr = axes

    any_curve = False

    for model_key in MODEL_ORDER:
        files = sorted(pred_dir.glob(f"{model_key}_binary_seed*_predictions.csv"))

        if not files:
            print(f"[skip] no repeated-seed binary files found for {model_key}")
            continue

        roc_info = mean_roc_curve_from_files(files)
        pr_info = mean_pr_curve_from_files(files)

        if roc_info is None or pr_info is None:
            continue

        color = MODEL_COLORS[model_key]
        model_name = MODEL_DISPLAY[model_key]

        roc_label = f"{model_name}, ROC-AUC={roc_info['auc_mean']:.3f}±{roc_info['auc_std']:.3f}"
        ax_roc.plot(roc_info["fpr_grid"], roc_info["tpr_mean"], color=color, linewidth=2, label=roc_label)
        ax_roc.fill_between(
            roc_info["fpr_grid"],
            np.maximum(0, roc_info["tpr_mean"] - roc_info["tpr_std"]),
            np.minimum(1, roc_info["tpr_mean"] + roc_info["tpr_std"]),
            color=color, alpha=0.10
        )

        pr_label = f"{model_name}, PR-AUC={pr_info['ap_mean']:.3f}±{pr_info['ap_std']:.3f}"
        ax_pr.plot(pr_info["recall_grid"], pr_info["precision_mean"], color=color, linewidth=2, label=pr_label)
        ax_pr.fill_between(
            pr_info["recall_grid"],
            np.maximum(0, pr_info["precision_mean"] - pr_info["precision_std"]),
            np.minimum(1, pr_info["precision_mean"] + pr_info["precision_std"]),
            color=color, alpha=0.10
        )

        any_curve = True

    if not any_curve:
        print("[error] no binary repeated-seed prediction files were found.")
        plt.close(fig)
        return

    ax_roc.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)
    ax_roc.set_title("ROC Curve", fontsize=11)
    ax_roc.set_xlabel("False Positive Rate")
    ax_roc.set_ylabel("True Positive Rate")
    ax_roc.set_xlim(0, 1)
    ax_roc.set_ylim(0, 1.02)
    ax_roc.grid(alpha=0.35)
    ax_roc.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), frameon=True, fontsize=8.5)

    ax_pr.set_title("PR Curve", fontsize=11)
    ax_pr.set_xlabel("Recall")
    ax_pr.set_ylabel("Precision")
    ax_pr.set_xlim(0, 1)
    ax_pr.set_ylim(0, 1.02)
    ax_pr.grid(alpha=0.35)
    ax_pr.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), frameon=True, fontsize=8.5)

    fig.suptitle("Binary Attack Detection - ROC Curve (left) and PR Curve (right)", fontsize=12, y=0.98)
    fig.subplots_adjust(top=0.87, bottom=0.27, wspace=0.28)

    save_path = fig_dir / "supervised_binary_roc_pr_curves.png"
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {save_path}")


def resolve_multiclass_probability_csv_for_model(model_key, out_dir):
    out_dir = Path(out_dir)
    results_path = out_dir / "repeated_seed_results.csv"
    if not results_path.exists():
        print(f"[skip] missing {results_path}; cannot resolve multiclass probabilities for {model_key}")
        return None
    df = pd.read_csv(results_path)
    sub = df[(df["task"] == "multiclass") & (df["model"] == model_key)].copy()
    if sub.empty:
        print(f"[skip] no multiclass repeated-seed results found for {model_key}")
        return None
    best_row = sub.sort_values("macro_f1", ascending=False).iloc[0]
    best_seed = int(best_row["seed"])
    probability_path = out_dir / "predictions" / f"{model_key}_multiclass_seed{best_seed}_probabilities.csv"
    if not probability_path.exists():
        print(f"[skip] missing multiclass probability file: {probability_path}")
        return None
    print(f"[info] using repeated-seed multiclass probability file: {probability_path.name}")
    return probability_path


def plot_supervised_multiclass_micro_roc(out_dir, figure_dir):
    """
    Plot micro-average multiclass ROC curves for the supervised models.

    Uses the saved multiclass probability outputs and reproduces the
    final visualization style used for the supervised multiclass ROC figure.
    """
    out_dir = Path(out_dir)
    fig_dir = Path(figure_dir)
    fig, ax = plt.subplots(figsize=(10, 8))
    any_curve = False

    for model_key in MODEL_ORDER:
        resolved = resolve_multiclass_probability_csv_for_model(model_key, out_dir)

        if resolved is None:
            print(f"[skip] no multiclass probability file found for {model_key}")
            continue

        # Support the existing resolver whether it returns only a Path
        # or a tuple containing the Path plus metadata.
        probability_path = Path(resolved)
        df = pd.read_csv(probability_path)

        if "y_true" not in df.columns:
            print(f"[skip] missing y_true in {probability_path.name}")
            continue

        prob_cols = [c for c in df.columns if c.startswith("class_") and c.endswith("_probability")]

        if not prob_cols:
            print(f"[skip] no per-class probability columns found in {probability_path.name}")
            continue

        y_true = df["y_true"].astype(int).to_numpy()
        y_score = df[prob_cols].astype(float).to_numpy()
        labels = list(range(len(prob_cols)))
        y_true_bin = label_binarize(y_true, classes=labels)
        fpr, tpr, _ = roc_curve(y_true_bin.ravel(), y_score.ravel())
        micro_auc = auc(fpr, tpr)

        ax.plot(
            fpr,
            tpr,
            linewidth=2.5,
            color=MODEL_COLORS[model_key],
            label=f"{MODEL_DISPLAY[model_key]}, micro ROC-AUC={micro_auc:.3f}"
        )
        any_curve = True

    if not any_curve:
        plt.close(fig)
        print("[skip] no multiclass probability files available for micro-average ROC curves")
        return

    # Random-classifier reference line
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1.5)

    # Match the original figure layout
    ax.set_title("Micro-average ROC Curve", fontsize=17, pad=10)
    ax.set_xlabel("False Positive Rate", fontsize=15)
    ax.set_ylabel("True Positive Rate", fontsize=15)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.tick_params(axis="both", labelsize=13)
    ax.grid(alpha=0.30)

    # Legend below the plot, as in the original figure
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=1, frameon=True, fontsize=12)

    fig.suptitle("Multiclass Attack Classification - Micro-average ROC Curves", fontsize=18, y=0.98)

    # Leave room for title and legend
    fig.subplots_adjust(top=0.84, bottom=0.25, left=0.11, right=0.97)

    save_path = fig_dir / "supervised_multiclass_micro_roc_curves.png"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {save_path}")

def plot_repeated_seed_stability(out_dir, fig_dir):
    """Final paper-ready repeated-seed stability figure.

    Left: Binary F1-score across five stratified seeds.
    Right: Multiclass Macro F1-score across five stratified seeds.
    Legend: Model (Binary mean±SD / Multiclass mean±SD)

    Same implementation/style as the former paper_figures.plot_repeated_seed_stability().
    """
    out_dir = Path(out_dir)
    fig_dir = Path(fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)

    results_path = out_dir / "repeated_seed_results.csv"
    if not results_path.exists():
        print(f"[skip] repeated-seed results not found: {results_path}")
        return

    df = pd.read_csv(results_path)

    required_cols = {"model", "task", "seed", "objective_score"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in repeated_seed_results.csv: {missing}")

    seed_order = DEFAULT_SEEDS
    seed_labels = [str(seed) for seed in seed_order]
    x = np.arange(len(seed_order))

    task_settings = {
        "binary": {"title": "Binary Attack Detection", "ylabel": "F1-score (%)"},
        "multiclass": {"title": "Multiclass Attack Classification", "ylabel": "Macro F1-score (%)"},
    }

    fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.7), sharey=False)
    fig.suptitle("Repeated-Seed Stability of Tuned Supervised Models", fontsize=17, y=0.975)

    legend_handles = []
    legend_labels = []
    model_summaries = {}

    for model_key in MODEL_ORDER:
        model_summaries[model_key] = {}
        for task in ["binary", "multiclass"]:
            sub = df[(df["model"] == model_key) & (df["task"] == task)].copy()
            vals = pd.to_numeric(sub["objective_score"], errors="coerce").dropna()
            if len(vals) > 0:
                model_summaries[model_key][task] = {
                    "mean": vals.mean() * 100,
                    "std": vals.std(ddof=1) * 100 if len(vals) > 1 else 0.0,
                }

    for ax, task in zip(axes, ["binary", "multiclass"]):
        task_df = df[df["task"] == task].copy()
        all_values = []

        for model_key in MODEL_ORDER:
            model_df = task_df[task_df["model"] == model_key].copy()
            if model_df.empty:
                continue

            model_df["seed"] = model_df["seed"].astype(int)
            values = []
            for seed in seed_order:
                seed_row = model_df[model_df["seed"] == seed]
                values.append(seed_row["objective_score"].iloc[0] * 100 if not seed_row.empty else np.nan)

            valid_values = [v for v in values if not np.isnan(v)]
            all_values.extend(valid_values)

            line, = ax.plot(
                x, values, marker="o", markersize=7.5, linewidth=2.6,
                color=MODEL_COLORS[model_key], label=MODEL_DISPLAY[model_key]
            )

            if task == "binary":
                legend_handles.append(line)

        ax.set_title(task_settings[task]["title"], fontsize=14, pad=9)
        ax.set_ylabel(task_settings[task]["ylabel"], fontsize=12.5)
        ax.set_xlabel("Seed", fontsize=12.5)
        ax.set_xticks(x)
        ax.set_xticklabels(seed_labels, fontsize=11.5)
        ax.tick_params(axis="y", labelsize=11.5)
        ax.grid(alpha=0.28, linewidth=0.8)

        if all_values:
            ymin = min(all_values)
            ymax = max(all_values)
            spread = ymax - ymin
            padding = max(1.2, spread * 0.14)
            ax.set_ylim(ymin - padding, ymax + padding)

    for model_key in MODEL_ORDER:
        binary_summary = model_summaries[model_key].get("binary")
        multiclass_summary = model_summaries[model_key].get("multiclass")

        if binary_summary and multiclass_summary:
            label = (
                f"{MODEL_DISPLAY[model_key]} "
                f"({binary_summary['mean']:.2f}±{binary_summary['std']:.2f} / "
                f"{multiclass_summary['mean']:.2f}±{multiclass_summary['std']:.2f})"
            )
        else:
            label = MODEL_DISPLAY[model_key]

        legend_labels.append(label)

    fig.legend(
        legend_handles, legend_labels, loc="lower center", bbox_to_anchor=(0.5, 0.015),
        ncol=3, frameon=True, fontsize=10.2, handlelength=2.3, columnspacing=1.8
    )

    fig.subplots_adjust(left=0.075, right=0.985, top=0.855, bottom=0.205, wspace=0.21)

    save_path = fig_dir / "repeated_seed_stability.png"
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {save_path}")


def make_final_plots(out_dir, fig_dir):
    """Generate the final repeated-seed paper figures only."""
    out_dir = Path(out_dir)
    fig_dir = Path(fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)
    pred_dir = out_dir / "predictions"
    plot_supervised_binary_roc_pr(pred_dir, fig_dir)
    plot_supervised_multiclass_micro_roc(out_dir, fig_dir)
    plot_repeated_seed_stability(out_dir, fig_dir)


def regenerate_plots_from_existing(out_dir, fig_dir):
    results_path = out_dir / "repeated_seed_results.csv"
    summary_path = out_dir / "repeated_seed_summary.csv"
    if not results_path.exists():
        raise FileNotFoundError("Missing repeated_seed_results.csv. Run the study once before --plot-only.")
    results = pd.read_csv(results_path)
    results = add_derived_binary_metrics(results)
    results.to_csv(results_path, index=False)
    summary = summarize_results(results)
    summary.to_csv(summary_path, index=False)
    make_paper_table(summary, out_dir / "repeated_seed_summary_paper_format.csv")
    make_final_plots(out_dir, fig_dir)
    print(f"[saved figures] {fig_dir.resolve()}")


def main():
    args = parse_args()
    data_dir = Path(args.data_dir)
    out_dir = Path(args.output_dir)
    fig_dir = config.FIGURES_DIR / "repeated_seeds"
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_dir = out_dir / "predictions"

    if args.plot_only:
        utils.print_header("Regenerating repeated-seed figures only")
        regenerate_plots_from_existing(out_dir, fig_dir)
        print("Done.")
        return

    utils.print_header("Repeated-seed robustness study on time-ablated benchmark")
    print("Data:", data_dir.resolve())
    print("Best params:", Path(args.best_params).resolve())
    print("Output:", out_dir.resolve())
    print("Models:", args.models)
    print("Tasks:", args.tasks)
    print("Seeds:", args.seeds)

    best_params = load_best_params(args.best_params)
    cached_data = {task: load_full_dataset(data_dir, task) for task in args.tasks}

    rows = []
    for task in args.tasks:
        for model_key in args.models:
            key = (model_key, task)
            if key not in best_params:
                print(f"[skip] missing best params for {model_key} {task}")
                continue
            base_params = best_params[key]
            utils.print_header(f"{MODEL_DISPLAY[model_key]} | {TASK_DISPLAY[task]} | tuned repeated seeds")
            for seed in args.seeds:
                start = time.time()
                metrics = run_one(model_key, task, base_params, seed, data_dir, cached_data, pred_dir=pred_dir)
                rows.append(flatten_metrics(metrics))
                print(f"seed={seed} objective={metrics['objective_score']:.4f} runtime={time.time() - start:.1f}s")

    results = pd.DataFrame(rows)
    results_path = out_dir / "repeated_seed_results.csv"
    results.to_csv(results_path, index=False)
    print(f"[saved] {results_path}")

    summary = summarize_results(results)
    summary_path = out_dir / "repeated_seed_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"[saved] {summary_path}")

    paper_path = out_dir / "repeated_seed_summary_paper_format.csv"
    make_paper_table(summary, paper_path)
    print(f"[saved] {paper_path}")

    make_final_plots(out_dir, fig_dir)
    print(f"[saved figures] {fig_dir.resolve()}")
    print("Done.")


if __name__ == "__main__":
    main()