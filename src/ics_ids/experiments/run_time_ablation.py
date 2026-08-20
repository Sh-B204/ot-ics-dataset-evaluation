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
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import average_precision_score, precision_recall_curve
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
BENCHMARKS = [
    ("full_feature", "Full-feature", config.WITH_TIME_DATA_DIR, 49),
    ("time_ablated", "Time-ablated", config.NO_TIME_DATA_DIR, 41),
]

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
    "balanced_accuracy": "Balanced accuracy",
    "false_positive_rate": "FPR",
    "roc_auc": "ROC-AUC",
    "pr_auc": "PR-AUC",
    "macro_precision": "Macro precision",
    "macro_recall": "Macro recall",
    "macro_f1": "Macro F1",
    "weighted_f1": "Weighted F1",
    "micro_roc_auc": "Micro ROC-AUC",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Tuned full-feature vs time-ablated benchmark comparison.")
    parser.add_argument("--models", nargs="+", default=["random_forest", "xgboost", "mlp"], choices=["random_forest", "xgboost", "mlp"])
    parser.add_argument("--tasks", nargs="+", default=["binary", "multiclass"], choices=["binary", "multiclass"])
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--best-params", default=str(config.RESULTS_DIR / "hyperparameter_study" / "best_hyperparameters.json"))
    parser.add_argument("--output-dir", default=str(config.RESULTS_DIR / "time_ablation_tuned"))
    parser.add_argument("--plot-only", action="store_true", help="Regenerate figures from existing CSV files without retraining.")
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


def run_one(model_key, task, base_params, seed, benchmark_key, benchmark_label, data_dir, expected_features, cached_data, pred_dir=None):
    params = dict(base_params)
    params["_seed"] = seed
    X, y, features = cached_data[(benchmark_key, task)]
    X_train, X_test, y_train, y_test = split_and_scale(X, y, seed)

    start = time.time()
    _, y_pred, y_prob = train_predict(model_key, task, params, X_train, y_train, X_test)
    training_time = time.time() - start

    if task == "binary":
        y_score = y_prob[:, 1] if y_prob is not None and y_prob.ndim == 2 else None
        metrics = utils.binary_metrics(y_test, y_pred, y_score)
        metrics["balanced_accuracy"] = float((metrics["recall"] + (1.0 - metrics["false_positive_rate"])) / 2.0)
        metrics["pr_auc"] = float(average_precision_score(y_test, y_score)) if y_score is not None else None
        if pred_dir is not None and y_score is not None:
            save_binary_prediction_scores(pred_dir, benchmark_key, model_key, seed, y_test, y_pred, y_score)
        objective_metric = "f1_score"
        objective_score = metrics["f1_score"]
    else:
        labels = sorted(pd.unique(y))
        metrics = utils.multiclass_metrics(y_test, y_pred, labels=labels, y_prob=y_prob, data_dir=data_dir)
        objective_metric = "macro_f1"
        objective_score = metrics["macro_f1"]

    metrics.update({
        "benchmark": benchmark_key,
        "benchmark_display": benchmark_label,
        "model": model_key,
        "model_display": MODEL_DISPLAY[model_key],
        "task": task,
        "task_display": TASK_DISPLAY[task],
        "seed": seed,
        "expected_features": expected_features,
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
    for (benchmark, model, task), group in results_df.groupby(["benchmark", "model", "task"]):
        row = {
            "benchmark": benchmark,
            "benchmark_display": group["benchmark_display"].iloc[0],
            "model": model,
            "model_display": MODEL_DISPLAY.get(model, model),
            "task": task,
            "task_display": TASK_DISPLAY.get(task, task),
            "n_runs": len(group),
            "n_features": int(group["n_features"].iloc[0]),
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


def add_drop_columns(summary):
    out = summary.copy()
    metric_cols = ["accuracy", "precision", "recall", "f1_score", "balanced_accuracy", "roc_auc", "pr_auc", "macro_precision", "macro_recall", "macro_f1", "weighted_f1", "micro_roc_auc", "false_positive_rate"]
    for metric in metric_cols:
        out[f"{metric}_drop_full_to_ablated_mean"] = np.nan

    for (model, task), group in out.groupby(["model", "task"]):
        full = group[group["benchmark"] == "full_feature"]
        ablated = group[group["benchmark"] == "time_ablated"]
        if full.empty or ablated.empty:
            continue
        full_row = full.iloc[0]
        ablated_idx = ablated.index[0]
        for metric in metric_cols:
            full_col = f"{metric}_mean"
            ablated_col = f"{metric}_mean"
            if full_col in out.columns and ablated_col in out.columns:
                full_val = full_row.get(full_col)
                ablated_val = out.loc[ablated_idx, ablated_col]
                if pd.notna(full_val) and pd.notna(ablated_val):
                    out.loc[ablated_idx, f"{metric}_drop_full_to_ablated_mean"] = full_val - ablated_val
    return out


def make_paper_table(summary, out_path):
    rows = []
    order = {"full_feature": 0, "time_ablated": 1, "mlp": 0, "random_forest": 1, "xgboost": 2, "binary": 0, "multiclass": 1}
    summary = summary.copy()
    summary["sort_key"] = summary.apply(lambda r: (order.get(r["task"], 9), order.get(r["model"], 9), order.get(r["benchmark"], 9)), axis=1)
    summary = summary.sort_values("sort_key")

    for _, r in summary.iterrows():
        if r["task"] == "binary":
            selected = ["accuracy", "precision", "recall", "f1_score", "balanced_accuracy", "pr_auc", "false_positive_rate"]
        else:
            selected = ["accuracy", "macro_precision", "macro_recall", "macro_f1", "weighted_f1", "micro_roc_auc"]
        row = {
            "Benchmark": r["benchmark_display"],
            "Model": r["model_display"],
            "Task": r["task_display"],
            "Features": int(r["n_features"]),
            "Runs": int(r["n_runs"]),
        }
        for metric in selected:
            mean_col = f"{metric}_mean"
            std_col = f"{metric}_std"
            if mean_col in r and pd.notna(r[mean_col]):
                row[METRIC_LABELS.get(metric, metric)] = f"{r[mean_col] * 100:.2f} ± {r[std_col] * 100:.2f}%"
        rows.append(row)
    paper = pd.DataFrame(rows)
    paper.to_csv(out_path, index=False)
    return paper



def save_binary_prediction_scores(pred_dir, benchmark_key, model_key, seed, y_true, y_pred, y_score):
    pred_dir = Path(pred_dir)
    pred_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame({
        "benchmark": benchmark_key,
        "model": model_key,
        "seed": int(seed),
        "y_true": np.asarray(y_true),
        "y_pred": np.asarray(y_pred),
        "y_score": np.asarray(y_score) if y_score is not None else np.nan,
    })
    out_path = pred_dir / f"{benchmark_key}_{model_key}_binary_seed{seed}_predictions.csv"
    df.to_csv(out_path, index=False)


def _mean_pr_curve_from_files(files):
    recall_grid = np.linspace(0.0, 1.0, 300)
    precision_rows = []
    ap_scores = []
    for file_path in files:
        df = pd.read_csv(file_path)
        if "y_score" not in df.columns or df["y_score"].isna().all():
            continue
        y_true = df["y_true"].astype(int).to_numpy()
        y_score = df["y_score"].astype(float).to_numpy()
        precision, recall, _ = precision_recall_curve(y_true, y_score)
        ap_scores.append(average_precision_score(y_true, y_score))

        recall_asc = recall[::-1]
        precision_asc = precision[::-1]
        precision_interp = np.interp(recall_grid, recall_asc, precision_asc)
        precision_rows.append(precision_interp)

    if not precision_rows:
        return None

    precision_arr = np.vstack(precision_rows)
    return {
        "recall_grid": recall_grid,
        "precision_mean": precision_arr.mean(axis=0),
        "precision_std": precision_arr.std(axis=0, ddof=1) if precision_arr.shape[0] > 1 else np.zeros_like(recall_grid),
        "ap_mean": float(np.mean(ap_scores)),
        "ap_std": float(np.std(ap_scores, ddof=1)) if len(ap_scores) > 1 else 0.0,
        "n_curves": len(ap_scores),
    }


def plot_binary_pr_curves(pred_dir, benchmark_key, benchmark_label, save_path):
    pred_dir = Path(pred_dir)
    fig, ax = plt.subplots(figsize=(7, 5))

    any_curve = False
    for model_key in ["mlp", "random_forest", "xgboost"]:
        files = sorted(pred_dir.glob(f"{benchmark_key}_{model_key}_binary_seed*_predictions.csv"))
        curve = _mean_pr_curve_from_files(files)
        if curve is None:
            continue

        label = f"{MODEL_DISPLAY[model_key]} (PR-AUC={curve['ap_mean']:.3f}±{curve['ap_std']:.3f})"
        ax.plot(curve["recall_grid"], curve["precision_mean"], linewidth=2, label=label)
        ax.fill_between(
            curve["recall_grid"],
            np.maximum(0, curve["precision_mean"] - curve["precision_std"]),
            np.minimum(1, curve["precision_mean"] + curve["precision_std"]),
            alpha=0.12,
        )
        any_curve = True

    if not any_curve:
        plt.close(fig)
        print(f"[skip] no prediction score files found for PR curves: {benchmark_key}")
        return

    ax.set_title(f"Tuned Binary PR Curves: {benchmark_label} Benchmark")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.3)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), frameon=True)
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_ablation_metric(summary, task, metrics, title, save_path):
    df = summary[summary["task"] == task].copy()
    if df.empty:
        return

    model_order = ["mlp", "random_forest", "xgboost"]
    benchmark_order = ["full_feature", "time_ablated"]
    labels = []
    centers = []
    x = []
    pos = 0
    for m in model_order:
        for b in benchmark_order:
            labels.append(f"{MODEL_DISPLAY[m]}\n{df[df['benchmark']==b]['benchmark_display'].iloc[0] if not df[df['benchmark']==b].empty else b}")
            x.append(pos)
            pos += 1
        pos += 0.6
    x = np.array(x)
    width = 0.8 / len(metrics)

    fig, ax = plt.subplots(figsize=(11, 5))
    for i, metric in enumerate(metrics):
        means, stds = [], []
        for label_idx, m in enumerate(model_order):
            for b in benchmark_order:
                row = df[(df["model"] == m) & (df["benchmark"] == b)]
                if row.empty:
                    means.append(np.nan); stds.append(np.nan)
                else:
                    means.append(row.iloc[0].get(f"{metric}_mean", np.nan) * 100)
                    stds.append(row.iloc[0].get(f"{metric}_std", np.nan) * 100)
        ax.bar(x + (i - (len(metrics) - 1) / 2) * width, means, width, yerr=stds, capsize=4, label=METRIC_LABELS.get(metric, metric))

    ax.set_title(title)
    ax.set_ylabel("Score (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylim(0, 105)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=len(metrics), frameon=True)
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_drop_summary(summary, save_path):
    rows = []
    for _, r in summary[summary["benchmark"] == "time_ablated"].iterrows():
        if r["task"] == "binary":
            metric = "f1_score_drop_full_to_ablated_mean"
            label = "Binary F1 drop"
        else:
            metric = "macro_f1_drop_full_to_ablated_mean"
            label = "Multiclass macro F1 drop"
        if metric in r and pd.notna(r[metric]):
            rows.append({"model": r["model_display"], "task_metric": label, "drop": r[metric] * 100})
    if not rows:
        return
    df = pd.DataFrame(rows)
    pivot = df.pivot(index="model", columns="task_metric", values="drop").reset_index()
    labels = pivot["model"].tolist()
    metrics = [c for c in pivot.columns if c != "model"]
    x = np.arange(len(labels))
    width = 0.8 / len(metrics)

    fig, ax = plt.subplots(figsize=(9, 5))
    for i, metric in enumerate(metrics):
        ax.bar(x + (i - (len(metrics) - 1) / 2) * width, pivot[metric], width, label=metric)
    ax.set_title("Performance Change After Removing Time-Derived Features")
    ax.set_ylabel("Score drop (percentage points)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=len(metrics), frameon=True)
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def make_plots(summary, fig_dir, pred_dir=None):
    fig_dir.mkdir(parents=True, exist_ok=True)
    plot_ablation_metric(
        summary, "binary", ["f1_score", "recall", "precision"],
        "Tuned Binary Detection Results",
        fig_dir / "tuned_ablation_binary_f1_recall_precision.png"
    )
    if pred_dir is not None:
        plot_binary_pr_curves(pred_dir, "time_ablated", "Time-ablated", fig_dir / "tuned_time_ablated_binary_pr_curves.png")
        plot_binary_pr_curves(pred_dir, "full_feature", "Full-feature", fig_dir / "tuned_full_feature_binary_pr_curves.png")

    binary_auc_metrics = ["balanced_accuracy", "false_positive_rate"]
    binary_auc_title = "Tuned Binary Imbalance-Aware Metrics"
    binary_auc_path = fig_dir / "tuned_ablation_binary_balanced_accuracy_fpr.png"
    if "pr_auc_mean" in summary.columns and summary["pr_auc_mean"].notna().any():
        binary_auc_metrics = ["pr_auc", "balanced_accuracy", "false_positive_rate"]
        binary_auc_title = "Tuned Binary PR-AUC, Balanced Accuracy, and FPR"
        binary_auc_path = fig_dir / "tuned_ablation_binary_pr_auc_balanced_accuracy_fpr.png"
    plot_ablation_metric(summary, "binary", binary_auc_metrics, binary_auc_title, binary_auc_path)

    plot_ablation_metric(
        summary, "multiclass", ["macro_f1", "macro_recall", "macro_precision"],
        "Tuned Multiclass Macro Metrics",
        fig_dir / "tuned_ablation_multiclass_macro_metrics.png"
    )
    plot_ablation_metric(
        summary, "multiclass", ["weighted_f1", "micro_roc_auc"],
        "Tuned Multiclass Aggregate Metrics",
        fig_dir / "tuned_ablation_multiclass_weighted_micro.png"
    )
    plot_drop_summary(summary, fig_dir / "tuned_ablation_score_difference.png")


def regenerate_plots_from_existing(out_dir, fig_dir):
    results_path = out_dir / "time_ablation_tuned_repeated_results.csv"
    summary_path = out_dir / "time_ablation_tuned_repeated_summary.csv"

    if results_path.exists():
        results = pd.read_csv(results_path)
        results = add_derived_binary_metrics(results)
        results.to_csv(results_path, index=False)
        summary = add_drop_columns(summarize_results(results))
        summary.to_csv(summary_path, index=False)
        make_paper_table(summary, out_dir / "time_ablation_tuned_repeated_summary_paper_format.csv")
        print("[updated] derived balanced-accuracy columns and summary files")
    elif summary_path.exists():
        summary = pd.read_csv(summary_path)
    else:
        raise FileNotFoundError("Missing time_ablation_tuned_repeated_results.csv. Run the study once before --plot-only.")

    make_plots(summary, fig_dir, out_dir / "predictions")
    if "pr_auc_mean" not in summary.columns or summary["pr_auc_mean"].isna().all():
        print("[note] PR-AUC was not in the existing CSV, so PR-AUC figures cannot be generated without rerunning the models with score saving enabled.")
    print(f"[saved figures] {fig_dir.resolve()}")


def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    fig_dir = config.FIGURES_DIR / "time_ablation_tuned"
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_dir = out_dir / "predictions"

    if args.plot_only:
        utils.print_header("Regenerating tuned time-ablation figures only")
        regenerate_plots_from_existing(out_dir, fig_dir)
        print("Done.")
        return

    utils.print_header("Tuned full-feature vs time-ablated repeated-seed comparison")
    print("Best params:", Path(args.best_params).resolve())
    print("Output:", out_dir.resolve())
    print("Models:", args.models)
    print("Tasks:", args.tasks)
    print("Seeds:", args.seeds)

    best_params = load_best_params(args.best_params)

    cached_data = {}
    for benchmark_key, _, data_dir, _ in BENCHMARKS:
        for task in args.tasks:
            cached_data[(benchmark_key, task)] = load_full_dataset(data_dir, task)

    rows = []
    for task in args.tasks:
        for model_key in args.models:
            key = (model_key, task)
            if key not in best_params:
                print(f"[skip] missing best params for {model_key} {task}")
                continue
            base_params = best_params[key]
            for benchmark_key, benchmark_label, data_dir, expected_features in BENCHMARKS:
                utils.print_header(f"{benchmark_label} | {MODEL_DISPLAY[model_key]} | {TASK_DISPLAY[task]} | tuned repeated seeds")
                for seed in args.seeds:
                    start = time.time()
                    metrics = run_one(model_key, task, base_params, seed, benchmark_key, benchmark_label, data_dir, expected_features, cached_data, pred_dir=pred_dir)
                    rows.append(flatten_metrics(metrics))
                    print(f"seed={seed} objective={metrics['objective_score']:.4f} runtime={time.time() - start:.1f}s")

    results = pd.DataFrame(rows)
    results_path = out_dir / "time_ablation_tuned_repeated_results.csv"
    results.to_csv(results_path, index=False)
    print(f"[saved] {results_path}")

    summary = summarize_results(results)
    summary = add_drop_columns(summary)
    summary_path = out_dir / "time_ablation_tuned_repeated_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"[saved] {summary_path}")

    paper_path = out_dir / "time_ablation_tuned_repeated_summary_paper_format.csv"
    make_paper_table(summary, paper_path)
    print(f"[saved] {paper_path}")

    make_plots(summary, fig_dir, out_dir / "predictions")
    print(f"[saved figures] {fig_dir.resolve()}")
    print("Done.")


if __name__ == "__main__":
    main()