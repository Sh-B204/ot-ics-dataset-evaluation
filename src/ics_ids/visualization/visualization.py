import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
from sklearn.preprocessing import label_binarize
from .. import config
from .. import utils


BINARY_MODELS = ["random_forest", "xgboost", "mlp"]
MULTICLASS_MODELS = ["random_forest", "xgboost", "mlp"]
MODEL_LABELS = {"random_forest": "RF", "xgboost": "XGBoost", "mlp": "MLP"}
BENCHMARK_LABELS = {"with_time": "With time", "no_time": "No time"}

BINARY_METRIC_KEYS = ["accuracy", "precision", "recall", "f1_score", "roc_auc", "false_positive_rate"]
MULTICLASS_METRIC_KEYS = ["accuracy", "macro_precision", "macro_recall", "macro_f1", "micro_roc_auc", "macro_roc_auc", "weighted_f1"]


RUN_SUFFIX = "no_time_ablation"
TIME_ABLATION_DIR = getattr(config, "ABLATION_DIR", config.RESULTS_DIR / "time_ablation")


def suffix(run_name=None):
    return f"_{run_name}" if run_name else ""


def load_metrics(model, task, run_name=None):
    tag = suffix(run_name)
    path = config.METRICS_DIR / f"{model}_{task}{tag}_metrics.json"
    if not path.exists():
        print(f"[skip] missing metrics file: {path}")
        return None
    return json.loads(path.read_text())


def load_predictions(model, task, run_name=None):
    tag = suffix(run_name)
    path = config.PREDICTIONS_DIR / f"{model}_{task}{tag}_predictions.csv"
    if not path.exists():
        print(f"[skip] missing predictions file: {path}")
        return None
    return pd.read_csv(path)


def load_probabilities(model, task="multiclass", run_name=None):
    tag = suffix(run_name)
    path = config.PREDICTIONS_DIR / f"{model}_{task}{tag}_probabilities.csv"
    if not path.exists():
        print(f"[skip] missing probability file: {path}")
        return None
    return pd.read_csv(path)


def shorten_class_names(class_names):
    short_names = []
    for name in class_names:
        name = str(name)
        if name.lower() in ["normal", "normal (0)"]:
            short_names.append("Normal")
        elif "attack" in name.lower() and "(1)" in name:
            short_names.append("Attack")
        elif "_T" in name:
            short_names.append("T" + name.split("_T")[-1])
        else:
            short_names.append(name)
    return short_names


def plot_metric_comparison(models, task, metric_keys, title, save_name, run_name=RUN_SUFFIX):
    rows = []
    for model in models:
        metrics = load_metrics(model, task, run_name=run_name)
        if metrics is None:
            continue
        row = {"model": MODEL_LABELS.get(model, model)}
        for key in metric_keys:
            row[key] = metrics.get(key)
        rows.append(row)

    if not rows:
        print(f"[skip] no metrics available for {title}")
        return

    df = pd.DataFrame(rows).set_index("model")
    df.to_csv(config.METRICS_DIR / f"{save_name}_summary.csv")

    fig, ax = plt.subplots(figsize=(11, 6))
    df.plot(kind="bar", ax=ax)
    ax.set_title(title)
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="lower right", ncol=2)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    fig.tight_layout(pad=2)
    fig.savefig(config.FIGURES_DIR / f"{save_name}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {save_name}.png")


def plot_final_no_time_roc_curves(run_name=RUN_SUFFIX):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    ax_binary = axes[0]
    ax_multi = axes[1]

    any_binary_curve = False
    for model in BINARY_MODELS:
        preds = load_predictions(model, "binary", run_name=run_name)
        if preds is None or "y_score" not in preds.columns:
            continue
        y_true, y_score = preds["y_true"], preds["y_score"]
        fpr, tpr, _ = roc_curve(y_true, y_score)
        roc_auc = auc(fpr, tpr)
        ax_binary.plot(fpr, tpr, linewidth=2, label=f"{MODEL_LABELS.get(model, model)} (AUC = {roc_auc:.3f})")
        any_binary_curve = True

    if any_binary_curve:
        ax_binary.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=2, label="Random")
        ax_binary.set_xlabel("False Positive Rate")
        ax_binary.set_ylabel("True Positive Rate")
        ax_binary.set_title("Binary Attack Detection ROC (No Time)")
        ax_binary.legend(loc="lower right")
        ax_binary.grid(alpha=0.4)
    else:
        ax_binary.set_title("Binary Attack Detection ROC (No Time)")
        ax_binary.text(0.5, 0.5, "No binary ROC data", ha="center", va="center")
        ax_binary.axis("off")

    any_multiclass_curve = False
    for model in MULTICLASS_MODELS:
        preds = load_probabilities(model, "multiclass", run_name=run_name)
        if preds is None:
            continue
        prob_cols = [c for c in preds.columns if c.startswith("class_") and c.endswith("_probability")]
        if not prob_cols:
            continue
        y_true = preds["y_true"].to_numpy()
        y_score = preds[prob_cols].to_numpy()
        labels = list(range(len(prob_cols)))
        y_true_bin = label_binarize(y_true, classes=labels)
        fpr, tpr, _ = roc_curve(y_true_bin.ravel(), y_score.ravel())
        roc_auc = auc(fpr, tpr)
        ax_multi.plot(fpr, tpr, linewidth=2, label=f"{MODEL_LABELS.get(model, model)} micro-AUC = {roc_auc:.3f}")
        any_multiclass_curve = True

    if any_multiclass_curve:
        ax_multi.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=2, label="Random")
        ax_multi.set_xlabel("False Positive Rate")
        ax_multi.set_ylabel("True Positive Rate")
        ax_multi.set_title("Multiclass Attack Classification ROC (No Time)")
        ax_multi.legend(loc="lower right")
        ax_multi.grid(alpha=0.4)
    else:
        ax_multi.set_title("Multiclass Attack Classification ROC (No Time)")
        ax_multi.text(0.5, 0.5, "No multiclass ROC data", ha="center", va="center")
        ax_multi.axis("off")

    fig.tight_layout(pad=2)
    fig.savefig(config.FIGURES_DIR / "no_time_roc_curves_binary_multiclass_models.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("[saved] no_time_roc_curves_binary_multiclass_models.png")


def create_results_table(run_name=RUN_SUFFIX, save_name="no_time_overall_results_table"):
    rows = []
    experiments = [
        ("random_forest", "binary"), ("random_forest", "multiclass"),
        ("xgboost", "binary"), ("xgboost", "multiclass"),
        ("mlp", "binary"), ("mlp", "multiclass"),
    ]

    for model, task in experiments:
        metrics = load_metrics(model, task, run_name=run_name)
        if metrics is None:
            continue
        if task == "binary":
            rows.append([MODEL_LABELS.get(model, model), task, metrics.get("n_features"), metrics.get("accuracy"), metrics.get("precision"), metrics.get("recall"), metrics.get("f1_score"), metrics.get("roc_auc"), metrics.get("false_positive_rate"), None])
        else:
            rows.append([MODEL_LABELS.get(model, model), task, metrics.get("n_features"), metrics.get("accuracy"), metrics.get("macro_precision"), metrics.get("macro_recall"), metrics.get("macro_f1"), metrics.get("micro_roc_auc"), None, metrics.get("weighted_f1")])

    if not rows:
        print("[skip] no no-time metrics available for results table")
        return

    columns = ["Model", "Task", "Features", "Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC", "FPR", "Weighted F1"]
    df = pd.DataFrame(rows, columns=columns)
    df.to_csv(config.METRICS_DIR / f"{save_name}.csv", index=False)

    display_df = df.copy()
    for col in ["Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC", "FPR", "Weighted F1"]:
        display_df[col] = display_df[col].apply(lambda x: "--" if pd.isna(x) else f"{x * 100:.2f}%")

    fig, ax = plt.subplots(figsize=(12, 3.8))
    ax.axis("off")
    table = ax.table(cellText=display_df.values, colLabels=display_df.columns, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.45)
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / f"{save_name}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {save_name}.png")


def plot_final_xgboost_no_time_confusion_matrix_normalized(run_name=RUN_SUFFIX):
    metrics = load_metrics("xgboost", "multiclass", run_name=run_name)
    if metrics is None:
        print("[skip] missing XGBoost no-time multiclass metrics")
        return

    cm = np.array(metrics["confusion_matrix"], dtype=float)
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_pct = np.divide(cm, row_sums, out=np.zeros_like(cm, dtype=float), where=row_sums != 0) * 100
    class_names = shorten_class_names(metrics["confusion_matrix_labels"])

    fig, ax = plt.subplots(figsize=(9.5, 8))
    im = ax.imshow(cm_pct, cmap="Blues", vmin=0, vmax=100)
    ax.set_title("XGBoost - Normalized Multiclass Confusion Matrix (No Time)")
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(class_names, fontsize=9)

    for i in range(cm_pct.shape[0]):
        for j in range(cm_pct.shape[1]):
            ax.text(j, i, f"{cm_pct[i, j]:.1f}%", ha="center", va="center", color="white" if cm_pct[i, j] > 50 else "black", fontsize=8)

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Percentage (%)")
    fig.tight_layout(pad=2)
    fig.savefig(config.FIGURES_DIR / "xgboost_multiclass_no_time_confusion_matrix_normalized.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("[saved] xgboost_multiclass_no_time_confusion_matrix_normalized.png")


def load_time_ablation_summary():
    path = TIME_ABLATION_DIR / "time_ablation_supervised_summary.csv"
    if not path.exists():
        print(f"[skip] missing time ablation summary: {path}")
        return None
    return pd.read_csv(path)


def plot_time_ablation_metric(summary, task, metrics, title, save_name):
    df = summary[summary["task"] == task].copy()
    if df.empty:
        print(f"[skip] no rows for {task}")
        return

    models = ["random_forest", "xgboost", "mlp"]
    x = np.arange(len(models))
    width = 0.18 if len(metrics) > 1 else 0.30

    fig, ax = plt.subplots(figsize=(12, 6))

    offsets = []
    total_bars = len(metrics) * 2
    for idx in range(total_bars):
        offsets.append((idx - (total_bars - 1) / 2) * width)

    bar_idx = 0
    for metric in metrics:
        for benchmark in ["with_time", "no_time"]:
            vals = []
            for model in models:
                row = df[(df["model_key"] == model) & (df["benchmark"] == benchmark)]
                vals.append(np.nan if row.empty else row.iloc[0].get(metric, np.nan))
            label = f"{BENCHMARK_LABELS[benchmark]} {metric.replace('_', ' ')}"
            ax.bar(x + offsets[bar_idx], vals, width, label=label)
            bar_idx += 1

    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_LABELS[m] for m in models])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.30), ncol=2, frameon=True)
    fig.subplots_adjust(bottom=0.28)
    fig.savefig(config.FIGURES_DIR / f"{save_name}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {save_name}.png")


def plot_time_ablation_drop(summary):
    rows = []
    for _, row in summary.iterrows():
        if row["benchmark"] != "no_time":
            continue
        if row["task"] == "binary":
            metric = "f1_score_drop_from_with_time"
            metric_name = "Binary F1 drop"
        else:
            metric = "f1_score_drop_from_with_time"
            metric_name = "Multiclass macro-F1 drop"
        rows.append({"model": MODEL_LABELS.get(row["model_key"], row["model_key"]), "task": row["task"], "metric_name": metric_name, "drop": row.get(metric)})

    df = pd.DataFrame(rows).dropna(subset=["drop"])
    if df.empty:
        print("[skip] no ablation drop values available")
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    labels = [f"{r.model}\n{r.task}" for r in df.itertuples()]
    ax.bar(labels, df["drop"])
    ax.axhline(0, color="gray", linewidth=1)
    ax.set_title("Performance Change After Removing Time Features")
    ax.set_ylabel("Drop from with-time score")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout(pad=2)
    fig.savefig(config.FIGURES_DIR / "time_ablation_score_drop.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("[saved] time_ablation_score_drop.png")


def plot_time_ablation_summary():
    summary = load_time_ablation_summary()
    if summary is None:
        return

    summary.to_csv(config.METRICS_DIR / "time_ablation_supervised_summary_copy.csv", index=False)

    plot_time_ablation_metric(
        summary,
        task="binary",
        metrics=["f1_score", "recall"],
        title="Time-Feature Ablation: Binary Attack Detection",
        save_name="time_ablation_binary_f1_recall",
    )
    plot_time_ablation_metric(
        summary,
        task="multiclass",
        metrics=["f1_score", "recall"],
        title="Time-Feature Ablation: Multiclass Attack Classification",
        save_name="time_ablation_multiclass_macro_f1_recall",
    )
    plot_time_ablation_metric(
        summary,
        task="binary",
        metrics=["roc_auc"],
        title="Time-Feature Ablation: Binary ROC-AUC",
        save_name="time_ablation_binary_roc_auc",
    )
    plot_time_ablation_drop(summary)


def plot_unsupervised_roc_pr_curves(run_name=None):
    models = [("isolation_forest", "IF"), ("stacked_autoencoder", "SAE")]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    ax_roc, ax_pr = axes
    any_curve = False

    for model, label in models:
        preds = load_predictions(model, "binary", run_name=run_name)
        if preds is None or "y_score" not in preds.columns:
            continue
        y_true, y_score = preds["y_true"], preds["y_score"]
        fpr, tpr, _ = roc_curve(y_true, y_score)
        roc_auc = auc(fpr, tpr)
        precision, recall, _ = precision_recall_curve(y_true, y_score)
        pr_auc = average_precision_score(y_true, y_score)
        ax_roc.plot(fpr, tpr, linewidth=2, label=f"{label}, ROC-AUC = {roc_auc:.3f}")
        ax_pr.step(recall, precision, where="post", linewidth=2, label=f"{label}, PR-AUC = {pr_auc:.3f}")
        any_curve = True

    if not any_curve:
        print("[skip] no unsupervised ROC/PR data available")
        plt.close(fig)
        return

    ax_roc.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=2, label="Random")
    ax_roc.set_xlabel("False Positive Rate")
    ax_roc.set_ylabel("True Positive Rate")
    ax_roc.set_title("ROC Curve")
    ax_roc.grid(alpha=0.4)
    ax_pr.set_xlabel("Recall")
    ax_pr.set_ylabel("Precision")
    ax_pr.set_title("PR Curve")
    ax_pr.grid(alpha=0.4)
    ax_roc.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), frameon=True)
    ax_pr.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), frameon=True)
    fig.suptitle("Unsupervised Binary Anomaly Detection")
    fig.subplots_adjust(bottom=0.28, top=0.84, wspace=0.25)
    fig.savefig(config.FIGURES_DIR / "unsupervised_roc_pr_curves.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("[saved] unsupervised_roc_pr_curves.png")



# ---------------------------------------------------------------------
# Final supervised evaluation figures (tuned + repeated-seed outputs)
# ---------------------------------------------------------------------

def regenerate_final_supervised_figures():
    """Regenerate final supervised figures from saved CSV/prediction outputs.

    This keeps the older visualization functions above available, but uses the final
    tuned outputs generated by run_repeated_seeds.py and run_time_ablation.py.
    No training is performed here.
    """
    from ..experiments import run_repeated_seeds
    from ..experiments import run_time_ablation
    from . import paper_figures

    repeated_out = config.RESULTS_DIR / "robustness"
    repeated_fig = config.FIGURES_DIR / "repeated_seeds"
    if (repeated_out / "repeated_seed_results.csv").exists():
        run_repeated_seeds.regenerate_plots_from_existing(repeated_out, repeated_fig)
    else:
        print(f"[skip] missing {repeated_out / 'repeated_seed_results.csv'}")

    ablation_out = config.RESULTS_DIR / "time_ablation_tuned"
    ablation_fig = config.FIGURES_DIR / "time_ablation_tuned"
    if (ablation_out / "time_ablation_tuned_repeated_results.csv").exists():
        run_time_ablation.regenerate_plots_from_existing(ablation_out, ablation_fig)
    else:
        print(f"[skip] missing {ablation_out / 'time_ablation_tuned_repeated_results.csv'}")

    # Final paper-ready figures (ROC/PR, best multiclass confusion matrix,
    # repeated-seed stability, time-ablation performance drop). Same code that
    # used to live in the standalone make_supervised_paper_figures.py script;
    # it's now part of this pipeline instead of a separate manual step.
    paper_figures.plot_supervised_binary_roc_pr()
    paper_figures.plot_best_multiclass_confusion_matrix()
    paper_figures.plot_repeated_seed_stability()
    paper_figures.plot_time_ablation_performance_drop()


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Generate visualization outputs for the ICS model evaluation.")
    parser.add_argument("--final-only", action="store_true", help="Only regenerate final tuned supervised figures from saved CSV outputs.")
    parser.add_argument("--include-legacy", action="store_true", help="Also generate older fixed-baseline/no-time visualizations if their metric files exist.")
    parser.add_argument("--include-unsupervised", action="store_true", help="Generate unsupervised ROC/PR curves if unsupervised prediction files exist.")
    args = parser.parse_args()

    utils.print_header("Generating evaluation visualizations")

    # Final tuned/repeated-seed supervised figures. These are the ones to use for
    # the updated paper results.
    regenerate_final_supervised_figures()

    # Keep the older visualization outputs available, but do not force them by
    # default because they may belong to the earlier fixed-baseline run.
    if args.include_legacy and not args.final_only:
        plot_time_ablation_summary()
        plot_metric_comparison(
            BINARY_MODELS, "binary", BINARY_METRIC_KEYS,
            "Binary Attack Detection - Time-Ablated Model Comparison",
            "time_ablated_binary_models_metric_comparison",
            run_name=RUN_SUFFIX,
        )
        plot_metric_comparison(
            MULTICLASS_MODELS, "multiclass", MULTICLASS_METRIC_KEYS,
            "Multiclass Attack Classification - Time-Ablated Model Comparison",
            "time_ablated_multiclass_models_metric_comparison",
            run_name=RUN_SUFFIX,
        )
        plot_final_no_time_roc_curves(run_name=RUN_SUFFIX)
        create_results_table(run_name=RUN_SUFFIX, save_name="time_ablated_overall_results_table")
        plot_final_xgboost_no_time_confusion_matrix_normalized(run_name=RUN_SUFFIX)

    # The unsupervised plot is independent of the final supervised tuned runs.
    # Use this when the IF/SAE prediction CSVs exist.
    if args.include_unsupervised:
        plot_unsupervised_roc_pr_curves()

    print("\nAll requested visualizations saved to:", config.FIGURES_DIR.resolve())


if __name__ == "__main__":
    main()