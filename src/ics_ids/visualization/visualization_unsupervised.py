import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, precision_recall_curve

from .. import config


UNSUPERVISED_MODELS = ["isolation_forest", "stacked_autoencoder"]
MODEL_LABELS = {"isolation_forest": "IF", "stacked_autoencoder": "SAE"}

DEFAULT_INPUT_DIR = config.RESULTS_DIR / "unsupervised_robustness"
DEFAULT_FIGURE_DIR = config.FIGURES_DIR / "unsupervised_robustness"


def load_unsupervised_outputs(input_dir):
    input_dir = Path(input_dir)
    predictions_path = input_dir / "unsupervised_repeated_seed_predictions.csv"
    summary_path = input_dir / "unsupervised_repeated_seed_summary.csv"
    paper_summary_path = input_dir / "unsupervised_repeated_seed_summary_paper_format.csv"

    if not predictions_path.exists():
        raise FileNotFoundError(f"Missing predictions file: {predictions_path}")

    if not summary_path.exists():
        raise FileNotFoundError(f"Missing summary file: {summary_path}")

    predictions_df = pd.read_csv(predictions_path)
    summary_df = pd.read_csv(summary_path)

    paper_summary_df = None
    if paper_summary_path.exists():
        paper_summary_df = pd.read_csv(paper_summary_path)

    return predictions_df, summary_df, paper_summary_df


def interpolate_roc(y_true, y_score, grid):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    interp = np.interp(grid, fpr, tpr)
    interp[0] = 0.0
    interp[-1] = 1.0
    return interp


def interpolate_pr(y_true, y_score, grid):
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    recall = recall[::-1]
    precision = precision[::-1]

    unique_recall, unique_idx = np.unique(recall, return_index=True)
    unique_precision = precision[unique_idx]

    interp = np.interp(grid, unique_recall, unique_precision)
    return interp


def plot_roc_pr_curves(predictions_df, summary_df, figure_dir):
    figure_dir = Path(figure_dir)
    figure_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    ax_roc, ax_pr = axes

    roc_grid = np.linspace(0, 1, 250)
    pr_grid = np.linspace(0, 1, 250)

    for model in UNSUPERVISED_MODELS:
        model_df = predictions_df[predictions_df["model"] == model]
        if model_df.empty:
            continue

        roc_curves = []
        pr_curves = []

        for seed, seed_df in model_df.groupby("seed"):
            y_true = seed_df["y_true"].to_numpy()
            y_score = seed_df["y_score"].to_numpy()

            roc_curves.append(interpolate_roc(y_true, y_score, roc_grid))
            pr_curves.append(interpolate_pr(y_true, y_score, pr_grid))

        if not roc_curves or not pr_curves:
            continue

        roc_curves = np.vstack(roc_curves)
        pr_curves = np.vstack(pr_curves)

        roc_mean = roc_curves.mean(axis=0)
        roc_std = roc_curves.std(axis=0)
        pr_mean = pr_curves.mean(axis=0)
        pr_std = pr_curves.std(axis=0)

        row = summary_df[summary_df["model"] == model].iloc[0]
        label = MODEL_LABELS.get(model, model)

        ax_roc.plot(roc_grid, roc_mean, linewidth=2, label=f"{label}, ROC-AUC = {row['roc_auc_mean']:.3f} ± {row['roc_auc_std']:.3f}")
        ax_roc.fill_between(roc_grid, np.maximum(0, roc_mean - roc_std), np.minimum(1, roc_mean + roc_std), alpha=0.15)

        ax_pr.plot(pr_grid, pr_mean, linewidth=2, label=f"{label}, PR-AUC = {row['pr_auc_mean']:.3f} ± {row['pr_auc_std']:.3f}")
        ax_pr.fill_between(pr_grid, np.maximum(0, pr_mean - pr_std), np.minimum(1, pr_mean + pr_std), alpha=0.15)

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

    fig.suptitle("Unsupervised Binary Anomaly Detection on Time-Ablated Benchmark")
    fig.subplots_adjust(bottom=0.30, top=0.84, wspace=0.25)
    fig.savefig(figure_dir / "unsupervised_repeated_seed_roc_pr_curves.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"[saved] {figure_dir / 'unsupervised_repeated_seed_roc_pr_curves.png'}")


def plot_metric_comparison(summary_df, figure_dir):
    figure_dir = Path(figure_dir)
    figure_dir.mkdir(parents=True, exist_ok=True)

    metrics = [("f1_score", "F1-score"), ("recall", "Recall"), ("precision", "Precision"), ("pr_auc", "PR-AUC"), ("roc_auc", "ROC-AUC")]
    labels = summary_df["model_label"].tolist()
    x = np.arange(len(labels))
    width = 0.16

    fig, ax = plt.subplots(figsize=(11, 6))

    for idx, (metric, label) in enumerate(metrics):
        means = summary_df[f"{metric}_mean"].to_numpy() * 100
        stds = summary_df[f"{metric}_std"].fillna(0).to_numpy() * 100
        ax.bar(x + (idx - 2) * width, means, width, yerr=stds, capsize=4, label=label)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Score (%)")
    ax.set_ylim(0, 105)
    ax.set_title("Unsupervised Repeated-Seed Results")
    ax.grid(axis="y", alpha=0.35)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=5, frameon=True)

    fig.tight_layout()
    fig.savefig(figure_dir / "unsupervised_repeated_seed_metrics.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"[saved] {figure_dir / 'unsupervised_repeated_seed_metrics.png'}")


def plot_results_table(paper_summary_df, figure_dir):
    if paper_summary_df is None or paper_summary_df.empty:
        print("[skip] missing paper-format summary table")
        return

    figure_dir = Path(figure_dir)
    figure_dir.mkdir(parents=True, exist_ok=True)

    display_df = paper_summary_df.copy()

    fig, ax = plt.subplots(figsize=(12, 2.8))
    ax.axis("off")

    table = ax.table(
        cellText=display_df.values,
        colLabels=display_df.columns,
        loc="center",
        cellLoc="center"
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.5)

    fig.tight_layout()
    fig.savefig(figure_dir / "unsupervised_repeated_seed_results_table.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"[saved] {figure_dir / 'unsupervised_repeated_seed_results_table.png'}")


def main():
    parser = argparse.ArgumentParser(description="Generate unsupervised anomaly-detection figures from saved repeated-seed results.")
    parser.add_argument("--input-dir", default=str(DEFAULT_INPUT_DIR), help="Directory containing unsupervised repeated-seed CSV files.")
    parser.add_argument("--figure-dir", default=str(DEFAULT_FIGURE_DIR), help="Directory where unsupervised figures will be saved.")
    parser.add_argument("--curves-only", action="store_true", help="Only regenerate the ROC/PR curve figure.")
    parser.add_argument("--metrics-only", action="store_true", help="Only regenerate the metric comparison figure.")
    parser.add_argument("--table-only", action="store_true", help="Only regenerate the table figure.")
    args = parser.parse_args()

    predictions_df, summary_df, paper_summary_df = load_unsupervised_outputs(args.input_dir)

    run_all = not args.curves_only and not args.metrics_only and not args.table_only

    if run_all or args.curves_only:
        plot_roc_pr_curves(predictions_df, summary_df, args.figure_dir)

    if run_all or args.metrics_only:
        plot_metric_comparison(summary_df, args.figure_dir)

    if run_all or args.table_only:
        plot_results_table(paper_summary_df, args.figure_dir)

    print("\nAll unsupervised visualizations saved to:", Path(args.figure_dir).resolve())


if __name__ == "__main__":
    main()
