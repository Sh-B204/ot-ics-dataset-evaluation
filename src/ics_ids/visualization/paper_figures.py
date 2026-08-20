from pathlib import Path
import re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import (
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
    confusion_matrix,
)
from matplotlib.ticker import FuncFormatter

# =========================================================
# Paths
# =========================================================
BASE_DIR = Path(__file__).resolve().parents[3]  # visualization -> ics_ids -> src -> repo root
RESULTS_DIR = BASE_DIR / "results"
ROBUSTNESS_DIR = RESULTS_DIR / "robustness"
PRED_DIR_BINARY = ROBUSTNESS_DIR / "predictions"   # repeated-seed binary predictions
PRED_DIR_STANDARD = RESULTS_DIR / "predictions"    # standard predictions folder
DIAG_DIR = RESULTS_DIR / "dataset_diagnostics"
OUT_DIR = RESULTS_DIR / "figures" / "supervised_results"
OUT_DIR.mkdir(parents=True, exist_ok=True)

REPEATED_SEED_RESULTS_CSV = ROBUSTNESS_DIR / "repeated_seed_results.csv"
CLASS_MAP_CSV = DIAG_DIR / "class_counts_train_test.csv"
TIME_ABLATION_SUMMARY_CSV = (
    RESULTS_DIR / "time_ablation_tuned" / "time_ablation_tuned_repeated_summary.csv"
)

# =========================================================
# Display settings
# =========================================================
MODEL_DISPLAY = {
    "mlp": "MLP",
    "random_forest": "Random Forest",
    "xgboost": "XGBoost",
}

MODEL_COLORS = {
    "mlp": "#1f77b4",            # blue
    "random_forest": "#ff7f0e",  # orange
    "xgboost": "#2ca02c",        # green
}

MODEL_ORDER = ["mlp", "random_forest", "xgboost"]

plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "legend.fontsize": 10,
})

# =========================================================
# Helpers
# =========================================================
def shorten_class_name(name: str) -> str:
    name = str(name)
    if name.lower() == "normal":
        return "Normal"
    match = re.search(r"T\d{4}", name)
    if match:
        return match.group(0)
    return name

def load_label_map():
    """
    Reads dataset_diagnostics/class_counts_train_test.csv
    and returns {multiclass_label: short_class_name}
    """
    df = pd.read_csv(CLASS_MAP_CSV)
    mapping = {}
    for _, row in df.iterrows():
        mapping[int(row["multiclass_label"])] = shorten_class_name(row["class_name"])
    return mapping

# =========================================================
# Binary ROC / PR aggregation across repeated seeds
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

        # interpolate TPR onto common FPR grid
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

        # precision_recall_curve returns recall descending/in awkward direction for interpolation
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

def plot_supervised_binary_roc_pr():
    fig, axes = plt.subplots(1, 2, figsize=(8.8, 4.8))
    ax_roc, ax_pr = axes

    any_curve = False

    for model_key in MODEL_ORDER:
        files = sorted(
            PRED_DIR_BINARY.glob(
                f"{model_key}_binary_seed*_predictions.csv"
            )
        )

        if not files:
            print(f"[skip] no repeated-seed binary files found for {model_key}")
            continue

        roc_info = mean_roc_curve_from_files(files)
        pr_info = mean_pr_curve_from_files(files)

        if roc_info is None or pr_info is None:
            continue

        color = MODEL_COLORS[model_key]
        model_name = MODEL_DISPLAY[model_key]

        # -------------------------
        # ROC curve
        # -------------------------
        roc_label = (
            f"{model_name}, "
            f"ROC-AUC={roc_info['auc_mean']:.3f}"
            f"±{roc_info['auc_std']:.3f}"
        )

        ax_roc.plot(
            roc_info["fpr_grid"],
            roc_info["tpr_mean"],
            color=color,
            linewidth=2,
            label=roc_label
        )

        ax_roc.fill_between(
            roc_info["fpr_grid"],
            np.maximum(
                0,
                roc_info["tpr_mean"] - roc_info["tpr_std"]
            ),
            np.minimum(
                1,
                roc_info["tpr_mean"] + roc_info["tpr_std"]
            ),
            color=color,
            alpha=0.10
        )

        # -------------------------
        # PR curve
        # -------------------------
        pr_label = (
            f"{model_name}, "
            f"PR-AUC={pr_info['ap_mean']:.3f}"
            f"±{pr_info['ap_std']:.3f}"
        )

        ax_pr.plot(
            pr_info["recall_grid"],
            pr_info["precision_mean"],
            color=color,
            linewidth=2,
            label=pr_label
        )

        ax_pr.fill_between(
            pr_info["recall_grid"],
            np.maximum(
                0,
                pr_info["precision_mean"] - pr_info["precision_std"]
            ),
            np.minimum(
                1,
                pr_info["precision_mean"] + pr_info["precision_std"]
            ),
            color=color,
            alpha=0.10
        )

        any_curve = True

    if not any_curve:
        print("[error] no binary repeated-seed prediction files were found.")
        plt.close(fig)
        return

    # -------------------------
    # ROC styling
    # -------------------------
    ax_roc.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        color="gray",
        linewidth=1
    )

    ax_roc.set_title("ROC Curve", fontsize=11)
    ax_roc.set_xlabel("False Positive Rate")
    ax_roc.set_ylabel("True Positive Rate")
    ax_roc.set_xlim(0, 1)
    ax_roc.set_ylim(0, 1.02)
    ax_roc.grid(alpha=0.35)

    ax_roc.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.18),
        frameon=True,
        fontsize=8.5
    )

    # -------------------------
    # PR styling
    # -------------------------
    ax_pr.set_title("PR Curve", fontsize=11)
    ax_pr.set_xlabel("Recall")
    ax_pr.set_ylabel("Precision")
    ax_pr.set_xlim(0, 1)
    ax_pr.set_ylim(0, 1.02)
    ax_pr.grid(alpha=0.35)

    ax_pr.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.18),
        frameon=True,
        fontsize=8.5
    )

    # -------------------------
    # Overall figure title
    # -------------------------
    fig.suptitle(
        "Binary Attack Detection - ROC Curve (left) and PR Curve (right)",
        fontsize=12,
        y=0.98
    )

    # Space for legends below plots
    fig.subplots_adjust(
        top=0.87,
        bottom=0.27,
        wspace=0.28
    )

    save_path = (
        OUT_DIR /
        "supervised_binary_roc_pr_curves.png"
    )

    fig.savefig(
        save_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig)

    print(f"[saved] {save_path}")

# =========================================================
# Multiclass confusion matrix
# =========================================================
def choose_best_multiclass_model_and_seed():
    """
    Uses repeated_seed_results.csv to identify the best multiclass run
    according to macro_f1.
    """
    df = pd.read_csv(REPEATED_SEED_RESULTS_CSV)
    sub = df[df["task"] == "multiclass"].copy()

    if sub.empty:
        raise ValueError("No multiclass rows found in repeated_seed_results.csv")

    best_row = sub.sort_values("macro_f1", ascending=False).iloc[0]
    return {
        "model": str(best_row["model"]),
        "seed": int(best_row["seed"]),
        "macro_f1": float(best_row["macro_f1"]),
    }

def resolve_multiclass_prediction_csv():
    """
    Tries to locate the best multiclass repeated-seed prediction file first.
    If it does not exist, falls back to the standard with-time-ablation
    multiclass prediction file (currently available in your results/predictions folder).
    """
    best = choose_best_multiclass_model_and_seed()
    model = best["model"]
    seed = best["seed"]

    # Expected location if you later export/save repeated-seed multiclass predictions
    repeated_seed_path = ROBUSTNESS_DIR / "predictions" / f"{model}_multiclass_seed{seed}_predictions.csv"

    if repeated_seed_path.exists():
        print(f"[info] using repeated-seed multiclass prediction file: {repeated_seed_path.name}")
        return repeated_seed_path, model, seed, True

    # Fallback to currently available standard prediction file
    fallback_path = PRED_DIR_STANDARD / f"{model}_multiclass_with_time_ablation_predictions.csv"
    if fallback_path.exists():
        print(f"[info] repeated-seed multiclass file not found, using fallback: {fallback_path.name}")
        return fallback_path, model, seed, False

    raise FileNotFoundError(
        f"Could not find either:\n"
        f"  {repeated_seed_path}\n"
        f"or\n"
        f"  {fallback_path}"
    )

def plot_best_multiclass_confusion_matrix():
    pred_csv, best_model, best_seed, is_seed_specific = resolve_multiclass_prediction_csv()

    df = pd.read_csv(pred_csv)
    if "y_true" not in df.columns or "y_pred" not in df.columns:
        raise ValueError(f"{pred_csv.name} must contain y_true and y_pred columns.")

    y_true = df["y_true"].astype(int).to_numpy()
    y_pred = df["y_pred"].astype(int).to_numpy()

    label_map = load_label_map()

    labels = sorted(np.unique(np.concatenate([y_true, y_pred])))
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    # normalize by row (true class)
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_pct = np.divide(cm, row_sums, out=np.zeros_like(cm, dtype=float), where=row_sums != 0) * 100.0

    class_names = [label_map.get(lbl, str(lbl)) for lbl in labels]

    fig, ax = plt.subplots(figsize=(9.2, 7.4))
    im = ax.imshow(cm_pct, cmap="Blues", vmin=0, vmax=100)

    model_title = MODEL_DISPLAY.get(best_model, best_model)
    if is_seed_specific:
        title = f"{model_title} - Normalized Multiclass Confusion Matrix (Seed {best_seed})"
    else:
        title = f"{model_title} - Normalized Multiclass Confusion Matrix"

    ax.set_title(title, pad=12)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")

    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(class_names, fontsize=9)

    for i in range(cm_pct.shape[0]):
        for j in range(cm_pct.shape[1]):
            value = cm_pct[i, j]
            text_color = "white" if value >= 50 else "black"
            ax.text(j, i, f"{value:.1f}%", ha="center", va="center", color=text_color, fontsize=8)

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Percentage (%)")

    fig.tight_layout()
    save_path = OUT_DIR / f"{best_model}_multiclass_confusion_matrix_normalized.png"
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {save_path}")




# =========================================================
# Repeated-seed stability
# =========================================================
def plot_repeated_seed_stability():
    """
    Final paper-ready repeated-seed stability figure.

    Left:
        Binary F1-score across five stratified seeds.

    Right:
        Multiclass Macro F1-score across five stratified seeds.

    Legend:
        Model (Binary mean±SD / Multiclass mean±SD)
    """

    results_path = RESULTS_DIR / "robustness" / "repeated_seed_results.csv"

    if not results_path.exists():
        print(f"[skip] repeated-seed results not found: {results_path}")
        return

    df = pd.read_csv(results_path)

    required_cols = {"model", "task", "seed", "objective_score"}
    missing = required_cols - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing required columns in repeated_seed_results.csv: {missing}"
        )

    seed_order = [42, 7, 21, 100, 2026]
    seed_labels = [str(seed) for seed in seed_order]
    x = np.arange(len(seed_order))

    task_settings = {
        "binary": {
            "title": "Binary Attack Detection",
            "ylabel": "F1-score (%)",
        },
        "multiclass": {
            "title": "Multiclass Attack Classification",
            "ylabel": "Macro F1-score (%)",
        },
    }

    # More compact canvas, but substantially larger plot text/elements
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(12.8, 5.7),
        sharey=False
    )

    fig.suptitle(
        "Repeated-Seed Stability of Tuned Supervised Models",
        fontsize=17,
        y=0.975
    )

    legend_handles = []
    legend_labels = []
    model_summaries = {}

    # ---------------------------------------------------------
    # Compute mean ± SD
    # ---------------------------------------------------------
    for model_key in MODEL_ORDER:
        model_summaries[model_key] = {}

        for task in ["binary", "multiclass"]:
            sub = df[
                (df["model"] == model_key) &
                (df["task"] == task)
            ].copy()

            vals = pd.to_numeric(
                sub["objective_score"],
                errors="coerce"
            ).dropna()

            if len(vals) > 0:
                model_summaries[model_key][task] = {
                    "mean": vals.mean() * 100,
                    "std": (
                        vals.std(ddof=1) * 100
                        if len(vals) > 1 else 0.0
                    ),
                }

    # ---------------------------------------------------------
    # Plot task panels
    # ---------------------------------------------------------
    for ax, task in zip(axes, ["binary", "multiclass"]):

        task_df = df[df["task"] == task].copy()
        all_values = []

        for model_key in MODEL_ORDER:

            model_df = task_df[
                task_df["model"] == model_key
            ].copy()

            if model_df.empty:
                continue

            model_df["seed"] = model_df["seed"].astype(int)

            values = []

            for seed in seed_order:
                seed_row = model_df[
                    model_df["seed"] == seed
                ]

                if seed_row.empty:
                    values.append(np.nan)
                else:
                    values.append(
                        seed_row["objective_score"].iloc[0] * 100
                    )

            valid_values = [
                v for v in values
                if not np.isnan(v)
            ]
            all_values.extend(valid_values)

            line, = ax.plot(
                x,
                values,
                marker="o",
                markersize=7.5,
                linewidth=2.6,
                color=MODEL_COLORS[model_key],
                label=MODEL_DISPLAY[model_key]
            )

            if task == "binary":
                legend_handles.append(line)

        # Panel title
        ax.set_title(
            task_settings[task]["title"],
            fontsize=14,
            pad=9
        )

        # Axis labels
        ax.set_ylabel(
            task_settings[task]["ylabel"],
            fontsize=12.5
        )

        ax.set_xlabel(
            "Seed",
            fontsize=12.5
        )

        # Tick labels
        ax.set_xticks(x)
        ax.set_xticklabels(
            seed_labels,
            fontsize=11.5
        )

        ax.tick_params(
            axis="y",
            labelsize=11.5
        )

        ax.grid(
            alpha=0.28,
            linewidth=0.8
        )

        # Controlled vertical spacing
        if all_values:
            ymin = min(all_values)
            ymax = max(all_values)

            spread = ymax - ymin
            padding = max(1.2, spread * 0.14)

            ax.set_ylim(
                ymin - padding,
                ymax + padding
            )

    # ---------------------------------------------------------
    # Short shared legend
    # ---------------------------------------------------------
    for model_key in MODEL_ORDER:

        binary_summary = model_summaries[
            model_key
        ].get("binary")

        multiclass_summary = model_summaries[
            model_key
        ].get("multiclass")

        if binary_summary and multiclass_summary:
            label = (
                f"{MODEL_DISPLAY[model_key]} "
                f"({binary_summary['mean']:.2f}±"
                f"{binary_summary['std']:.2f} / "
                f"{multiclass_summary['mean']:.2f}±"
                f"{multiclass_summary['std']:.2f})"
            )
        else:
            label = MODEL_DISPLAY[model_key]

        legend_labels.append(label)

    fig.legend(
        legend_handles,
        legend_labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.015),
        ncol=3,
        frameon=True,
        fontsize=10.2,
        handlelength=2.3,
        columnspacing=1.8
    )

    # Less wasted whitespace
    fig.subplots_adjust(
        left=0.075,
        right=0.985,
        top=0.855,
        bottom=0.205,
        wspace=0.21
    )

    save_path = OUT_DIR / "repeated_seed_stability.png"

    fig.savefig(
        save_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig)

    print(f"[saved] {save_path}")


# =========================================================
# Full-feature vs time-ablated comparison
# =========================================================
def plot_time_ablation_performance_drop():
    """
    Paper-ready comparison of full-feature and time-ablated performance.

    Left: Binary F1-score change.
    Right: Multiclass Macro F1-score change.

    Bars extend downward when performance decreases. Each bar is labeled
    directly at the zero line with the percentage change after removing
    time-derived features.
    """
    if not TIME_ABLATION_SUMMARY_CSV.exists():
        print(f"[skip] time-ablation summary not found: {TIME_ABLATION_SUMMARY_CSV}")
        return

    df = pd.read_csv(TIME_ABLATION_SUMMARY_CSV)

    required_cols = {
        "benchmark",
        "model",
        "task",
        "f1_score_mean",
        "macro_f1_mean",
    }
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(
            f"Missing required columns in {TIME_ABLATION_SUMMARY_CSV.name}: {sorted(missing)}"
        )

    task_settings = {
        "binary": {
            "title": "Binary Attack Detection",
            "metric": "f1_score_mean",
            "ylabel": "F1-score change (%)",
        },
        "multiclass": {
            "title": "Multiclass Attack Classification",
            "metric": "macro_f1_mean",
            "ylabel": "Macro F1-score change (%)",
        },
    }

    fig, axes = plt.subplots(1, 2, figsize=(11.8, 5.8), sharey=False)

    for ax, task in zip(axes, ["binary", "multiclass"]):
        setting = task_settings[task]
        metric = setting["metric"]

        labels = []
        changes = []
        colors = []

        for model_key in MODEL_ORDER:
            model_df = df[(df["model"] == model_key) & (df["task"] == task)].copy()

            full_row = model_df[model_df["benchmark"] == "full_feature"]
            ablated_row = model_df[model_df["benchmark"] == "time_ablated"]

            if full_row.empty or ablated_row.empty:
                print(f"[skip] missing ablation rows for {model_key} / {task}")
                continue

            full_val = full_row.iloc[0].get(metric, np.nan)
            ablated_val = ablated_row.iloc[0].get(metric, np.nan)

            if pd.isna(full_val) or pd.isna(ablated_val):
                print(f"[skip] missing {metric} for {model_key} / {task}")
                continue

            full_score = float(full_val) * 100
            ablated_score = float(ablated_val) * 100
            change = ablated_score - full_score

            labels.append(MODEL_DISPLAY[model_key])
            changes.append(change)
            colors.append(MODEL_COLORS[model_key])

        x = np.arange(len(labels))
        bars = ax.bar(x, changes, width=0.58, color=colors)

        # Strong zero-reference line, matching the drop-chart style.
        ax.axhline(0, color="gray", linewidth=1.2)

        # Put only the percentage change on the zero line above each bar.
        for i, bar in enumerate(bars):
            change = changes[i]
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                0.75,
                f"{change:.1f}%",
                ha="center",
                va="bottom",
                fontsize=10,
                fontweight="bold",
            )

        ax.set_title(setting["title"], fontsize=12)
        ax.set_ylabel(setting["ylabel"])
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.grid(axis="y", alpha=0.30)
        
        ax.yaxis.set_major_formatter(
            FuncFormatter(lambda value, pos: f"{value:.0f}%")
        )

        if changes:
            lowest = min(changes)
            # Enough room above zero for the value labels, and below for the bars.
            ax.set_ylim(min(lowest - 4.0, -8.0), 5.0)
            ticks = ax.get_yticks()
            ax.set_yticks([t for t in ticks if t not in (5, 10)])

    fig.suptitle(
        "Performance Change After Removing Time-Derived Features",
        fontsize=14,
        y=0.98,
    )

    fig.subplots_adjust(
        top=0.84,
        bottom=0.14,
        wspace=0.28,
    )

    save_path = OUT_DIR / "time_ablation_performance_drop.png"
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {save_path}")


# =========================================================
# Run
# =========================================================
if __name__ == "__main__":
    plot_supervised_binary_roc_pr()
    plot_best_multiclass_confusion_matrix()
    plot_repeated_seed_stability()
    plot_time_ablation_performance_drop()