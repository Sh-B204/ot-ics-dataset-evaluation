import argparse
import time
from pathlib import Path
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, roc_curve, precision_recall_curve, average_precision_score, balanced_accuracy_score
from sklearn.model_selection import train_test_split
from .. import config
from .. import utils


SEEDS = [42, 7, 21, 100, 2026]
UNSUPERVISED_MODELS = ["isolation_forest", "stacked_autoencoder"]
MODEL_LABELS = {"isolation_forest": "IF", "stacked_autoencoder": "SAE"}
THRESHOLD_PERCENTILES = [80, 85, 90, 92, 95, 97, 99]
VAL_SIZE = 0.2
TEST_SIZE = 0.2
EPOCHS = 100
BATCH_SIZE = 256

DEFAULT_DATA_DIR = config.TIME_ABLATED_DATA_DIR
OUTPUT_DIR = config.RESULTS_DIR / "unsupervised_robustness"
FIGURE_DIR = config.FIGURES_DIR / "unsupervised_robustness"


def as_list(value):
    if isinstance(value, list):
        return value
    return [int(v.strip()) for v in str(value).split(",") if v.strip()]


def normalize_binary_labels(y):
    y = pd.Series(y).reset_index(drop=True)
    if y.dtype == object:
        lowered = y.astype(str).str.lower()
        if lowered.isin(["normal", "benign", "0", "normal (0)"]).any():
            return lowered.apply(lambda v: 0 if v in ["normal", "benign", "0", "normal (0)"] else 1).astype(int).to_numpy()
    return y.astype(int).to_numpy()


def clean_features(X):
    X = X.copy()
    drop_cols = [c for c in X.columns if str(c).lower() in ["label", "binary_label", "multiclass_label", "attack_type_label", "is_attack", "target", "y"]]
    if drop_cols:
        X = X.drop(columns=drop_cols)
    X = X.replace([np.inf, -np.inf], np.nan)
    for col in X.columns:
        if X[col].dtype == object:
            X[col] = pd.to_numeric(X[col], errors="ignore")
    object_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    if object_cols:
        X = pd.get_dummies(X, columns=object_cols, dummy_na=False)
    X = X.fillna(X.median(numeric_only=True))
    X = X.fillna(0)
    return X.astype("float32")


def read_y(path):
    df = pd.read_csv(path)
    if df.shape[1] == 1:
        return normalize_binary_labels(df.iloc[:, 0])
    for col in ["y", "label", "binary_label", "is_attack", "target"]:
        if col in df.columns:
            return normalize_binary_labels(df[col])
    return normalize_binary_labels(df.iloc[:, -1])


def find_existing(data_dir, names):
    for name in names:
        path = data_dir / name
        if path.exists():
            return path
    return None


def load_from_presplit(data_dir):
    x_train_path = find_existing(data_dir, ["X_train_binary.csv", "binary_X_train.csv", "X_binary_train.csv", "X_train.csv", "binary_train_features.csv"])
    x_test_path = find_existing(data_dir, ["X_test_binary.csv", "binary_X_test.csv", "X_binary_test.csv", "X_test.csv", "binary_test_features.csv"])
    y_train_path = find_existing(data_dir, ["y_train_binary.csv", "binary_y_train.csv", "y_binary_train.csv", "y_train.csv", "binary_train_labels.csv"])
    y_test_path = find_existing(data_dir, ["y_test_binary.csv", "binary_y_test.csv", "y_binary_test.csv", "y_test.csv", "binary_test_labels.csv"])

    if not all([x_train_path, x_test_path, y_train_path, y_test_path]):
        return None

    X_train = clean_features(pd.read_csv(x_train_path))
    X_test = clean_features(pd.read_csv(x_test_path))
    y_train = read_y(y_train_path)
    y_test = read_y(y_test_path)

    X_train, X_test = X_train.align(X_test, join="left", axis=1, fill_value=0)
    return X_train, y_train, X_test, y_test


def load_from_full_file(data_dir, seed):
    direct_names = ["binary_dataset.csv", "binary_benchmark.csv", "processed_binary_dataset.csv", "network_flow_binary.csv", "flow_binary_dataset.csv", "all_binary.csv"]
    candidates = [data_dir / name for name in direct_names if (data_dir / name).exists()]

    if not candidates:
        for path in data_dir.glob("*.csv"):
            try:
                cols = pd.read_csv(path, nrows=5).columns
            except Exception:
                continue
            lower = [str(c).lower() for c in cols]
            if any(c in lower for c in ["binary_label", "is_attack", "target", "label", "y"]):
                candidates.append(path)

    if not candidates:
        return None

    df = pd.read_csv(candidates[0])
    label_col = None
    for col in ["binary_label", "is_attack", "target", "label", "y"]:
        if col in df.columns:
            label_col = col
            break
    if label_col is None:
        for col in df.columns:
            if str(col).lower() in ["binary_label", "is_attack", "target", "label", "y"]:
                label_col = col
                break
    if label_col is None:
        return None

    y = normalize_binary_labels(df[label_col])
    X = clean_features(df.drop(columns=[label_col]))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, stratify=y, random_state=seed)
    return X_train.reset_index(drop=True), y_train, X_test.reset_index(drop=True), y_test


def load_binary_data(data_dir, seed):
    data_dir = Path(data_dir)

    try:
        loaded = utils.load_binary_datasets(data_dir=data_dir, random_state=seed)
        if loaded is not None:
            X_train, y_train, X_test, y_test, *_ = loaded
            return clean_features(pd.DataFrame(X_train)), normalize_binary_labels(y_train), clean_features(pd.DataFrame(X_test)), normalize_binary_labels(y_test)
    except TypeError:
        pass
    except Exception:
        pass

    try:
        loaded = utils.load_binary_datasets(data_dir=data_dir)
        if loaded is not None:
            X_train, y_train, X_test, y_test, *_ = loaded
            return clean_features(pd.DataFrame(X_train)), normalize_binary_labels(y_train), clean_features(pd.DataFrame(X_test)), normalize_binary_labels(y_test)
    except TypeError:
        pass
    except Exception:
        pass

    full_loaded = load_from_full_file(data_dir, seed)
    if full_loaded is not None:
        return full_loaded

    split_loaded = load_from_presplit(data_dir)
    if split_loaded is not None:
        return split_loaded

    loaded = utils.load_binary_datasets()
    X_train, y_train, X_test, y_test, *_ = loaded
    return clean_features(pd.DataFrame(X_train)), normalize_binary_labels(y_train), clean_features(pd.DataFrame(X_test)), normalize_binary_labels(y_test)


def binary_metrics(y_true, y_pred, y_score):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    fpr = fp / (fp + tn) if (fp + tn) else 0.0

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
        "false_positive_rate": float(fpr),
        "roc_auc": float(roc_auc_score(y_true, y_score)),
        "pr_auc": float(average_precision_score(y_true, y_score)),
        "confusion_matrix": cm.tolist(),
        "confusion_matrix_labels": ["Normal (0)", "Attack (1)"],
    }


def build_autoencoder(input_dim, seed):
    import tensorflow as tf
    from tensorflow.keras import layers, models

    tf.random.set_seed(seed)
    inputs = layers.Input(shape=(input_dim,))
    x = layers.Dense(32, activation="relu")(inputs)
    x = layers.Dense(16, activation="relu")(x)
    bottleneck = layers.Dense(8, activation="relu", name="bottleneck")(x)
    x = layers.Dense(16, activation="relu")(bottleneck)
    x = layers.Dense(32, activation="relu")(x)
    outputs = layers.Dense(input_dim, activation="linear")(x)

    model = models.Model(inputs, outputs, name="stacked_autoencoder")
    model.compile(optimizer="adam", loss="mse")
    return model


def reconstruction_error(model, X):
    recon = model.predict(X, verbose=0)
    return np.mean(np.square(X - recon), axis=1)


def select_threshold(train_normal_scores, val_scores, y_val):
    threshold_rows = []
    best_result = None
    best_f1 = -1

    for percentile in THRESHOLD_PERCENTILES:
        threshold = float(np.percentile(train_normal_scores, percentile))
        y_val_pred = (val_scores > threshold).astype(int)
        result = binary_metrics(y_val, y_val_pred, val_scores)
        result["threshold_percentile"] = percentile
        result["threshold"] = threshold
        threshold_rows.append(result)

        if result["f1_score"] > best_f1:
            best_f1 = result["f1_score"]
            best_result = result

    return best_result["threshold"], best_result["threshold_percentile"], threshold_rows


def run_isolation_forest(X_train, y_train, X_test, y_test, seed):
    X_fit, X_val, y_fit, y_val = train_test_split(X_train, y_train, test_size=VAL_SIZE, stratify=y_train, random_state=seed)
    X_fit_normal = X_fit[np.asarray(y_fit) == 0]

    model = IsolationForest(n_estimators=300, contamination="auto", random_state=seed, n_jobs=-1)
    start = time.time()
    model.fit(X_fit_normal)
    training_time = time.time() - start

    train_normal_scores = -model.decision_function(X_fit_normal)
    val_scores = -model.decision_function(X_val)
    test_scores = -model.decision_function(X_test)

    threshold, percentile, threshold_rows = select_threshold(train_normal_scores, val_scores, y_val)
    y_pred = (test_scores > threshold).astype(int)
    metrics = binary_metrics(y_test, y_pred, test_scores)

    return model, y_pred, test_scores, metrics, threshold, percentile, threshold_rows, training_time


def run_stacked_autoencoder(X_train, y_train, X_test, y_test, seed):
    import tensorflow as tf
    from tensorflow.keras import callbacks

    tf.random.set_seed(seed)
    np.random.seed(seed)

    X_train_arr = X_train.to_numpy(dtype="float32")
    X_test_arr = X_test.to_numpy(dtype="float32")
    X_fit, X_val, y_fit, y_val = train_test_split(X_train_arr, y_train, test_size=VAL_SIZE, stratify=y_train, random_state=seed)
    X_fit_normal = X_fit[np.asarray(y_fit) == 0]

    X_es_fit, X_es_val = train_test_split(X_fit_normal, test_size=0.1, random_state=seed)
    model = build_autoencoder(X_fit_normal.shape[1], seed)

    early_stop = callbacks.EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)
    start = time.time()
    history = model.fit(X_es_fit, X_es_fit, validation_data=(X_es_val, X_es_val), epochs=EPOCHS, batch_size=BATCH_SIZE, callbacks=[early_stop], verbose=0)
    training_time = time.time() - start

    train_normal_errors = reconstruction_error(model, X_fit_normal)
    val_errors = reconstruction_error(model, X_val)
    test_errors = reconstruction_error(model, X_test_arr)

    threshold, percentile, threshold_rows = select_threshold(train_normal_errors, val_errors, y_val)
    y_pred = (test_errors > threshold).astype(int)
    metrics = binary_metrics(y_test, y_pred, test_errors)
    metrics["epochs_trained"] = len(history.history["loss"])

    return model, y_pred, test_errors, metrics, threshold, percentile, threshold_rows, training_time


def run_one_model(model_name, X_train, y_train, X_test, y_test, seed):
    if model_name == "isolation_forest":
        return run_isolation_forest(X_train, y_train, X_test, y_test, seed)
    if model_name == "stacked_autoencoder":
        return run_stacked_autoencoder(X_train, y_train, X_test, y_test, seed)
    raise ValueError(f"Unknown model: {model_name}")


def summarize_results(results_df):
    metric_cols = ["accuracy", "balanced_accuracy", "precision", "recall", "f1_score", "false_positive_rate", "roc_auc", "pr_auc", "training_time_seconds"]
    rows = []

    for model, group in results_df.groupby("model"):
        row = {"model": model, "model_label": MODEL_LABELS.get(model, model), "task": "binary", "n_seeds": len(group)}
        for metric in metric_cols:
            row[f"{metric}_mean"] = group[metric].mean()
            row[f"{metric}_std"] = group[metric].std(ddof=1)
            row[f"{metric}_min"] = group[metric].min()
            row[f"{metric}_max"] = group[metric].max()
        rows.append(row)

    return pd.DataFrame(rows)


def create_paper_summary(summary_df):
    rows = []
    for _, row in summary_df.iterrows():
        rows.append({
            "Model": row["model_label"],
            "Accuracy": f"{row['accuracy_mean']*100:.2f} ± {row['accuracy_std']*100:.2f}",
            "Balanced accuracy": f"{row['balanced_accuracy_mean']*100:.2f} ± {row['balanced_accuracy_std']*100:.2f}",
            "Precision": f"{row['precision_mean']*100:.2f} ± {row['precision_std']*100:.2f}",
            "Recall": f"{row['recall_mean']*100:.2f} ± {row['recall_std']*100:.2f}",
            "F1-score": f"{row['f1_score_mean']*100:.2f} ± {row['f1_score_std']*100:.2f}",
            "FPR": f"{row['false_positive_rate_mean']*100:.2f} ± {row['false_positive_rate_std']*100:.2f}",
            "ROC-AUC": f"{row['roc_auc_mean']*100:.2f} ± {row['roc_auc_std']*100:.2f}",
            "PR-AUC": f"{row['pr_auc_mean']*100:.2f} ± {row['pr_auc_std']*100:.2f}",
            "Training time (s)": f"{row['training_time_seconds_mean']:.2f} ± {row['training_time_seconds_std']:.2f}",
        })
    return pd.DataFrame(rows)


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


def plot_unsupervised_curves(predictions_df, summary_df, figure_dir):
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


def regenerate_plots_from_existing(output_dir=OUTPUT_DIR, figure_dir=FIGURE_DIR):
    output_dir = Path(output_dir)
    figure_dir = Path(figure_dir)
    predictions_path = output_dir / "unsupervised_repeated_seed_predictions.csv"
    summary_path = output_dir / "unsupervised_repeated_seed_summary.csv"

    if not predictions_path.exists() or not summary_path.exists():
        print(f"[skip] missing {predictions_path} or {summary_path}")
        return

    predictions_df = pd.read_csv(predictions_path)
    summary_df = pd.read_csv(summary_path)
    plot_unsupervised_curves(predictions_df, summary_df, figure_dir)
    print(f"[saved figures] {figure_dir}")


def main():
    parser = argparse.ArgumentParser(description="Repeated-seed unsupervised binary anomaly detection on the time-ablated benchmark.")
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR), help="Path to the time-ablated processed benchmark.")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR), help="Output directory for unsupervised result CSVs.")
    parser.add_argument("--figure-dir", default=str(FIGURE_DIR), help="Output directory for unsupervised figures.")
    parser.add_argument("--models", nargs="+", default=UNSUPERVISED_MODELS, choices=UNSUPERVISED_MODELS)
    parser.add_argument("--seeds", default=",".join(str(s) for s in SEEDS), help="Comma-separated seeds.")
    parser.add_argument("--plot-only", action="store_true", help="Regenerate figures from saved CSVs without retraining.")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    figure_dir = Path(args.figure_dir)
    seeds = as_list(args.seeds)

    output_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    if args.plot_only:
        regenerate_plots_from_existing(output_dir, figure_dir)
        return

    utils.print_header("Unsupervised repeated-seed evaluation on time-ablated benchmark")
    print("Data:", data_dir)
    print("Output:", output_dir)
    print("Models:", args.models)
    print("Seeds:", seeds)

    all_results = []
    all_predictions = []
    all_thresholds = []

    for model_name in args.models:
        utils.print_header(f"{MODEL_LABELS.get(model_name, model_name)} | normal-only anomaly detection")

        for seed in seeds:
            X_train, y_train, X_test, y_test = load_binary_data(data_dir, seed)
            start = time.time()
            model, y_pred, y_score, metrics, threshold, percentile, threshold_rows, training_time = run_one_model(model_name, X_train, y_train, X_test, y_test, seed)
            runtime = time.time() - start

            metrics["model"] = model_name
            metrics["model_label"] = MODEL_LABELS.get(model_name, model_name)
            metrics["task"] = "binary"
            metrics["seed"] = seed
            metrics["benchmark"] = "time_ablated"
            metrics["benchmark_display"] = "Time-ablated"
            metrics["n_features"] = X_train.shape[1]
            metrics["threshold"] = threshold
            metrics["threshold_percentile"] = percentile
            metrics["threshold_selected_on"] = "validation_split (stratified, from train)"
            metrics["trained_on"] = "normal_samples_only"
            metrics["training_time_seconds"] = training_time
            metrics["runtime_seconds"] = runtime

            all_results.append(metrics)

            pred_df = pd.DataFrame({"model": model_name, "model_label": MODEL_LABELS.get(model_name, model_name), "seed": seed, "y_true": y_test, "y_pred": y_pred, "y_score": y_score})
            all_predictions.append(pred_df)

            for row in threshold_rows:
                row = row.copy()
                row["model"] = model_name
                row["seed"] = seed
                all_thresholds.append(row)

            print(f"seed={seed} f1={metrics['f1_score']:.4f} recall={metrics['recall']:.4f} precision={metrics['precision']:.4f} roc_auc={metrics['roc_auc']:.4f} pr_auc={metrics['pr_auc']:.4f}")

            if model_name == "isolation_forest":
                joblib.dump(model, config.MODELS_DIR / f"isolation_forest_binary_seed_{seed}.pkl")
            else:
                model.save(config.MODELS_DIR / f"stacked_autoencoder_binary_seed_{seed}.keras")

    results_df = pd.DataFrame(all_results)
    predictions_df = pd.concat(all_predictions, ignore_index=True)
    thresholds_df = pd.DataFrame(all_thresholds)
    summary_df = summarize_results(results_df)
    paper_df = create_paper_summary(summary_df)

    results_df.to_csv(output_dir / "unsupervised_repeated_seed_results.csv", index=False)
    predictions_df.to_csv(output_dir / "unsupervised_repeated_seed_predictions.csv", index=False)
    thresholds_df.to_csv(output_dir / "unsupervised_threshold_search_by_seed.csv", index=False)
    summary_df.to_csv(output_dir / "unsupervised_repeated_seed_summary.csv", index=False)
    paper_df.to_csv(output_dir / "unsupervised_repeated_seed_summary_paper_format.csv", index=False)

    plot_unsupervised_curves(predictions_df, summary_df, figure_dir)

    print(f"[saved] {output_dir / 'unsupervised_repeated_seed_results.csv'}")
    print(f"[saved] {output_dir / 'unsupervised_repeated_seed_summary.csv'}")
    print(f"[saved] {output_dir / 'unsupervised_repeated_seed_summary_paper_format.csv'}")
    print(f"[saved figures] {figure_dir}")
    print("Done.")


if __name__ == "__main__":
    main()
