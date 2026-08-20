import argparse
import json
import time
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import ParameterSampler, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier
from .. import config
from .. import utils


RANDOM_STATE = getattr(config, "RANDOM_STATE", 42)
OUT_DIR = config.RESULTS_DIR / "hyperparameter_study"
FIG_DIR = config.RESULTS_DIR / "figures" / "hyperparameter_study"

RF_ITER = 30
XGB_ITER = 36
MLP_ITER = 28
VAL_SIZE = 0.2

TASKS = ["binary", "multiclass"]
MODELS = ["random_forest", "xgboost", "mlp"]


RF_SPACE = {
    "n_estimators": [200, 300, 500, 700],
    "max_depth": [None, 10, 20, 30, 40],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "max_features": ["sqrt", "log2", None],
    "bootstrap": [True],
}

XGB_SPACE = {
    "n_estimators": [200, 300, 400, 600, 800],
    "max_depth": [3, 4, 5, 6, 8],
    "learning_rate": [0.01, 0.03, 0.05, 0.1, 0.2],
    "subsample": [0.7, 0.8, 0.9, 1.0],
    "colsample_bytree": [0.7, 0.8, 0.9, 1.0],
    "min_child_weight": [1, 3, 5],
    "gamma": [0, 0.1, 0.5],
    "reg_lambda": [1, 2, 5],
}

MLP_SPACE = {
    "hidden_layer_sizes": [(128, 64, 32), (256, 128, 64), (128, 64), (64, 32), (256, 128, 64, 32)],
    "activation": ["relu", "tanh"],
    "alpha": [0.0001, 0.0005, 0.001, 0.005],
    "learning_rate_init": [0.0001, 0.0005, 0.001, 0.003],
    "batch_size": [128, 256, 512],
    "early_stopping": [True],
    "validation_fraction": [0.1],
}


def normalize_requested(value, allowed):
    if value == "all":
        return allowed
    values = [v.strip() for v in value.split(",") if v.strip()]
    bad = [v for v in values if v not in allowed]
    if bad:
        raise ValueError(f"Invalid values {bad}. Allowed: {allowed} or all")
    return values


def sampled_configs(space, n_iter, seed):
    all_count = np.prod([len(v) for v in space.values()])
    n = min(n_iter, int(all_count))
    return list(ParameterSampler(space, n_iter=n, random_state=seed))


def load_task_data(task):
    if task == "binary":
        X_train, y_train, _, _, features = utils.load_binary_datasets(config.TIME_ABLATED_DATA_DIR)
    else:
        X_train, y_train, _, _, features = utils.load_multiclass_datasets(config.TIME_ABLATED_DATA_DIR)
    return X_train, y_train, features


def split_validation(X, y):
    return train_test_split(X, y, test_size=VAL_SIZE, stratify=y, random_state=RANDOM_STATE)


def binary_eval(y_true, y_pred, y_score=None):
    row = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1_score": f1_score(y_true, y_pred, zero_division=0),
    }
    if y_score is not None:
        try:
            row["roc_auc"] = roc_auc_score(y_true, y_score)
        except ValueError:
            row["roc_auc"] = np.nan
    return row


def multiclass_eval(y_true, y_pred, y_prob=None):
    row = {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "macro_recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
    }
    if y_prob is not None:
        scores = utils.multiclass_roc_auc_scores(y_true, y_prob, labels=sorted(np.unique(y_true)))
        row.update(scores)
    return row


def objective_metric(task, metrics):
    if task == "binary":
        return metrics.get("f1_score", np.nan)
    return metrics.get("macro_f1", np.nan)


def build_rf(params, task):
    if task == "binary":
        class_weight = "balanced_subsample"
    else:
        class_weight = "balanced_subsample"
    return RandomForestClassifier(**params, class_weight=class_weight, random_state=RANDOM_STATE, n_jobs=-1)


def build_xgb(params, task, y_fit):
    base = {
        "objective": "binary:logistic" if task == "binary" else "multi:softprob",
        "eval_metric": "logloss" if task == "binary" else "mlogloss",
        "tree_method": "hist",
        "random_state": RANDOM_STATE,
        "n_jobs": -1,
    }
    if task == "binary":
        neg = int((y_fit == 0).sum())
        pos = int((y_fit == 1).sum())
        base["scale_pos_weight"] = neg / pos if pos else 1.0
    else:
        base["num_class"] = int(pd.Series(y_fit).nunique())
    base.update(params)
    return XGBClassifier(**base)


def build_mlp(params):
    return MLPClassifier(**params, solver="adam", max_iter=120, n_iter_no_change=10, random_state=RANDOM_STATE, verbose=False)


def run_one_config(model_name, task, params, config_id):
    X, y, features = load_task_data(task)
    X_fit, X_val, y_fit, y_val = split_validation(X, y)

    start = time.time()

    if model_name == "random_forest":
        model = build_rf(params, task)
        model.fit(X_fit, y_fit)
        y_pred = model.predict(X_val)
        if task == "binary":
            y_score = model.predict_proba(X_val)[:, 1]
            metrics = binary_eval(y_val, y_pred, y_score)
        else:
            y_prob = model.predict_proba(X_val)
            metrics = multiclass_eval(y_val, y_pred, y_prob)

    elif model_name == "xgboost":
        model = build_xgb(params, task, y_fit)
        if task == "binary":
            model.fit(X_fit, y_fit)
        else:
            sample_weights = compute_sample_weight(class_weight="balanced", y=y_fit)
            model.fit(X_fit, y_fit, sample_weight=sample_weights)
        y_pred = model.predict(X_val)
        if task == "binary":
            y_score = model.predict_proba(X_val)[:, 1]
            metrics = binary_eval(y_val, y_pred, y_score)
        else:
            y_prob = model.predict_proba(X_val)
            metrics = multiclass_eval(y_val, y_pred, y_prob)

    elif model_name == "mlp":
        model = build_mlp(params)
        sample_weights = compute_sample_weight(class_weight="balanced", y=y_fit)
        try:
            model.fit(X_fit, y_fit, sample_weight=sample_weights)
        except TypeError:
            model.fit(X_fit, y_fit)
        y_pred = model.predict(X_val)
        if task == "binary":
            y_score = model.predict_proba(X_val)[:, 1]
            metrics = binary_eval(y_val, y_pred, y_score)
        else:
            y_prob = model.predict_proba(X_val)
            metrics = multiclass_eval(y_val, y_pred, y_prob)

    else:
        raise ValueError(model_name)

    runtime = time.time() - start
    score = objective_metric(task, metrics)

    row = {
        "model": model_name,
        "task": task,
        "benchmark": "time_ablated",
        "config_id": config_id,
        "objective_metric": "f1_score" if task == "binary" else "macro_f1",
        "objective_score": score,
        "n_features": len(features),
        "runtime_seconds": runtime,
    }
    row.update(metrics)
    row.update({f"param_{k}": str(v) if isinstance(v, tuple) else v for k, v in params.items()})
    return row


def run_model_task(model_name, task, max_configs=None):
    if model_name == "random_forest":
        configs = sampled_configs(RF_SPACE, max_configs or RF_ITER, RANDOM_STATE + 1)
    elif model_name == "xgboost":
        configs = sampled_configs(XGB_SPACE, max_configs or XGB_ITER, RANDOM_STATE + 2)
    elif model_name == "mlp":
        configs = sampled_configs(MLP_SPACE, max_configs or MLP_ITER, RANDOM_STATE + 3)
    else:
        raise ValueError(model_name)

    rows = []
    utils.print_header(f"Hyperparameter study | {model_name} | {task} | {len(configs)} configs")

    for i, params in enumerate(configs, start=1):
        try:
            row = run_one_config(model_name, task, params, i)
            rows.append(row)
            print(f"[{i}/{len(configs)}] score={row['objective_score']:.4f} runtime={row['runtime_seconds']:.1f}s params={params}")
        except Exception as e:
            fail = {"model": model_name, "task": task, "benchmark": "time_ablated", "config_id": i, "error": str(e)}
            fail.update({f"param_{k}": str(v) if isinstance(v, tuple) else v for k, v in params.items()})
            rows.append(fail)
            print(f"[{i}/{len(configs)}] FAILED: {e}")

    df = pd.DataFrame(rows)
    path = OUT_DIR / f"{model_name}_{task}_hyperparameter_results.csv"
    df.to_csv(path, index=False)
    print(f"[saved] {path}")
    return df


def display_axis_name(col):
    name = col.replace("param_", "")
    mapping = {
        "n_estimators": "Estimators",
        "max_depth": "Max depth",
        "min_samples_split": "Min split",
        "min_samples_leaf": "Min leaf",
        "max_features": "Max features",
        "bootstrap": "Bootstrap",
        "learning_rate": "Learning rate",
        "subsample": "Subsample",
        "colsample_bytree": "Column sample",
        "min_child_weight": "Min child weight",
        "gamma": "Gamma",
        "reg_lambda": "L2 reg.",
        "hidden_layer_sizes": "MLP layers",
        "activation": "Activation",
        "alpha": "Alpha",
        "learning_rate_init": "Learning rate",
        "batch_size": "Batch size",
        "early_stopping": "Early stopping",
        "validation_fraction": "Val. fraction",
    }
    return mapping.get(name, name.replace("_", " ").title())


def display_model_name(model_name):
    mapping = {
        "random_forest": "Random Forest",
        "xgboost": "XGBoost",
        "mlp": "MLP",
    }
    return mapping.get(model_name, model_name.replace("_", " ").title())


def prepare_parallel_data(df, top_n=40):
    valid = df[pd.to_numeric(df.get("objective_score"), errors="coerce").notna()].copy()
    if valid.empty:
        return None, None, None, None, None

    valid["objective_score"] = pd.to_numeric(valid["objective_score"])
    valid = valid.sort_values("objective_score", ascending=False).head(top_n).reset_index(drop=True)

    param_cols = [c for c in valid.columns if c.startswith("param_") and valid[c].notna().any()]
    if not param_cols:
        return None, None, None, None, None

    norm_df = pd.DataFrame(index=valid.index)
    tick_info = {}

    force_categorical = {"param_max_depth", "param_max_features", "param_hidden_layer_sizes", "param_activation", "param_bootstrap", "param_early_stopping"}

    for col in param_cols:
        s = valid[col].copy()

        # Important for RF: max_depth=None is saved as blank/NaN in CSV.
        # Treat it as a real category named "None" and place it at the bottom of the axis,
        # so the line remains connected instead of disappearing.
        if col in force_categorical:
            s = s.where(s.notna(), "None").astype(str)
            s = s.replace({"nan": "None", "None": "None", "": "None"})
            categories = sorted(s.unique(), key=lambda x: (x != "None", str(x)))
            mapping = {v: i for i, v in enumerate(categories)}
            numeric = s.map(mapping).astype(float)
            if len(categories) == 1:
                norm_df[col] = 0.5
            else:
                norm_df[col] = numeric / (len(categories) - 1)
            tick_info[col] = {"type": "categorical", "categories": categories}
            continue

        numeric = pd.to_numeric(s, errors="coerce")

        # If numeric conversion creates NaN, keep those values as a category instead of breaking lines.
        if numeric.isna().any():
            s = s.where(s.notna(), "None").astype(str)
            s = s.replace({"nan": "None", "None": "None", "": "None"})
            categories = sorted(s.unique(), key=lambda x: (x != "None", str(x)))
            mapping = {v: i for i, v in enumerate(categories)}
            numeric_cat = s.map(mapping).astype(float)
            if len(categories) == 1:
                norm_df[col] = 0.5
            else:
                norm_df[col] = numeric_cat / (len(categories) - 1)
            tick_info[col] = {"type": "categorical", "categories": categories}
            continue

        min_v, max_v = numeric.min(), numeric.max()
        if pd.isna(min_v) or pd.isna(max_v) or min_v == max_v:
            norm_df[col] = 0.5
        else:
            norm_df[col] = (numeric - min_v) / (max_v - min_v)
        tick_info[col] = {"type": "numeric", "min": min_v, "max": max_v}

    return valid, norm_df[param_cols], param_cols, tick_info, valid["objective_score"]


def format_tick_value(value):
    if pd.isna(value):
        return ""
    try:
        value = float(value)
        if abs(value) >= 100:
            return f"{value:.0f}"
        if abs(value) >= 10:
            return f"{value:.1f}"
        if abs(value) >= 1:
            return f"{value:.2f}"
        return f"{value:.3f}"
    except Exception:
        return str(value)


def objective_label_for_task(task):
    return "Validation F1-score" if task == "binary" else "Validation macro F1-score"


def normalize_score(score, score_min, score_max):
    if score_max == score_min:
        return 0.5
    return (float(score) - score_min) / (score_max - score_min)


def plot_parallel(df, model_name, task):
    valid, plot_df, cols, tick_info, scores = prepare_parallel_data(df, top_n=40)
    if plot_df is None or plot_df.empty:
        return

    score_min = float(scores.min())
    score_max = float(scores.max())

    if score_min == score_max:
        score_min = max(0.0, score_min - 0.01)
        score_max = min(1.0, score_max + 0.01)

    cmap = plt.cm.plasma
    norm = plt.Normalize(vmin=score_min, vmax=score_max)

    n_param_axes = len(cols)
    xs = np.arange(n_param_axes, dtype=float)
    strip_x = float(n_param_axes)
    strip_w = 0.18
    strip_center = strip_x + strip_w / 2

    fig, ax = plt.subplots(figsize=(15.5, 6.5))

    gradient = np.linspace(score_min, score_max, 256).reshape(-1, 1)
    ax.imshow(
        gradient,
        aspect="auto",
        cmap=cmap,
        norm=norm,
        origin="lower",
        extent=[strip_x, strip_x + strip_w, 0, 1],
        zorder=0,
    )
    ax.add_patch(plt.Rectangle((strip_x, 0), strip_w, 1, fill=False, edgecolor="black", linewidth=1.0, zorder=4))

    for i, row in plot_df.iterrows():
        score = float(valid.loc[i, "objective_score"])
        y_score = normalize_score(score, score_min, score_max)
        is_best = i == 0

        x_line = list(xs) + [strip_x]
        y_line = list(row.values) + [y_score]

        ax.plot(
            x_line,
            y_line,
            color=cmap(norm(score)),
            alpha=0.98 if is_best else 0.58,
            linewidth=3.2 if is_best else 1.35,
            zorder=5 if is_best else 3,
            solid_capstyle="round",
        )

    for x in xs:
        ax.axvline(x, color="black", linewidth=1.0, alpha=0.85, zorder=1)

    ax.set_xlim(xs[0] - 0.25, strip_x + strip_w + 0.55)
    ax.set_ylim(-0.04, 1.04)
    ax.set_yticks([])

    tick_positions = list(xs) + [strip_center]
    tick_labels = [display_axis_name(c) for c in cols] + [objective_label_for_task(task)]
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, rotation=25, ha="right", fontsize=9)

    for x, col in zip(xs, cols):
        info = tick_info[col]
        if info["type"] == "categorical":
            cats = info["categories"]
            if len(cats) <= 6:
                ys = [0.5] if len(cats) == 1 else np.linspace(0, 1, len(cats))
                for y, label in zip(ys, cats):
                    ax.text(x - 0.03, y, str(label), ha="right", va="center", fontsize=7, color="dimgray")
            else:
                ax.text(x - 0.03, 0, str(cats[0]), ha="right", va="center", fontsize=7, color="dimgray")
                ax.text(x - 0.03, 1, str(cats[-1]), ha="right", va="center", fontsize=7, color="dimgray")
        else:
            ax.text(x - 0.03, 0, format_tick_value(info["min"]), ha="right", va="center", fontsize=7, color="dimgray")
            ax.text(x - 0.03, 1, format_tick_value(info["max"]), ha="right", va="center", fontsize=7, color="dimgray")

    ax.text(strip_x + strip_w + 0.04, 0, f"{score_min:.3f}", ha="left", va="center", fontsize=7, color="dimgray")
    ax.text(strip_x + strip_w + 0.04, 1, f"{score_max:.3f}", ha="left", va="center", fontsize=7, color="dimgray")
    ax.text(strip_center, 1.035, f"{score_max:.3f}", ha="center", va="bottom", fontsize=7, color="dimgray")

    best_score = float(valid.iloc[0]["objective_score"])
    title = f"{display_model_name(model_name)} {task.title()} Hyperparameter Study"
    ax.set_title(title, fontsize=13, pad=14)
    ax.grid(axis="x", alpha=0.12)
    ax.spines[["left", "right", "top"]].set_visible(False)
    ax.spines["bottom"].set_alpha(0.25)

    ax.text(
        0.01,
        -0.18,
        f"Best validation score = {best_score:.4f} | Showing top {len(valid)} configurations",
        transform=ax.transAxes,
        fontsize=9,
        ha="left",
        va="center",
    )

    fig.subplots_adjust(left=0.05, right=0.98, bottom=0.26, top=0.86)
    out = FIG_DIR / f"{model_name}_{task}_hyperparameter_parallel.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {out}")


def read_existing_results(models, tasks):
    results = []
    missing = []
    for model_name in models:
        for task in tasks:
            path = OUT_DIR / f"{model_name}_{task}_hyperparameter_results.csv"
            if path.exists():
                df = pd.read_csv(path)
                results.append(df)
                print(f"[loaded] {path}")
            else:
                missing.append(path)
                print(f"[missing] {path}")
    return results, missing


def save_best(results):
    best = {}
    summary_rows = []

    for df in results:
        if df.empty or "objective_score" not in df.columns:
            continue
        valid = df[pd.to_numeric(df["objective_score"], errors="coerce").notna()].copy()
        if valid.empty:
            continue

        valid["objective_score"] = pd.to_numeric(valid["objective_score"])
        row = valid.sort_values("objective_score", ascending=False).iloc[0]
        model = row["model"]
        task = row["task"]
        key = f"{model}_{task}"

        params = {}
        for c in valid.columns:
            if c.startswith("param_"):
                name = c.replace("param_", "")
                val = row[c]
                if pd.isna(val):
                    val = None
                params[name] = val

        best[key] = {
            "model": model,
            "task": task,
            "benchmark": "time_ablated",
            "objective_metric": row.get("objective_metric"),
            "objective_score": float(row["objective_score"]),
            "params": params,
        }
        summary_rows.append(row.to_dict())

    (OUT_DIR / "best_hyperparameters.json").write_text(json.dumps(best, indent=2, default=str))
    pd.DataFrame(summary_rows).to_csv(OUT_DIR / "best_hyperparameter_summary.csv", index=False)
    print("[saved] best_hyperparameters.json")
    print("[saved] best_hyperparameter_summary.csv")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", default="all", help="all or comma-separated: random_forest,xgboost,mlp")
    parser.add_argument("--tasks", default="all", help="all or comma-separated: binary,multiclass")
    parser.add_argument("--max-configs", type=int, default=None, help="optional quick-test limit per model-task")
    parser.add_argument("--rebuild-from-results", action="store_true", help="do not retrain; reload saved hyperparameter CSVs, regenerate figures, all_hyperparameter_results.csv, and best_hyperparameters.json")
    parser.add_argument("--plot-only", action="store_true", help="do not retrain; reload saved hyperparameter CSVs and regenerate only figures")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    models = normalize_requested(args.models, MODELS)
    tasks = normalize_requested(args.tasks, TASKS)

    print("Hyperparameter study on time-ablated benchmark")
    print("Data:", Path(config.TIME_ABLATED_DATA_DIR).resolve())
    print("Output:", OUT_DIR.resolve())
    print("Models:", models)
    print("Tasks:", tasks)

    if args.rebuild_from_results or args.plot_only:
        utils.print_header("Rebuilding hyperparameter outputs from saved CSV files")
        all_results, missing = read_existing_results(models, tasks)

        if not all_results:
            raise FileNotFoundError("No saved hyperparameter result CSVs were found.")

        for df in all_results:
            if df.empty or "model" not in df.columns or "task" not in df.columns:
                continue
            model_name = str(df["model"].dropna().iloc[0])
            task = str(df["task"].dropna().iloc[0])
            plot_parallel(df, model_name, task)

        if not args.plot_only:
            combined = pd.concat(all_results, ignore_index=True)
            combined.to_csv(OUT_DIR / "all_hyperparameter_results.csv", index=False)
            print("[saved] all_hyperparameter_results.csv")
            save_best(all_results)

        print("\nDone.")
        return

    all_results = []
    for model_name in models:
        for task in tasks:
            df = run_model_task(model_name, task, max_configs=args.max_configs)
            all_results.append(df)
            plot_parallel(df, model_name, task)

    if all_results:
        combined = pd.concat(all_results, ignore_index=True)
        combined.to_csv(OUT_DIR / "all_hyperparameter_results.csv", index=False)
        print("[saved] all_hyperparameter_results.csv")
        save_best(all_results)

    print("\nDone.")


if __name__ == "__main__":
    main()