import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from .. import config
from .. import utils

MODELS = ["random_forest", "xgboost", "mlp"]
TASKS = ["binary", "multiclass"]

MODEL_DISPLAY = {"random_forest": "Random Forest", "xgboost": "XGBoost", "mlp": "MLP"}
TASK_DISPLAY = {"binary": "Binary", "multiclass": "Multiclass"}

ROBUSTNESS_DIR = config.RESULTS_DIR / "robustness"
RESULTS_CSV = ROBUSTNESS_DIR / "repeated_seed_results.csv"
PREDICTIONS_DIR = ROBUSTNESS_DIR / "predictions"
CONFUSION_MATRIX_DIR = config.FIGURES_DIR / "confusion_matrices"

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

def load_multiclass_label_map():
    path = config.TIME_ABLATED_DATA_DIR / "multiclass_label_map.json"
    if not path.exists():
        return None
    raw = json.loads(path.read_text())
    return {int(value): name for name, value in raw.items()}

def find_best_seed(model, task):
    if not RESULTS_CSV.exists():
        raise FileNotFoundError(f"Missing repeated-seed results: {RESULTS_CSV}")
    results = pd.read_csv(RESULTS_CSV)
    subset = results[(results["model"] == model) & (results["task"] == task)].copy()
    if subset.empty:
        raise ValueError(f"No repeated-seed results found for {model} / {task}")
    best_row = subset.sort_values("objective_score", ascending=False).iloc[0]
    return int(best_row["seed"])

def load_prediction_file(model, task, seed):
    if task == "binary":
        path = PREDICTIONS_DIR / f"{model}_binary_seed{seed}_predictions.csv"
    else:
        path = PREDICTIONS_DIR / f"{model}_multiclass_seed{seed}_probabilities.csv"
    if not path.exists():
        print(f"[skip] missing prediction file: {path}")
        return None
    return pd.read_csv(path)

def get_class_names(task, labels):
    if task == "binary":
        mapping = {0: "Normal", 1: "Attack"}
        return [mapping.get(int(label), str(label)) for label in labels]
    label_map = load_multiclass_label_map()
    if label_map is None:
        return [str(label) for label in labels]
    names = [label_map.get(int(label), str(label)) for label in labels]
    return shorten_class_names(names)

def plot_confusion_matrix_normalized(model, task):
    seed = find_best_seed(model, task)
    predictions = load_prediction_file(model, task, seed)
    if predictions is None:
        return
    if "y_true" not in predictions.columns or "y_pred" not in predictions.columns:
        print(f"[skip] prediction file for {model} / {task} does not contain y_true and y_pred")
        return
    y_true = predictions["y_true"].astype(int).to_numpy()
    y_pred = predictions["y_pred"].astype(int).to_numpy()
    labels = sorted(np.unique(np.concatenate([y_true, y_pred])))
    cm = confusion_matrix(y_true, y_pred, labels=labels).astype(float)
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_pct = np.divide(cm, row_sums, out=np.zeros_like(cm, dtype=float), where=row_sums != 0) * 100.0
    class_names = get_class_names(task, labels)
    figsize = (9.5, 8) if task == "multiclass" else (6.6, 5.2)
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(cm_pct, cmap="Blues", vmin=0, vmax=100)
    model_title = MODEL_DISPLAY.get(model, model)
    task_title = TASK_DISPLAY.get(task, task)
    ax.set_title(f"{model_title} - Normalized {task_title} Confusion Matrix", fontsize=13 if task == "binary" else 14, pad=16)
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
    cbar = fig.colorbar(im, ax=ax, pad=0.04)
    cbar.set_label("Percentage (%)")
    fig.tight_layout(pad=2.5)
    save_path = CONFUSION_MATRIX_DIR / f"{model}_{task}_confusion_matrix_normalized.png"
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {save_path} (best seed: {seed})")

def main():
    utils.print_header("Generating supervised confusion matrices")
    CONFUSION_MATRIX_DIR.mkdir(parents=True, exist_ok=True)
    for model in MODELS:
        for task in TASKS:
            plot_confusion_matrix_normalized(model, task)
    print("\nAll confusion matrices saved to:", CONFUSION_MATRIX_DIR.resolve())

if __name__ == "__main__":
    main()