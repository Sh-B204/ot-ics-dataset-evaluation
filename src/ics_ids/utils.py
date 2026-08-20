import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, precision_recall_fscore_support,
    classification_report,
)
from sklearn.preprocessing import label_binarize
from . import config


def resolve_data_dir(data_dir=None):
    if data_dir is None:
        return Path(config.DATA_DIR)
    return Path(data_dir)


def binary_paths(data_dir=None):
    data_dir = resolve_data_dir(data_dir)
    return data_dir / "binary_detection_train.csv", data_dir / "binary_detection_test.csv"


def multiclass_paths(data_dir=None):
    data_dir = resolve_data_dir(data_dir)
    return data_dir / "multiclass_classification_train.csv", data_dir / "multiclass_classification_test.csv"


def load_binary_datasets(data_dir=None):
    train_path, test_path = binary_paths(data_dir)
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    feature_cols = [c for c in train.columns if c != config.BINARY_LABEL_COL]
    X_train, y_train = train[feature_cols], train[config.BINARY_LABEL_COL]
    X_test, y_test = test[feature_cols], test[config.BINARY_LABEL_COL]
    return X_train, y_train, X_test, y_test, feature_cols


def load_multiclass_datasets(data_dir=None):
    train_path, test_path = multiclass_paths(data_dir)
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    feature_cols = [c for c in train.columns if c != config.MULTICLASS_LABEL_COL]
    X_train, y_train = train[feature_cols], train[config.MULTICLASS_LABEL_COL]
    X_test, y_test = test[feature_cols], test[config.MULTICLASS_LABEL_COL]
    return X_train, y_train, X_test, y_test, feature_cols


def load_multiclass_label_map(data_dir=None):
    data_dir = resolve_data_dir(data_dir)
    candidates = [data_dir / "multiclass_label_map.json"]
    for path in candidates:
        if path.exists():
            return json.loads(path.read_text())
    return None


def class_names_for_labels(labels, data_dir=None, short=False):
    label_map = load_multiclass_label_map(data_dir)
    if label_map is None:
        names = [str(l) for l in labels]
    else:
        inv_map = {int(v): k for k, v in label_map.items()}
        names = [inv_map.get(int(l), str(l)) for l in labels]
    if short:
        return [shorten_label_name(name) for name in names]
    return names


def shorten_label_name(name):
    if name == "Normal" or str(name).startswith("Normal"):
        return "Normal"
    name = str(name)
    if "_T" in name:
        return "T" + name.split("_T")[-1]
    return name


def binary_metrics(y_true, y_pred, y_score=None):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
        "false_positive_rate": float(fpr),
        "confusion_matrix": cm.tolist(),
        "confusion_matrix_labels": ["Normal (0)", "Attack (1)"],
    }
    if y_score is not None:
        try:
            metrics["roc_auc"] = float(roc_auc_score(y_true, y_score))
        except ValueError:
            metrics["roc_auc"] = None
    else:
        metrics["roc_auc"] = None
    return metrics


def multiclass_roc_auc_scores(y_true, y_prob, labels=None):
    if labels is None:
        labels = sorted(pd.unique(pd.Series(y_true)))
    y_true_bin = label_binarize(y_true, classes=labels)
    scores = {"micro_roc_auc": None, "macro_roc_auc": None, "weighted_roc_auc": None}
    try:
        scores["micro_roc_auc"] = float(roc_auc_score(y_true_bin, y_prob, average="micro", multi_class="ovr"))
    except ValueError:
        pass
    try:
        scores["macro_roc_auc"] = float(roc_auc_score(y_true_bin, y_prob, average="macro", multi_class="ovr"))
    except ValueError:
        pass
    try:
        scores["weighted_roc_auc"] = float(roc_auc_score(y_true_bin, y_prob, average="weighted", multi_class="ovr"))
    except ValueError:
        pass
    return scores


def multiclass_metrics(y_true, y_pred, labels=None, y_prob=None, data_dir=None):
    if labels is None:
        labels = sorted(pd.unique(pd.concat([pd.Series(y_true), pd.Series(y_pred)])))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average="macro", zero_division=0
    )
    weighted_p, weighted_r, weighted_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average="weighted", zero_division=0
    )
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(macro_p),
        "macro_recall": float(macro_r),
        "macro_f1": float(macro_f1),
        "weighted_precision": float(weighted_p),
        "weighted_recall": float(weighted_r),
        "weighted_f1": float(weighted_f1),
        "confusion_matrix": cm.tolist(),
        "confusion_matrix_labels": class_names_for_labels(labels, data_dir=data_dir),
        "label_order": [int(l) for l in labels],
    }
    if y_prob is not None:
        metrics.update(multiclass_roc_auc_scores(y_true, y_prob, labels=labels))
    return metrics


def save_json(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=str))


def save_predictions_csv(path, y_true, y_pred, y_score=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame({"y_true": np.asarray(y_true), "y_pred": np.asarray(y_pred)})
    if y_score is not None:
        df["y_score"] = np.asarray(y_score)
    df.to_csv(path, index=False)


def save_probabilities_csv(path, y_true, y_pred, y_prob):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    prob_df = pd.DataFrame(y_prob, columns=[f"class_{i}_probability" for i in range(y_prob.shape[1])])
    prob_df.insert(0, "y_true", np.asarray(y_true))
    prob_df.insert(1, "y_pred", np.asarray(y_pred))
    prob_df.to_csv(path, index=False)


def save_per_class_report(path, y_true, y_pred, labels=None, data_dir=None):
    if labels is None:
        labels = sorted(pd.unique(pd.concat([pd.Series(y_true), pd.Series(y_pred)])))
    names = class_names_for_labels(labels, data_dir=data_dir)
    report = classification_report(y_true, y_pred, labels=labels, target_names=names, output_dict=True, zero_division=0)
    rows = []
    for label, name in zip(labels, names):
        row = report.get(name, {})
        rows.append({
            "label": int(label),
            "class_name": name,
            "precision": row.get("precision", 0.0),
            "recall": row.get("recall", 0.0),
            "f1_score": row.get("f1-score", 0.0),
            "support": int(row.get("support", 0)),
        })
    df = pd.DataFrame(rows)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return df


def plot_confusion_matrix(cm, class_names, title, save_path, figsize=(6, 5), normalize=False, percent=False, vmax=None):
    cm = np.array(cm, dtype=float)
    if normalize:
        row_sums = cm.sum(axis=1, keepdims=True)
        cm_plot = np.divide(cm, row_sums, out=np.zeros_like(cm), where=row_sums != 0)
        if percent:
            cm_plot = cm_plot * 100
    else:
        cm_plot = cm

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(cm_plot, cmap="Blues", vmin=0, vmax=vmax)
    ax.set_title(title)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)

    thresh = (vmax / 2.0) if vmax is not None else (cm_plot.max() / 2.0 if cm_plot.max() > 0 else 0.5)
    for i in range(cm_plot.shape[0]):
        for j in range(cm_plot.shape[1]):
            if normalize and percent:
                text = f"{cm_plot[i, j]:.1f}%"
            elif normalize:
                text = f"{cm_plot[i, j]:.2f}"
            else:
                text = format(int(cm_plot[i, j]), "d")
            ax.text(j, i, text, ha="center", va="center", color="white" if cm_plot[i, j] > thresh else "black")

    cbar = fig.colorbar(im, ax=ax)
    if normalize and percent:
        cbar.set_label("Percentage (%)")
    fig.tight_layout()
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def print_header(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)