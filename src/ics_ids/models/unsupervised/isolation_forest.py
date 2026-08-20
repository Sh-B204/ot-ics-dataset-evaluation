import time

import joblib
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.metrics import average_precision_score
from sklearn.model_selection import train_test_split

from ... import config
from ... import utils

THRESHOLD_PERCENTILES = [80, 85, 90, 92, 95, 97, 99]
VAL_SIZE = 0.2 


def run():
    utils.print_header("Isolation Forest - Binary Anomaly Detection (trained on normal only)")
    X_train, y_train, X_test, y_test, _ = utils.load_binary_datasets()

    # Labeled validation split used only for threshold selection.
    X_fit, X_thresh_val, y_fit, y_thresh_val = train_test_split(
        X_train, y_train, test_size=VAL_SIZE,
        stratify=y_train, random_state=config.RANDOM_STATE,
    )
    X_fit_normal = X_fit[y_fit == 0]

    model = IsolationForest(
        n_estimators=300, contamination="auto",
        random_state=config.RANDOM_STATE, n_jobs=-1,
    )
    start = time.time()
    model.fit(X_fit_normal)
    training_time = time.time() - start

    # Higher score = more anomalous.
    train_normal_scores = -model.decision_function(X_fit_normal)
    val_scores = -model.decision_function(X_thresh_val)
    test_scores = -model.decision_function(X_test)

    threshold_search = []
    best_result = None
    best_f1 = -1

    for percentile in THRESHOLD_PERCENTILES:
        threshold = float(np.percentile(train_normal_scores, percentile))
        y_val_pred = (val_scores > threshold).astype(int)
        val_metrics = utils.binary_metrics(y_thresh_val.to_numpy(), y_val_pred, val_scores)
        val_metrics["threshold_percentile"] = percentile
        val_metrics["threshold"] = threshold

        threshold_search.append(val_metrics)
        if val_metrics["f1_score"] > best_f1:
            best_f1 = val_metrics["f1_score"]
            best_result = val_metrics

    threshold = best_result["threshold"]

    # Final, unbiased evaluation on the untouched test set.
    y_pred = (test_scores > threshold).astype(int)
    metrics = utils.binary_metrics(y_test.to_numpy(), y_pred, test_scores)
    metrics["pr_auc"] = float(average_precision_score(y_test, test_scores))
    metrics["model"] = "isolation_forest"
    metrics["task"] = "binary"
    metrics["threshold"] = threshold
    metrics["threshold_percentile"] = best_result["threshold_percentile"]
    metrics["threshold_selected_on"] = "validation_split (stratified, from train)"
    metrics["trained_on"] = "normal_samples_only"
    metrics["training_time_seconds"] = training_time
    print(metrics)

    joblib.dump(model, config.MODELS_DIR / "isolation_forest_binary.pkl")
    utils.save_json(metrics, config.METRICS_DIR / "isolation_forest_binary_metrics.json")
    utils.save_json(
        threshold_search,
        config.METRICS_DIR / "isolation_forest_threshold_search.json",
    )
    utils.save_predictions_csv(
        config.PREDICTIONS_DIR / "isolation_forest_binary_predictions.csv",
        y_test, y_pred, test_scores,
    )
    utils.plot_confusion_matrix(
        metrics["confusion_matrix"], metrics["confusion_matrix_labels"],
        "Isolation Forest - Binary Confusion Matrix",
        config.FIGURES_DIR / "isolation_forest_binary_confusion_matrix.png",
    )


if __name__ == "__main__":
    run()
