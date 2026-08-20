import time
import joblib
import numpy as np
import pandas as pd
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier
from ... import config
from ... import utils


BINARY_DEFAULT_PARAMS = {
    "n_estimators": 400,
    "max_depth": 6,
    "learning_rate": 0.1,
    "subsample": 1.0,
    "colsample_bytree": 1.0,
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "tree_method": "hist",
    "random_state": config.RANDOM_STATE,
    "n_jobs": -1,
}

MULTICLASS_DEFAULT_PARAMS = {
    "n_estimators": 400,
    "max_depth": 6,
    "learning_rate": 0.1,
    "subsample": 1.0,
    "colsample_bytree": 1.0,
    "objective": "multi:softprob",
    "eval_metric": "mlogloss",
    "tree_method": "hist",
    "random_state": config.RANDOM_STATE,
    "n_jobs": -1,
}


def suffix(run_name=None):
    return f"_{run_name}" if run_name else ""


def make_binary_model(y_train, params=None):
    model_params = BINARY_DEFAULT_PARAMS.copy()
    neg = int((y_train == 0).sum())
    pos = int((y_train == 1).sum())
    model_params["scale_pos_weight"] = neg / pos if pos > 0 else 1.0
    if params:
        model_params.update(params)
    return XGBClassifier(**model_params)


def make_multiclass_model(num_classes, params=None):
    model_params = MULTICLASS_DEFAULT_PARAMS.copy()
    model_params["num_class"] = int(num_classes)
    if params:
        model_params.update(params)
    return XGBClassifier(**model_params)


def save_feature_importance(model, feature_cols, path):
    df = pd.DataFrame({
        "feature": feature_cols,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def run_binary(data_dir=None, params=None, run_name=None):
    utils.print_header("XGBoost - Binary Attack Detection")
    data_dir = utils.resolve_data_dir(data_dir)
    tag = suffix(run_name)

    X_train, y_train, X_test, y_test, feature_cols = utils.load_binary_datasets(data_dir)

    model = make_binary_model(y_train, params)
    start = time.time()
    model.fit(X_train, y_train)
    training_time = time.time() - start

    y_pred = model.predict(X_test)
    y_score = model.predict_proba(X_test)[:, 1]

    metrics = utils.binary_metrics(y_test, y_pred, y_score)
    metrics["model"] = "xgboost"
    metrics["task"] = "binary"
    metrics["benchmark"] = data_dir.name
    metrics["n_features"] = len(feature_cols)
    metrics["training_time_seconds"] = training_time
    metrics["params"] = model.get_params()
    print(metrics)

    joblib.dump(model, config.MODELS_DIR / f"xgboost_binary{tag}.pkl")
    utils.save_json(metrics, config.METRICS_DIR / f"xgboost_binary{tag}_metrics.json")
    utils.save_predictions_csv(config.PREDICTIONS_DIR / f"xgboost_binary{tag}_predictions.csv", y_test, y_pred, y_score)
    save_feature_importance(model, feature_cols, config.METRICS_DIR / f"xgboost_binary{tag}_feature_importance.csv")
    utils.plot_confusion_matrix(
        metrics["confusion_matrix"], metrics["confusion_matrix_labels"],
        "XGBoost - Binary Confusion Matrix",
        config.FIGURES_DIR / f"xgboost_binary{tag}_confusion_matrix.png",
    )
    return metrics


def run_multiclass(data_dir=None, params=None, run_name=None):
    utils.print_header("XGBoost - Multiclass Attack Classification")
    data_dir = utils.resolve_data_dir(data_dir)
    tag = suffix(run_name)

    X_train, y_train, X_test, y_test, feature_cols = utils.load_multiclass_datasets(data_dir)
    num_classes = int(max(y_train.max(), y_test.max()) + 1)

    sample_weights = compute_sample_weight(class_weight="balanced", y=y_train)
    model = make_multiclass_model(num_classes, params)

    start = time.time()
    model.fit(X_train, y_train, sample_weight=sample_weights)
    training_time = time.time() - start

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)
    labels = list(range(num_classes))

    metrics = utils.multiclass_metrics(y_test, y_pred, labels=labels, y_prob=y_prob, data_dir=data_dir)
    metrics["model"] = "xgboost"
    metrics["task"] = "multiclass"
    metrics["benchmark"] = data_dir.name
    metrics["n_features"] = len(feature_cols)
    metrics["training_time_seconds"] = training_time
    metrics["params"] = model.get_params()
    print(metrics)

    joblib.dump(model, config.MODELS_DIR / f"xgboost_multiclass{tag}.pkl")
    utils.save_json(metrics, config.METRICS_DIR / f"xgboost_multiclass{tag}_metrics.json")
    utils.save_predictions_csv(config.PREDICTIONS_DIR / f"xgboost_multiclass{tag}_predictions.csv", y_test, y_pred)
    utils.save_probabilities_csv(config.PREDICTIONS_DIR / f"xgboost_multiclass{tag}_probabilities.csv", y_test, y_pred, y_prob)
    utils.save_per_class_report(config.METRICS_DIR / f"xgboost_multiclass{tag}_per_class_report.csv", y_test, y_pred, labels=labels, data_dir=data_dir)
    save_feature_importance(model, feature_cols, config.METRICS_DIR / f"xgboost_multiclass{tag}_feature_importance.csv")
    utils.plot_confusion_matrix(
        metrics["confusion_matrix"], metrics["confusion_matrix_labels"],
        "XGBoost - Multiclass Confusion Matrix",
        config.FIGURES_DIR / f"xgboost_multiclass{tag}_confusion_matrix.png",
        figsize=(9, 8),
    )
    return metrics


def run(data_dir=None, binary_params=None, multiclass_params=None, run_name=None):
    return {
        "binary": run_binary(data_dir=data_dir, params=binary_params, run_name=run_name),
        "multiclass": run_multiclass(data_dir=data_dir, params=multiclass_params, run_name=run_name),
    }


if __name__ == "__main__":
    run_binary()
    run_multiclass()