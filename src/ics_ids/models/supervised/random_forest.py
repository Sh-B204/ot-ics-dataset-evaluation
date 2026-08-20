import time
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from ... import config
from ... import utils


DEFAULT_PARAMS = {
    "n_estimators": 300,
    "max_depth": None,
    "n_jobs": -1,
    "random_state": config.RANDOM_STATE,
    "class_weight": "balanced_subsample",
}


def suffix(run_name=None):
    return f"_{run_name}" if run_name else ""


def make_model(params=None):
    model_params = DEFAULT_PARAMS.copy()
    if params:
        model_params.update(params)
    return RandomForestClassifier(**model_params)


def save_feature_importance(model, feature_cols, path):
    df = pd.DataFrame({
        "feature": feature_cols,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def run_binary(data_dir=None, params=None, run_name=None):
    utils.print_header("Random Forest - Binary Attack Detection")
    data_dir = utils.resolve_data_dir(data_dir)
    tag = suffix(run_name)

    X_train, y_train, X_test, y_test, feature_cols = utils.load_binary_datasets(data_dir)

    model = make_model(params)
    start = time.time()
    model.fit(X_train, y_train)
    training_time = time.time() - start

    y_pred = model.predict(X_test)
    y_score = model.predict_proba(X_test)[:, 1]

    metrics = utils.binary_metrics(y_test, y_pred, y_score)
    metrics["model"] = "random_forest"
    metrics["task"] = "binary"
    metrics["benchmark"] = data_dir.name
    metrics["n_features"] = len(feature_cols)
    metrics["training_time_seconds"] = training_time
    metrics["params"] = model.get_params()
    print(metrics)

    joblib.dump(model, config.MODELS_DIR / f"random_forest_binary{tag}.pkl")
    utils.save_json(metrics, config.METRICS_DIR / f"random_forest_binary{tag}_metrics.json")
    utils.save_predictions_csv(config.PREDICTIONS_DIR / f"random_forest_binary{tag}_predictions.csv", y_test, y_pred, y_score)
    save_feature_importance(model, feature_cols, config.METRICS_DIR / f"random_forest_binary{tag}_feature_importance.csv")
    utils.plot_confusion_matrix(
        metrics["confusion_matrix"], metrics["confusion_matrix_labels"],
        "Random Forest - Binary Confusion Matrix",
        config.FIGURES_DIR / f"random_forest_binary{tag}_confusion_matrix.png",
    )
    return metrics


def run_multiclass(data_dir=None, params=None, run_name=None):
    utils.print_header("Random Forest - Multiclass Attack Classification")
    data_dir = utils.resolve_data_dir(data_dir)
    tag = suffix(run_name)

    X_train, y_train, X_test, y_test, feature_cols = utils.load_multiclass_datasets(data_dir)

    model = make_model(params)
    start = time.time()
    model.fit(X_train, y_train)
    training_time = time.time() - start

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)
    labels = list(model.classes_)

    metrics = utils.multiclass_metrics(y_test, y_pred, labels=labels, y_prob=y_prob, data_dir=data_dir)
    metrics["model"] = "random_forest"
    metrics["task"] = "multiclass"
    metrics["benchmark"] = data_dir.name
    metrics["n_features"] = len(feature_cols)
    metrics["training_time_seconds"] = training_time
    metrics["params"] = model.get_params()
    print(metrics)

    joblib.dump(model, config.MODELS_DIR / f"random_forest_multiclass{tag}.pkl")
    utils.save_json(metrics, config.METRICS_DIR / f"random_forest_multiclass{tag}_metrics.json")
    utils.save_predictions_csv(config.PREDICTIONS_DIR / f"random_forest_multiclass{tag}_predictions.csv", y_test, y_pred)
    utils.save_probabilities_csv(config.PREDICTIONS_DIR / f"random_forest_multiclass{tag}_probabilities.csv", y_test, y_pred, y_prob)
    utils.save_per_class_report(config.METRICS_DIR / f"random_forest_multiclass{tag}_per_class_report.csv", y_test, y_pred, labels=labels, data_dir=data_dir)
    save_feature_importance(model, feature_cols, config.METRICS_DIR / f"random_forest_multiclass{tag}_feature_importance.csv")
    utils.plot_confusion_matrix(
        metrics["confusion_matrix"], metrics["confusion_matrix_labels"],
        "Random Forest - Multiclass Confusion Matrix",
        config.FIGURES_DIR / f"random_forest_multiclass{tag}_confusion_matrix.png",
        figsize=(9, 8),
    )
    return metrics


def run(data_dir=None, params=None, run_name=None):
    return {
        "binary": run_binary(data_dir=data_dir, params=params, run_name=run_name),
        "multiclass": run_multiclass(data_dir=data_dir, params=params, run_name=run_name),
    }


if __name__ == "__main__":
    run_binary()
    run_multiclass()