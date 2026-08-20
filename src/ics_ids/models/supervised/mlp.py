import time
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras import callbacks, layers, models, optimizers
import config
import utils


EPOCHS = 100
BATCH_SIZE = 256


BINARY_DEFAULT_PARAMS = {
    "layers_units": [128, 64, 32],
    "activation": "relu",
    "dropout_rates": [0.3, 0.2, 0.0],
    "learning_rate": 0.001,
    "batch_size": BATCH_SIZE,
}

MULTICLASS_DEFAULT_PARAMS = {
    "layers_units": [128, 64, 32],
    "activation": "relu",
    "dropout_rates": [0.3, 0.2, 0.0],
    "learning_rate": 0.001,
    "batch_size": BATCH_SIZE,
    "max_class_weight": 10.0,
}


def suffix(run_name=None):
    return f"_{run_name}" if run_name else ""


def make_params(defaults, params=None):
    final = defaults.copy()
    if params:
        final.update(params)
    return final


def add_dense_block(model, units, activation, dropout_rate):
    model.add(layers.Dense(units, activation=activation))
    model.add(layers.BatchNormalization())
    if dropout_rate and dropout_rate > 0:
        model.add(layers.Dropout(dropout_rate))


def build_binary_mlp(input_dim, params=None):
    params = make_params(BINARY_DEFAULT_PARAMS, params)
    model = models.Sequential()
    model.add(layers.Input(shape=(input_dim,)))
    for idx, units in enumerate(params["layers_units"]):
        dropout_rate = params["dropout_rates"][idx] if idx < len(params["dropout_rates"]) else 0.0
        add_dense_block(model, units, params["activation"], dropout_rate)
    model.add(layers.Dense(1, activation="sigmoid"))
    model.compile(
        optimizer=optimizers.Adam(learning_rate=params["learning_rate"]),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model, params


def build_multiclass_mlp(input_dim, num_classes, params=None):
    params = make_params(MULTICLASS_DEFAULT_PARAMS, params)
    model = models.Sequential()
    model.add(layers.Input(shape=(input_dim,)))
    for idx, units in enumerate(params["layers_units"]):
        dropout_rate = params["dropout_rates"][idx] if idx < len(params["dropout_rates"]) else 0.0
        add_dense_block(model, units, params["activation"], dropout_rate)
    model.add(layers.Dense(num_classes, activation="softmax"))
    model.compile(
        optimizer=optimizers.Adam(learning_rate=params["learning_rate"]),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model, params


def make_class_weight_dict(y_train):
    classes = np.unique(y_train)
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
    return {int(cls): float(weight) for cls, weight in zip(classes, weights)}


def make_soft_class_weight_dict(y_train, max_weight=10.0):
    classes = np.unique(y_train)
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
    weights = np.sqrt(weights)
    weights = np.minimum(weights, max_weight)
    return {int(cls): float(weight) for cls, weight in zip(classes, weights)}


def run_binary(data_dir=None, params=None, run_name=None):
    utils.print_header("MLP - Binary Attack Detection")
    data_dir = utils.resolve_data_dir(data_dir)
    tag = suffix(run_name)

    tf.keras.backend.clear_session()
    tf.random.set_seed(config.RANDOM_STATE)
    np.random.seed(config.RANDOM_STATE)

    X_train, y_train, X_test, y_test, feature_cols = utils.load_binary_datasets(data_dir)

    X_train_arr = X_train.to_numpy(dtype="float32")
    X_test_arr = X_test.to_numpy(dtype="float32")
    y_train_arr = y_train.to_numpy(dtype="int32")
    y_test_arr = y_test.to_numpy(dtype="int32")

    X_fit, X_val, y_fit, y_val = train_test_split(
        X_train_arr, y_train_arr, test_size=0.1, stratify=y_train_arr, random_state=config.RANDOM_STATE
    )

    class_weights = make_class_weight_dict(y_fit)
    model, final_params = build_binary_mlp(input_dim=X_train_arr.shape[1], params=params)
    early_stop = callbacks.EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)

    start = time.time()
    history = model.fit(
        X_fit, y_fit,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=final_params["batch_size"],
        callbacks=[early_stop],
        class_weight=class_weights,
        verbose=2,
    )
    training_time = time.time() - start

    y_score = model.predict(X_test_arr, verbose=0).ravel()
    y_pred = (y_score >= 0.5).astype(int)

    metrics = utils.binary_metrics(y_test_arr, y_pred, y_score)
    metrics["model"] = "mlp"
    metrics["task"] = "binary"
    metrics["benchmark"] = data_dir.name
    metrics["n_features"] = len(feature_cols)
    metrics["training_time_seconds"] = training_time
    metrics["epochs_trained"] = len(history.history["loss"])
    metrics["params"] = final_params
    print(metrics)

    model.save(config.MODELS_DIR / f"mlp_binary{tag}.keras")
    utils.save_json(metrics, config.METRICS_DIR / f"mlp_binary{tag}_metrics.json")
    utils.save_json(history.history, config.METRICS_DIR / f"mlp_binary{tag}_training_history.json")
    utils.save_predictions_csv(config.PREDICTIONS_DIR / f"mlp_binary{tag}_predictions.csv", y_test_arr, y_pred, y_score)
    utils.plot_confusion_matrix(
        metrics["confusion_matrix"], metrics["confusion_matrix_labels"],
        "MLP - Binary Confusion Matrix",
        config.FIGURES_DIR / f"mlp_binary{tag}_confusion_matrix.png",
    )
    return metrics


def run_multiclass(data_dir=None, params=None, run_name=None):
    utils.print_header("MLP - Multiclass Attack Classification")
    data_dir = utils.resolve_data_dir(data_dir)
    tag = suffix(run_name)

    tf.keras.backend.clear_session()
    tf.random.set_seed(config.RANDOM_STATE)
    np.random.seed(config.RANDOM_STATE)

    X_train, y_train, X_test, y_test, feature_cols = utils.load_multiclass_datasets(data_dir)

    X_train_arr = X_train.to_numpy(dtype="float32")
    X_test_arr = X_test.to_numpy(dtype="float32")
    y_train_arr = y_train.to_numpy(dtype="int32")
    y_test_arr = y_test.to_numpy(dtype="int32")

    num_classes = int(max(y_train_arr.max(), y_test_arr.max()) + 1)
    model, final_params = build_multiclass_mlp(input_dim=X_train_arr.shape[1], num_classes=num_classes, params=params)
    class_weights = make_soft_class_weight_dict(y_train_arr, max_weight=final_params.get("max_class_weight", 10.0))
    early_stop = callbacks.EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)

    X_fit, X_val, y_fit, y_val = train_test_split(
        X_train_arr, y_train_arr, test_size=0.1, stratify=y_train_arr, random_state=config.RANDOM_STATE
    )

    start = time.time()
    history = model.fit(
        X_fit, y_fit,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=final_params["batch_size"],
        callbacks=[early_stop],
        class_weight=class_weights,
        verbose=2,
    )
    training_time = time.time() - start

    y_prob = model.predict(X_test_arr, verbose=0)
    y_pred = np.argmax(y_prob, axis=1)
    labels = list(range(num_classes))

    metrics = utils.multiclass_metrics(y_test_arr, y_pred, labels=labels, y_prob=y_prob, data_dir=data_dir)
    metrics["model"] = "mlp"
    metrics["task"] = "multiclass"
    metrics["benchmark"] = data_dir.name
    metrics["n_features"] = len(feature_cols)
    metrics["training_time_seconds"] = training_time
    metrics["epochs_trained"] = len(history.history["loss"])
    metrics["params"] = final_params
    print(metrics)

    model.save(config.MODELS_DIR / f"mlp_multiclass{tag}.keras")
    utils.save_json(metrics, config.METRICS_DIR / f"mlp_multiclass{tag}_metrics.json")
    utils.save_json(history.history, config.METRICS_DIR / f"mlp_multiclass{tag}_training_history.json")
    utils.save_predictions_csv(config.PREDICTIONS_DIR / f"mlp_multiclass{tag}_predictions.csv", y_test_arr, y_pred)
    utils.save_probabilities_csv(config.PREDICTIONS_DIR / f"mlp_multiclass{tag}_probabilities.csv", y_test_arr, y_pred, y_prob)
    utils.save_per_class_report(config.METRICS_DIR / f"mlp_multiclass{tag}_per_class_report.csv", y_test_arr, y_pred, labels=labels, data_dir=data_dir)
    utils.plot_confusion_matrix(
        metrics["confusion_matrix"], metrics["confusion_matrix_labels"],
        "MLP - Multiclass Confusion Matrix",
        config.FIGURES_DIR / f"mlp_multiclass{tag}_confusion_matrix.png",
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