import time

import numpy as np
import tensorflow as tf
from sklearn.metrics import average_precision_score
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models, callbacks

from ... import config
from ... import utils

THRESHOLD_PERCENTILES = [80, 85, 90, 92, 95, 97, 99]
VAL_SIZE = 0.2  # stratified split carved out of TRAIN only, for threshold selection
EPOCHS = 100
BATCH_SIZE = 256


def build_autoencoder(input_dim):
    # Bottleneck restored to 8 (was over-compressed to 3, which hurt reconstruction
    # fidelity on normal traffic and made the error less discriminative).
    inputs = layers.Input(shape=(input_dim,))
    x = layers.Dense(32, activation="relu")(inputs)
    x = layers.Dense(16, activation="relu")(x)
    bottleneck = layers.Dense(8, activation="relu", name="bottleneck")(x)
    x = layers.Dense(16, activation="relu")(bottleneck)
    x = layers.Dense(32, activation="relu")(x)
    outputs = layers.Dense(input_dim, activation="linear")(x)

    autoencoder = models.Model(inputs, outputs, name="stacked_autoencoder")
    autoencoder.compile(optimizer="adam", loss="mse")
    return autoencoder


def run():
    utils.print_header("Stacked Autoencoder - Binary Anomaly Detection (trained on normal only)")
    tf.random.set_seed(config.RANDOM_STATE)
    np.random.seed(config.RANDOM_STATE)

    X_train, y_train, X_test, y_test, _ = utils.load_binary_datasets()
    X_test_arr = X_test.to_numpy(dtype="float32")
    y_test_arr = y_test.to_numpy()

    # Labeled validation split used ONLY for threshold selection (fixes test-set leakage).
    X_fit, X_thresh_val, y_fit, y_thresh_val = train_test_split(
        X_train, y_train, test_size=VAL_SIZE,
        stratify=y_train, random_state=config.RANDOM_STATE,
    )
    X_fit_normal = X_fit[y_fit == 0].to_numpy(dtype="float32")
    X_thresh_val_arr = X_thresh_val.to_numpy(dtype="float32")

    # Separate held-out split of normal-only data, used purely for early stopping.
    X_es_fit, X_es_val = train_test_split(
        X_fit_normal, test_size=0.1, random_state=config.RANDOM_STATE
    )

    model = build_autoencoder(input_dim=X_fit_normal.shape[1])

    early_stop = callbacks.EarlyStopping(
        monitor="val_loss", patience=10, restore_best_weights=True
    )
    start = time.time()
    history = model.fit(
        X_es_fit, X_es_fit,
        validation_data=(X_es_val, X_es_val),
        epochs=EPOCHS, batch_size=BATCH_SIZE,
        callbacks=[early_stop], verbose=2,
    )
    training_time = time.time() - start

    def reconstruction_error(X):
        recon = model.predict(X, verbose=0)
        return np.mean(np.square(X - recon), axis=1)

    train_normal_errors = reconstruction_error(X_fit_normal)
    thresh_val_errors = reconstruction_error(X_thresh_val_arr)
    test_errors = reconstruction_error(X_test_arr)

    threshold_search = []
    best_result = None
    best_f1 = -1

    for percentile in THRESHOLD_PERCENTILES:
        threshold = float(np.percentile(train_normal_errors, percentile))
        y_val_pred = (thresh_val_errors > threshold).astype(int)
        val_metrics = utils.binary_metrics(y_thresh_val.to_numpy(), y_val_pred, thresh_val_errors)
        val_metrics["threshold_percentile"] = percentile
        val_metrics["threshold"] = threshold

        threshold_search.append(val_metrics)
        if val_metrics["f1_score"] > best_f1:
            best_f1 = val_metrics["f1_score"]
            best_result = val_metrics

    threshold = best_result["threshold"]

    # Final, unbiased evaluation on the untouched test set.
    y_pred = (test_errors > threshold).astype(int)
    metrics = utils.binary_metrics(y_test_arr, y_pred, test_errors)
    metrics["pr_auc"] = float(average_precision_score(y_test_arr, test_errors))
    metrics["model"] = "stacked_autoencoder"
    metrics["task"] = "binary"
    metrics["threshold"] = threshold
    metrics["threshold_percentile"] = best_result["threshold_percentile"]
    metrics["threshold_selected_on"] = "validation_split (stratified, from train)"
    metrics["trained_on"] = "normal_samples_only"
    metrics["epochs_trained"] = len(history.history["loss"])
    metrics["training_time_seconds"] = training_time
    print(metrics)

    model.save(config.MODELS_DIR / "stacked_autoencoder_binary.keras")
    utils.save_json(metrics, config.METRICS_DIR / "stacked_autoencoder_binary_metrics.json")
    utils.save_json(
        threshold_search,
        config.METRICS_DIR / "stacked_autoencoder_threshold_search.json",
    )
    utils.save_json(
        {"loss": history.history["loss"], "val_loss": history.history["val_loss"]},
        config.METRICS_DIR / "stacked_autoencoder_training_history.json",
    )
    utils.save_predictions_csv(
        config.PREDICTIONS_DIR / "stacked_autoencoder_binary_predictions.csv",
        y_test_arr, y_pred, test_errors,
    )
    utils.plot_confusion_matrix(
        metrics["confusion_matrix"], metrics["confusion_matrix_labels"],
        "Stacked Autoencoder - Binary Confusion Matrix",
        config.FIGURES_DIR / "stacked_autoencoder_binary_confusion_matrix.png",
    )


if __name__ == "__main__":
    run()
