import json
import warnings
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler

warnings.filterwarnings("ignore")

# Paths are anchored to the repo root (this file lives in preprocessing/) so
# the script works the same regardless of the current working directory it's
# launched from. Same folder names/meaning as before, just relocated under
# data/ instead of the repo root.
REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = REPO_ROOT / "data" / "raw" / "ICS_Dataset"
OUTPUT_ROOT = REPO_ROOT / "data" / "processed"
WITH_TIME_OUTPUT_DIR = OUTPUT_ROOT / "with-time"
NO_TIME_OUTPUT_DIR = OUTPUT_ROOT / "no_time"

TEST_SIZE = 0.20
RANDOM_STATE = 42

NORMAL_ROOT_NAME = "Normal_Data"
ATTACK_ROOT_NAME = "Attack_Data"

REQUIRED_ATTACK_PATH_TOKENS = ["Network_Level", "Flow"]

EXCLUDE_PATH_TOKENS = ["windows", "window", "__MACOSX"]

RAW_LABEL_COLS = ["label", "Label", "attack", "Attack"]
PROTOCOL_COL_CANDIDATES = ["protocol", "Protocol", "proto"]

IDENTIFIER_COLS = [
    "sender_address", "receiver_address",
    "src_ip", "dst_ip", "source_ip", "destination_ip",
    "source_address", "destination_address",
    "src_mac", "dst_mac",
    "flow_id", "flowID", "Flow ID",
    "id", "ID", "session_id"
]

ABSOLUTE_TIME_COLS = ["start", "end", "timestamp", "Timestamp", "Time", "time"]

TIME_LEAKAGE_COLS = [
    "startOffset",
    "endOffset",
    "raw_start_time",
    "raw_end_time",
    "event_time_seconds",
    "event_end_seconds",
    "start_seconds_of_day",
    "end_seconds_of_day",
    "timestamp_seconds_of_day",
    "Timestamp_seconds_of_day",
    "Time_seconds_of_day",
    "time_seconds_of_day"
]

META_COLS = [
    "source_file",
    "source_path",
    "scenario",
    "binary_label",
    "attack_type_label",
    "multiclass_label"
]


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def path_has_any_token(path: Path, tokens):
    lowered = [p.lower() for p in path.parts]
    return any(tok.lower() in p for p in lowered for tok in tokens)


def find_label_column(df: pd.DataFrame):
    for col in RAW_LABEL_COLS:
        if col in df.columns:
            return col
    return None


def find_protocol_column(df: pd.DataFrame):
    for col in PROTOCOL_COL_CANDIDATES:
        if col in df.columns:
            return col
    return None


def seconds_of_day_from_series(series: pd.Series):
    numeric = pd.to_numeric(series, errors="coerce")

    if numeric.notna().mean() > 0.8:
        return numeric % 86400

    dt = pd.to_datetime(series, errors="coerce")
    return (
        dt.dt.hour * 3600
        + dt.dt.minute * 60
        + dt.dt.second
        + dt.dt.microsecond / 1_000_000
    )


def discover_flow_csv_files(data_root: Path):
    if not data_root.exists():
        raise FileNotFoundError(f"DATA_ROOT does not exist: {data_root}")

    files = []

    for csv_path in sorted(data_root.rglob("*.csv")):
        rel_path = csv_path.relative_to(data_root)
        parts = rel_path.parts

        if path_has_any_token(rel_path, EXCLUDE_PATH_TOKENS):
            continue

        if NORMAL_ROOT_NAME in parts:
            if not all(tok in parts for tok in REQUIRED_ATTACK_PATH_TOKENS):
                continue

            files.append({
                "path": csv_path,
                "category": "normal",
                "scenario": "Normal",
                "relative_path": str(rel_path)
            })
            continue

        if ATTACK_ROOT_NAME in parts:
            if not all(tok in parts for tok in REQUIRED_ATTACK_PATH_TOKENS):
                continue

            if "Network_Level" in parts:
                idx = parts.index("Network_Level")
                scenario = parts[idx - 1] if idx > 0 else csv_path.parent.name
            else:
                scenario = csv_path.parent.name

            files.append({
                "path": csv_path,
                "category": "attack",
                "scenario": scenario,
                "relative_path": str(rel_path)
            })

    if not files:
        raise FileNotFoundError(f"No usable network-flow CSV files found under {data_root}")

    print("\nDetected files:")
    for f in files[:10]:
        print(f["relative_path"], "----", f["scenario"])
    print("Total detected files:", len(files))

    return files


def load_one_file(info):
    df = pd.read_csv(info["path"], low_memory=False)
    df["source_file"] = info["path"].name
    df["source_path"] = info["relative_path"]
    df["scenario"] = info["scenario"]

    if info["category"] == "normal":
        df["binary_label"] = 0
        df["attack_type_label"] = "Normal"
        return df

    label_col = find_label_column(df)
    if label_col is None:
        raise ValueError(
            f"Attack file has no row-level label column: {info['path']}"
        )

    raw_label = pd.to_numeric(df[label_col], errors="coerce").fillna(0)
    is_attack_row = raw_label > 0

    df["binary_label"] = np.where(is_attack_row, 1, 0).astype(int)
    df["attack_type_label"] = np.where(is_attack_row, info["scenario"], "Normal")

    return df


def load_and_label_all(files_info):
    frames = []
    inventory = []

    for info in files_info:
        df = load_one_file(info)
        frames.append(df)

        inventory.append({
            "file": info["relative_path"],
            "category": info["category"],
            "scenario": info["scenario"],
            "rows": int(len(df)),
            "normal_rows_after_labeling": int((df["binary_label"] == 0).sum()),
            "attack_rows_after_labeling": int((df["binary_label"] == 1).sum())
        })

    combined = pd.concat(frames, ignore_index=True, sort=False)
    return combined, pd.DataFrame(inventory)


def convert_absolute_time_columns(df: pd.DataFrame):
    created_cols = []

    for col in ABSOLUTE_TIME_COLS:
        if col in df.columns:
            new_col = f"{col}_seconds_of_day"
            df[new_col] = seconds_of_day_from_series(df[col])
            created_cols.append(new_col)
            df = df.drop(columns=[col])

    return df, created_cols


def drop_identifier_and_raw_label_columns(df: pd.DataFrame):
    drop_cols = []

    for col in df.columns:
        if col in IDENTIFIER_COLS or col in RAW_LABEL_COLS:
            drop_cols.append(col)

    drop_cols = sorted(set(drop_cols) & set(df.columns))
    df = df.drop(columns=drop_cols)

    return df, drop_cols


def encode_protocol(df: pd.DataFrame):
    proto_col = find_protocol_column(df)

    if proto_col is None:
        return df, None, []

    dummies = pd.get_dummies(df[proto_col].astype(str), prefix="proto")
    dummy_cols = list(dummies.columns)

    df = pd.concat([df.drop(columns=[proto_col]), dummies], axis=1)
    return df, proto_col, dummy_cols


def encode_remaining_categoricals(df: pd.DataFrame):
    encoded_cols = []
    dropped_text_cols = []

    object_cols = [
        c for c in df.columns
        if c not in META_COLS and df[c].dtype == object
    ]

    for col in object_cols:
        nunique = df[col].nunique(dropna=True)

        if nunique <= 20:
            dummies = pd.get_dummies(df[col].astype(str), prefix=col)
            encoded_cols.extend(list(dummies.columns))
            df = pd.concat([df.drop(columns=[col]), dummies], axis=1)
        else:
            dropped_text_cols.append(col)
            df = df.drop(columns=[col])

    return df, encoded_cols, dropped_text_cols


def clean_numeric_features(df: pd.DataFrame, feature_cols):
    for col in feature_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df[feature_cols] = df[feature_cols].replace([np.inf, -np.inf], np.nan)

    missing_before = {
        col: int(df[col].isna().sum())
        for col in feature_cols
        if int(df[col].isna().sum()) > 0
    }

    for col in feature_cols:
        median = df[col].median()
        df[col] = df[col].fillna(median if pd.notna(median) else 0)

    nunique = df[feature_cols].nunique(dropna=False)
    constant_cols = sorted(nunique[nunique <= 1].index.tolist())
    kept_cols = [c for c in feature_cols if c not in constant_cols]

    return df, kept_cols, constant_cols, missing_before


def build_multiclass_label_map(df: pd.DataFrame):
    classes = sorted(df["attack_type_label"].unique().tolist())
    classes = ["Normal"] + [c for c in classes if c != "Normal"]
    label_map = {name: idx for idx, name in enumerate(classes)}
    df["multiclass_label"] = df["attack_type_label"].map(label_map).astype(int)
    return df, label_map


def stratified_split(X, y):
    counts = y.value_counts()
    if counts.min() < 2:
        print("Warning: at least one class has fewer than 2 samples; using non-stratified split.")
        stratify = None
    else:
        stratify = y

    return train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=stratify
    )


def build_no_time_feature_list(feature_cols):
    removed_time_cols = [c for c in TIME_LEAKAGE_COLS if c in feature_cols]
    no_time_feature_cols = [c for c in feature_cols if c not in removed_time_cols]
    return no_time_feature_cols, removed_time_cols


def save_common_diagnostics(df, inventory_df, label_map, output_dir: Path):
    scenario_distribution = (
        df["attack_type_label"]
        .value_counts()
        .sort_index()
        .rename_axis("attack_type_label")
        .reset_index(name="rows")
    )

    binary_distribution = (
        df["binary_label"]
        .value_counts()
        .sort_index()
        .rename_axis("binary_label")
        .reset_index(name="rows")
    )

    inventory_df.to_csv(output_dir / "source_file_inventory.csv", index=False)
    scenario_distribution.to_csv(output_dir / "scenario_distribution.csv", index=False)
    binary_distribution.to_csv(output_dir / "binary_distribution.csv", index=False)
    (output_dir / "multiclass_label_map.json").write_text(json.dumps(label_map, indent=2))

    return scenario_distribution, binary_distribution


def scale_split_save(df, feature_cols, label_col, dataset_name, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)

    X = df[feature_cols].copy()
    y = df[label_col].copy()

    X_train, X_test, y_train, y_test = stratified_split(X, y)

    scaler = RobustScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=feature_cols,
        index=X_train.index
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test),
        columns=feature_cols,
        index=X_test.index
    )

    train_out = X_train_scaled.copy()
    train_out[label_col] = y_train.values

    test_out = X_test_scaled.copy()
    test_out[label_col] = y_test.values

    train_out.to_csv(output_dir / f"{dataset_name}_train.csv", index=False)
    test_out.to_csv(output_dir / f"{dataset_name}_test.csv", index=False)

    joblib.dump(scaler, output_dir / f"{dataset_name}_robust_scaler.pkl")
    (output_dir / f"{dataset_name}_feature_list.json").write_text(
        json.dumps(feature_cols, indent=2)
    )

    return {
        "dataset_name": dataset_name,
        "label_col": label_col,
        "n_features": int(len(feature_cols)),
        "train_rows": int(len(train_out)),
        "test_rows": int(len(test_out)),
        "train_distribution": {
            str(k): int(v)
            for k, v in y_train.value_counts().sort_index().items()
        },
        "test_distribution": {
            str(k): int(v)
            for k, v in y_test.value_counts().sort_index().items()
        }
    }


def main():
    ensure_dir(WITH_TIME_OUTPUT_DIR)
    ensure_dir(NO_TIME_OUTPUT_DIR)

    files_info = discover_flow_csv_files(DATA_ROOT)
    df, inventory_df = load_and_label_all(files_info)

    df, created_time_cols = convert_absolute_time_columns(df)
    df, dropped_identifier_cols = drop_identifier_and_raw_label_columns(df)

    df, protocol_source_col, protocol_dummy_cols = encode_protocol(df)
    df, extra_encoded_cols, dropped_text_cols = encode_remaining_categoricals(df)

    df, label_map = build_multiclass_label_map(df)

    feature_cols = [
        c for c in df.columns
        if c not in META_COLS
    ]

    df, feature_cols, constant_cols, missing_before = clean_numeric_features(
        df, feature_cols
    )

    no_time_feature_cols, removed_time_cols = build_no_time_feature_list(feature_cols)

    # Save the original with-time benchmark.
    full_cols_with_time = META_COLS + feature_cols
    df[full_cols_with_time].to_csv(WITH_TIME_OUTPUT_DIR / "full_labeled_flow_dataset.csv", index=False)

    binary_summary_with_time = scale_split_save(
        df=df,
        feature_cols=feature_cols,
        label_col="binary_label",
        dataset_name="binary_detection",
        output_dir=WITH_TIME_OUTPUT_DIR
    )

    multiclass_summary_with_time = scale_split_save(
        df=df,
        feature_cols=feature_cols,
        label_col="multiclass_label",
        dataset_name="multiclass_classification",
        output_dir=WITH_TIME_OUTPUT_DIR
    )

    scenario_distribution, binary_distribution = save_common_diagnostics(
        df=df,
        inventory_df=inventory_df,
        label_map=label_map,
        output_dir=WITH_TIME_OUTPUT_DIR
    )

    metadata_with_time = {
        "benchmark_variant": "with_time",
        "data_root": str(DATA_ROOT),
        "output_dir": str(WITH_TIME_OUTPUT_DIR),
        "random_state": RANDOM_STATE,
        "test_size": TEST_SIZE,
        "sample_unit": "one network flow row",
        "windowing_used": False,
        "padding_used": False,
        "masking_used": False,
        "process_telemetry_used": False,
        "time_or_capture_position_features_used": True,
        "labeling_policy": {
            "normal_files": "all rows labeled Normal",
            "attack_files": "folder path defines scenario; row-level raw label defines whether the row is attack or normal"
        },
        "n_source_files": int(len(files_info)),
        "n_total_rows": int(len(df)),
        "n_features_final": int(len(feature_cols)),
        "label_map": label_map,
        "created_time_features": created_time_cols,
        "protocol_source_column": protocol_source_col,
        "protocol_dummy_columns": protocol_dummy_cols,
        "extra_encoded_columns": extra_encoded_cols,
        "dropped_identifier_or_raw_label_columns": dropped_identifier_cols,
        "dropped_high_cardinality_text_columns": dropped_text_cols,
        "constant_columns_dropped": constant_cols,
        "missing_values_before_imputation": missing_before,
        "binary_dataset": binary_summary_with_time,
        "multiclass_dataset": multiclass_summary_with_time
    }

    (WITH_TIME_OUTPUT_DIR / "preprocessing_metadata.json").write_text(
        json.dumps(metadata_with_time, indent=2, default=str)
    )

    # Save the no-time benchmark used for temporal-leakage ablation.
    full_cols_no_time = META_COLS + no_time_feature_cols
    df[full_cols_no_time].to_csv(NO_TIME_OUTPUT_DIR / "full_labeled_flow_dataset.csv", index=False)

    binary_summary_no_time = scale_split_save(
        df=df,
        feature_cols=no_time_feature_cols,
        label_col="binary_label",
        dataset_name="binary_detection",
        output_dir=NO_TIME_OUTPUT_DIR
    )

    multiclass_summary_no_time = scale_split_save(
        df=df,
        feature_cols=no_time_feature_cols,
        label_col="multiclass_label",
        dataset_name="multiclass_classification",
        output_dir=NO_TIME_OUTPUT_DIR
    )

    save_common_diagnostics(
        df=df,
        inventory_df=inventory_df,
        label_map=label_map,
        output_dir=NO_TIME_OUTPUT_DIR
    )

    metadata_no_time = {
        "benchmark_variant": "no_time",
        "data_root": str(DATA_ROOT),
        "output_dir": str(NO_TIME_OUTPUT_DIR),
        "random_state": RANDOM_STATE,
        "test_size": TEST_SIZE,
        "sample_unit": "one network flow row",
        "windowing_used": False,
        "padding_used": False,
        "masking_used": False,
        "process_telemetry_used": False,
        "time_or_capture_position_features_used": False,
        "time_or_capture_position_features_removed": removed_time_cols,
        "n_time_or_capture_position_features_removed": int(len(removed_time_cols)),
        "labeling_policy": {
            "normal_files": "all rows labeled Normal",
            "attack_files": "folder path defines scenario; row-level raw label defines whether the row is attack or normal"
        },
        "n_source_files": int(len(files_info)),
        "n_total_rows": int(len(df)),
        "n_features_original_with_time": int(len(feature_cols)),
        "n_features_final": int(len(no_time_feature_cols)),
        "label_map": label_map,
        "created_time_features": created_time_cols,
        "protocol_source_column": protocol_source_col,
        "protocol_dummy_columns": protocol_dummy_cols,
        "extra_encoded_columns": extra_encoded_cols,
        "dropped_identifier_or_raw_label_columns": dropped_identifier_cols,
        "dropped_high_cardinality_text_columns": dropped_text_cols,
        "constant_columns_dropped": constant_cols,
        "missing_values_before_imputation": missing_before,
        "binary_dataset": binary_summary_no_time,
        "multiclass_dataset": multiclass_summary_no_time
    }

    (NO_TIME_OUTPUT_DIR / "preprocessing_metadata.json").write_text(
        json.dumps(metadata_no_time, indent=2, default=str)
    )

    print("Done.")
    print(f"Source files used: {len(files_info)}")
    print(f"Total rows: {len(df)}")
    print(f"With-time features: {len(feature_cols)}")
    print(f"Removed time/capture-position features: {len(removed_time_cols)}")
    print(f"No-time features: {len(no_time_feature_cols)}")
    print(f"With-time output directory: {WITH_TIME_OUTPUT_DIR.resolve()}")
    print(f"No-time output directory: {NO_TIME_OUTPUT_DIR.resolve()}")
    print("\nRemoved time/capture-position columns:")
    print(removed_time_cols)
    print("\nScenario distribution:")
    print(scenario_distribution.to_string(index=False))

if __name__ == "__main__":
    main()