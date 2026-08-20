import json
from pathlib import Path

import pandas as pd

from .. import config


TIME_LEAKAGE_COLS = [
    "startOffset",
    "endOffset",
    "raw_start_time",
    "raw_end_time",
    "event_time_seconds",
    "event_end_seconds",
    "start_seconds_of_day",
    "end_seconds_of_day",
]


def get_path_from_config(name, fallback):
    return Path(getattr(config, name, fallback))


def load_label_map(data_dir):
    label_map_path = data_dir / "multiclass_label_map.json"
    if not label_map_path.exists():
        return None
    label_map = json.loads(label_map_path.read_text())
    return {int(v): k for k, v in label_map.items()}


def add_class_name(df, label_map, label_col="multiclass_label"):
    if label_map is None or label_col not in df.columns:
        return df
    df["class_name"] = df[label_col].map(label_map)
    return df


def count_full_classes(data_dir, out_dir):
    full_path = data_dir / "full_labeled_flow_dataset.csv"
    if not full_path.exists():
        print(f"[skip] missing {full_path}")
        return None

    df = pd.read_csv(full_path)

    if "attack_type_label" in df.columns:
        label_col = "attack_type_label"
    elif "multiclass_label_name" in df.columns:
        label_col = "multiclass_label_name"
    elif "multiclass_label" in df.columns:
        label_col = "multiclass_label"
    else:
        raise ValueError("Could not find a full-dataset class label column.")

    counts = df[label_col].value_counts().reset_index()
    counts.columns = ["class_name", "full_count"]
    counts["percentage"] = counts["full_count"] / counts["full_count"].sum() * 100
    counts = counts.sort_values("class_name")
    counts.to_csv(out_dir / "class_counts_full.csv", index=False)

    print("[saved] class_counts_full.csv")
    return counts


def count_train_test_classes(data_dir, out_dir):
    train_path = data_dir / "multiclass_classification_train.csv"
    test_path = data_dir / "multiclass_classification_test.csv"

    if not train_path.exists() or not test_path.exists():
        print(f"[skip] missing train/test files in {data_dir}")
        return None

    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    label_col = getattr(config, "MULTICLASS_LABEL_COL", "multiclass_label")
    label_map = load_label_map(data_dir)

    train_counts = train[label_col].value_counts().rename("train_count")
    test_counts = test[label_col].value_counts().rename("test_count")
    table = pd.concat([train_counts, test_counts], axis=1).fillna(0).astype(int).reset_index()
    table = table.rename(columns={"index": label_col})
    table["full_count"] = table["train_count"] + table["test_count"]
    table["train_percentage"] = table["train_count"] / table["full_count"] * 100
    table["test_percentage"] = table["test_count"] / table["full_count"] * 100

    if label_map is not None:
        table["class_name"] = table[label_col].map(label_map)
        table = table[[label_col, "class_name", "train_count", "test_count", "full_count", "train_percentage", "test_percentage"]]
    else:
        table = table[[label_col, "train_count", "test_count", "full_count", "train_percentage", "test_percentage"]]

    table = table.sort_values(label_col)
    table.to_csv(out_dir / "class_counts_train_test.csv", index=False)

    print("[saved] class_counts_train_test.csv")
    return table


def count_binary_distribution(data_dir, out_dir):
    train_path = data_dir / "binary_detection_train.csv"
    test_path = data_dir / "binary_detection_test.csv"

    if not train_path.exists() or not test_path.exists():
        print(f"[skip] missing binary train/test files in {data_dir}")
        return None

    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    label_col = getattr(config, "BINARY_LABEL_COL", "binary_label")

    train_counts = train[label_col].value_counts().rename("train_count")
    test_counts = test[label_col].value_counts().rename("test_count")
    table = pd.concat([train_counts, test_counts], axis=1).fillna(0).astype(int).reset_index()
    table = table.rename(columns={"index": label_col})
    table["class_name"] = table[label_col].map({0: "Normal", 1: "Attack"})
    table["full_count"] = table["train_count"] + table["test_count"]
    table["full_percentage"] = table["full_count"] / table["full_count"].sum() * 100
    table = table[[label_col, "class_name", "train_count", "test_count", "full_count", "full_percentage"]]
    table = table.sort_values(label_col)
    table.to_csv(out_dir / "binary_counts_train_test.csv", index=False)

    normal_count = int(table.loc[table[label_col] == 0, "full_count"].iloc[0])
    attack_count = int(table.loc[table[label_col] == 1, "full_count"].iloc[0])
    ratio = normal_count / attack_count if attack_count else None

    ratio_df = pd.DataFrame([{
        "normal_count": normal_count,
        "attack_count": attack_count,
        "total_count": normal_count + attack_count,
        "normal_percentage": normal_count / (normal_count + attack_count) * 100,
        "attack_percentage": attack_count / (normal_count + attack_count) * 100,
        "normal_to_attack_ratio": ratio,
    }])
    ratio_df.to_csv(out_dir / "binary_normal_attack_ratio.csv", index=False)

    print("[saved] binary_counts_train_test.csv")
    print("[saved] binary_normal_attack_ratio.csv")
    return table, ratio_df


def feature_count_summary(with_time_dir, no_time_dir, out_dir):
    with_feature_path = with_time_dir / "binary_detection_feature_list.json"
    no_time_feature_path = no_time_dir / "binary_detection_feature_list.json"

    if not with_feature_path.exists() or not no_time_feature_path.exists():
        print("[skip] missing feature list json files")
        return None

    with_features = json.loads(with_feature_path.read_text())
    no_time_features = json.loads(no_time_feature_path.read_text())

    removed = [c for c in with_features if c not in no_time_features]
    expected_removed = [c for c in TIME_LEAKAGE_COLS if c in with_features]

    summary = pd.DataFrame([
        {"benchmark_version": "with_time", "feature_count": len(with_features), "removed_feature_count": 0, "removed_features": ""},
        {"benchmark_version": "no_time", "feature_count": len(no_time_features), "removed_feature_count": len(removed), "removed_features": "; ".join(removed)},
    ])
    summary.to_csv(out_dir / "feature_count_original_vs_no_time.csv", index=False)

    removed_df = pd.DataFrame({"removed_feature": removed})
    removed_df["is_expected_time_or_position_feature"] = removed_df["removed_feature"].isin(TIME_LEAKAGE_COLS)
    removed_df.to_csv(out_dir / "removed_time_features.csv", index=False)

    print("[saved] feature_count_original_vs_no_time.csv")
    print("[saved] removed_time_features.csv")
    print(f"Original model-input features: {len(with_features)}")
    print(f"Removed time/capture-position features: {len(removed)}")
    print(f"No-time model-input features: {len(no_time_features)}")

    missing_expected = [c for c in expected_removed if c not in removed]
    unexpected_removed = [c for c in removed if c not in TIME_LEAKAGE_COLS]

    if missing_expected:
        print("[warning] expected time columns present in with-time but not listed as removed:", missing_expected)
    if unexpected_removed:
        print("[warning] removed columns not in TIME_LEAKAGE_COLS:", unexpected_removed)

    return summary


def source_file_summary(data_dir, out_dir):
    inventory_path = data_dir / "source_file_inventory.csv"
    if not inventory_path.exists():
        print(f"[skip] missing {inventory_path}")
        return None

    inv = pd.read_csv(inventory_path)
    inv.to_csv(out_dir / "source_file_inventory_used.csv", index=False)
    print("[saved] source_file_inventory_used.csv")
    return inv


def main():
    base_data_dir = get_path_from_config("DATA_ROOT_DIR", Path(config.BASE_DIR).parent / "processed_flow_benchmark")
    with_time_dir = get_path_from_config("WITH_TIME_DATA_DIR", base_data_dir / "with-time")
    no_time_dir = get_path_from_config("NO_TIME_DATA_DIR", base_data_dir / "no_time")

    results_dir = Path(getattr(config, "RESULTS_DIR", Path(config.BASE_DIR) / "results"))
    out_dir = results_dir / "dataset_diagnostics"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Dataset diagnostics")
    print("With-time directory:", with_time_dir.resolve())
    print("No-time directory:", no_time_dir.resolve())
    print("Output directory:", out_dir.resolve())
    print()

    feature_count_summary(with_time_dir, no_time_dir, out_dir)
    count_full_classes(no_time_dir, out_dir)
    count_train_test_classes(no_time_dir, out_dir)
    count_binary_distribution(no_time_dir, out_dir)
    source_file_summary(no_time_dir, out_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()
