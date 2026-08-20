# Dataset preparation

The raw and processed OT/ICS datasets are not committed to this repository.

## Raw data

Place the raw dataset under:

```text
data/raw/ICS_Dataset/
```

Keep the original dataset hierarchy intact so the preprocessing script can discover normal and attack flow files correctly.

## Build the processed benchmarks

From the repository root:

```bash
python preprocessing/build_flow_benchmark.py
```

The script creates:

```text
data/processed/
├── full_features/
└── time_ablated/
```

The processed outputs include the binary and multiclass train/test CSVs, feature lists, label mapping, preprocessing metadata, diagnostics inputs, and the full labeled flow dataset required by the repeated-seed experiments.

`data/raw/` and `data/processed/` are intentionally excluded from Git. The committed `results/` directory contains the final experiment outputs used by the evaluation code.
