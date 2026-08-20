# Data

The raw ICS/OT dataset is **not included** in this repository. It comes from
Kaggle and needs to be downloaded separately.

## 1. Download the raw dataset

Source: `<KAGGLE_DATASET_URL>` (fill in the exact Kaggle dataset link — it
could not be determined from the supplied project files).

Unzip it so the flow-level CSVs end up at:

```
data/raw/ICS_Dataset/
├── Normal_Data/...
└── Attack_Data/...      (only files under paths containing "Network_Level"
                           and "Flow" are used by the preprocessing script)
```

This path is what `preprocessing/preprocess_ics_flow_dataset.py` expects by
default (`data/raw/ICS_Dataset`).

## 2. Run preprocessing

```bash
python preprocessing/preprocess_ics_flow_dataset.py
```

This reads the raw flow CSVs and writes two processed benchmark variants to
`data/processed/`:

```
data/processed/
├── with-time/    # full feature set, including time-derived columns
└── no_time/      # time-ablated benchmark used for the ablation study
```

Each variant contains the train/test splits and label maps used throughout
`src/ics_ids/` (binary and multiclass detection tasks).

## Already have the processed data?

If you already have `with-time/` and `no_time/` generated, just drop them
into `data/processed/` in the layout shown above — no code changes needed.
Both folders are gitignored under `data/raw/`; the processed folders are not
git-ignored by default since they're derived data you may choose to publish
alongside this repo, but they are not included in this ZIP.
