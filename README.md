# OT/ICS Dataset Evaluation

Reproducible machine-learning evaluation pipeline for an OT/ICS network-flow dataset. The repository covers preprocessing, supervised binary attack detection and multiclass attack classification, unsupervised anomaly detection, repeated-seed robustness analysis, hyperparameter studies, time-feature ablation, and final result visualizations.

## Repository structure

```text
preprocessing/                 Build the processed flow benchmarks
src/ics_ids/
  models/                      Supervised and unsupervised model implementations
  evaluation/                  Dataset diagnostics
  experiments/                 Hyperparameter, repeated-seed, ablation, and robustness studies
  visualization/               Normalized confusion matrices
results/                       Final experiment outputs and figures
data/README.md                 Dataset preparation instructions
scripts/run_all.py             Regenerate the final evaluation outputs
```

## Installation

```bash
git clone https://github.com/Sh-B204/ot-ics-dataset-evaluation.git
cd ot-ics-dataset-evaluation
python -m venv .venv
pip install -r requirements.txt
```

Activate the environment, then expose the `src` package:

```bash
# Windows PowerShell
$env:PYTHONPATH = "$PWD\src"

# Linux/macOS
export PYTHONPATH="$PWD/src"
```

## Dataset preparation

Raw and processed datasets are not committed to the repository. See [`data/README.md`](data/README.md).

To build the two processed benchmarks:

```bash
python preprocessing/build_flow_benchmark.py
```

This creates:

```text
data/processed/full_features/
data/processed/time_ablated/
```

## Experiments

```bash
python -m ics_ids.evaluation.dataset_diagnostics
python -m ics_ids.experiments.run_hyperparameter_study
python -m ics_ids.experiments.run_repeated_seeds
python -m ics_ids.experiments.run_time_ablation
python -m ics_ids.experiments.run_unsupervised_repeated_seeds
python -m ics_ids.visualization.visualization
```

The repeated-seed, time-ablation, and unsupervised experiment scripts support `--plot-only` for regenerating figures from saved outputs without retraining.

To regenerate the final evaluation batch:

```bash
python scripts/run_all.py --plot-only
```

## Final outputs

The retained results are organized under:

```text
results/
  dataset_diagnostics/
  hyperparameter_study/
  robustness/
  time_ablation_tuned/
  unsupervised_robustness/
  figures/
```

Final figures are grouped into `confusion_matrices`, `hyperparameter_study`, `repeated_seeds`, `time_ablation_tuned`, and `unsupervised_robustness`.

## Reproducibility

The repeated-seed experiments use seeds `42, 7, 21, 100, 2026`. The default random state is `42`. Neural-network runs may show minor hardware-dependent numerical variation.

## Citation

Citation metadata is provided in [`CITATION.cff`](CITATION.cff).
