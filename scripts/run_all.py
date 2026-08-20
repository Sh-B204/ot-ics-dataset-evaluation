import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"

def run_step(description, module_name, extra_args=None):
    cmd = [sys.executable, "-m", module_name]
    if extra_args:
        cmd.extend(extra_args)
    print("\n" + "=" * 70)
    print(description)
    print("=" * 70)
    print("Command:", " ".join(cmd), f"(from {SRC_DIR})")
    subprocess.run(cmd, check=True, cwd=str(SRC_DIR))


def main():
    parser = argparse.ArgumentParser(
        description="Run the final evaluation batch: tuned repeated seeds, tuned full-feature vs time-ablated comparison, and final figures."
    )
    parser.add_argument("--plot-only", action="store_true", help="Regenerate figures/tables from existing CSVs without retraining.")
    parser.add_argument("--skip-repeated-seeds", action="store_true", help="Skip the repeated-seed robustness study.")
    parser.add_argument("--skip-time-ablation", action="store_true", help="Skip the full-feature vs time-ablated comparison.")
    parser.add_argument("--skip-visualization", action="store_true", help="Skip the final visualization wrapper.")
    parser.add_argument("--models", nargs="+", choices=["random_forest", "xgboost", "mlp"], help="Optional supervised model subset.")
    parser.add_argument("--tasks", nargs="+", choices=["binary", "multiclass"], help="Optional supervised task subset.")
    parser.add_argument("--seeds", nargs="+", type=int, help="Optional seed subset.")
    args = parser.parse_args()

    forwarded = []
    if args.plot_only:
        forwarded.append("--plot-only")
    if args.models:
        forwarded.extend(["--models", *args.models])
    if args.tasks:
        forwarded.extend(["--tasks", *args.tasks])
    if args.seeds:
        forwarded.extend(["--seeds", *[str(s) for s in args.seeds]])

    if not args.skip_repeated_seeds:
        run_step("Final repeated-seed robustness study", "ics_ids.experiments.run_repeated_seeds", forwarded)

    if not args.skip_time_ablation:
        run_step("Final tuned full-feature vs time-ablated comparison", "ics_ids.experiments.run_time_ablation", forwarded)

    if not args.skip_visualization:
        run_step("Final confusion-matrix generation", "ics_ids.visualization.visualization")

    print("\nFinal evaluation batch completed.")


if __name__ == "__main__":
    main()