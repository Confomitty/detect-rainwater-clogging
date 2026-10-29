"""
run_pipeline.py
===============
Single entry point that runs the full ML pipeline end-to-end.

Steps
-----
1. Load and validate the raw data CSV
2. Split into train / test (group-aware by event_id)
3. Preprocess features (fit on train, apply to test)
4. Train both models
5. Evaluate and compare both models
6. Save both models to models/

Usage
-----
    python run_pipeline.py --data data/raw/silk_board_data.csv

Optional flags
--------------
    --test-size 0.2          fraction of events for test set (default: 0.2)
    --models-dir models/     where to save .pkl files (default: models/)
    --no-plot                skip confusion matrix plots
    --save-processed         write processed CSVs to data/processed/
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make sure the project root is on the Python path when run directly
sys.path.insert(0, str(Path(__file__).parent))

from src.validate import load_and_validate
from src.preprocess import (
    build_preprocessor,
    apply_preprocessor,
    extract_target,
    split_by_group,
)
from src.train import train_models, save_model
from src.evaluate import evaluate_model, print_report, compare_models, plot_confusion_matrix


# ------------------------------------------------------------------
# Main pipeline
# ------------------------------------------------------------------

def run(
    data_path: str,
    test_size: float = 0.2,
    models_dir: str = "models",
    save_plots: bool = True,
    save_processed: bool = False,
) -> None:
    """
    Execute the full pipeline.

    Parameters
    ----------
    data_path      : path to the raw CSV file
    test_size      : fraction of events held out for testing
    models_dir     : directory where trained .pkl files are saved
    save_plots     : if True, save confusion matrix PNGs to models_dir
    save_processed : if True, write train/test CSVs to data/processed/
    """
    print("\n" + "=" * 60)
    print("  Waterlogging Risk Prediction — ML Pipeline")
    print("  Silk Board, Bengaluru")
    print("=" * 60)

    # ── Step 1: Validate ──────────────────────────────────────────
    print(f"\n[1/5] Validating data: {data_path}")
    df, report = load_and_validate(data_path, require_target=True)

    if report["warnings"]:
        for w in report["warnings"]:
            print(f"  ⚠️  {w}")

    if not report["valid"]:
        print("\n❌ Validation failed. Fix the errors below before training:\n")
        for e in report["errors"]:
            print(f"  • {e}")
        sys.exit(1)

    print(f"  ✓ {report['stats']['rows']} rows, "
          f"{report['stats']['columns']} columns. "
          f"Waterlogged rate: {report['stats'].get('waterlogged_rate_pct', 'N/A')}%")

    # ── Step 2: Train/test split ──────────────────────────────────
    print(f"\n[2/5] Splitting data (test_size={test_size}, group=event_id)...")
    df_train, df_test = split_by_group(df, test_size=test_size)
    print(f"  ✓ Train: {len(df_train)} rows | Test: {len(df_test)} rows")

    if save_processed:
        proc_dir = Path("data/processed")
        proc_dir.mkdir(parents=True, exist_ok=True)
        df_train.to_csv(proc_dir / "train.csv", index=False)
        df_test.to_csv(proc_dir / "test.csv", index=False)
        print(f"  ✓ Saved to data/processed/train.csv and test.csv")

    # ── Step 3: Preprocess ────────────────────────────────────────
    print("\n[3/5] Preprocessing features...")
    preprocessor = build_preprocessor(df_train)
    X_train, feature_names = apply_preprocessor(preprocessor, df_train, fit=True)
    X_test, _ = apply_preprocessor(preprocessor, df_test, fit=False)
    y_train = extract_target(df_train)
    y_test = extract_target(df_test)
    print(f"  ✓ Features: {len(feature_names)} | "
          f"Train shape: {X_train.shape} | Test shape: {X_test.shape}")

    # ── Step 4: Train ─────────────────────────────────────────────
    print("\n[4/5] Training models...")
    models = train_models(X_train, y_train)

    # ── Step 5: Evaluate & save ───────────────────────────────────
    print("\n[5/5] Evaluating models...")
    all_metrics: list[dict] = []
    models_path = Path(models_dir)
    models_path.mkdir(parents=True, exist_ok=True)

    for model_name, model in models.items():
        metrics = evaluate_model(model, X_test, y_test, model_name=model_name)
        print_report(metrics)
        all_metrics.append(metrics)

        # Save model + preprocessor bundle
        pkl_path = models_path / f"{model_name}.pkl"
        save_model(model, preprocessor, feature_names, pkl_path)

        # Save confusion matrix plot
        if save_plots:
            plot_path = str(models_path / f"{model_name}_confusion_matrix.png")
            plot_confusion_matrix(metrics, save_path=plot_path)

    compare_models(all_metrics)

    print("\n✅ Pipeline complete.")
    print(f"   Saved models: {[str(models_path / n) + '.pkl' for n in models]}")
    print("   Next steps:")
    print("   • Share these results with your team.")
    print("   • If test metrics look too good, check for data leakage.")
    print("   • When you have more data, add GridSearchCV for hyperparameter tuning.")


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Waterlogging Risk Prediction ML Pipeline"
    )
    parser.add_argument(
        "--data",
        required=True,
        help="Path to the raw CSV data file (e.g. data/raw/silk_board_data.csv)",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Fraction of events to hold out for testing (default: 0.2)",
    )
    parser.add_argument(
        "--models-dir",
        default="models",
        help="Directory to save trained model .pkl files (default: models/)",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Skip saving confusion matrix plots",
    )
    parser.add_argument(
        "--save-processed",
        action="store_true",
        help="Save train/test CSVs to data/processed/",
    )

    args = parser.parse_args()
    run(
        data_path=args.data,
        test_size=args.test_size,
        models_dir=args.models_dir,
        save_plots=not args.no_plot,
        save_processed=args.save_processed,
    )

