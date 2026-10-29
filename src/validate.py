"""
validate.py
===========
Checks a raw CSV against the schema before any ML work happens.

Usage
-----
    from src.validate import validate_dataframe, load_and_validate

    # From a file path:
    df, report = load_and_validate("data/raw/silk_board_data.csv", require_target=True)

    # From an existing DataFrame (e.g. inside tests):
    report = validate_dataframe(df, require_target=False)
    if report["valid"]:
        print("Data is clean!")
    else:
        print(report["errors"])

Returns
-------
report : dict with keys
    "valid"    : bool
    "errors"   : list[str]  — blockers (pipeline will not continue)
    "warnings" : list[str]  — non-fatal issues worth checking
    "stats"    : dict       — basic summary (row count, missing %, etc.)
"""

from __future__ import annotations

import pandas as pd

from src.schema import (
    ALLOWED_CATEGORIES,
    NUMERIC_BOUNDS,
    REQUIRED_COLUMNS,
    TARGET_COLUMN,
    DTYPE_EXPECTATIONS,
)


# ------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------

def load_and_validate(
    filepath: str,
    require_target: bool = True,
) -> tuple[pd.DataFrame, dict]:
    """
    Load a CSV from *filepath* and run full validation.

    Parameters
    ----------
    filepath      : path to the raw CSV file
    require_target: if True, the 'waterlogged' column must be present
                    (needed for training; set False for prediction)

    Returns
    -------
    df     : the loaded DataFrame (empty if load fails)
    report : validation report dict
    """
    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        return pd.DataFrame(), {
            "valid": False,
            "errors": [f"File not found: {filepath}"],
            "warnings": [],
            "stats": {},
        }
    except Exception as exc:
        return pd.DataFrame(), {
            "valid": False,
            "errors": [f"Could not read file: {exc}"],
            "warnings": [],
            "stats": {},
        }

    report = validate_dataframe(df, require_target=require_target)
    return df, report


def validate_dataframe(
    df: pd.DataFrame,
    require_target: bool = True,
) -> dict:
    """
    Run all validation checks on an already-loaded DataFrame.

    Parameters
    ----------
    df             : input DataFrame
    require_target : if True, TARGET_COLUMN must be present

    Returns
    -------
    report : dict — see module docstring for keys
    """
    errors: list[str] = []
    warnings: list[str] = []

    # ── 1. Empty check ────────────────────────────────────────────
    if df.empty:
        return {
            "valid": False,
            "errors": ["DataFrame is empty."],
            "warnings": [],
            "stats": {},
        }

    # ── 2. Required columns ───────────────────────────────────────
    missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_cols:
        errors.append(f"Missing required columns: {missing_cols}")

    # ── 3. Target column ──────────────────────────────────────────
    if require_target:
        if TARGET_COLUMN not in df.columns:
            errors.append(
                f"Missing target column '{TARGET_COLUMN}'. "
                "Supervised training cannot begin without waterlogging labels. "
                "Each row must have 0 (not waterlogged) or 1 (waterlogged) "
                "confirmed from field observations or official flood records."
            )
    else:
        if TARGET_COLUMN not in df.columns:
            warnings.append(
                f"'{TARGET_COLUMN}' column absent — running in prediction mode."
            )

    # ── 4. Numeric type checks ────────────────────────────────────
    for col, expected in DTYPE_EXPECTATIONS.items():
        if col not in df.columns:
            continue
        if expected == "numeric":
            if not pd.api.types.is_numeric_dtype(df[col]):
                errors.append(
                    f"Column '{col}' must be numeric but contains non-numeric values. "
                    "Check for text like 'N/A' or '--' in that column."
                )

    # ── 5. Missing value report ───────────────────────────────────
    missing_pct = (df.isnull().mean() * 100).round(2)
    high_missing = missing_pct[missing_pct > 20].to_dict()
    if high_missing:
        warnings.append(
            f"Columns with >20% missing values (will be imputed): {high_missing}"
        )

    any_missing = missing_pct[missing_pct > 0].to_dict()
    if any_missing:
        warnings.append(f"Columns with some missing values: {any_missing}")

    # ── 6. Categorical value checks ───────────────────────────────
    for col, allowed in ALLOWED_CATEGORIES.items():
        if col not in df.columns:
            continue
        actual_vals = df[col].dropna().unique().tolist()
        unknown = [v for v in actual_vals if v not in allowed]
        if unknown:
            errors.append(
                f"Column '{col}' contains unrecognised values: {unknown}. "
                f"Allowed values are: {allowed}"
            )

    # ── 7. Numeric range checks ───────────────────────────────────
    for col, (lo, hi) in NUMERIC_BOUNDS.items():
        if col not in df.columns:
            continue
        if not pd.api.types.is_numeric_dtype(df[col]):
            continue  # already flagged above
        col_data = df[col].dropna()
        if lo is not None and (col_data < lo).any():
            bad_count = int((col_data < lo).sum())
            warnings.append(
                f"'{col}': {bad_count} value(s) below expected minimum {lo}. "
                "Possible data entry errors."
            )
        if hi is not None and (col_data > hi).any():
            bad_count = int((col_data > hi).sum())
            warnings.append(
                f"'{col}': {bad_count} value(s) above expected maximum {hi}. "
                "Possible data entry errors."
            )

    # ── 8. Target value check ─────────────────────────────────────
    if TARGET_COLUMN in df.columns:
        target_vals = df[TARGET_COLUMN].dropna().unique()
        invalid_target = [v for v in target_vals if v not in (0, 1, 0.0, 1.0)]
        if invalid_target:
            errors.append(
                f"'{TARGET_COLUMN}' must contain only 0 or 1, "
                f"but found: {invalid_target}"
            )
        # Class balance
        if TARGET_COLUMN in df.columns and pd.api.types.is_numeric_dtype(
            df[TARGET_COLUMN]
        ):
            pos_rate = df[TARGET_COLUMN].mean()
            if pos_rate < 0.05:
                warnings.append(
                    f"Only {pos_rate*100:.1f}% of rows are waterlogged (class=1). "
                    "Severe class imbalance — consider stratified sampling or "
                    "class_weight='balanced' (already enabled in the pipeline)."
                )
            elif pos_rate > 0.95:
                warnings.append(
                    f"{pos_rate*100:.1f}% of rows are waterlogged. "
                    "Almost all rows are positive — check your labelling process."
                )

    # ── 9. Minimum row count ──────────────────────────────────────
    if len(df) < 50:
        warnings.append(
            f"Only {len(df)} rows found. Reliable ML typically needs "
            "at least a few hundred labelled examples."
        )

    # ── 10. Stats summary ─────────────────────────────────────────
    stats: dict = {
        "rows": len(df),
        "columns": len(df.columns),
        "missing_pct": any_missing,
    }
    if TARGET_COLUMN in df.columns and pd.api.types.is_numeric_dtype(
        df[TARGET_COLUMN]
    ):
        stats["waterlogged_rate_pct"] = round(df[TARGET_COLUMN].mean() * 100, 2)

    valid = len(errors) == 0
    return {"valid": valid, "errors": errors, "warnings": warnings, "stats": stats}


# ------------------------------------------------------------------
# CLI helper — run directly: python -m src.validate data/raw/myfile.csv
# ------------------------------------------------------------------
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m src.validate <path/to/data.csv>")
        sys.exit(1)

    filepath = sys.argv[1]
    require_target = "--no-target" not in sys.argv
    df, report = load_and_validate(filepath, require_target=require_target)

    print(f"\n{'='*60}")
    print(f"Validation report for: {filepath}")
    print(f"{'='*60}")
    print(f"Valid : {report['valid']}")
    print(f"Rows  : {report['stats'].get('rows', 'N/A')}")

    if report["errors"]:
        print("\n❌ ERRORS (must fix before training):")
        for e in report["errors"]:
            print(f"  • {e}")

    if report["warnings"]:
        print("\n⚠️  WARNINGS (non-fatal):")
        for w in report["warnings"]:
            print(f"  • {w}")

    if report["valid"]:
        print("\n✅ Data passed all checks. Ready for preprocessing.")

