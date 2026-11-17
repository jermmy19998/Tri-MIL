"""Split a survival feature manifest by case into a Tri-MIL survival CSV."""

import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from utils.survival_metrics import validate_outcomes


def prepare_survival_csv(manifest, output, seed=2024, val_fraction=0.2, test_fraction=0.2):
    """Split slide_path,time,event,case_id rows without leaking cases across splits.

    Fractions apply to unique cases, with rounding performed by scikit-learn.
    Explicit case IDs are required so multiple slides from a patient stay together.
    """
    frame = pd.read_csv(manifest)
    columns = ["slide_path", "time", "event", "case_id"]
    if any(column not in frame for column in columns) or frame[columns].isna().any().any():
        raise ValueError(f"Manifest requires nonmissing {columns}.")
    validate_outcomes(frame.time, frame.event)
    if frame.slide_path.duplicated().any():
        raise ValueError("Duplicate slide paths are not allowed.")
    if any(not Path(path).is_file() for path in frame.slide_path):
        raise ValueError("Every slide_path must reference an existing feature file.")
    if not (0 < val_fraction < 1 and 0 < test_fraction < 1 and val_fraction + test_fraction < 1):
        raise ValueError("Validation/test fractions must be positive and sum to less than one.")
    if (frame.groupby("case_id")[["time", "event"]].nunique() > 1).any().any():
        raise ValueError("Slides in the same case must share time and event outcomes.")
    train_val_indices, test_indices = next(
        GroupShuffleSplit(n_splits=1, test_size=test_fraction, random_state=seed).split(
            frame, groups=frame.case_id
        )
    )
    train_val = frame.iloc[train_val_indices]
    train_indices, val_indices = next(
        GroupShuffleSplit(
            n_splits=1, test_size=val_fraction / (1 - test_fraction), random_state=seed
        ).split(train_val, groups=train_val.case_id)
    )
    splits = {
        "train": train_val.iloc[train_indices],
        "val": train_val.iloc[val_indices],
        "test": frame.iloc[test_indices],
    }
    result = pd.DataFrame(
        {
            f"{split}_{column}": rows[column].reset_index(drop=True)
            for split, rows in splits.items()
            for column in columns
        }
    )
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False)
    return result


def main():
    """Parse manifest and case split options, then write the survival split CSV."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--val_fraction", type=float, default=0.2)
    parser.add_argument("--test_fraction", type=float, default=0.2)
    args = parser.parse_args()
    prepare_survival_csv(
        args.manifest, args.output, args.seed, args.val_fraction, args.test_fraction
    )


if __name__ == "__main__":
    main()
