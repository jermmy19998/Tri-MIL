"""Feature loading, Cox loss, and datasets for right-censored MIL survival."""

from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from utils.survival_metrics import validate_outcomes


def load_survival_features(path):
    """Load a nonempty [patches, dimensions] feature bag from PT or HDF5."""
    if Path(path).suffix.lower() == ".h5":
        with h5py.File(path, "r") as handle:
            features = torch.from_numpy(handle["features"][:])
    else:
        features = torch.load(path, map_location="cpu", weights_only=True)
        if isinstance(features, dict):
            features = features.get("features", features.get("feats"))
    if features is None:
        raise ValueError(f"Missing features/feats in {path}.")
    features = torch.as_tensor(features).float()
    if features.ndim == 3 and features.shape[0] == 1:
        features = features.squeeze(0)
    if features.ndim != 2 or min(features.shape) == 0 or not torch.isfinite(features).all():
        raise ValueError(f"Expected finite nonempty [patches, dimensions] features: {path}.")
    return features


class SurvivalDataset(Dataset):
    """Read split-prefixed CSV columns with row-aligned survival outcomes.

    Required columns: {split}_slide_path, {split}_time, {split}_event.
    Prediction requires only paths. Empty padding rows are allowed, incomplete
    outcome rows are rejected. Optional {split}_case_id enables leakage checks.
    """

    def __init__(self, csv_path, split, require_outcomes=True):
        """Read a split and reject missing paths, outcomes, or invalid censoring."""
        self.split = split
        self.require_outcomes = require_outcomes
        frame = pd.read_csv(csv_path)
        path_col = f"{split}_slide_path"
        if path_col not in frame:
            self.frame = pd.DataFrame(columns=[path_col, f"{split}_time", f"{split}_event"])
            return
        valid = frame[path_col].notna()
        for field in ["time", "event"]:
            column = f"{split}_{field}"
            if column in frame and (frame[column].notna() & ~valid).any():
                raise ValueError(f"{column} contains outcomes without a slide path.")
        self.frame = frame.loc[valid].reset_index(drop=True)
        if require_outcomes:
            columns = [f"{split}_time", f"{split}_event"]
            if any(column not in frame for column in columns):
                raise ValueError(f"Missing survival columns: {columns}.")
            validate_outcomes(self.frame[columns[0]].to_numpy(), self.frame[columns[1]].to_numpy())
        self.require_outcomes = require_outcomes

    def __len__(self):
        """Return the number of feature bags in this split."""
        return len(self.frame)

    def __getitem__(self, index):
        """Return a feature bag and its source path; outcomes stay cohort aligned."""
        path = str(self.frame.iloc[index][f"{self.split}_slide_path"])
        return load_survival_features(path), path

    def outcomes(self):
        """Return follow-up times and event indicators in feature iteration order."""
        return validate_outcomes(
            self.frame[f"{self.split}_time"], self.frame[f"{self.split}_event"]
        )


def validate_split_separation(datasets):
    """Reject slide overlap and optional case overlap between dataset splits."""
    for index, left in enumerate(datasets):
        for right in datasets[index + 1 :]:
            for suffix in ["slide_path", "case_id"]:
                left_col, right_col = f"{left.split}_{suffix}", f"{right.split}_{suffix}"
                if left_col not in left.frame or right_col not in right.frame:
                    continue
                first = set(left.frame[left_col].dropna().astype(str))
                second = set(right.frame[right_col].dropna().astype(str))
                if suffix == "slide_path":
                    first = {str(Path(path).resolve()) for path in first}
                    second = {str(Path(path).resolve()) for path in second}
                if first & second:
                    raise ValueError(f"{suffix} leakage between {left.split} and {right.split}.")


def cox_ph_loss(risks, times, events):
    """Compute event-normalized Cox negative partial likelihood with Breslow ties.

    The risk set includes all individuals with time >= an observed event time.
    Sorting and logcumsumexp keep memory linear and avoid exponential overflow.
    At least one event is required; censored-only training is not identifiable.
    """
    risks = risks.reshape(-1)
    times = torch.as_tensor(times, device=risks.device, dtype=torch.float64).reshape(-1)
    events = torch.as_tensor(events, device=risks.device).reshape(-1)
    if not (risks.shape == times.shape == events.shape) or risks.numel() == 0:
        raise ValueError("Cox loss requires equally sized nonempty risk, time, and event vectors.")
    if not torch.isfinite(risks).all() or not torch.isfinite(times).all() or (times < 0).any():
        raise ValueError("Cox inputs must be finite, with nonnegative times.")
    if not ((events == 0) | (events == 1)).all() or not events.bool().any():
        raise ValueError("Cox loss requires binary events and at least one observed event.")
    order = torch.argsort(times, descending=True)
    sorted_times, scores, observed = times[order], risks[order], events[order].bool()
    end_indices = torch.searchsorted(-sorted_times, -sorted_times, right=True) - 1
    log_risk_sets = torch.logcumsumexp(scores, dim=0)[end_indices]
    return (log_risk_sets[observed] - scores[observed]).mean()


def model_risk(model, features):
    """Extract one scalar log hazard per bag from a compatible MIL backbone."""
    output = model(features.unsqueeze(0))
    logits = output.get("logits") if isinstance(output, dict) else output
    if not isinstance(logits, torch.Tensor) or logits.numel() != 1:
        raise ValueError(
            "Survival models must return one scalar logits value per bag (num_classes=1)."
        )
    return logits.reshape(())
