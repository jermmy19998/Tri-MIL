"""Regression tests for censoring, Cox ties, data validation, and survival entrypoints."""

import json
import math
import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pytest
import torch
import yaml

from prepare_survival_csv import prepare_survival_csv
from utils.survival_metrics import concordance_index
from utils.survival_utils import (
    SurvivalDataset,
    cox_ph_loss,
    load_survival_features,
    validate_split_separation,
)


@pytest.mark.parametrize("risk,expected", [([3, 2, 1], 1), ([1, 2, 3], 0), ([1, 1, 1], 0.5)])
def test_cindex_order_and_risk_ties(risk, expected):
    """Verify risk direction and half-credit for tied risk scores."""
    assert concordance_index([1, 2, 3], risk, [1, 1, 0]) == expected


def test_cindex_censoring_and_time_ties():
    """Exclude unobserved early events, and include tied event/censor pairs."""
    assert concordance_index([1, 2, 3], [99, 2, 1], [0, 1, 1]) == 1
    assert concordance_index([1, 1], [2, 1], [1, 0]) == 1
    assert math.isnan(concordance_index([1, 1], [2, 1], [1, 1]))
    assert math.isnan(concordance_index([1, 2], [2, 1], [0, 0]))


@pytest.mark.parametrize(
    "times,risks,events",
    [
        ([1, -1], [1, 2], [1, 0]),
        ([1, 2], [1, float("nan")], [1, 0]),
        ([1, 2], [1, 2], [1, 2]),
        ([1], [1, 2], [1]),
    ],
)
def test_invalid_metric_inputs(times, risks, events):
    """Reject invalid shapes, event flags, times, and nonfinite risks."""
    with pytest.raises(ValueError):
        concordance_index(times, risks, events)


def test_cox_breslow_matches_explicit_risk_sets_and_gradients():
    """Match a direct risk-set implementation under tied event times."""
    risks = torch.tensor([1.2, -0.4, 0.5, 0.8], dtype=torch.float64, requires_grad=True)
    times = torch.tensor([2.0, 2.0, 3.0, 1.0])
    events = torch.tensor([1, 1, 0, 1])
    expected = torch.stack(
        [torch.logsumexp(risks[times >= times[i]], 0) - risks[i] for i in range(4) if events[i]]
    ).mean()
    loss = cox_ph_loss(risks, times, events)
    torch.testing.assert_close(loss, expected)
    torch.testing.assert_close(
        torch.autograd.grad(loss, risks)[0], torch.autograd.grad(expected, risks)[0]
    )
    torch.testing.assert_close(loss, cox_ph_loss(risks + 10000, times, events))
    order = torch.tensor([3, 1, 0, 2])
    torch.testing.assert_close(loss, cox_ph_loss(risks[order], times[order], events[order]))


def test_all_censored_cox_is_rejected():
    """Reject a Cox training cohort with no observed events."""
    with pytest.raises(ValueError, match="observed event"):
        cox_ph_loss(torch.zeros(2), [1, 2], [0, 0])


def test_dataset_alignment_and_feature_formats(tmp_path):
    """Keep sparse CSV rows aligned and load both PT dictionaries and HDF5."""
    features = torch.randn(4, 8)
    pt = tmp_path / "bag.pt"
    h5 = tmp_path / "bag.h5"
    torch.save({"feats": features}, pt)
    with h5py.File(h5, "w") as handle:
        handle.create_dataset("features", data=features.numpy())
    torch.testing.assert_close(load_survival_features(pt), features)
    torch.testing.assert_close(load_survival_features(h5), features)
    csv = tmp_path / "splits.csv"
    pd.DataFrame(
        {
            "train_slide_path": [str(pt), None, str(h5)],
            "train_time": [2.0, None, 3.0],
            "train_event": [1, None, 0],
        }
    ).to_csv(csv, index=False)
    dataset = SurvivalDataset(csv, "train")
    np.testing.assert_array_equal(dataset.outcomes()[0], [2, 3])
    assert len(dataset) == 2
    frame = pd.read_csv(csv)
    frame.loc[2, "train_time"] = np.nan
    frame.to_csv(csv, index=False)
    with pytest.raises(ValueError):
        SurvivalDataset(csv, "train")


def test_case_split_and_leakage_guard(tmp_path):
    """Keep all slides of each case together and detect later leakage."""
    rows = []
    for index in range(12):
        for slide in range(2):
            path = tmp_path / f"{index}_{slide}.pt"
            torch.save(torch.randn(3, 8), path)
            rows.append(
                {
                    "slide_path": str(path),
                    "time": index + 1,
                    "event": index % 2,
                    "case_id": f"case{index}",
                }
            )
    manifest, output = tmp_path / "manifest.csv", tmp_path / "splits.csv"
    pd.DataFrame(rows).to_csv(manifest, index=False)
    result = prepare_survival_csv(manifest, output)
    datasets = [SurvivalDataset(output, split) for split in ["train", "val", "test"]]
    validate_split_separation(datasets)
    assert sum(map(len, datasets)) == 24
    result.loc[0, "val_case_id"] = result.loc[0, "train_case_id"]
    result.to_csv(output, index=False)
    with pytest.raises(ValueError, match="case_id leakage"):
        validate_split_separation([SurvivalDataset(output, split) for split in ["train", "val"]])


def test_train_valid_infer_survival_entrypoints(tmp_path):
    """Train a Cox MIL model, reload its checkpoint, validate, and predict unlabeled bags."""
    root = Path(__file__).resolve().parents[1]
    config = yaml.safe_load((root / "configs/SURVIVAL_AB_MIL.yaml").read_text(encoding="utf-8"))
    config["Model"].update(in_dim=8, L=8, D=4, dropout=0)
    config["General"].update(num_epochs=2, device="cpu")
    columns = {}
    for split, count in [("train", 4), ("val", 3), ("test", 3)]:
        paths = []
        for index in range(count):
            path = tmp_path / f"{split}_{index}.pt"
            torch.save(torch.randn(index + 3, 8), path)
            paths.append(str(path))
        columns.update(
            {
                f"{split}_slide_path": pd.Series(paths),
                f"{split}_time": pd.Series(range(1, count + 1)),
                f"{split}_event": pd.Series([1] * (count - 1) + [0]),
            }
        )
    csv = tmp_path / "splits.csv"
    pd.DataFrame(columns).to_csv(csv, index=False)
    config["Dataset"]["dataset_csv_path"] = str(csv)
    model_yaml = tmp_path / "model.yaml"
    model_yaml.write_text(yaml.safe_dump(config), encoding="utf-8")
    output = tmp_path / "output"
    checkpoint = output / "Best_survival.pth"
    pipeline = {
        "Common": {
            "task": "survival",
            "model_yaml_path": str(model_yaml),
            "dataset_csv_path": str(csv),
            "model_weight_path": str(checkpoint),
            "device": "cpu",
        },
        "Train": {"output_dir": str(output)},
        "Valid": {"output_dir": str(output / "valid")},
        "Infer": {"output_dir": str(output / "infer")},
    }
    pipeline_yaml = tmp_path / "pipeline.yaml"
    pipeline_yaml.write_text(yaml.safe_dump(pipeline), encoding="utf-8")
    for entry in ["train_mil.py", "valid_mil.py", "infer_mil.py"]:
        if entry == "infer_mil.py":
            pd.DataFrame({"test_slide_path": columns["test_slide_path"]}).to_csv(csv, index=False)
        completed = subprocess.run(
            [sys.executable, str(root / entry), "--yaml_path", str(pipeline_yaml)],
            cwd=root,
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert completed.returncode == 0, completed.stdout + completed.stderr
    assert checkpoint.exists()
    metrics = json.loads((output / "survival_metrics.json").read_text())
    assert 0 <= metrics["val"]["cindex"] <= 1
    assert metrics["best_epoch"] in [1, 2]
    expected = pd.read_csv(output / "valid/valid_survival_predictions.csv")
    predictions = pd.read_csv(output / "infer/infer_survival_predictions.csv")
    np.testing.assert_allclose(expected.risk, predictions.risk)
    assert "time" not in predictions and len(predictions) == 3
