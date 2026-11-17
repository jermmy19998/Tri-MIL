"""Preserve label-free inference contracts when merging main and release."""

import h5py
import pandas as pd
import pytest
import torch

from utils.wsi_utils import CDP_MIL_WSI_Dataset, LONG_MIL_WSI_Dataset, WSI_Dataset


@pytest.mark.parametrize("kind", ["standard", "cdp", "long"])
def test_inference_datasets_do_not_require_labels(kind, tmp_path):
    """Return feature bags and source paths without reading a label column."""
    path = tmp_path / "slide.pt"
    features = torch.randn(4, 8)
    torch.save(features, path)
    csv = tmp_path / "inference.csv"
    pd.DataFrame({"test_slide_path": [str(path)]}).to_csv(csv, index=False)
    if kind == "cdp":
        cached_path = tmp_path / "slide_bayesian_gaussian.pt"
        torch.save(features, cached_path)
        dataset = CDP_MIL_WSI_Dataset(csv, str(tmp_path), "test", mode="infer")
        expected_path = cached_path
    elif kind == "long":
        h5_path = tmp_path / "slide.h5"
        coords = torch.arange(8).reshape(4, 2)
        with h5py.File(h5_path, "w") as handle:
            handle.create_dataset("coords", data=coords.numpy())
        h5_csv = tmp_path / "coordinates.csv"
        pd.DataFrame({"h5_path": [str(h5_path)]}).to_csv(h5_csv, index=False)
        dataset = LONG_MIL_WSI_Dataset(csv, h5_csv, "test", mode="infer")
        features = torch.cat([features, coords], dim=-1)
        expected_path = path
    else:
        dataset = WSI_Dataset(csv, "test", mode="infer")
        expected_path = path
    actual, actual_path = dataset[0]
    torch.testing.assert_close(actual, features)
    assert actual_path == str(expected_path)
