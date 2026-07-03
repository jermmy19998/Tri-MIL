import copy
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.model_selection import StratifiedKFold


SUPPORTED_FEATURE_SUFFIXES = (".h5", ".pt")
PIPELINE_SECTION_NAMES = ("Common", "Train", "Test", "Infer", "Heatmap")


def read_plain_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    return data or {}


def is_pipeline_yaml(config: dict) -> bool:
    if not isinstance(config, dict):
        return False
    if "General" in config and "Model" in config:
        return False
    return any(section in config for section in PIPELINE_SECTION_NAMES)


def get_pipeline_section(config: dict, section_name: str) -> dict:
    section = config.get(section_name, {})
    return section if isinstance(section, dict) else {}


def resolve_model_yaml_path(yaml_path: str, pipeline_config: dict | None = None) -> str:
    pipeline_config = pipeline_config or {}
    common_cfg = get_pipeline_section(pipeline_config, "Common")
    model_yaml_path = common_cfg.get("model_yaml_path") or common_cfg.get("yaml_path")
    return str(Path(model_yaml_path).expanduser().resolve()) if model_yaml_path else str(Path(yaml_path).expanduser().resolve())


def merge_nested_dict(base: dict, overrides: dict) -> dict:
    merged = copy.deepcopy(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = merge_nested_dict(merged[key], value)
        else:
            merged[key] = value
    return merged


def collect_feature_files(feature_dir: str, recursive: bool = False) -> list[str]:
    root = Path(feature_dir).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"Feature directory does not exist: {feature_dir}")
    if not root.is_dir():
        raise NotADirectoryError(f"Feature directory is not a directory: {feature_dir}")

    iterator = root.rglob("*") if recursive else root.iterdir()
    feature_files = [
        str(path.resolve())
        for path in iterator
        if path.is_file() and path.suffix.lower() in SUPPORTED_FEATURE_SUFFIXES
    ]
    feature_files.sort()

    if not feature_files:
        raise FileNotFoundError(
            f"No feature files with suffixes {SUPPORTED_FEATURE_SUFFIXES} were found in {feature_dir}"
        )

    return feature_files


def resolve_first_feature_path(
    csv_path: str | None = None,
    feature_dir: str | None = None,
    recursive: bool = False,
) -> str:
    if feature_dir:
        return collect_feature_files(feature_dir, recursive=recursive)[0]

    if not csv_path:
        raise ValueError("Either csv_path or feature_dir must be provided to resolve a feature file.")

    df = pd.read_csv(csv_path)
    candidate_columns = [
        "test_slide_path",
        "slide_path",
        "test_slide_path",
        "val_slide_path",
        "train_slide_path",
    ]
    for column in candidate_columns:
        if column in df.columns:
            values = df[column].dropna().tolist()
            if values:
                return str(values[0])

    raise ValueError(
        f"Unable to find a feature-path column in CSV: {csv_path}. "
        "Expected one of test_slide_path, slide_path, train_slide_path, val_slide_path."
    )


def _torch_load_any(path: str):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except Exception:
        return torch.load(path, map_location="cpu", weights_only=False)


def load_torch_checkpoint(path: str, map_location="cpu"):
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except Exception:
        return torch.load(path, map_location=map_location, weights_only=False)


def infer_feature_dim(feature_path: str) -> int:
    if feature_path.endswith(".h5"):
        with h5py.File(feature_path, "r") as handle:
            if "features" not in handle:
                raise KeyError(f"`features` dataset not found in {feature_path}")
            features = handle["features"]
            if len(features.shape) < 2:
                raise ValueError(f"Unexpected feature shape in {feature_path}: {features.shape}")
            return int(features.shape[-1])

    feature_obj = _torch_load_any(feature_path)
    if isinstance(feature_obj, dict):
        if "feats" in feature_obj:
            feature_obj = feature_obj["feats"]
        elif "features" in feature_obj:
            feature_obj = feature_obj["features"]
        else:
            raise ValueError(
                f"Unsupported feature dict format in {feature_path}. Keys: {list(feature_obj.keys())}"
            )

    if not torch.is_tensor(feature_obj):
        feature_obj = torch.as_tensor(feature_obj)

    if feature_obj.ndim < 2:
        raise ValueError(f"Unexpected feature tensor shape in {feature_path}: {tuple(feature_obj.shape)}")

    return int(feature_obj.shape[-1])


def select_runtime_device(requested_device=None) -> tuple[torch.device, str]:
    if isinstance(requested_device, torch.device):
        return requested_device, f"Using device {requested_device}."

    requested = "auto" if requested_device in (None, "", "auto") else str(requested_device).strip()
    cuda_count = torch.cuda.device_count()
    has_cuda = torch.cuda.is_available() and cuda_count > 0

    if requested.lower() == "cpu":
        return torch.device("cpu"), "Using CPU because --device cpu was requested."

    if requested == "auto":
        if has_cuda:
            return torch.device("cuda:0"), "Auto-selected cuda:0."
        return torch.device("cpu"), "CUDA is unavailable; falling back to CPU."

    if requested.isdigit():
        idx = int(requested)
        if has_cuda and idx < cuda_count:
            return torch.device(f"cuda:{idx}"), f"Using requested cuda:{idx}."
        if has_cuda:
            return torch.device("cuda:0"), (
                f"Requested CUDA device {idx} is unavailable; falling back to cuda:0."
            )
        return torch.device("cpu"), (
            f"Requested CUDA device {idx} is unavailable because CUDA is not available; falling back to CPU."
        )

    if requested.lower().startswith("cuda:"):
        try:
            idx = int(requested.split(":", 1)[1])
        except ValueError:
            idx = None

        if idx is not None and has_cuda and idx < cuda_count:
            return torch.device(f"cuda:{idx}"), f"Using requested cuda:{idx}."
        if has_cuda:
            return torch.device("cuda:0"), (
                f"Requested device {requested} is unavailable; falling back to cuda:0."
            )
        return torch.device("cpu"), (
            f"Requested device {requested} is unavailable because CUDA is not available; falling back to CPU."
        )

    return torch.device(requested), f"Using requested device {requested}."


def create_infer_csv_from_feature_dir(
    feature_dir: str,
    output_csv: str,
    recursive: bool = False,
) -> str:
    feature_files = collect_feature_files(feature_dir, recursive=recursive)
    output_path = Path(output_csv).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"test_slide_path": feature_files}).to_csv(output_path, index=False)
    return str(output_path)


def clone_config_with_overrides(base_config_path: str, output_path: str, overrides: dict) -> str:
    with open(base_config_path, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    updated = copy.deepcopy(config)
    for dotted_key, value in overrides.items():
        keys = dotted_key.split(".")
        cursor = updated
        for key in keys[:-1]:
            cursor = cursor[key]
        cursor[keys[-1]] = value

    target = Path(output_path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    with open(target, "w", encoding="utf-8") as handle:
        yaml.safe_dump(updated, handle, sort_keys=False, allow_unicode=True)
    return str(target)


def clone_config_with_updates(base_config_path: str, output_path: str, updater) -> str:
    with open(base_config_path, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    updated = copy.deepcopy(config)
    updated = updater(updated)

    target = Path(output_path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    with open(target, "w", encoding="utf-8") as handle:
        yaml.safe_dump(updated, handle, sort_keys=False, allow_unicode=True)
    return str(target)


def infer_encoder_name_from_feature_dir(feature_dir: str | None) -> str:
    if not feature_dir:
        return "unknown_encoder"
    name = Path(feature_dir).expanduser().resolve().name
    if name.startswith("features_"):
        return name.replace("features_", "", 1)
    return name


def build_train_base_csv_from_reference(
    reference_csv: str,
    feature_dir: str,
    output_csv: str,
    slide_col: str,
    label_col: str,
    feature_recursive: bool = False,
) -> tuple[str, dict[str, int]]:
    feature_files = collect_feature_files(feature_dir, recursive=feature_recursive)
    feature_map = {Path(path).stem: path for path in feature_files}

    ref_df = pd.read_csv(reference_csv)
    if slide_col not in ref_df.columns:
        raise ValueError(f"Missing slide column in reference CSV: {slide_col}")
    if label_col not in ref_df.columns:
        raise ValueError(f"Missing label column in reference CSV: {label_col}")

    label_values = [value for value in ref_df[label_col].dropna().tolist()]
    unique_labels = sorted(dict.fromkeys(label_values))
    label_map = {str(label): idx for idx, label in enumerate(unique_labels)}

    rows = []
    for _, row in ref_df.iterrows():
        slide_value = row[slide_col]
        label_value = row[label_col]
        if pd.isna(slide_value) or pd.isna(label_value):
            continue
        slide_id = Path(str(slide_value)).stem
        feature_path = feature_map.get(slide_id)
        if feature_path is None:
            continue
        rows.append({"slide_path": feature_path, "label": label_map[str(label_value)]})

    if not rows:
        raise ValueError("No rows matched between reference CSV and feature directory.")

    output_path = Path(output_csv).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_path, index=False)
    return str(output_path), label_map


def build_train_base_csv_from_source_dirs(
    source_dir: str,
    feature_dir: str,
    output_csv: str,
    feature_recursive: bool = False,
    source_recursive: bool = True,
) -> tuple[str, dict[str, int]]:
    source_root = Path(source_dir).expanduser().resolve()
    if not source_root.exists():
        raise FileNotFoundError(f"Source directory does not exist: {source_dir}")

    feature_files = collect_feature_files(feature_dir, recursive=feature_recursive)
    feature_map = {Path(path).stem: path for path in feature_files}

    label_dirs = [path for path in sorted(source_root.iterdir()) if path.is_dir()]
    if not label_dirs:
        raise ValueError("No label subdirectories found under source_dir.")

    label_map = {label_dir.name: idx for idx, label_dir in enumerate(label_dirs)}
    source_suffixes = {
        ".svs", ".tif", ".tiff", ".ndpi", ".mrxs", ".scn", ".vms", ".vmu", ".bif",
        ".svslide", ".png", ".jpg", ".jpeg", ".bmp", ".czi",
    }

    rows = []
    for label_dir in label_dirs:
        iterator = label_dir.rglob("*") if source_recursive else label_dir.iterdir()
        for path in iterator:
            if not path.is_file() or path.suffix.lower() not in source_suffixes:
                continue
            feature_path = feature_map.get(path.stem)
            if feature_path is None:
                continue
            rows.append({"slide_path": feature_path, "label": label_map[label_dir.name]})

    if not rows:
        raise ValueError("No labeled rows were generated from source label directories.")

    output_path = Path(output_csv).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_path, index=False)
    return str(output_path), label_map


def build_kfold_dataset_dir(
    base_csv_path: str,
    output_dir: str,
    dataset_name: str,
    seed: int = 42,
    k: int = 3,
) -> str:
    df = pd.read_csv(base_csv_path)
    if "slide_path" not in df.columns or "label" not in df.columns:
        raise ValueError("Base training CSV must contain slide_path and label columns.")

    X = df["slide_path"].tolist()
    y = df["label"].tolist()
    save_root = Path(output_dir).expanduser().resolve() / dataset_name
    save_root.mkdir(parents=True, exist_ok=True)

    skf = StratifiedKFold(n_splits=k, random_state=seed, shuffle=True)
    for fold_idx, (train_index, val_index) in enumerate(skf.split(X, y), start=1):
        train_paths = [X[i] for i in train_index]
        train_labels = [y[i] for i in train_index]
        val_paths = [X[i] for i in val_index]
        val_labels = [y[i] for i in val_index]

        max_len = max(len(train_paths), len(val_paths))
        frame = pd.DataFrame(
            {
                "train_slide_path": train_paths + [np.nan] * (max_len - len(train_paths)),
                "train_label": train_labels + [np.nan] * (max_len - len(train_labels)),
                "val_slide_path": val_paths + [np.nan] * (max_len - len(val_paths)),
                "val_label": val_labels + [np.nan] * (max_len - len(val_labels)),
                "test_slide_path": [np.nan] * max_len,
                "test_label": [np.nan] * max_len,
            }
        )
        frame.to_csv(save_root / f"Total_{k}-fold_{dataset_name}_{fold_idx}fold.csv", index=False)

    return str(save_root)


def write_label_map_json(label_map: dict[str, int], output_path: str) -> str:
    target = Path(output_path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    with open(target, "w", encoding="utf-8") as handle:
        json.dump(label_map, handle, indent=2, ensure_ascii=False)
    return str(target)
