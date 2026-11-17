"""Route existing Tri-MIL entrypoints to the survival workflow when configured."""

import json
from pathlib import Path

import torch

from utils.runtime_utils import (
    get_pipeline_section,
    get_pipeline_section_compat,
    is_pipeline_yaml,
    read_plain_yaml,
    resolve_model_yaml_path,
    select_runtime_device,
)
from utils.yaml_utils import read_yaml, update_config_from_options


def maybe_run_survival(cli_args, mode):
    """Run survival train/valid/infer for General.task=survival; return routing status.

    Both model YAMLs and the unified Common/Train/Valid/Infer YAML are accepted.
    Survival input CSVs already contain splits, times and events; classification
    label encoding and stratified class splitting are deliberately bypassed.
    """
    plain = read_plain_yaml(cli_args.yaml_path)
    pipeline = is_pipeline_yaml(plain)
    common = get_pipeline_section(plain, "Common") if pipeline else {}
    section = (
        get_pipeline_section_compat(
            plain,
            {"train": "Train", "valid": "Valid", "infer": "Infer"}[mode],
            ("Test",) if mode == "valid" else (),
        )
        if pipeline
        else {}
    )
    model_path = resolve_model_yaml_path(cli_args.yaml_path, plain if pipeline else None)
    config = read_yaml(model_path)
    options = getattr(cli_args, "options", None)
    if options:
        config = update_config_from_options(config, options)
    if (
        str(
            section.get("task", common.get("task", config.General.get("task", "classification")))
        ).lower()
        != "survival"
    ):
        return False
    from process.survival import build_survival_model, evaluate_survival, train_survival
    from utils.survival_utils import SurvivalDataset

    device_override = getattr(cli_args, "device", None)
    config.General.device = (
        device_override
        if device_override is not None
        else section.get("device", common.get("device", config.General.device))
    )
    csv_path = (
        getattr(cli_args, "valid_dataset_csv", None)
        or getattr(cli_args, "test_dataset_csv", None)
        or getattr(cli_args, "infer_dataset_csv", None)
        or section.get("dataset_csv_path")
        or section.get("valid_dataset_csv")
        or section.get("test_dataset_csv")
        or section.get("infer_dataset_csv")
        or common.get("dataset_csv_path")
        or config.Dataset.get("dataset_csv_path")
    )
    if not csv_path:
        raise ValueError(
            "Survival requires a split CSV via Dataset.dataset_csv_path or the pipeline section."
        )
    config.Dataset.dataset_csv_path = csv_path
    output_dir = (
        getattr(cli_args, "output_dir", None)
        or getattr(cli_args, "valid_log_dir", None)
        or getattr(cli_args, "test_log_dir", None)
        or section.get("output_dir")
        or common.get("output_dir")
        or config.Logs.get("log_root_dir", "./log_dir/survival")
    )
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    if mode == "train":
        config.General.num_epochs = section.get("epochs", config.General.num_epochs)
        config.Logs.now_log_dir = output_dir
        result = train_survival(config)
    else:
        checkpoint_path = (
            getattr(cli_args, "model_weight_path", None)
            or section.get("model_weight_path")
            or common.get("model_weight_path")
        )
        if not checkpoint_path:
            raise ValueError("Survival evaluation/prediction requires model_weight_path.")
        device, _ = select_runtime_device(config.General.device)
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
        if (
            checkpoint.get("task") != "survival"
            or checkpoint.get("model_name") != config.General.MODEL_NAME
        ):
            raise ValueError("Checkpoint task/backbone does not match this survival configuration.")
        model = build_survival_model(config, device)
        model.load_state_dict(checkpoint["model_state_dict"])
        split = section.get("split", "test")
        dataset = SurvivalDataset(csv_path, split, require_outcomes=mode == "valid")
        if not len(dataset):
            raise ValueError(f"No samples in survival {split} split.")
        result = evaluate_survival(
            model, dataset, device, Path(output_dir) / f"{mode}_survival_predictions.csv"
        )
        (Path(output_dir) / f"{mode}_survival_metrics.json").write_text(
            json.dumps(result, indent=2, allow_nan=False), encoding="utf-8"
        )
    print(json.dumps(result, indent=2, allow_nan=False))
    return True
