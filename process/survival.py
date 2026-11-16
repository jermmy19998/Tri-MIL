"""Train and evaluate MIL Cox models without mixing survival and class labels."""

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from utils.model_utils import get_model_from_yaml, get_optimizer
from utils.runtime_utils import select_runtime_device
from utils.survival_metrics import concordance_index
from utils.survival_utils import (
    SurvivalDataset,
    cox_ph_loss,
    model_risk,
    validate_split_separation,
)


def build_survival_model(args, device):
    """Build a single-head MIL backbone with scalar Cox output on the device."""
    if args.General.MODEL_NAME in {"DTFD_MIL", "CDP_MIL", "LONG_MIL", "CLAM_MB_MIL"}:
        raise ValueError(
            "This backbone requires a specialized survival adapter; choose AB_MIL or another scalar-logit model."
        )
    args.General.num_classes = 1
    model = get_model_from_yaml(args)
    if not isinstance(model, torch.nn.Module):
        raise ValueError("Survival requires a single torch.nn.Module backbone.")
    return model.to(device)


def evaluate_survival(model, dataset, device, output_path=None):
    """Export cohort risks and compute C-index and Cox loss when outcomes exist."""
    model.eval()
    paths, risks = [], []
    with torch.no_grad():
        for features, path in dataset:
            paths.append(path)
            risks.append(float(model_risk(model, features.to(device)).cpu()))
    predictions = pd.DataFrame({"slide_path": paths, "risk": risks})
    metrics = {"n_samples": len(risks)}
    if dataset.require_outcomes and len(dataset):
        times, events = dataset.outcomes()
        predictions["time"], predictions["event"] = times, events.astype(int)
        cindex = concordance_index(times, risks, events)
        loss = (
            float(cox_ph_loss(torch.tensor(risks, dtype=torch.float64), times, events))
            if events.any()
            else None
        )
        metrics.update(
            cindex=cindex if math.isfinite(cindex) else None,
            cox_loss=loss,
            n_events=int(events.sum()),
        )
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        predictions.to_csv(output_path, index=False)
    return metrics


def train_survival(args):
    """Optimize full-cohort Cox risk sets and select checkpoints by validation C-index.

    Each epoch retains one graph per training bag until the cohort loss backward
    pass. This gives exact risk sets; GPU memory therefore scales with the cohort.
    Test outcomes are evaluated once after checkpoint selection, never for tuning.
    """
    torch.manual_seed(int(args.General.seed))
    np.random.seed(int(args.General.seed))
    device, _ = select_runtime_device(args.General.device)
    datasets = [
        SurvivalDataset(args.Dataset.dataset_csv_path, split) for split in ["train", "val", "test"]
    ]
    train, validation, test = datasets
    validate_split_separation(datasets)
    if len(train) < 2 or not len(validation):
        raise ValueError(
            "Survival training requires at least two training bags and a nonempty validation split."
        )
    times, events = train.outcomes()
    if not events.any():
        raise ValueError("Training requires at least one observed event.")
    val_times, val_events = validation.outcomes()
    if not math.isfinite(concordance_index(val_times, np.zeros(len(validation)), val_events)):
        raise ValueError("Validation requires at least one comparable pair for C-index selection.")
    model = build_survival_model(args, device)
    optimizer, _ = get_optimizer(args, model)
    epochs = int(args.General.num_epochs)
    if epochs < 1:
        raise ValueError("num_epochs must be positive.")
    output_dir = Path(args.Logs.now_log_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    best, logs, stale = -float("inf"), [], 0
    earlystop = args.General.get("earlystop", {})
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        risks = torch.stack([model_risk(model, features.to(device)) for features, _ in train])
        loss = cox_ph_loss(risks, times, events)
        loss.backward()
        optimizer.step()
        metrics = evaluate_survival(model, validation, device)
        if metrics["cindex"] is None:
            raise ValueError("Validation produced no finite C-index.")
        logs.append(
            {
                "epoch": epoch,
                "train_cox_loss": float(loss.detach().cpu()),
                "val_cox_loss": metrics["cox_loss"],
                "val_cindex": metrics["cindex"],
            }
        )
        checkpoint = {
            "model_state_dict": model.state_dict(),
            "epoch": epoch,
            "task": "survival",
            "model_name": args.General.MODEL_NAME,
            "config": args.to_dict(),
            "val_cindex": metrics["cindex"],
        }
        torch.save(checkpoint, output_dir / "Last_survival.pth")
        if metrics["cindex"] > best:
            best, stale = metrics["cindex"], 0
            torch.save(checkpoint, output_dir / "Best_survival.pth")
        else:
            stale += 1
        print(
            f"Epoch {epoch}: train Cox={logs[-1]['train_cox_loss']:.5f}, val C-index={metrics['cindex']:.5f}"
        )
        if earlystop.get("use", False) and stale >= int(earlystop.get("patience", 5)):
            break
    pd.DataFrame(logs).to_csv(output_dir / "survival_log.csv", index=False)
    checkpoint = torch.load(
        output_dir / "Best_survival.pth", map_location=device, weights_only=True
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    result = {
        "best_epoch": checkpoint["epoch"],
        "val": evaluate_survival(model, validation, device, output_dir / "val_predictions.csv"),
    }
    if len(test):
        result["test"] = evaluate_survival(model, test, device, output_dir / "test_predictions.csv")
    (output_dir / "survival_metrics.json").write_text(
        json.dumps(result, indent=2, allow_nan=False), encoding="utf-8"
    )
    return result
