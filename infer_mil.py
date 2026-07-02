import argparse
import warnings
import torch
import shutil
import os
import tempfile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json

from sklearn.metrics import (
    roc_curve,
    auc,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from sklearn.preprocessing import label_binarize
from itertools import cycle

from utils.yaml_utils import read_yaml
from torch.utils.data import DataLoader
from utils.loop_utils import test_loop
from utils.runtime_utils import (
    clone_config_with_overrides,
    create_infer_csv_from_feature_dir,
    infer_feature_dim,
    load_torch_checkpoint,
    resolve_first_feature_path,
    select_runtime_device,
)
from utils.wsi_utils import (
    WSI_Dataset,
    CDP_MIL_WSI_Dataset,
    LONG_MIL_WSI_Dataset,
)
from utils.model_utils import get_model_from_yaml, get_criterion

warnings.filterwarnings("ignore")


# =====================================================
# 统一提取 logits（兼容 dict / tensor）
# =====================================================
def extract_logits(output):
    if isinstance(output, dict):
        if "logits" in output:
            return output["logits"]
        elif "Y_prob" in output:
            return output["Y_prob"]
        else:
            raise ValueError("Model output dict missing 'logits' or 'Y_prob'")
    return output


# =====================================================
# Universal ROC computation（支持2类+多类）
# =====================================================
def compute_roc_auc(y_true, probs):

    y_true = np.asarray(y_true).astype(int)
    probs = np.asarray(probs)

    fpr = {}
    tpr = {}
    roc_auc = {}

    # sigmoid binary
    if probs.ndim == 1 or (probs.ndim == 2 and probs.shape[1] == 1):
        probs = probs.reshape(-1)
        fpr[0], tpr[0], _ = roc_curve(y_true, probs)
        roc_auc[0] = auc(fpr[0], tpr[0])
        return fpr, tpr, roc_auc

    # softmax binary
    if probs.ndim == 2 and probs.shape[1] == 2:
        pos_probs = probs[:, 1]
        fpr[0], tpr[0], _ = roc_curve(y_true, pos_probs)
        roc_auc[0] = auc(fpr[0], tpr[0])
        return fpr, tpr, roc_auc

    # multi-class
    num_classes = probs.shape[1]
    y_true_bin = label_binarize(y_true, classes=np.arange(num_classes))

    for i in range(num_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], probs[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    fpr["micro"], tpr["micro"], _ = roc_curve(
        y_true_bin.ravel(), probs.ravel()
    )
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    return fpr, tpr, roc_auc


# =====================================================
# JSON safe
# =====================================================
def to_json_serializable(obj):
    if isinstance(obj, dict):
        return {k: to_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [to_json_serializable(v) for v in obj]
    elif isinstance(obj, tuple):
        return [to_json_serializable(v) for v in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    elif torch.is_tensor(obj):
        return obj.detach().cpu().tolist()
    else:
        return obj


# =====================================================
# Main
# =====================================================
def test(args):

    yaml_args = read_yaml(args.yaml_path)
    model_name = yaml_args.General.MODEL_NAME
    num_classes = yaml_args.General.num_classes

    dataset_csv_path = args.test_dataset_csv
    temp_dir = None
    temp_yaml_path = None

    if not dataset_csv_path:
        if not args.feature_dir:
            raise ValueError("Provide either --test_dataset_csv or --feature_dir for inference.")
        temp_dir = tempfile.mkdtemp(prefix="tri_mil_infer_")
        dataset_csv_path = create_infer_csv_from_feature_dir(
            feature_dir=args.feature_dir,
            output_csv=os.path.join(temp_dir, "test.csv"),
            recursive=args.feature_recursive,
        )
        print(f"[INFO] Generated internal inference CSV: {dataset_csv_path}")

    first_feature_path = resolve_first_feature_path(
        csv_path=dataset_csv_path,
        feature_dir=args.feature_dir,
        recursive=args.feature_recursive,
    )
    inferred_in_dim = infer_feature_dim(first_feature_path)

    runtime_device, device_message = select_runtime_device(
        args.device if args.device is not None else getattr(yaml_args.General, "device", "auto")
    )
    print(f"[INFO] {device_message}")

    config_overrides = {}
    if int(yaml_args.Model.in_dim) != inferred_in_dim:
        config_overrides["Model.in_dim"] = inferred_in_dim
        print(
            f"[INFO] Overriding Model.in_dim from {yaml_args.Model.in_dim} to {inferred_in_dim} "
            f"based on feature file {first_feature_path}."
        )

    yaml_device_value = "cpu"
    if runtime_device.type == "cuda":
        yaml_device_value = runtime_device.index if runtime_device.index is not None else 0
    config_overrides["General.device"] = yaml_device_value

    if args.num_classes is not None:
        config_overrides["General.num_classes"] = args.num_classes

    if config_overrides:
        if temp_dir is None:
            temp_dir = tempfile.mkdtemp(prefix="tri_mil_infer_")
        temp_yaml_path = clone_config_with_overrides(
            base_config_path=args.yaml_path,
            output_path=os.path.join(temp_dir, "runtime_infer.yaml"),
            overrides=config_overrides,
        )
        yaml_args = read_yaml(temp_yaml_path)
        num_classes = yaml_args.General.num_classes
        print(f"[INFO] Generated runtime config: {temp_yaml_path}")

    label_map = None
    if hasattr(yaml_args, "Label"):
        label_map = {v: k for k, v in yaml_args.Label.items()}

    class_names = (
        [label_map[i] for i in range(num_classes)]
        if label_map
        else [str(i) for i in range(num_classes)]
    )

    print("Class names:", class_names)

    # Dataset
    if model_name == "CDP_MIL":
        test_ds = CDP_MIL_WSI_Dataset(
            dataset_csv_path,
            yaml_args.Dataset.BeyesGuassian_pt_dir,
            "test",
        )
    elif model_name == "LONG_MIL":
        test_ds = LONG_MIL_WSI_Dataset(
            dataset_csv_path,
            yaml_args.Dataset.h5_csv_path,
            "test",
        )
    else:
        if args.no_label:
            test_ds = WSI_Dataset(dataset_csv_path, "test", mode="infer")
        else:
            test_ds = WSI_Dataset(dataset_csv_path, "test")

    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False)

    # Model
    device = runtime_device
    model = get_model_from_yaml(yaml_args).to(device)
    model.load_state_dict(
        load_torch_checkpoint(args.model_weight_path, map_location=device)
    )
    model.eval()

    out_dir = args.test_log_dir
    os.makedirs(out_dir, exist_ok=True)

    shutil.copyfile(temp_yaml_path or args.yaml_path, os.path.join(out_dir, "test.yaml"))
    shutil.copyfile(dataset_csv_path, os.path.join(out_dir, "test_dataset.csv"))

    # =====================================================
    # 无标签推理
    # =====================================================
    if args.no_label:

        print("Running inference without labels...")

        slide_paths = []
        probs_list = []

        with torch.no_grad():
            for batch in test_loader:

                feat, slide_path = batch
                feat = feat.to(device)

                output = model(feat)
                logits = extract_logits(output)

                if num_classes == 1:
                    prob = torch.sigmoid(logits)
                else:
                    prob = torch.softmax(logits, dim=1)

                probs_list.append(prob.cpu().numpy())
                slide_paths.extend(slide_path)

        probs = np.vstack(probs_list)

        # Prediction
        if probs.ndim == 1 or probs.shape[1] == 1:
            probs = probs.reshape(-1, 1)
            y_pred = (probs.squeeze() > 0.5).astype(int)
        else:
            y_pred = probs.argmax(axis=1)

        # Save CSV
        df = pd.DataFrame({
            "wsi_path": slide_paths,
            "y_pred": y_pred,
        })

        if probs.ndim == 1 or probs.shape[1] == 1:
            df["prob"] = probs.reshape(-1)
        else:
            for i in range(probs.shape[1]):
                df[f"prob_{class_names[i]}"] = probs[:, i]

        df.to_csv(os.path.join(out_dir, "test_predictions.csv"), index=False)

        print("Inference finished.")
        return

    # =====================================================
    # 有标签评估
    # =====================================================
    criterion = get_criterion(yaml_args.Model.criterion)

    (
        test_loss,
        test_metrics,
        slide_paths,
        y_true,
        logits,
        probs,
    ) = test_loop(device, num_classes, model, test_loader, criterion)

    probs = np.asarray(probs)
    y_true = np.asarray(y_true).astype(int)

    # Prediction
    if probs.ndim == 1 or probs.shape[1] == 1:
        probs = probs.reshape(-1, 1)
        y_pred = (probs.squeeze() > 0.5).astype(int)
    else:
        y_pred = probs.argmax(axis=1)

    # Save CSV
    df = pd.DataFrame({
        "wsi_path": slide_paths,
        "y_true": y_true,
        "y_pred": y_pred,
    })

    if probs.ndim == 1 or probs.shape[1] == 1:
        df["prob"] = probs.reshape(-1)
    else:
        for i in range(probs.shape[1]):
            df[f"prob_{class_names[i]}"] = probs[:, i]

    df.to_csv(os.path.join(out_dir, "test_predictions.csv"), index=False)

    # Save metrics
    safe_metrics = to_json_serializable(test_metrics)
    with open(os.path.join(out_dir, "test_metrics.json"), "w") as f:
        json.dump(safe_metrics, f, indent=4)

    # ROC
    print("Drawing ROC curve...")
    fpr, tpr, roc_auc = compute_roc_auc(y_true, probs)

    plt.figure(figsize=(7, 7))

    if len(roc_auc) == 1:
        plt.plot(fpr[0], tpr[0], lw=2, label=f"AUC={roc_auc[0]:.3f}")
    else:
        colors = cycle(["aqua", "darkorange", "cornflowerblue", "red"])
        for i, color in zip(range(num_classes), colors):
            plt.plot(
                fpr[i],
                tpr[i],
                color=color,
                lw=2,
                label=f"{class_names[i]} (AUC={roc_auc[i]:.3f})",
            )
        plt.plot(
            fpr["micro"],
            tpr["micro"],
            linestyle="--",
            color="black",
            lw=2,
            label=f"micro-average (AUC={roc_auc['micro']:.3f})",
        )

    plt.plot([0, 1], [0, 1], "k--", lw=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "roc_curve.png"), dpi=300)
    plt.close()

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(cm)
    disp.plot(cmap="Blues", values_format="d")
    plt.savefig(os.path.join(out_dir, "confusion_matrix.png"), dpi=300)
    plt.close()

    print("Test with evaluation finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--yaml_path", type=str, required=True)
    parser.add_argument("--test_dataset_csv", type=str, default=None)
    parser.add_argument("--feature_dir", type=str, default=None,
                        help="Feature directory for folder-first inference. Internal CSV will be generated automatically.")
    parser.add_argument("--feature_recursive", action="store_true",
                        help="Recursively scan --feature_dir for feature files.")
    parser.add_argument("--model_weight_path", type=str, required=True)
    parser.add_argument("--test_log_dir", type=str, required=True)
    parser.add_argument("--device", type=str, default=None,
                        help="Runtime device override. Examples: auto, cpu, 0, cuda:0.")
    parser.add_argument("--num_classes", type=int, default=None,
                        help="Optional runtime override for General.num_classes.")
    parser.add_argument("--no_label", action="store_true",
                        help="Enable inference without labels")
    args = parser.parse_args()

    test(args)
