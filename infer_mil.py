import argparse
import warnings
import torch
import shutil
import os
import tempfile
import numpy as np
import pandas as pd

from utils.yaml_utils import read_yaml
from torch.utils.data import DataLoader
from utils.runtime_utils import (
    clone_config_with_overrides,
    create_infer_csv_from_feature_dir,
    get_pipeline_section,
    get_pipeline_section_compat,
    infer_feature_dim,
    is_pipeline_yaml,
    load_torch_checkpoint,
    read_plain_yaml,
    resolve_first_feature_path,
    resolve_model_yaml_path,
    select_runtime_device,
)
from utils.wsi_utils import (
    WSI_Dataset,
    CDP_MIL_WSI_Dataset,
    LONG_MIL_WSI_Dataset,
)
from utils.model_utils import get_model_from_yaml

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


def dtfd_predict_logits(feat, model_list, model_args):
    classifier, attention, dim_reduction, att_cls = model_list
    num_groups = int(model_args.num_Group)
    instance_per_group = max(1, int(model_args.total_instance) // num_groups)
    pseudo_features = []

    for sub_features in torch.chunk(feat.squeeze(0), num_groups, dim=0):
        mid_features = dim_reduction(sub_features)
        attention_weights = attention(mid_features).squeeze(0)
        attended_features = torch.einsum(
            "ns,n->ns", mid_features, attention_weights
        )
        bag_feature = attended_features.sum(dim=0, keepdim=True)
        classifier_weight = list(classifier.parameters())[-2]
        patch_logits = torch.einsum(
            "bgf,cf->bcg", attended_features.unsqueeze(0), classifier_weight
        ).squeeze(0).transpose(0, 1)
        patch_probs = torch.softmax(patch_logits, dim=1)
        sort_idx = torch.argsort(patch_probs[:, -1], descending=True)
        count = min(instance_per_group, len(sort_idx))
        top_idx = sort_idx[:count]

        if model_args.distill == "MaxMinS":
            bottom_idx = sort_idx[-count:]
            pseudo_features.append(
                mid_features.index_select(0, torch.cat([top_idx, bottom_idx]))
            )
        elif model_args.distill == "MaxS":
            pseudo_features.append(mid_features.index_select(0, top_idx))
        elif model_args.distill == "AFS":
            pseudo_features.append(bag_feature)
        else:
            raise ValueError(f"Unsupported DTFD distill mode: {model_args.distill}")

    return att_cls(torch.cat(pseudo_features, dim=0))["logits"]


# =====================================================
# Main
# =====================================================
def test(args):
    plain_cfg = read_plain_yaml(args.yaml_path)
    model_yaml_path = resolve_model_yaml_path(args.yaml_path, plain_cfg if is_pipeline_yaml(plain_cfg) else None)
    infer_cfg = get_pipeline_section_compat(plain_cfg, "Infer") if is_pipeline_yaml(plain_cfg) else {}
    common_cfg = get_pipeline_section(plain_cfg, "Common") if is_pipeline_yaml(plain_cfg) else {}

    yaml_args = read_yaml(model_yaml_path)
    model_name = yaml_args.General.MODEL_NAME
    num_classes = yaml_args.General.num_classes

    dataset_csv_path = args.test_dataset_csv or infer_cfg.get("test_dataset_csv") or common_cfg.get("test_dataset_csv")
    temp_dir = None
    temp_yaml_path = None

    if not dataset_csv_path:
        feature_dir = args.feature_dir or infer_cfg.get("feature_dir") or common_cfg.get("feature_dir")
        feature_recursive = args.feature_recursive or bool(infer_cfg.get("feature_recursive")) or bool(common_cfg.get("feature_recursive"))
        if not feature_dir:
            raise ValueError("Provide either --test_dataset_csv or --feature_dir for inference.")
        temp_dir = tempfile.mkdtemp(prefix="tri_mil_infer_")
        dataset_csv_path = create_infer_csv_from_feature_dir(
            feature_dir=feature_dir,
            output_csv=os.path.join(temp_dir, "test.csv"),
            recursive=feature_recursive,
        )
        print(f"[INFO] Generated internal inference CSV: {dataset_csv_path}")
    else:
        feature_dir = args.feature_dir or infer_cfg.get("feature_dir") or common_cfg.get("feature_dir")
        feature_recursive = args.feature_recursive or bool(infer_cfg.get("feature_recursive")) or bool(common_cfg.get("feature_recursive"))

    first_feature_path = resolve_first_feature_path(
        csv_path=dataset_csv_path,
        feature_dir=feature_dir,
        recursive=feature_recursive,
    )
    inferred_in_dim = infer_feature_dim(first_feature_path)

    runtime_device, device_message = select_runtime_device(
        args.device if args.device is not None else infer_cfg.get("device", common_cfg.get("device", getattr(yaml_args.General, "device", "auto")))
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

    runtime_num_classes = args.num_classes if args.num_classes is not None else infer_cfg.get("num_classes")
    if runtime_num_classes is not None:
        config_overrides["General.num_classes"] = runtime_num_classes

    if config_overrides:
        if temp_dir is None:
            temp_dir = tempfile.mkdtemp(prefix="tri_mil_infer_")
        temp_yaml_path = clone_config_with_overrides(
            base_config_path=model_yaml_path,
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
            mode="infer",
        )
    elif model_name == "LONG_MIL":
        test_ds = LONG_MIL_WSI_Dataset(
            dataset_csv_path,
            yaml_args.Dataset.h5_csv_path,
            "test",
            mode="infer",
        )
    else:
        test_ds = WSI_Dataset(dataset_csv_path, "test", mode="infer")

    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False)

    # Model
    device = runtime_device
    model_weight_path = args.model_weight_path or infer_cfg.get("model_weight_path") or common_cfg.get("model_weight_path")
    if not model_weight_path:
        raise ValueError("Model weight path is required for infer_mil.")
    checkpoint = load_torch_checkpoint(model_weight_path, map_location=device)
    if model_name == "DTFD_MIL":
        classifier, attention, dim_reduction, att_cls = get_model_from_yaml(yaml_args)
        model_list = [classifier, attention, dim_reduction, att_cls]
        checkpoint_keys = ["classifier", "attention", "dimReduction", "attCls"]
        for component, key in zip(model_list, checkpoint_keys):
            component.load_state_dict(checkpoint[key])
            component.to(device).eval()
    else:
        model = get_model_from_yaml(yaml_args).to(device)
        model.load_state_dict(checkpoint)
        model.eval()

    out_dir = args.test_log_dir or infer_cfg.get("test_log_dir") or common_cfg.get("test_log_dir")
    if not out_dir:
        raise ValueError("test_log_dir is required for infer_mil.")
    os.makedirs(out_dir, exist_ok=True)

    shutil.copyfile(temp_yaml_path or model_yaml_path, os.path.join(out_dir, "test.yaml"))
    shutil.copyfile(dataset_csv_path, os.path.join(out_dir, "test_dataset.csv"))

    print("Running inference without labels...")

    slide_paths = []
    probs_list = []

    with torch.no_grad():
        for feat, slide_path in test_loader:
            feat = feat.to(device)
            if model_name == "DTFD_MIL":
                logits = dtfd_predict_logits(feat, model_list, yaml_args.Model)
            else:
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--yaml_path", type=str, required=True)
    parser.add_argument("--test_dataset_csv", type=str, default=None)
    parser.add_argument("--feature_dir", type=str, default=None,
                        help="Feature directory for folder-first inference. Internal CSV will be generated automatically.")
    parser.add_argument("--feature_recursive", action="store_true",
                        help="Recursively scan --feature_dir for feature files.")
    parser.add_argument("--model_weight_path", type=str, default=None)
    parser.add_argument("--test_log_dir", type=str, default=None)
    parser.add_argument("--device", type=str, default=None,
                        help="Runtime device override. Examples: auto, cpu, 0, cuda:0.")
    parser.add_argument("--num_classes", type=int, default=None,
                        help="Optional runtime override for General.num_classes.")
    args = parser.parse_args()

    test(args)
