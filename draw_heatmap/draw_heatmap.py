import os
import pdb
import sys
import csv
import argparse
import torch
import h5py
import tempfile
from tqdm import tqdm

# -----------------------------
# Project path
# -----------------------------
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
sys.path.insert(0, CURRENT_DIR)
sys.path.insert(0, PROJECT_ROOT)

# -----------------------------
# Trident
# -----------------------------
from trident import load_wsi, visualize_heatmap
from trident.segmentation_models import segmentation_model_factory
from trident.patch_encoder_models import encoder_factory as patch_encoder_factory

# -----------------------------
# MIL utils
# -----------------------------
from IO import dict_to_namespace
from utils.yaml_utils import read_yaml
from utils.model_utils import get_model_from_yaml
from utils.runtime_utils import (
    clone_config_with_overrides,
    infer_encoder_name_from_feature_dir,
    infer_feature_dim,
    load_torch_checkpoint,
    select_runtime_device,
)


# -----------------------------
# Args
# -----------------------------
def parse_args():
    parser = argparse.ArgumentParser("WSI Heatmap + MIL (Full Pipeline)")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--wsi_dir", type=str, default=None, help="Override input WSI directory.")
    parser.add_argument("--feature_dir", type=str, default=None, help="Precomputed feature directory.")
    parser.add_argument("--coord_dir", type=str, default=None, help="Precomputed coordinate directory.")
    parser.add_argument("--model_ckpt", type=str, default=None, help="MIL checkpoint path override.")
    parser.add_argument("--model_yaml", type=str, default=None, help="MIL YAML path override.")
    parser.add_argument("--job_dir", type=str, default=None, help="Output directory override.")
    parser.add_argument("--device", type=str, default=None, help="Runtime device override. Examples: auto, cpu, 0, cuda:0.")
    parser.add_argument("--mpp", type=float, default=None, help="Override WSI mpp, useful for png/jpg inputs.")
    parser.add_argument("--reader_type", type=str, default=None, help="Override WSI reader type. Example: image.")
    parser.add_argument("--num_top_patches", type=int, default=None, help="Override number of top patches to save.")
    parser.add_argument("--vis_level", type=int, default=None, help="Override visualization level.")
    parser.add_argument("--blur", dest="blur", action="store_true", help="Enable gaussian blur on the heatmap overlay.")
    parser.add_argument("--no_blur", dest="blur", action="store_false", help="Disable gaussian blur on the heatmap overlay.")
    parser.set_defaults(blur=None)
    return parser.parse_args()


# -----------------------------
# Utils
# -----------------------------
def collect_wsi_files(wsi_dir, extensions):
    paths = []
    for root, _, files in os.walk(wsi_dir):
        for name in files:
            if any(name.lower().endswith(ext) for ext in extensions):
                paths.append(os.path.join(root, name))
    return sorted(paths)


def is_image_input(path):
    return os.path.splitext(path)[1].lower() in {".png", ".jpg", ".jpeg"}


def sanitize_path_token(value):
    token = str(value).replace("\\", "__").replace("/", "__").replace(" ", "_")
    token = token.replace(":", "")
    return token.strip("._") or "root"


def build_slide_output_prefix(wsi_path, wsi_root):
    parent_dir = os.path.dirname(wsi_path)
    rel_parent = os.path.relpath(parent_dir, wsi_root)
    if rel_parent in (".", ""):
        return ""
    return sanitize_path_token(rel_parent)


def find_h5_in_dir(h5_dir, slide_id, require_patch):
    matches = []
    for name in os.listdir(h5_dir):
        if not name.endswith(".h5"):
            continue
        if not name.startswith(slide_id):
            continue

        has_patch = "patch" in name.lower()
        if require_patch and not has_patch:
            continue
        if not require_patch and has_patch:
            continue

        matches.append(os.path.join(h5_dir, name))

    if len(matches) == 0:
        rule = "with patch" if require_patch else "without patch"
        raise FileNotFoundError(
            f"No .h5 {rule} file found for slide {slide_id} in {h5_dir}"
        )

    if len(matches) > 1:
        raise RuntimeError(
            f"Multiple .h5 files found for slide {slide_id}: {matches}"
        )

    return matches[0]


def try_find_local_h5(slide_job_dir, encoder_name):
    """
    Find existing h5 files in:
      - features_{encoder_name}/*.h5
      - patches/*.h5
    """
    feat_dir = os.path.join(slide_job_dir, f"features_{encoder_name}")
    coord_dir = os.path.join(slide_job_dir, "patches")

    if not os.path.isdir(feat_dir) or not os.path.isdir(coord_dir):
        return None, None

    feat_h5 = [
        os.path.join(feat_dir, f)
        for f in os.listdir(feat_dir)
        if f.endswith(".h5")
    ]
    coord_h5 = [
        os.path.join(coord_dir, f)
        for f in os.listdir(coord_dir)
        if f.endswith(".h5")
    ]

    if len(feat_h5) == 1 and len(coord_h5) == 1:
        return feat_h5[0], coord_h5[0]

    return None, None


# -----------------------------
# Main
# -----------------------------
def main():
    args = parse_args()
    cfg_path = args.config
    cfg = read_yaml(cfg_path)

    runtime_overrides = {}
    if args.wsi_dir is not None:
        runtime_overrides["wsi.wsi_dir"] = args.wsi_dir
    if args.feature_dir is not None:
        runtime_overrides["precomputed.feature_path"] = args.feature_dir
        runtime_overrides["patch_encoder.model_name"] = infer_encoder_name_from_feature_dir(args.feature_dir)
    if args.coord_dir is not None:
        runtime_overrides["precomputed.coord_path"] = args.coord_dir
    if args.model_ckpt is not None:
        runtime_overrides["model.ckpt_path"] = args.model_ckpt
    if args.model_yaml is not None:
        runtime_overrides["model.yaml_path"] = args.model_yaml
    if args.job_dir is not None:
        runtime_overrides["job_dir"] = args.job_dir
    if args.mpp is not None:
        runtime_overrides["wsi.mpp"] = args.mpp
    if args.reader_type is not None:
        runtime_overrides["wsi.reader_type"] = args.reader_type
    if args.num_top_patches is not None:
        runtime_overrides["visualization.num_top_patches_to_save"] = args.num_top_patches
    if args.vis_level is not None:
        runtime_overrides["visualization.vis_level"] = args.vis_level
    if args.blur is not None:
        runtime_overrides["visualization.blur"] = args.blur

    runtime_device, device_message = select_runtime_device(
        args.device if args.device is not None else cfg.get("device", "auto")
    )
    print(f"[INFO] {device_message}")
    runtime_overrides["device"] = str(runtime_device)

    temp_cfg_dir = None
    if runtime_overrides:
        temp_cfg_dir = tempfile.mkdtemp(prefix="tri_mil_heatmap_")
        cfg_path = clone_config_with_overrides(
            base_config_path=args.config,
            output_path=os.path.join(temp_cfg_dir, "runtime_heatmap.yaml"),
            overrides=runtime_overrides,
        )
        print(f"[INFO] Generated runtime heatmap config: {cfg_path}")
        cfg = read_yaml(cfg_path)

    device = str(runtime_device)
    job_dir = cfg["job_dir"]
    os.makedirs(job_dir, exist_ok=True)

    # -------------------------------------------------
    # Collect WSIs
    # -------------------------------------------------
    wsi_cfg = cfg["wsi"]
    wsi_reader_type = wsi_cfg.get("reader_type", None)
    wsi_mpp = wsi_cfg.get("mpp", None)
    wsi_custom_mpp_keys = wsi_cfg.get("custom_mpp_keys", None)
    wsi_paths = collect_wsi_files(
        wsi_cfg["wsi_dir"],
        wsi_cfg.get("extensions", [".svs", ".tif", ".tiff"])
    )

    if len(wsi_paths) == 0:
        raise RuntimeError("No WSI files found")

    print(f"[INFO] FOUND {len(wsi_paths)} WSIs")

    # -------------------------------------------------
    # Global precomputed
    # -------------------------------------------------
    pre_cfg = cfg.get("precomputed", {})
    feature_dir = pre_cfg.get("feature_path", "")
    coord_dir = pre_cfg.get("coord_path", "")

    use_global_precomputed = (
        feature_dir
        and coord_dir
        and os.path.isdir(feature_dir)
        and os.path.isdir(coord_dir)
    )

    if not use_global_precomputed:
        segmentation_model = segmentation_model_factory(
            cfg["segmentation"]["model_name"]
        )
        patch_encoder = (
            patch_encoder_factory(cfg["patch_encoder"]["model_name"])
            .eval()
            .to(device)
        )

    # -------------------------------------------------
    # Load MIL model
    # -------------------------------------------------
    model_cfg = cfg["model"]
    mil_yaml = read_yaml(model_cfg["yaml_path"])

    if "in_dim" in mil_yaml.get("Model", {}) and use_global_precomputed:
        feature_example_path = find_h5_in_dir(
            feature_dir,
            os.path.splitext(os.path.basename(wsi_paths[0]))[0],
            False,
        )
        inferred_in_dim = infer_feature_dim(feature_example_path)
        if int(mil_yaml["Model"]["in_dim"]) != inferred_in_dim:
            print(
                f"[INFO] Overriding Model.in_dim from {mil_yaml['Model']['in_dim']} to {inferred_in_dim} "
                f"based on precomputed features."
            )
            mil_yaml["Model"]["in_dim"] = inferred_in_dim

    mil_args = dict_to_namespace(mil_yaml)
    model_name = mil_args.General.MODEL_NAME
    label_map = {k: v for k, v in mil_yaml.get("Label", {}).items()}

    print(f"[INFO] MODEL: {model_name}")
    print(f"[INFO] LABEL MAP: {label_map}")

    if model_name == "DTFD_MIL":
        classifier, attention, dimReduction, attCls = get_model_from_yaml(mil_args)
        state = load_torch_checkpoint(model_cfg["ckpt_path"], map_location=device)

        classifier.load_state_dict(state["classifier"])
        attention.load_state_dict(state["attention"])
        dimReduction.load_state_dict(state["dimReduction"])
        attCls.load_state_dict(state["attCls"])

        mil_model = tuple(
            m.to(device).eval()
            for m in [classifier, attention, dimReduction, attCls]
        )

        attention_kwargs = {
            "total_instance": mil_args.Model.total_instance,
            "num_Group": mil_args.Model.num_Group,
            "grad_clipping": mil_args.Model.grad_clipping,
            "distill": mil_args.Model.distill,
        }
    else:
        mil_model = get_model_from_yaml(mil_args)
        mil_model.load_state_dict(
            load_torch_checkpoint(model_cfg["ckpt_path"], map_location=device)
        )
        mil_model = mil_model.to(device).eval()
        attention_kwargs = {}

    print("[INFO] MIL MODEL READY")

    # -------------------------------------------------
    # CSV
    # -------------------------------------------------
    csv_path = os.path.join(job_dir, "predictions.csv")
    csv_rows = []

    encoder_name = cfg["patch_encoder"]["model_name"]

    # -------------------------------------------------
    # WSI loop
    # -------------------------------------------------
    wsi_root = os.path.abspath(wsi_cfg["wsi_dir"])
    for wsi_path in tqdm(wsi_paths, desc="Processing WSIs"):
        slide_id = os.path.splitext(os.path.basename(wsi_path))[0]
        slide_prefix = build_slide_output_prefix(wsi_path, wsi_root)
        output_slide_id = f"{slide_prefix}__{slide_id}" if slide_prefix else slide_id
        slide_job_dir = os.path.join(job_dir, output_slide_id)
        os.makedirs(slide_job_dir, exist_ok=True)

        # -------------------------------------------------
        # Feature / coord decision
        # -------------------------------------------------
        if use_global_precomputed:
            feature_path = find_h5_in_dir(feature_dir, slide_id, False)
            coord_path = find_h5_in_dir(coord_dir, slide_id, True)
            print(f"[INFO] {slide_id}: USING GLOBAL PRECOMPUTED")
        else:
            feature_path, coord_path = try_find_local_h5(
                slide_job_dir, encoder_name
            )

            if feature_path and coord_path:
                print(f"[INFO] {slide_id}: REUSE LOCAL FEATURES")
            else:
                print(f"[INFO] {slide_id}: RUN FULL PIPELINE")

                slide = load_wsi(
                    slide_path=wsi_path,
                    lazy_init=False,
                    reader_type=wsi_reader_type,
                    mpp=wsi_mpp,
                    default_mpp=wsi_mpp,
                    custom_mpp_keys=wsi_custom_mpp_keys,
                )

                slide.segment_tissue(
                    segmentation_model=segmentation_model,
                    job_dir=slide_job_dir,
                    device=device
                )

                coord_cfg = cfg["patch_coords"]
                coord_path = slide.extract_tissue_coords(
                    target_mag=coord_cfg["target_mag"],
                    patch_size=coord_cfg["patch_size"],
                    overlap=coord_cfg["overlap"],
                    save_coords=slide_job_dir
                )

                feature_path = slide.extract_patch_features(
                    patch_encoder=patch_encoder,
                    coords_path=coord_path,
                    save_features=os.path.join(
                        slide_job_dir, f"features_{encoder_name}"
                    ),
                    device=device
                )

        # -------------------------------------------------
        # Load H5
        # -------------------------------------------------
        with h5py.File(feature_path, "r") as f:
            patch_features = f["features"][:]

        with h5py.File(coord_path, "r") as f:
            coords = f["coords"][:]
            coords_attrs = dict(f["coords"].attrs)

        # -------------------------------------------------
        # WSI init (only for heatmap / top-patch visualization)
        # -------------------------------------------------
        slide_mpp = wsi_mpp
        if slide_mpp is None and is_image_input(wsi_path):
            slide_mpp = cfg.get("mpp", None)

        try:
            slide = load_wsi(
                slide_path=wsi_path,
                lazy_init=False,
                reader_type=wsi_reader_type,
                mpp=slide_mpp,
                default_mpp=slide_mpp,
                custom_mpp_keys=wsi_custom_mpp_keys,
            )
        except Exception as exc:
            if is_image_input(wsi_path) and slide_mpp is None:
                raise RuntimeError(
                    "PNG/JPG heatmap visualization requires `mpp` to be set in the heatmap config, "
                    "for example under `wsi.mpp: 0.5`."
                ) from exc
            raise

        # -------------------------------------------------
        # MIL inference
        # -------------------------------------------------
        with torch.no_grad():
            x = torch.from_numpy(patch_features).float().to(device).unsqueeze(0)

            if isinstance(mil_model, tuple):
                _, attention_model, _, _ = mil_model
                output = attention_model(x, **attention_kwargs)
            else:
                output = mil_model(x, return_WSI_attn=True)

            logits = output["logits"].squeeze(0)
            attn = output["WSI_attn"].squeeze(0)

        probs = torch.softmax(logits, dim=0)
        pred_idx = int(torch.argmax(probs))
        pred_prob = float(probs[pred_idx])
        pred_label = label_map.get(pred_idx, f"Class_{pred_idx}")

        print(f"[PRED] {output_slide_id} -> {pred_label} ({pred_prob:.4f})")

        csv_rows.append([
            output_slide_id,
            wsi_path,
            slide_prefix,
            pred_idx,
            pred_label,
            pred_prob,
            len(coords),
            logits.cpu().numpy().tolist()
        ])

        # -------------------------------------------------
        # Heatmap
        # -------------------------------------------------
        heatmap_filename = (
            f"{output_slide_id}_heatmap_{encoder_name}_{model_name}_{pred_label}.png"
        )
        topk_dir_name = (
            f"{output_slide_id}_topk_{encoder_name}_{model_name}"
        )
        
        if len(attn.shape) == 2:
            attn = attn.squeeze()
            
        vis_cfg = cfg["visualization"]
        print("[HEATMAP] Generating Heatmap ......")
        visualize_heatmap(
            wsi=slide,
            scores=attn.cpu().numpy(),
            coords=coords,
            vis_level=vis_cfg["vis_level"],
            patch_size_level0=coords_attrs["patch_size_level0"],
            normalize=vis_cfg["normalize"],
            num_top_patches_to_save=vis_cfg["num_top_patches_to_save"],
            output_dir=slide_job_dir,
            blur=vis_cfg["blur"],
            filename=heatmap_filename,
            topk_dir_name=topk_dir_name
        )
        print("[HEATMAP]  Heatmap Generate Success......")

    # -------------------------------------------------
    # Write CSV
    # -------------------------------------------------
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "slide_id",
            "slide_path",
            "slide_prefix",
            "pred_idx",
            "pred_label",
            "pred_prob",
            "num_patches",
            "logits"
        ])
        writer.writerows(csv_rows)

    print("[INFO] ALL DONE")


if __name__ == "__main__":
    main()
