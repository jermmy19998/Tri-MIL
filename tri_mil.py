import argparse
import sys


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tri_mil.py",
        description="Simplified folder-first workflows for Tri-MIL inference and heatmap generation.",
    )
    subparsers = parser.add_subparsers(dest="command")
    subparsers.required = True

    infer_parser = subparsers.add_parser(
        "infer",
        help="Run MIL inference from a feature folder or an existing CSV.",
    )
    infer_parser.add_argument("--yaml_path", type=str, required=True, help="MIL YAML config path.")
    infer_parser.add_argument("--model_weight_path", type=str, required=True, help="MIL checkpoint path.")
    infer_parser.add_argument("--test_log_dir", type=str, required=True, help="Output directory.")
    infer_parser.add_argument("--feature_dir", type=str, default=None, help="Feature directory for folder-first inference.")
    infer_parser.add_argument("--test_dataset_csv", type=str, default=None, help="Optional existing dataset CSV.")
    infer_parser.add_argument("--feature_recursive", action="store_true", help="Recursively scan feature_dir.")
    infer_parser.add_argument("--device", type=str, default="auto", help="Runtime device. Examples: auto, cpu, 0, cuda:0.")
    infer_parser.add_argument("--num_classes", type=int, default=None, help="Optional runtime override for class count.")
    infer_parser.add_argument("--no_label", action="store_true", help="Enable unlabeled inference mode.")

    train_parser = subparsers.add_parser(
        "train",
        help="Run MIL training from feature folders with internal CSV/fold generation.",
    )
    train_parser.add_argument("--yaml_path", type=str, required=True, help="MIL YAML config path.")
    train_parser.add_argument("--feature_dir", type=str, required=True, help="Extracted feature directory.")
    train_parser.add_argument("--output_dir", type=str, required=True, help="Working directory for generated CSVs and splits.")
    train_parser.add_argument("--dataset_name", type=str, required=True, help="Dataset name used in logs and generated files.")
    train_parser.add_argument("--reference_csv", type=str, default=None, help="Reference CSV containing slide ids and labels.")
    train_parser.add_argument("--slide_col", type=str, default="slide_id", help="Slide-id column in reference CSV.")
    train_parser.add_argument("--label_col", type=str, default="label", help="Label column in reference CSV.")
    train_parser.add_argument("--source_dir", type=str, default=None, help="Raw-data directory with one label subfolder per class.")
    train_parser.add_argument("--feature_recursive", action="store_true", help="Recursively scan feature_dir.")
    train_parser.add_argument("--source_recursive", action="store_true", help="Recursively scan source_dir label folders.")
    train_parser.add_argument("--k", type=int, default=3, help="Number of folds.")
    train_parser.add_argument("--seed", type=int, default=42, help="Split seed.")
    train_parser.add_argument("--device", type=str, default="auto", help="Runtime device. Examples: auto, cpu, 0, cuda:0.")
    train_parser.add_argument("--epochs", type=int, default=None, help="Optional runtime override for number of epochs.")
    train_parser.add_argument("--num_classes", type=int, default=None, help="Optional runtime override for class count.")
    train_parser.add_argument("--log_dir", type=str, default=None, help="Optional runtime override for Logs.log_root_dir.")

    heatmap_parser = subparsers.add_parser(
        "heatmap",
        help="Generate WSI heatmaps with minimal command-line overrides.",
    )
    heatmap_parser.add_argument("--config", type=str, default="./draw_heatmap/heatmap.yaml", help="Heatmap config path.")
    heatmap_parser.add_argument("--wsi_dir", type=str, required=True, help="WSI directory.")
    heatmap_parser.add_argument("--feature_dir", type=str, required=True, help="Precomputed feature directory.")
    heatmap_parser.add_argument("--coord_dir", type=str, required=True, help="Precomputed coordinate directory.")
    heatmap_parser.add_argument("--model_ckpt", type=str, required=True, help="MIL checkpoint path.")
    heatmap_parser.add_argument("--model_yaml", type=str, required=True, help="MIL YAML config path.")
    heatmap_parser.add_argument("--job_dir", type=str, required=True, help="Output directory.")
    heatmap_parser.add_argument("--device", type=str, default="auto", help="Runtime device. Examples: auto, cpu, 0, cuda:0.")
    heatmap_parser.add_argument("--mpp", type=float, default=None, help="MPP override for png/jpg inputs.")
    heatmap_parser.add_argument("--reader_type", type=str, default=None, help="Reader type override, for example image.")
    heatmap_parser.add_argument("--num_top_patches", type=int, default=None, help="Top-k patch crops to export.")
    heatmap_parser.add_argument("--vis_level", type=int, default=None, help="Heatmap visualization level.")
    heatmap_parser.add_argument("--blur", dest="blur", action="store_true", help="Enable gaussian blur.")
    heatmap_parser.add_argument("--no_blur", dest="blur", action="store_false", help="Disable gaussian blur.")
    heatmap_parser.set_defaults(blur=None)

    return parser


def run_infer(args: argparse.Namespace) -> None:
    if args.feature_dir is None and args.test_dataset_csv is None:
        raise ValueError("Provide either --feature_dir or --test_dataset_csv for tri_mil.py infer.")
    from infer_mil import test as infer_entry

    infer_entry(args)


def run_train(args: argparse.Namespace) -> None:
    if args.reference_csv is None and args.source_dir is None:
        raise ValueError("Provide either --reference_csv or --source_dir for tri_mil.py train.")

    import os
    import tempfile

    from train_mil import main as train_entry
    from utils.runtime_utils import (
        build_kfold_dataset_dir,
        build_train_base_csv_from_reference,
        build_train_base_csv_from_source_dirs,
        clone_config_with_updates,
        collect_feature_files,
        infer_feature_dim,
        select_runtime_device,
        write_label_map_json,
    )

    work_dir = os.path.abspath(args.output_dir)
    os.makedirs(work_dir, exist_ok=True)

    if args.reference_csv is not None:
        base_csv_path, label_map = build_train_base_csv_from_reference(
            reference_csv=args.reference_csv,
            feature_dir=args.feature_dir,
            output_csv=os.path.join(work_dir, "train_base.csv"),
            slide_col=args.slide_col,
            label_col=args.label_col,
            feature_recursive=args.feature_recursive,
        )
    else:
        base_csv_path, label_map = build_train_base_csv_from_source_dirs(
            source_dir=args.source_dir,
            feature_dir=args.feature_dir,
            output_csv=os.path.join(work_dir, "train_base.csv"),
            feature_recursive=args.feature_recursive,
            source_recursive=args.source_recursive,
        )

    dataset_root_dir = build_kfold_dataset_dir(
        base_csv_path=base_csv_path,
        output_dir=os.path.join(work_dir, "datasets"),
        dataset_name=args.dataset_name,
        seed=args.seed,
        k=args.k,
    )
    write_label_map_json(label_map, os.path.join(work_dir, "label_map.json"))

    first_feature_path = collect_feature_files(args.feature_dir, recursive=args.feature_recursive)[0]
    first_feature_dim = infer_feature_dim(first_feature_path)

    runtime_device, _ = select_runtime_device(args.device)
    yaml_device_value = "cpu" if runtime_device.type == "cpu" else (
        runtime_device.index if runtime_device.index is not None else 0
    )

    runtime_num_classes = args.num_classes if args.num_classes is not None else len(label_map)
    log_dir = os.path.abspath(args.log_dir) if args.log_dir else os.path.join(work_dir, "logs")

    temp_dir = tempfile.mkdtemp(prefix="tri_mil_train_")
    index_to_label = {index: label for label, index in label_map.items()}

    def update_runtime_config(config):
        config["Dataset"]["DATASET_NAME"] = args.dataset_name
        config["Dataset"]["dataset_root_dir"] = dataset_root_dir
        config["Dataset"]["dataset_csv_path"] = None
        config["Logs"]["log_root_dir"] = log_dir
        config["General"]["seed"] = args.seed
        config["General"]["device"] = yaml_device_value
        config["General"]["num_classes"] = runtime_num_classes
        config["Model"]["in_dim"] = first_feature_dim
        if args.epochs is not None:
            config["General"]["num_epochs"] = args.epochs
        config["Label"] = {index_to_label[idx]: idx for idx in sorted(index_to_label)}
        return config

    runtime_yaml_path = clone_config_with_updates(
        base_config_path=args.yaml_path,
        output_path=os.path.join(temp_dir, "runtime_train.yaml"),
        updater=update_runtime_config,
    )

    cli_args = argparse.Namespace(yaml_path=runtime_yaml_path, options=None)
    train_entry(cli_args)


def run_heatmap(args: argparse.Namespace) -> None:
    from draw_heatmap.draw_heatmap import main as heatmap_entry

    forwarded = ["draw_heatmap.py", "--config", args.config]
    optional_pairs = [
        ("--wsi_dir", args.wsi_dir),
        ("--feature_dir", args.feature_dir),
        ("--coord_dir", args.coord_dir),
        ("--model_ckpt", args.model_ckpt),
        ("--model_yaml", args.model_yaml),
        ("--job_dir", args.job_dir),
        ("--device", args.device),
        ("--reader_type", args.reader_type),
    ]
    for flag, value in optional_pairs:
        if value is not None:
            forwarded.extend([flag, str(value)])

    if args.mpp is not None:
        forwarded.extend(["--mpp", str(args.mpp)])
    if args.num_top_patches is not None:
        forwarded.extend(["--num_top_patches", str(args.num_top_patches)])
    if args.vis_level is not None:
        forwarded.extend(["--vis_level", str(args.vis_level)])
    if args.blur is True:
        forwarded.append("--blur")
    if args.blur is False:
        forwarded.append("--no_blur")

    previous_argv = sys.argv[:]
    try:
        sys.argv = forwarded
        heatmap_entry()
    finally:
        sys.argv = previous_argv


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "infer":
        run_infer(args)
        return

    if args.command == "train":
        run_train(args)
        return

    if args.command == "heatmap":
        run_heatmap(args)
        return

    parser.error(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
