import argparse
from pathlib import Path

import pandas as pd


SUPPORTED_FEATURE_SUFFIXES = (".h5", ".pt")
SUPPORTED_SOURCE_SUFFIXES = (
    ".svs",
    ".tif",
    ".tiff",
    ".ndpi",
    ".mrxs",
    ".scn",
    ".vms",
    ".vmu",
    ".bif",
    ".svslide",
    ".png",
    ".jpg",
    ".jpeg",
    ".bmp",
    ".czi",
)


def _normalize_stem(name: str) -> str:
    return Path(name).stem


def _ensure_exists(path: str, path_name: str) -> Path:
    resolved = Path(path)
    if not resolved.exists():
        raise FileNotFoundError(f"{path_name} does not exist: {path}")
    return resolved


def _iter_files(directory: Path, recursive: bool, suffixes: tuple[str, ...]) -> list[Path]:
    iterator = directory.rglob("*") if recursive else directory.iterdir()
    files = []
    for path in iterator:
        if path.is_file() and path.suffix.lower() in suffixes:
            files.append(path.resolve())
    return sorted(files)


def _find_feature_files(feature_dir: str, recursive: bool = False) -> list[str]:
    feature_root = _ensure_exists(feature_dir, "Feature directory")
    return [str(path) for path in _iter_files(feature_root, recursive, SUPPORTED_FEATURE_SUFFIXES)]


def _build_feature_index(feature_dir: str, recursive: bool = False) -> dict[str, str]:
    feature_files = _find_feature_files(feature_dir, recursive=recursive)
    feature_map = {}
    duplicate_stems = []

    for feature_path in feature_files:
        stem = _normalize_stem(Path(feature_path).name)
        if stem in feature_map:
            duplicate_stems.append(stem)
            continue
        feature_map[stem] = feature_path

    if duplicate_stems:
        duplicates = ", ".join(sorted(set(duplicate_stems))[:10])
        raise ValueError(
            "Duplicate feature file stems were found. Please ensure feature filenames are unique. "
            f"Examples: {duplicates}"
        )

    return feature_map


def _read_reference_csv(reference_csv: str, slide_col: str, label_col: str) -> pd.DataFrame:
    reference_path = _ensure_exists(reference_csv, "Reference CSV")
    df = pd.read_csv(reference_path)

    if slide_col not in df.columns:
        raise ValueError(f"Reference CSV must contain slide column: {slide_col}")
    if label_col not in df.columns:
        raise ValueError(f"Reference CSV must contain label column: {label_col}")

    return df


def _write_dataframe(out_df: pd.DataFrame, output_csv: str) -> None:
    output_path = Path(output_csv)
    if output_path.parent != Path("."):
        output_path.parent.mkdir(parents=True, exist_ok=True)

    out_df.to_csv(output_path, index=False)
    print(f"CSV saved to {output_path}")
    print(f"Total rows: {len(out_df)}")
    print(out_df.head())


def build_labeled_csv_from_reference_csv(
    reference_csv: str,
    feature_dir: str,
    output_csv: str,
    slide_col: str,
    label_col: str,
    feature_recursive: bool = False,
) -> None:
    feature_map = _build_feature_index(feature_dir, recursive=feature_recursive)
    ref_df = _read_reference_csv(reference_csv, slide_col, label_col)

    rows = []
    for _, row in ref_df.iterrows():
        slide_name = row[slide_col]
        if not isinstance(slide_name, str):
            continue

        slide_id = _normalize_stem(slide_name)
        feature_path = feature_map.get(slide_id)
        if feature_path is None:
            print(f"Warning: feature not found for {slide_id}")
            continue

        rows.append(
            {
                "slide_path": feature_path,
                "label": row[label_col],
            }
        )

    if not rows:
        raise ValueError("No labeled rows were generated. Check feature_dir and reference_csv alignment.")

    _write_dataframe(pd.DataFrame(rows), output_csv)


def build_labeled_csv_from_feature_label_dirs(
    feature_dir: str,
    output_csv: str,
) -> None:
    feature_root = _ensure_exists(feature_dir, "Feature directory")
    label_dirs = [path for path in sorted(feature_root.iterdir()) if path.is_dir()]
    if not label_dirs:
        raise ValueError(
            "No label subdirectories were found. Expected a structure like "
            "`feature_dir/class_a/*.h5`, `feature_dir/class_b/*.h5`."
        )

    rows = []
    label_to_index = {label_dir.name: idx for idx, label_dir in enumerate(label_dirs)}
    print("Detected labels:")
    for label_name, label_idx in label_to_index.items():
        print(f"  {label_name} -> {label_idx}")

    for label_dir in label_dirs:
        label_idx = label_to_index[label_dir.name]
        feature_files = _iter_files(label_dir, recursive=True, suffixes=SUPPORTED_FEATURE_SUFFIXES)
        for feature_path in feature_files:
            rows.append(
                {
                    "slide_path": str(feature_path),
                    "label": label_idx,
                }
            )

    if not rows:
        raise ValueError("No feature files were found under label subdirectories.")

    _write_dataframe(pd.DataFrame(rows), output_csv)


def build_labeled_csv_from_source_label_dirs(
    source_dir: str,
    feature_dir: str,
    output_csv: str,
    feature_recursive: bool = False,
    source_recursive: bool = True,
) -> None:
    source_root = _ensure_exists(source_dir, "Source directory")
    feature_map = _build_feature_index(feature_dir, recursive=feature_recursive)

    label_dirs = [path for path in sorted(source_root.iterdir()) if path.is_dir()]
    if not label_dirs:
        raise ValueError(
            "No label subdirectories were found under source_dir. Expected a structure like "
            "`source_dir/class_a/*.svs`, `source_dir/class_b/*.png`."
        )

    rows = []
    label_to_index = {label_dir.name: idx for idx, label_dir in enumerate(label_dirs)}
    print("Detected labels:")
    for label_name, label_idx in label_to_index.items():
        print(f"  {label_name} -> {label_idx}")

    for label_dir in label_dirs:
        label_idx = label_to_index[label_dir.name]
        source_files = _iter_files(label_dir, recursive=source_recursive, suffixes=SUPPORTED_SOURCE_SUFFIXES)

        for source_path in source_files:
            slide_id = _normalize_stem(source_path.name)
            feature_path = feature_map.get(slide_id)
            if feature_path is None:
                print(f"Warning: feature not found for {slide_id}")
                continue

            rows.append(
                {
                    "slide_path": feature_path,
                    "label": label_idx,
                }
            )

    if not rows:
        raise ValueError(
            "No labeled rows were generated from source label directories. "
            "Check source_dir / feature_dir alignment."
        )

    _write_dataframe(pd.DataFrame(rows), output_csv)


def build_infer_csv_from_feature_dir(
    feature_dir: str,
    output_csv: str,
    recursive: bool = False,
) -> None:
    feature_files = _find_feature_files(feature_dir, recursive=recursive)
    if not feature_files:
        raise ValueError(f"No feature files (.h5 or .pt) found in {feature_dir}")

    _write_dataframe(pd.DataFrame({"test_slide_path": feature_files}), output_csv)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare dataset CSVs for Tri-MIL from a flat feature directory + reference CSV, "
            "label-organized directories, or a plain feature directory for inference."
        )
    )
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["train_flat", "train_label_dirs", "infer"],
        help=(
            "train_flat: flat feature folder + reference CSV with labels; "
            "train_label_dirs: labels inferred from subdirectories; "
            "infer: plain feature folder to build unlabeled inference CSV."
        ),
    )
    parser.add_argument(
        "--feature_dir",
        type=str,
        required=True,
        help=(
            "Feature directory. For train_label_dirs, this can either be the labeled feature root "
            "or the flat extracted feature directory to be matched with --source_dir."
        ),
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        required=True,
        help="Output CSV path.",
    )
    parser.add_argument(
        "--reference_csv",
        type=str,
        default=None,
        help="Reference CSV path, required when --mode train_flat.",
    )
    parser.add_argument(
        "--slide_col",
        type=str,
        default="slide_id",
        help="Slide-id column name in reference CSV for --mode train_flat.",
    )
    parser.add_argument(
        "--label_col",
        type=str,
        default="label",
        help="Label column name in reference CSV for --mode train_flat.",
    )
    parser.add_argument(
        "--source_dir",
        type=str,
        default=None,
        help=(
            "Optional raw-data directory with one label subfolder per class. "
            "Use this with --mode train_label_dirs when extracted features are stored flat."
        ),
    )
    parser.add_argument(
        "--feature_recursive",
        action="store_true",
        help="Recursively scan feature_dir for .h5/.pt files.",
    )
    parser.add_argument(
        "--source_recursive",
        action="store_true",
        help="Recursively scan inside each source label folder when using --source_dir.",
    )

    args = parser.parse_args()

    print("Preparing dataset CSV...")
    print(f"Mode             : {args.mode}")
    print(f"Feature dir      : {args.feature_dir}")
    print(f"Output CSV       : {args.output_csv}")
    print(f"Reference CSV    : {args.reference_csv}")
    print(f"Slide column     : {args.slide_col}")
    print(f"Label column     : {args.label_col}")
    print(f"Source dir       : {args.source_dir}")
    print(f"Feature recursive: {args.feature_recursive}")
    print(f"Source recursive : {args.source_recursive}")
    print("-" * 60)

    if args.mode == "train_flat":
        if args.reference_csv is None:
            raise ValueError("--reference_csv is required when --mode train_flat")
        build_labeled_csv_from_reference_csv(
            reference_csv=args.reference_csv,
            feature_dir=args.feature_dir,
            output_csv=args.output_csv,
            slide_col=args.slide_col,
            label_col=args.label_col,
            feature_recursive=args.feature_recursive,
        )
        return

    if args.mode == "train_label_dirs":
        if args.source_dir is not None:
            build_labeled_csv_from_source_label_dirs(
                source_dir=args.source_dir,
                feature_dir=args.feature_dir,
                output_csv=args.output_csv,
                feature_recursive=args.feature_recursive,
                source_recursive=args.source_recursive,
            )
        else:
            build_labeled_csv_from_feature_label_dirs(
                feature_dir=args.feature_dir,
                output_csv=args.output_csv,
            )
        return

    build_infer_csv_from_feature_dir(
        feature_dir=args.feature_dir,
        output_csv=args.output_csv,
        recursive=args.feature_recursive,
    )


if __name__ == "__main__":
    main()
