import os
import pandas as pd
import argparse


def generate_test_csv_from_reference_csv(
    reference_csv,
    feature_dir,
    output_csv,
    no_label=False
):
    """
    Generate test CSV.

    If no_label=False:
        - test_slide_path
        - test_label

    If no_label=True:
        - test_slide_path only
    """

    if not os.path.exists(reference_csv):
        print(f"Error: CSV '{reference_csv}' does not exist")
        return

    if not os.path.exists(feature_dir):
        print(f"Error: Feature directory '{feature_dir}' does not exist")
        return

    df = pd.read_csv(reference_csv)

    if "slide_id" not in df.columns:
        raise ValueError("reference CSV must contain column: slide_id")

    if not no_label and "censorship" not in df.columns:
        raise ValueError("reference CSV must contain column: censorship")

    out_data = []

    for _, row in df.iterrows():

        slide_id = row["slide_id"]

        if not isinstance(slide_id, str):
            continue

        slide_base = os.path.splitext(slide_id)[0]
        slide_path = os.path.join(feature_dir, f"{slide_base}.h5")

        if not os.path.exists(slide_path):
            print(f"Warning: feature not found for {slide_base}")
            continue

        if no_label:
            out_data.append({
                "test_slide_path": slide_path
            })
        else:
            censorship = row["censorship"]
            out_data.append({
                "test_slide_path": slide_path,
                "test_label": censorship
            })

    if not out_data:
        print("Warning: no valid rows generated")
        return

    out_df = pd.DataFrame(out_data)
    out_df.to_csv(output_csv, index=False)

    print(f"CSV saved to {output_csv}")
    print(f"Total slides: {len(out_df)}")
    print(out_df.head())


def main():

    parser = argparse.ArgumentParser(
        description="Generate test CSV (support label / no-label mode)"
    )

    parser.add_argument(
        "--reference_csv",
        type=str,
        required=True,
        help="CSV containing slide_id (and optionally censorship)"
    )

    parser.add_argument(
        "--feature_dir",
        type=str,
        required=True,
        help="Folder containing pre-extracted feature files (.h5)"
    )

    parser.add_argument(
        "--output_csv",
        type=str,
        required=True,
        help="Output CSV path"
    )

    parser.add_argument(
        "--no_label",
        action="store_true",
        help="Generate CSV without test_label column"
    )

    args = parser.parse_args()

    print("Generating test CSV...")
    print(f"Reference CSV: {args.reference_csv}")
    print(f"Feature dir  : {args.feature_dir}")
    print(f"Output CSV   : {args.output_csv}")
    print(f"No label mode: {args.no_label}")
    print("-" * 60)

    generate_test_csv_from_reference_csv(
        args.reference_csv,
        args.feature_dir,
        args.output_csv,
        no_label=args.no_label
    )


if __name__ == "__main__":
    main()
