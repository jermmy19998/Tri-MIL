import os
import pandas as pd
import argparse


def generate_slide_labels_csv(reference_csv, feature_dir, output_csv):
    """
    Generate CSV file with feature paths (.pt) and censorship labels.

    Args:
        reference_csv (str): CSV containing slide_id and censorship columns
        feature_dir (str): Directory containing .pt feature files
        output_csv (str): Output CSV path
    """

    if not os.path.exists(reference_csv):
        print(f"Error: Reference CSV '{reference_csv}' does not exist")
        return

    if not os.path.exists(feature_dir):
        print(f"Error: Feature directory '{feature_dir}' does not exist")
        return

    # Load reference CSV
    ref_df = pd.read_csv(reference_csv)

    if "slide_id" not in ref_df.columns:
        raise ValueError("reference CSV must contain column: slide_id")

    if "censorship" not in ref_df.columns:
        raise ValueError("reference CSV must contain column: censorship")

    print(f"Loaded reference CSV: {len(ref_df)} rows")

    slide_data = []

    for _, row in ref_df.iterrows():
        slide_name = row["slide_id"]
        censorship = row["censorship"]

        # Remove .svs if present
        slide_base = os.path.splitext(slide_name)[0]

        # Feature file is .pt
        feature_path = os.path.join(feature_dir, f"{slide_base}.h5")

        if os.path.exists(feature_path):
            slide_data.append({
                "slide_path": feature_path,
                "label": censorship
            })
        else:
            print(f"Warning: Feature not found for {slide_base}")

    if slide_data:
        df = pd.DataFrame(slide_data)
        df.to_csv(output_csv, index=False)
        print(f"CSV saved to {output_csv}")
        print(f"Total valid slides: {len(df)}")
        print(df.head())
    else:
        print("Warning: No matching feature files found")


def main():
    parser = argparse.ArgumentParser(
        description="Generate CSV for .h5 slide features using reference censorship labels"
    )

    parser.add_argument(
        "--reference_csv",
        type=str,
        required=True,
        help="CSV file containing slide_id and censorship columns"
    )

    parser.add_argument(
        "--feature_dir",
        type=str,
        required=True,
        help="Directory containing .h5 feature files"
    )

    parser.add_argument(
        "--output_csv",
        type=str,
        required=True,
        help="Path to save generated CSV"
    )

    args = parser.parse_args()

    print("Generating feature CSV from reference CSV...")
    print(f"Reference CSV: {args.reference_csv}")
    print(f"Feature directory: {args.feature_dir}")
    print(f"Output CSV: {args.output_csv}")
    print("-" * 50)

    generate_slide_labels_csv(
        args.reference_csv,
        args.feature_dir,
        args.output_csv
    )


if __name__ == "__main__":
    main()
