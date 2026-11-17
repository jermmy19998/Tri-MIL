"""Inventory local upstream changes without modifying either upstream tree."""

import hashlib
import json
from pathlib import Path


def inventory(root, folders):
    """Return relative source paths and SHA-256 hashes under selected folders."""
    result = {}
    for folder in folders:
        path = root / folder
        paths = path.rglob("*") if path.is_dir() else [path]
        for source in paths:
            if source.is_file() and "__pycache__" not in source.parts:
                result[source.relative_to(root).as_posix()] = hashlib.sha256(
                    source.read_bytes().replace(b"\r\n", b"\n")
                ).hexdigest()
    return result


def main():
    """Print changed and added upstream source paths for integration review."""
    root = Path(__file__).resolve().parents[1]
    for name, folders in [
        ("MIL_BASELINE-main", ["modules", "process", "configs", "utils", "split_by_case", "tests"]),
        ("TRIDENT-main", ["trident", "run_batch_of_slides.py", "run_single_slide.py"]),
    ]:
        current = inventory(root, folders)
        upstream = inventory(root.parent / name, folders)
        print(
            json.dumps(
                {
                    "upstream": name,
                    "added": sorted(upstream.keys() - current.keys()),
                    "changed": sorted(
                        p for p in upstream.keys() & current.keys() if upstream[p] != current[p]
                    ),
                },
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
