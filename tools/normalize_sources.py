"""Normalize tracked and untracked project text sources to UTF-8 LF line endings."""

import subprocess
from pathlib import Path


def main():
    """Normalize source line endings without touching ignored artifacts or environments."""
    root = Path(__file__).resolve().parents[1]
    data = subprocess.check_output(
        [
            "git",
            "-c",
            f"safe.directory={root.as_posix()}",
            "ls-files",
            "--cached",
            "--others",
            "--exclude-standard",
            "-z",
        ],
        cwd=root,
    )
    suffixes = {".py", ".md", ".yaml", ".yml", ".toml", ".json", ".txt", ".sh"}
    total = 0
    for item in data.split(b"\0"):
        if not item:
            continue
        path = root / item.decode("utf-8")
        if path.is_file() and (
            path.suffix in suffixes or path.name in {".gitignore", ".gitattributes"}
        ):
            path.write_text(path.read_text(encoding="utf-8-sig"), encoding="utf-8", newline="\n")
            total += 1
    print(f"Normalized {total} text sources to UTF-8 LF.")


if __name__ == "__main__":
    main()
