"""Check function docstring coverage and add missing API summaries on request."""

import argparse
import ast
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXCLUDED = {".venv", ".git", "__pycache__", "build", "dist"}
SPECIAL = {
    "__init__": "Initialize the {owner} instance and configure its dependencies.",
    "__len__": "Return the number of entries exposed by this {owner} instance.",
    "__getitem__": "Return the {owner} entry at the requested index.",
    "__iter__": "Yield entries from this {owner} instance in iteration order.",
    "forward": "Compute the {owner} forward pass for the supplied input tensors.",
    "backward": "Compute gradients for the {owner} operation.",
    "setUp": "Create fixtures for the {owner} test case.",
    "tearDown": "Release fixtures created by the {owner} test case.",
    "main": "Run the command-line workflow defined by this module.",
}


def summary(node, owner):
    """Build a function-specific summary from its API name and owning class."""
    if node.name in SPECIAL:
        return SPECIAL[node.name].format(owner=owner)
    words = re.sub(r"(?<!^)(?=[A-Z][a-z])", " ", node.name.strip("_")).replace("_", " ")
    if words.startswith("test "):
        return "Verify that " + words[5:] + " behaves as expected."
    return words[:1].upper() + words[1:] + "."


def missing_docs(tree, owner="module"):
    """Yield undocumented functions with their nearest owning class name."""
    for node in ast.iter_child_nodes(tree):
        if (
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and ast.get_docstring(node) is None
        ):
            yield node, owner
        yield from missing_docs(node, node.name if isinstance(node, ast.ClassDef) else owner)


def main():
    """Check all project functions, optionally inserting missing summaries."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true")
    options = parser.parse_args()
    total = changed = 0
    for path in ROOT.rglob("*.py"):
        if any(part in EXCLUDED for part in path.relative_to(ROOT).parts):
            continue
        source = path.read_text(encoding="utf-8-sig")
        tree = ast.parse(source, filename=str(path))
        missing = list(missing_docs(tree))
        total += len(missing)
        if not missing:
            continue
        if not options.write:
            print(f"{path.relative_to(ROOT)}: {len(missing)} undocumented functions")
            continue
        lines = source.splitlines(keepends=True)
        for node, owner in sorted(missing, key=lambda item: item[0].body[0].lineno, reverse=True):
            statement = node.body[0]
            index = statement.lineno - 1
            if statement.lineno == node.lineno:
                original = lines[index]
                prefix, body = original[: statement.col_offset], original[statement.col_offset :]
                indentation = " " * (node.col_offset + 4)
                lines[index] = (
                    prefix.rstrip()
                    + "\n"
                    + indentation
                    + '"""'
                    + summary(node, owner)
                    + '"""\n'
                    + indentation
                    + body
                )
            else:
                indentation = lines[index][: len(lines[index]) - len(lines[index].lstrip())]
                lines.insert(index, indentation + '"""' + summary(node, owner) + '"""\n')
        updated = "".join(lines)
        ast.parse(updated, filename=str(path))
        path.write_text(updated, encoding="utf-8")
        changed += 1
    print(f"Missing function docstrings: {total}; files updated: {changed}.")
    if total and not options.write:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
