import os
from types import SimpleNamespace


def collect_wsi_files(wsi_dirs, extensions):
    """
    Collect WSI files from directory / file / list.
    """

    # ---- normalize to list ----
    if isinstance(wsi_dirs, str):
        wsi_dirs = [wsi_dirs]

    if not isinstance(wsi_dirs, (list, tuple)):
        raise TypeError("wsi_list must be a path or list of paths")

    wsi_files = []

    for wsi_dir in wsi_dirs:
        if not os.path.exists(wsi_dir):
            raise ValueError(f"WSI path not found: {wsi_dir}")

        # single file
        if os.path.isfile(wsi_dir):
            if wsi_dir.lower().endswith(tuple(extensions)):
                wsi_files.append(wsi_dir)
            continue

        # directory
        for root, _, files in os.walk(wsi_dir):
            for f in files:
                if f.lower().endswith(tuple(extensions)):
                    wsi_files.append(os.path.join(root, f))

    if len(wsi_files) == 0:
        raise RuntimeError("No WSI files found")

    return sorted(wsi_files)



# -----------------------------
# Dict → Namespace
# -----------------------------
def dict_to_namespace(d):
    if isinstance(d, dict):
        return SimpleNamespace(**{str(k): dict_to_namespace(v) for k, v in d.items()})
    elif isinstance(d, list):
        return [dict_to_namespace(x) for x in d]
    else:
        return d