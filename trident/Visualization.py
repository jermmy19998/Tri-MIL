import numpy as np
import pdb
import cv2
import matplotlib.pyplot as plt
from PIL import Image
from typing import Optional, Tuple, Union, Any
import os 
from shapely import Polygon, MultiPolygon


def create_overlay(
    scores: np.ndarray,
    coords: np.ndarray,
    patch_size_level0: int,
    scale: np.ndarray,
    region_size: Tuple[int, int]
) -> np.ndarray:
    """
    Create the heatmap overlay based on scores and coordinates.
    
    Parameters
    ----------
    scores : np.ndarray
        Normalized scores.
    coords : np.ndarray
        Coordinates of patches.
    patch_size_level0 : int
        Patch size at level 0.
    scale : np.ndarray
        Scaling factors.
    region_size : Tuple[int, int]
        Dimensions of the region.
    
    Returns
    -------
    np.ndarray
        Heatmap overlay.
    """
    patch_size = np.ceil(np.array([patch_size_level0, patch_size_level0]) * scale).astype(int)
    coords = np.ceil(coords * scale).astype(int)
    
    overlay = np.zeros(tuple(np.flip(region_size)), dtype=float)
    counter = np.zeros_like(overlay, dtype=np.uint16)
    
    for idx, coord in enumerate(coords):
        overlay[coord[1]:coord[1] + patch_size[1], coord[0]:coord[0] + patch_size[0]] += scores[idx]
        counter[coord[1]:coord[1] + patch_size[1], coord[0]:coord[0] + patch_size[0]] += 1
    
    zero_mask = counter == 0
    overlay[~zero_mask] /= counter[~zero_mask]
    overlay[zero_mask] = np.nan  # Set areas with no data to NaN
    
    return overlay


def apply_colormap(overlay: np.ndarray, cmap_name: str) -> np.ndarray:
    """
    Apply a colormap to the heatmap overlay.
    
    Parameters
    ----------
    overlay : np.ndarray
        Heatmap overlay.
    cmap_name : str
        Colormap name.

    Returns
    -------
    np.ndarray
        Colored overlay image.
    """
    cmap = plt.get_cmap(cmap_name)
    overlay_colored = np.zeros((*overlay.shape, 3), dtype=np.uint8)
    valid_mask = ~np.isnan(overlay)
    colored_valid = (cmap(overlay[valid_mask]) * 255).astype(np.uint8)[:, :3]
    overlay_colored[valid_mask] = colored_valid
    return overlay_colored


def visualize_heatmap(
    wsi: Any,
    scores: np.ndarray,
    coords: np.ndarray,
    patch_size_level0: int,
    vis_level: Optional[int] = 2,
    cmap: str = 'coolwarm',
    normalize: bool = True,
    num_top_patches_to_save: int = -1,
    output_dir: str = "output",
    vis_mag: Optional[int] = None,
    overlay_only: bool = False,
    blur: bool = True,
    filename: str = "heatmap.png",
    topk_dir_name: Optional[str] = "topk_patches"
) -> str:

    if normalize:
        from scipy.stats import rankdata
        scores = rankdata(scores) / len(scores)

    if vis_mag is None:
        downsample = wsi.level_downsamples[vis_level]
    else:
        downsample = wsi.mag / vis_mag
        if not overlay_only:
            vis_level, _ = wsi.get_best_level_and_custom_downsample(downsample)

    scale = np.array([1 / downsample, 1 / downsample])
    region_size = tuple((np.array(wsi.level_dimensions[0]) * scale).astype(int))

    overlay = create_overlay(
        scores, coords, patch_size_level0, scale, region_size
    )

    overlay_colored = apply_colormap(overlay, cmap)

    if blur:
        ksize = int(np.ceil(patch_size_level0 * scale[0]))
        ksize = max(3, ksize // 2 * 2 + 1)
        ksize = (ksize, ksize)

        overlay_colored = cv2.GaussianBlur(overlay_colored, ksize, 0)

    if overlay_only:
        blended = overlay_colored
    else:
        img = wsi.read_region(
            (0, 0),
            vis_level,
            wsi.level_dimensions[vis_level]
        ).convert("RGB")

        img = img.resize(region_size, Image.Resampling.BICUBIC)
        img = np.array(img)

        blended = cv2.addWeighted(img, 0.6, overlay_colored, 0.4, 0)

    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, filename)
    Image.fromarray(blended).save(save_path)

    if num_top_patches_to_save > 0:
        topk_dir = os.path.join(output_dir, topk_dir_name)
        os.makedirs(topk_dir, exist_ok=True)

        topk_idx = np.argsort(scores)[-num_top_patches_to_save:]
        for rank, i in enumerate(topk_idx):
            x, y = coords[i]
            patch = wsi.read_region(
                (int(x), int(y)), 0,
                (patch_size_level0, patch_size_level0)
            )
            patch.save(
                os.path.join(topk_dir, f"top{rank}_score_{scores[i]:.4f}.png")
            )

    return save_path
