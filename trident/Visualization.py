import numpy as np
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
    
    Parameters:
        scores (np.ndarray):
            Normalized scores.
        coords (np.ndarray):
            Coordinates of patches.
        patch_size_level0 (int):
            Patch size at level 0.
        scale (np.ndarray):
            Scaling factors.
        region_size (Tuple[int, int]):
            Dimensions of the region.

    Returns:
        np.ndarray: Heatmap overlay.
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
    
    Parameters:
        overlay (np.ndarray):
            Heatmap overlay.
        cmap_name (str):
            Colormap name.

    Returns:
        np.ndarray: Colored overlay image.
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
    output_dir: Optional[str] = "output",
    vis_mag: Optional[int] = None,
    overlay_only: bool = False,
    blur: bool = False,
    topk_dir_name: str = "topk_patches",
    filename: str = 'heatmap.png'
) -> str:
    """
    Generate a heatmap visualization overlayed on a whole slide image (WSI).
    
    Parameters:
        wsi (WSI):
            Whole slide image object.
        scores (np.ndarray):
            Scores associated with each coordinate.
        coords (np.ndarray):
            Coordinates of patches at level 0.
        patch_size_level0 (int):
            Patch size at level 0.
        vis_level (Optional[int]):
            Visualization level.
        cmap (str):
            Colormap to use for the heatmap.
        normalize (bool):
            Whether to normalize the scores.
        num_top_patches_to_save (int):
            Number of high-score patches to save. If set to -1, do not save any. Defaults to -1.
        output_dir (Optional[str]):
            Directory to save heatmap and top-k patches.
        vis_mag (Optional[int]):
            Visualization magnification. This overwrites `vis_level`.
        overlay_only (bool):
            Whether to save the overlay only. If True, saves the overlay on top of a downscaled version of the WSI.
            Defaults to False.
        blur (bool):
            Whether to apply gaussian blur to the heatmap overlay for smoother visualization.
            Defaults to False.
        topk_dir_name (str):
            Directory name used to save top-scoring patches under `output_dir`.
            Defaults to "topk_patches".
        filename (str):
            File will be saved in `output_dir`/`filename`.

    Returns:
        str: Path to the saved heatmap image.
    """

    if normalize:
        from scipy.stats import rankdata
        scores = rankdata(scores, 'average') / len(scores) * 100 / 100
    
    if vis_mag is None:
        downsample = wsi.level_downsamples[vis_level]
    else:
        src_mag = wsi.mag
        downsample = src_mag / vis_mag
        if not overlay_only:
            vis_level, _ = wsi.get_best_level_and_custom_downsample(downsample)
    
    scale = np.array([1 / downsample, 1 / downsample])
    region_size = tuple((np.array(wsi.level_dimensions[0]) * scale).astype(int))
    overlay = create_overlay(scores, coords, patch_size_level0, scale, region_size)

    if blur:
        nan_mask = np.isnan(overlay)
        patch_size_scaled = np.ceil(np.array([patch_size_level0, patch_size_level0]) * scale).astype(int)
        kernel = np.maximum(3, (patch_size_scaled * 2).astype(int))
        kernel = kernel + (kernel + 1) % 2  # ensure odd kernel sizes
        overlay_for_blur = np.nan_to_num(overlay, nan=0.0).astype(np.float32)
        overlay = cv2.GaussianBlur(overlay_for_blur, tuple(kernel.tolist()), 0)
        overlay = cv2.GaussianBlur(overlay, tuple(kernel.tolist()), 0)
        overlay[nan_mask] = np.nan

    overlay_colored = apply_colormap(overlay, cmap)
    
    if overlay_only:
        blended_img = overlay_colored
    else:
        img = wsi.read_region((0, 0), vis_level, wsi.level_dimensions[vis_level]).convert("RGB")
        img = img.resize(region_size, resample=Image.Resampling.BICUBIC)
        img = np.array(img)
        
        blended_img = cv2.addWeighted(img, 0.6, overlay_colored, 0.4, 0)
    
    blended_img = Image.fromarray(blended_img)

    os.makedirs(output_dir, exist_ok=True)
    heatmap_path = os.path.join(output_dir, filename)
    blended_img.save(heatmap_path)

    if num_top_patches_to_save > 0:
        topk_dir = os.path.join(output_dir, topk_dir_name)
        os.makedirs(topk_dir, exist_ok=True)
        topk_indices = np.argsort(scores)[-num_top_patches_to_save:]
        for idx, i in enumerate(topk_indices):
            x, y = coords[i]
            patch = wsi.read_region((x, y), 0, (patch_size_level0, patch_size_level0))
            patch.save(os.path.join(topk_dir, f"top_{idx}_score_{scores[i]:.4f}.png"))

    return heatmap_path
