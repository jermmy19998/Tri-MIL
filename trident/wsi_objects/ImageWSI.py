from __future__ import annotations
import numpy as np
from PIL import Image
from typing import List, Tuple, Union

from trident.wsi_objects.WSI import WSI, ReadMode


class ImageWSI(WSI):

    def __init__(self, slide_path, **kwargs) -> None:
        """
        Initialize a WSI object from a standard image file (e.g., PNG, JPEG, etc.).

        Parameters:
            slide_path (str):
                Path to the image file.
            mpp (float):
                Microns per pixel. Required since standard image formats do not store this metadata.
            name (str, optional):
                Optional name for the slide.
            lazy_init (bool, default=True):
                Whether to defer initialization until the WSI is accessed.

        Example
        -------
        >>> wsi = ImageWSI("path/to/image.png", lazy_init=False, mpp=0.51)
        >>> print(wsi)
        <width=5120, height=3840, backend=ImageWSI, mpp=0.51, mag=20>
        """
        #enable loading large images.
        from PIL import PngImagePlugin
        PngImagePlugin.MAX_TEXT_CHUNK = 2**30  # ~1GB
        PngImagePlugin.MAX_TEXT_MEMORY = 2**30
        PngImagePlugin.MAX_IMAGE_PIXELS = None  # Optional: disables large image warning

        self.img = None
        super().__init__(slide_path, **kwargs)

    def _lazy_initialize(self) -> None:
        """
        Lazily initialize the WSI using a standard image file (e.g., JPEG, PNG, etc.).

        This method loads the image using PIL and extracts relevant metadata such as
        dimensions and magnification. It assumes a single-resolution image (no pyramid).
        If a tissue segmentation mask is available, it is also loaded.

        Raises:
            FileNotFoundError:
                If the WSI file or tissue segmentation mask is not found.
            Exception:
                If an unexpected error occurs during initialization.

        Notes:
        After initialization, the following attributes are set:
        - `width` and `height`: dimensions of the image.
        - `dimensions`: (width, height) tuple of the image.
        - `level_downsamples`: set to `[1]` (single resolution).
        - `level_dimensions`: set to a list containing the image dimensions.
        - `level_count`: set to `1`.
        - `mag`: estimated magnification level.
        - `gdf_contours`: loaded from `tissue_seg_path`, if available.
        """

        super()._lazy_initialize()

        if not self._initialized:
            try:
                self._ensure_image_open()
                if self.mpp is None:
                    self.mpp = self._fetch_mpp(self.custom_mpp_keys)
                if self.mpp is None and self.default_mpp is not None:
                    self.mpp = round(float(self.default_mpp), 4)
                if self.mpp is None:
                    raise ValueError(
                        "Unable to determine `mpp` for image input. Provide `mpp` explicitly, "
                        "store it in config/CSV, or use an input image with readable DPI metadata."
                    )
                self.level_downsamples = [1]
                self.dimensions = (self.img.width, self.img.height)
                self.width, self.height = self.dimensions[0], self.dimensions[1]
                self.mag = self._fetch_magnification(self.custom_mpp_keys)
                self.dimensions = self.img.size
                self.level_dimensions = [(self.img.width, self.img.height)]
                self.level_count = 1
                self._initialized = True

            except Exception as e:
                raise Exception(f"Error initializing WSI with PIL.Image: {e}")

    def _ensure_image_open(self):
        if self.img is None:
            self.img = Image.open(self.slide_path).convert("RGB")

    def _fetch_mpp(self, custom_mpp_keys: List[str] | None = None) -> float | None:
        """
        Recover MPP for standard image files.

        Priority:
        1. Explicit custom metadata keys from PIL `Image.info`
        2. Common hand-authored MPP keys in `Image.info`
        3. DPI-style metadata (PNG/JPEG/TIFF via Pillow)
        """
        self._ensure_image_open()

        info = getattr(self.img, "info", {}) or {}
        candidate_keys = list(custom_mpp_keys or []) + [
            "mpp",
            "MPP",
            "microns_per_pixel",
            "microns-per-pixel",
            "um_per_px",
            "um-per-px",
        ]

        for key in candidate_keys:
            if key in info:
                try:
                    return round(float(info[key]), 4)
                except (TypeError, ValueError):
                    continue

        dpi = info.get("dpi")
        if isinstance(dpi, tuple) and len(dpi) >= 1:
            try:
                dpi_x = float(dpi[0])
                if dpi_x > 0:
                    return round(25400.0 / dpi_x, 4)
            except (TypeError, ValueError, ZeroDivisionError):
                pass

        jfif_density = info.get("jfif_density")
        jfif_unit = info.get("jfif_unit")
        if isinstance(jfif_density, tuple) and len(jfif_density) >= 1:
            try:
                density_x = float(jfif_density[0])
                if density_x > 0:
                    if jfif_unit == 1:  # pixels per inch
                        return round(25400.0 / density_x, 4)
                    if jfif_unit == 2:  # pixels per cm
                        return round(10000.0 / density_x, 4)
            except (TypeError, ValueError, ZeroDivisionError):
                pass

        return None

    def get_dimensions(self):
        return self.dimensions

    def get_thumbnail(self, size):
        """
        Generate a thumbnail of the image.

        Parameters:
            size (tuple[int, int]):
                Desired thumbnail size (width, height).

        Returns:
            PIL.Image.Image: RGB thumbnail image.
        """
        self._ensure_image_open()
        img = self.img.copy()
        img.thumbnail(size)
        return img

    def read_region(
        self,
        location: Tuple[int, int],
        level: int,
        size: Tuple[int, int],
        read_as: ReadMode = 'pil',
    ) -> Union[Image.Image, np.ndarray]:
        """
        Extract a specific region from a single-resolution image (e.g., JPEG, PNG, TIFF).

        Parameters:
            location (Tuple[int, int]):
                (x, y) coordinates of the top-left corner of the region to extract.
            level (int):
                Pyramid level to read from. Only level 0 is supported for non-pyramidal images.
            size (Tuple[int, int]):
                (width, height) of the region to extract.
            read_as ({'pil', 'numpy'}, optional):
                Output format for the region:
                - 'pil': returns a PIL Image (default)
                - 'numpy': returns a NumPy array (H, W, 3)

        Returns:
            Union[PIL.Image.Image, np.ndarray]: Extracted image region in the specified format.

        Raises:
            ValueError:
                If `level` is not 0 or if `read_as` is not one of the supported options.

        Example
        -------
        >>> region = wsi.read_region((0, 0), level=0, size=(512, 512), read_as='numpy')
        >>> print(region.shape)
        (512, 512, 3)
        """
        if level != 0:
            raise ValueError("ImageWSI only supports reading at level=0 (no pyramid levels).")

        self._ensure_image_open()
        region = self.img.crop((
            location[0],
            location[1],
            location[0] + size[0],
            location[1] + size[1]
        )).convert('RGB')

        if read_as == 'pil':
            return region
        elif read_as == 'numpy':
            return np.array(region)
        else:
            raise ValueError(f"Invalid `read_as` value: {read_as}. Must be 'pil' or 'numpy'.")

    def segment_tissue(self, *args, **kwargs):
        out = super().segment_tissue(*args, **kwargs)
        self.close()
        return out
    
    def extract_tissue_coords(self, *args, **kwargs):
        out = super().extract_tissue_coords(*args, **kwargs)
        self.close()
        return out

    def visualize_coords(self, *args, **kwargs):
        out = super().visualize_coords(*args, **kwargs)
        self.close()
        return out

    def extract_patch_features(self, *args, **kwargs):
        out = super().extract_patch_features(*args, **kwargs)
        self.close()
        return out

    def extract_slide_features(self, *args, **kwargs):
        out = super().extract_slide_features(*args, **kwargs)
        self.close()
        return out

    def close(self):
        """
        Close the internal image object to free memory. These can take several GB in RAM.
        """
        if self.img is not None:
            self.img.close()
            self.img = None
            self._initialized = False
