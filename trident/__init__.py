"""TRIDENT preprocessing exports, loaded lazily to keep optional readers optional."""

from importlib import import_module
from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("tri-mil")
except PackageNotFoundError:
    __version__ = "0.4.0"

_EXPORTS = {
    "Processor": "trident.Processor",
    "AnyToTiffConverter": "trident.Converter",
    "visualize_heatmap": "trident.Visualization",
    "deprecated": "trident.Maintenance",
    "load_wsi": "trident.wsi_objects.WSIFactory",
    "WSIReaderType": "trident.wsi_objects.WSIFactory",
    "WSIPatcher": "trident.wsi_objects.WSIPatcher",
    "OpenSlideWSIPatcher": "trident.wsi_objects.WSIPatcher",
    "WSIPatcherDataset": "trident.wsi_objects.WSIPatcherDataset",
}
for _reader in ["OpenSlideWSI", "CuCIMWSI", "ImageWSI", "SDPCWSI", "OMEZarrWSI", "CZIWSI"]:
    _EXPORTS[_reader] = f"trident.wsi_objects.{_reader}"

__all__ = list(_EXPORTS)


def __getattr__(name):
    """Import the requested preprocessing class only when it is first accessed."""
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(_EXPORTS[name]), name)
    globals()[name] = value
    return value
