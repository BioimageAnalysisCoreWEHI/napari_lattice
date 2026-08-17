from importlib.metadata import PackageNotFoundError, version as _package_version

try:
    __version__ = _package_version("napari-lattice")
except PackageNotFoundError:
    __version__ = "unknown"
