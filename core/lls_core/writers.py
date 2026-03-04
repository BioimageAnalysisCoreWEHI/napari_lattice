from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List, Optional

from lls_core.types import ArrayLike

from pydantic.v1 import NonNegativeInt

from numcodecs import Blosc
from pathlib import Path
import xarray as xr
import dask.array as da
import numpy as np
import zarr

from lls_core.utils import make_filename_suffix, get_zarr_compression

import logging
logger = logging.getLogger(__name__)

RoiIndex = Optional[NonNegativeInt]

if TYPE_CHECKING:
    from lls_core.models.lattice_data import LatticeData
    import npy2bdv
    from lls_core.models.results import ProcessedSlice, ImageSlice
    from pathlib import Path


@dataclass
class Writer(ABC):
    """
    A writer is an abstraction over the logic used to write image slices to disk.
    `Writer`s need to work incrementally, in order that we don't need the entire multidimensional
    image in memory at the same time.
    """
    lattice: LatticeData
    roi_index: RoiIndex
    written_files: List[Path] = field(default_factory=list, init=False)

    @abstractmethod
    def write_slice(self, slice: ProcessedSlice[ArrayLike]):
        """
        Writes a 3D image slice
        """
        pass

    def close(self):
        """
        Called when no more image slices are available, and the writer should finalise its output files
        """
        pass

@dataclass
class BdvWriter(Writer):
    """
    A writer for for Fiji BigDataViewer output format
    """
    bdv_writer: npy2bdv.BdvWriter = field(init=False)

    def __post_init__(self):
        import npy2bdv
        suffix = f"_{make_filename_suffix(roi_index=str(self.roi_index))}" if self.roi_index is not None else ""
        path = self.lattice.make_filepath(suffix)
        self.bdv_writer = npy2bdv.BdvWriter(
            filename=str(path),
            compression='gzip',
            nchannels=len(self.lattice.channel_range),
            subsamp=((1, 1, 1), (1, 2, 2), (2, 4, 4)),
            overwrite=False
        )
        self.written_files.append(path)

    def write_slice(self, slice: ProcessedSlice[ArrayLike]):
        import numpy as np
        self.bdv_writer.append_view(
            np.array(slice.data),
            # We need to use the indices here to ensure they start from 0 and 
            # are contiguous
            time=slice.time_index,
            channel=slice.channel_index,
            voxel_size_xyz=(self.lattice.dx, self.lattice.dy, self.lattice.new_dz),
            voxel_units='um'
        )

    def close(self):
        self.bdv_writer.write_xml()
        self.bdv_writer.close()

@dataclass
class TiffWriter(Writer):
    """
    A writer for for TIFF output format
    """
    pending_slices: List[ImageSlice] = field(default_factory=list, init=False)
    time: Optional[NonNegativeInt] = None

    def __post_init__(self):
        self.pending_slices = []
    
    def flush(self):
        "Write out all pending slices to a TIFF file"
        import tifffile
        if len(self.pending_slices) > 0:
            first_result = self.pending_slices[0]
            path = self.lattice.make_filepath(
                make_filename_suffix(
                    channel=first_result.channel,
                    time=first_result.time,
                    roi_index=first_result.roi_index
                )
            )

            n_channels = len(self.pending_slices)

            if n_channels == 1:
                # Single channel: write directly using imwrite.
                # For memmap-backed data, tifffile iterates Z-pages lazily,
                # avoiding loading the full volume into RAM.
                # Reshape to 5D (T,Z,C,Y,X) so tifffile correctly identifies
                # the Z dimension as slices (not channels). reshape() on a
                # memmap creates a view — no data is copied.
                data = np.asarray(self.pending_slices[0].data)
                z, y, x = data.shape
                data_5d = data.reshape(1, z, 1, y, x)
                tifffile.imwrite(
                    str(path),
                    data=data_5d,
                    bigtiff=True,
                    imagej=True,
                    resolution=(1./self.lattice.dx, 1./self.lattice.dy),
                    resolutionunit="MICROMETER",
                    metadata={'spacing': self.lattice.new_dz, 'unit': 'um', 'axes': 'TZCYX'},
                )
            else:
                # Multi-channel: build TZCYX array and write at once.
                # Preserve the data's native dtype (uint8 labels, uint16
                # intensity, float32 workflow outputs, etc.).
                images_array = np.swapaxes(
                    np.expand_dims(
                        [np.asarray(result.data) for result in self.pending_slices],
                        axis=0,
                    ),
                    1, 2,
                )
                tifffile.imwrite(
                    str(path),
                    data=images_array,
                    bigtiff=True,
                    imagej=True,
                    resolution=(1./self.lattice.dx, 1./self.lattice.dy),
                    resolutionunit="MICROMETER",
                    metadata={'spacing': self.lattice.new_dz, 'unit': 'um', 'axes': 'TZCYX'},
                )

            self.written_files.append(path)

            # Clean up memmap temp files if any slices were backed by memmap
            memmap_paths = set()
            for result in self.pending_slices:
                if isinstance(result.data, np.memmap):
                    p = getattr(result.data, 'filename', None)
                    if p:
                        memmap_paths.add(p)

            self.pending_slices = []

            if memmap_paths:
                import gc, os
                gc.collect()
                for p in memmap_paths:
                    try:
                        os.unlink(p)
                        logger.info(f"Cleaned up memmap temp file: {p}")
                    except OSError:
                        pass

    def write_slice(self, slice: ProcessedSlice[ArrayLike]):
        if slice.time != self.time:
            self.flush()

        self.time = slice.time
        self.pending_slices.append(slice)

    def close(self):
        self.flush()

@dataclass
class OMEZarrWriter(Writer):
    DEFAULT_CHUNK_ZYX = (64, 256, 256)
    def __init__(
        self,
        params,
        *,
        overwrite: bool = True,
        chunk_zyx: tuple[int, int, int] = DEFAULT_CHUNK_ZYX,
        compressor: Optional[Blosc] = None,
        roi_index: Optional[int] = None,         
        roi_label: Optional[str] = None,         
        **kwargs,                                
    ) -> None:
        self._roi_index = int(roi_index) if roi_index is not None else int(getattr(params, "roi_index", 0))

        super().__init__(params,roi_index=self._roi_index)
        self.params = params
        self.overwrite = overwrite
        self.chunk_zyx = chunk_zyx
        self.compressor = compressor or Blosc(cname="zstd", clevel=5, shuffle=Blosc.SHUFFLE)

        self._roi_label = roi_label

        self._save_dir = Path(self.params.save_dir)
        self._save_dir.mkdir(parents=True, exist_ok=True)

        suffix = f"_{make_filename_suffix(roi_index=str(self.roi_index))}" if self.roi_index is not None else ""
        path = self.lattice.make_filepath(suffix)

        self._base_name = path.name
        self._root_path = path

        self._arr = None
        self._root_group = None
        self._zyx = None
        self._t_len = None
        self._c_len = None
        self._dtype = np.uint16 #We are enforcing 16-bit, but may change in future

        self._pix_z, self._pix_y, self._pix_x = (self.lattice.new_dz, self.lattice.dy, self.lattice.dx)

    def ensure_initialized(self, zyx_shape: tuple[int, int, int], t_len: int, c_len: int, dtype) -> zarr.Array:
        """Pre-create the zarr store and return the backing array.

        This allows callers (e.g. the direct-write tiling path) to obtain
        the zarr array before any ``write_slice()`` calls, so they can
        write slabs directly into the array.

        Args:
            zyx_shape: (Z, Y, X) dimensions of a single 3D volume.
            t_len: Number of time points.
            c_len: Number of channels.
            dtype: Data type for the array.

        Returns:
            The zarr.Array backing this store (shape: ``(t, c, z, y, x)``).
        """
        self._zyx = (int(zyx_shape[0]), int(zyx_shape[1]), int(zyx_shape[2]))
        self._t_len = t_len
        self._c_len = c_len
        self._dtype = np.dtype(dtype)
        if self._arr is None:
            self._root_group, self._arr = self._create_store(t_len, c_len, self._zyx, self._dtype)
        return self._arr

    def write_slice(self, slice) -> Path:
        """Write a 3D (Z,Y,X) slice into (t,c,:,:,:) and return root .ome.zarr path."""
        data3d = self._to_numpy(getattr(slice, "data", slice))
        if data3d.ndim != 3:
            raise ValueError(f"Expected (Z,Y,X), got {data3d.shape}")

        if self._zyx is None:
            self._zyx = (int(data3d.shape[0]), int(data3d.shape[1]), int(data3d.shape[2]))

        # Preserve the data's native dtype so label images (uint8, uint32)
        # and float workflow outputs are not silently truncated.
        if np.issubdtype(data3d.dtype, np.integer) or np.issubdtype(data3d.dtype, np.floating):
            self._dtype = data3d.dtype
        else:
            raise TypeError(f"Unsupported data dtype: {data3d.dtype}")
                
        t_idx = int(getattr(slice, "time_index", 0))
        c_idx = int(getattr(slice, "channel_index", 0))
        t_len, c_len = self._resolve_t_c_lengths(slice)

        # If it's the first slice - initialize the full zarr array size
        if self._arr is None:
            self._root_group, self._arr = self._create_store(t_len, c_len, self._zyx, self._dtype)
        
        # Convert to the store dtype.  For integer targets, clip to the
        # valid range to avoid wrap-around; for floats, cast directly.
        if np.issubdtype(self._dtype, np.integer):
            info = np.iinfo(self._dtype)
            self._arr[t_idx, c_idx, :, :, :] = np.clip(
                data3d, float(info.min), float(info.max)
            ).astype(self._dtype)
        else:
            self._arr[t_idx, c_idx, :, :, :] = data3d.astype(self._dtype)
        return self._root_path

    # Optional hook if the framework ever calls it.
    def finalize(self) -> None:
        """No-op; multiscales metadata is written at creation."""
        return

    def _resolve_t_c_lengths(self, slice) -> tuple[int, int]:
        if self._t_len is not None and self._c_len is not None:
            return self._t_len, self._c_len
        t_len = len(getattr(self.params, "time_range", None) or [])
        c_len = len(getattr(self.params, "channel_range", None) or [])
        self._t_len, self._c_len = t_len, c_len 
        return t_len, c_len

    def _create_store(
        self, t_len: int, c_len: int, zyx: tuple[int, int, int], dtype: np.dtype
    ) -> tuple[zarr.Group, zarr.Array]:
        if self.overwrite and self._root_path.exists():
            import shutil
            shutil.rmtree(self._root_path)

        chunks = (1, 1, *self.chunk_zyx)

        zarr_major = int(zarr.__version__.split(".")[0])

        # Always write zarr v2-format stores for maximum compatibility
        # (Fiji, OMERO, QuPath, napari).  Use numcodecs compressor
        # regardless of zarr library version since zarr_format=2
        # requires numcodecs, not zarr.codecs.
        from numcodecs import Blosc as _Blosc
        compressor = _Blosc(cname="zstd", clevel=5, shuffle=_Blosc.SHUFFLE)

        if zarr_major >= 3:
            root = zarr.open_group(
                store=str(self._root_path),
                mode="a",
                zarr_format=2,
            )
        else:
            store = zarr.DirectoryStore(
                str(self._root_path),
                dimension_separator="/",
            )
            root = zarr.group(store=store)

        dataset_kwargs = {
            "shape": (t_len, c_len, zyx[0], zyx[1], zyx[2]),
            "chunks": chunks,
            "dtype": dtype,
            "compressor": compressor,
        }
        if zarr_major < 3:
            dataset_kwargs["overwrite"] = self.overwrite
            dataset_kwargs["dimension_separator"] = "/"

        arr = root.create_dataset("0", **dataset_kwargs)

        # _ARRAY_DIMENSIONS is expected by xarray and some NGFF readers
        arr.attrs["_ARRAY_DIMENSIONS"] = ["t", "c", "z", "y", "x"]

        self._write_ngff_attrs(root)
        return root, arr

    def _write_ngff_attrs(self, group: zarr.Group) -> None:
        # Minimal, valid NGFF (v0.4) with (t,c,z,y,x) and micrometer units
        z_ps = float(self._pix_z)
        y_ps = float(self._pix_y)
        x_ps = float(self._pix_x)
        group.attrs["multiscales"] = [
            {
                "version": "0.4",
                "name": self._base_name,
                "axes": [
                    {"name": "t", "type": "time"},
                    {"name": "c", "type": "channel"},
                    {"name": "z", "type": "space", "unit": "micrometer"},
                    {"name": "y", "type": "space", "unit": "micrometer"},
                    {"name": "x", "type": "space", "unit": "micrometer"},
                ],
                "datasets": [
                    {
                        "path": "0",
                        "coordinateTransformations": [
                            {"type": "scale", "scale": [1.0, 1.0, float(z_ps), float(y_ps), float(x_ps)]}
                        ],
                    }
                ],
            }
        ]
        # Optional: coarse OMERO display info
        cN = int(self._c_len or 1)
        group.attrs["omero"] = {
            "name": self._base_name,
            "version": "0.4",
            "channels": [{"label": f"C{c}"} for c in range(cN)],
        }

    @staticmethod
    def _to_numpy(data) -> np.ndarray:
        if isinstance(data, xr.DataArray):
            data = data.data
        if da is not None and isinstance(data, da.Array):
            return np.asarray(data.compute())
        return np.asarray(data)