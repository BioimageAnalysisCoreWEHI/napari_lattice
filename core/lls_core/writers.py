from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Iterable, List, Optional

from lls_core.types import ArrayLike

from pydantic.v1 import NonNegativeInt

from numcodecs import Blosc
from pathlib import Path
import xarray as xr
import dask.array as da
import numpy as np
import zarr

from lls_core.utils import make_filename_suffix, get_zarr_compression, ZARR_MAJOR_VERSION
RoiIndex = Optional[NonNegativeInt]

def resolve_output_dtype(dtype: np.dtype) -> np.dtype:
    """Pick the output dtype: keep float and small ints, cast larger ints (int32, int64) to uint16."""
    dtype = np.dtype(dtype)
    if np.issubdtype(dtype, np.integer):
        return dtype if np.iinfo(dtype).max < np.iinfo(np.uint16).max else np.dtype(np.uint16)
    if np.issubdtype(dtype, np.floating):
        return dtype
    raise TypeError(f"Unsupported data dtype: {dtype}")


def to_output_dtype(array: np.ndarray, out_dtype: np.dtype) -> np.ndarray:
    """Cast to ``out_dtype``, clipping to range only when casting to uint16."""
    out_dtype = np.dtype(out_dtype)
    if out_dtype == np.uint16 and array.dtype != np.uint16:
        return np.clip(array, 0.0, 65535.0).astype(np.uint16)
    return array.astype(out_dtype, copy=False)

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

    def write_all(self, slices: Iterable[ProcessedSlice[ArrayLike]]) -> None:
        """
        Write each slice, then finish the output.

        By default, each slice is passed to ``write_slice`` in order, and
        ``close`` is called at the end. Override this for custom behavior.
        """
        for slice in slices:
            self.write_slice(slice)
        self.close()

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
        # A MIP is a singleton-Z (1, Y, X) volume; the default (2, 4, 4) level
        # downsamples Z by 2, which is degenerate for Z=1. Drop Z-subsampling
        # levels in that case.
        if getattr(self.lattice, "save_mip", False):
            subsamp = ((1, 1, 1), (1, 2, 2))
        else:
            subsamp = ((1, 1, 1), (1, 2, 2), (2, 4, 4))
        self.bdv_writer = npy2bdv.BdvWriter(
            filename=str(path),
            compression='gzip',
            nchannels=len(self.lattice.channel_range),
            subsamp=subsamp,
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
    A writer for for TIFF output format.

    By default write a deflate-compressed OME-TIFF (compression='zlib),
    which keeps the compresses empty/black space at borders of deskewed images keeping
    Fiji/Bio-Formats readability. Set compression=None to fall back to the legacy
    uncompressed ImageJ TIFF.
    """
    pending_slices: List[ImageSlice] = field(default_factory=list, init=False)
    time: Optional[NonNegativeInt] = None
    #: tifffile compression codec for the OME-TIFF output. ``'zlib'`` (deflate) is
    #: the default; ``None`` writes an uncompressed ImageJ-TIFF as before.
    compression: Optional[str] = "zlib"

    def __post_init__(self):
        self.pending_slices = []

    def flush(self):
        "Write out all pending slices"
        import numpy as np
        import tifffile
        if len(self.pending_slices) == 0:
            return

        first_result = self.pending_slices[0]
        # One buffered slice per channel for this timepoint; each is a (Z, Y, X)
        # volume. flush() is called once the timepoint changes, so T == 1 here.
        channel_arrays = [np.asarray(result.data) for result in self.pending_slices]

        # Holds every channel for this timepoint (TCZYX / TZCYX), so
        # name will just be by ROI and time
        path = self.lattice.make_filepath(
            make_filename_suffix(
                time=first_result.time,
                roi_index=first_result.roi_index
            )
        )

        if self.compression is None:
            # Legacy uncompressed ImageJ-TIFF (TZCYX). ImageJ-TIFF cannot be
            # compressed, so this path only exists as a fallback. It is not an
            # OME-TIFF, so drop the ".ome" from the default ".ome.tif" name.
            if path.name.endswith(".ome.tif"):
                path = path.with_name(path.name[:-len(".ome.tif")] + ".tif")
            images_array = np.swapaxes(
                np.expand_dims(channel_arrays, axis=0), 1, 2
            ).astype("uint16")  # ImageJ TIFF can only handle 16-bit uints, not 32
            tifffile.imwrite(
                str(path),
                data=images_array,
                bigtiff=True,
                resolution=(1. / self.lattice.dx, 1. / self.lattice.dy),
                resolutionunit="MICROMETER",
                metadata={'spacing': self.lattice.new_dz, 'unit': 'um', 'axes': 'TZCYX'},
                imagej=True
            )
        else:
            # Compressed OME-TIFF. The full timepoint is already buffered in
            # channel_arrays, so assemble the (T, C, Z, Y, X) array (T == 1) and
            # write it with tifffile's native OME support, which lays out the
            # pages and OME-XML correctly for any Z depth. imagej=True is
            # incompatible with both compression and OME; physical pixel sizes
            # and channel names travel in the OME-XML instead.

            # Same dtype policy as OMEZarrWriter: preserve small ints and float,
            # standardise wider integers to uint16.
            out_dtype = resolve_output_dtype(np.result_type(*channel_arrays))
            stack = to_output_dtype(np.stack(channel_arrays, axis=0), out_dtype)  # (C, Z, Y, X)
            data5d = stack[np.newaxis, ...]  # (T=1, C, Z, Y, X), matches TCZYX

            ome_metadata = {
                "axes": "TCZYX",
                "PhysicalSizeX": float(self.lattice.dx), "PhysicalSizeXUnit": "µm",
                "PhysicalSizeY": float(self.lattice.dy), "PhysicalSizeYUnit": "µm",
                "PhysicalSizeZ": float(self.lattice.new_dz), "PhysicalSizeZUnit": "µm",
                "Channel": {"Name": [str(result.channel) for result in self.pending_slices]},
            }
            with tifffile.TiffWriter(str(path), bigtiff=True, ome=True) as tw:
                tw.write(data5d, compression=self.compression, metadata=ome_metadata)

        self.written_files.append(path)

        # Reinitialise
        self.pending_slices = []

    def write_slice(self, slice: ProcessedSlice[ArrayLike]):
        if slice.time != self.time:
            self.flush()

        self.time = slice.time
        self.pending_slices.append(slice)

    def close(self):
        self.flush()

    def write_all(self, slices: Iterable[ProcessedSlice[ArrayLike]]) -> None:
        """
        Stream every timepoint/channel into a single compressed OME-TIFF.

        tifffile only makes one OME series per ``write()`` call, so we stream
        planes from the slice iterator instead of buffering full volumes.
        The uncompressed path keeps the old per-timepoint behavior.
        """
        # Legacy ImageJ-TIFF: unchanged, one file per timepoint.
        if self.compression is None:
            super().write_all(slices) # default behaviour
            return

        import numpy as np
        import tifffile

        it = iter(slices)
        first = next(it, None)
        if first is None:
            # Empty ROI: write nothing.
            return

        first_vol = np.asarray(first.data)
        if first_vol.ndim != 3:
            raise ValueError(f"Expected (Z, Y, X) slice, got shape {first_vol.shape}")

        # Dtype policy matches OMEZarrWriter: fixed from the first slice.
        out_dtype = resolve_output_dtype(first_vol.dtype)
        z_len, y_len, x_len = (int(d) for d in first_vol.shape)
        t_len = len(self.lattice.time_range)
        c_len = len(self.lattice.channel_range)

        # One file per ROI, named like the other writers (no timepoint/channel
        # in the name, since the file holds them all).
        suffix = f"_{make_filename_suffix(roi_index=str(self.roi_index))}" if self.roi_index is not None else ""
        path = self.lattice.make_filepath(suffix)

        ome_metadata = {
            "axes": "TCZYX",
            "PhysicalSizeX": float(self.lattice.dx), "PhysicalSizeXUnit": "µm",
            "PhysicalSizeY": float(self.lattice.dy), "PhysicalSizeYUnit": "µm",
            "PhysicalSizeZ": float(self.lattice.new_dz), "PhysicalSizeZUnit": "µm",
            "Channel": {"Name": [str(ch) for ch in self.lattice.channel_range]},
        }

        def plane_generator():
            # First slice is already materialised; cast it once and emit its
            # Z-planes, then pull and cast the rest. Slices arrive time-major,
            # channel-minor, so flattening each (Z, Y, X) volume yields planes in
            # exactly the (t, c, z) order tifffile expects for shape=(T,C,Z,Y,X).
            first_cast = to_output_dtype(first_vol, out_dtype)
            for z in range(first_cast.shape[0]):
                yield first_cast[z]
            for sl in it:
                vol = to_output_dtype(np.asarray(sl.data), out_dtype)
                if vol.shape != (z_len, y_len, x_len):
                    raise ValueError(
                        f"Inconsistent slice shape {vol.shape}; expected {(z_len, y_len, x_len)}"
                    )
                for z in range(vol.shape[0]):
                    yield vol[z]

        # A plane count != T*C*Z makes tw.write raise, so a partial or misordered
        # run fails loudly instead of writing a silently wrong file.
        with tifffile.TiffWriter(str(path), bigtiff=True, ome=True) as tw:
            tw.write(
                plane_generator(),
                shape=(t_len, c_len, z_len, y_len, x_len),
                dtype=out_dtype,
                compression=self.compression,
                metadata=ome_metadata,
            )
        self.written_files.append(path)

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
        self._dtype = np.uint16 #Placeholder; resolved per data in write_slice

        self._pix_z, self._pix_y, self._pix_x = (self.lattice.new_dz, self.lattice.dy, self.lattice.dx)

    def write_slice(self, slice) -> Path:
        """Write a 3D (Z,Y,X) slice into (t,c,:,:,:) and return root .ome.zarr path."""
        data3d = self._to_numpy(getattr(slice, "data", slice))
        if data3d.ndim != 3:
            raise ValueError(f"Expected (Z,Y,X), got {data3d.shape}")

        if self._zyx is None:
            self._zyx = (int(data3d.shape[0]), int(data3d.shape[1]), int(data3d.shape[2]))

        # Same dtype policy as TiffWriter: preserve small ints and float,
        # standardise wider integers to uint16. Preserving float means label/
        # mask images (typically float32) keep their values instead of being
        # clipped to 16-bit.
        self._dtype = resolve_output_dtype(data3d.dtype)

        t_idx = int(getattr(slice, "time_index", 0))
        c_idx = int(getattr(slice, "channel_index", 0))
        t_len, c_len = self._resolve_t_c_lengths(slice)

        # If it's the first slice - initialize the full zarr array size
        if self._arr is None:
            self._root_group, self._arr = self._create_store(t_len, c_len, self._zyx, self._dtype)

        self._arr[t_idx, c_idx, :, :, :] = to_output_dtype(data3d, self._arr.dtype)
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

        dataset_kwargs = {
            "shape": (t_len, c_len, zyx[0], zyx[1], zyx[2]),
            "chunks": (1, 1, *self.chunk_zyx),
            "dtype": dtype,
            **get_zarr_compression(),
        }

        # Single version check for both group and array creation 
        if ZARR_MAJOR_VERSION >= 3:
            # zarr v3: group cannot be constructed from a path directly, and
            # create_array is the current API (create_dataset is deprecated).
            root = zarr.open_group(store=str(self._root_path), mode="a")
            arr = root.create_array("0", **dataset_kwargs)
        else:
            # zarr v2: build  group from a DirectoryStore; the group has no
            # create_array, so create_dataset is the equivalent.
            root = zarr.group(store=zarr.DirectoryStore(str(self._root_path)))
            dataset_kwargs["overwrite"] = self.overwrite
            arr = root.create_dataset("0", **dataset_kwargs)

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