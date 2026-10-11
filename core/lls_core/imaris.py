"""
Imaris (v 5.5; ``.ims``) writing, implemented directly on top of ``h5py``.

Why not PyImarisWriter
----------------------
Bitplane ship an official Python wrapper (``PyImarisWriter`` on PyPI), but 
bundles only Windows libraries. In linux and macOS, you'll haev to build the 
C++ project. Instead we create a custom imaris writer as 
``.ims`` is plain HDF5 with a documented layout. This
ensures no new dependency: ``h5py`` already is a part of ``npy2bdv``.

Provenance
----------
Every convention below is taken from the reference implementation,
``writer/bpWriterHDF5.cxx`` in https://github.com/imaris/ImarisWriter (Apache
2.0), so that it can be re-checked. Line references are to that file at the
time of writing. The layout is::

    /                                   attrs: ImarisDataSet, ImarisVersion, ...
    /DataSetInfo/Image                  attrs: X, Y, Z, ExtMin0..2, ExtMax0..2, Unit
    /DataSetInfo/Channel {c}            attrs: Name, Color, ColorRange, ...
    /DataSetInfo/TimeInfo               attrs: DatasetTimePoints, TimePoint1..N
    /DataSet/ResolutionLevel {r}/TimePoint {t}/Channel {c}/Data       (Z, Y, X)
                                        attrs: ImageSizeX/Y/Z, HistogramMin/Max
                                                             .../Histogram   (256 x uint64)

Two conventions that are easy to get wrong and explicitly mentioned here are:

* **Strings are 1-D arrays of single characters**, not HDF5 strings.
  ``bpWriterHDF5.cxx:670`` writes them as ``H5T_C_S1`` with the dataspace
  length set to the string length, i.e. ``numpy`` dtype ``S1``.
* **The ``Data`` dataspace is padded up to a whole number of chunks**
  (``bpWriterHDF5.cxx:460-470``: ``fileSize = chunkSize * numChunks``). The true
  size lives in the ``ImageSizeX/Y/Z`` attributes on the channel group, which is
  what readers use.
  
  Claude Opus 4.8 used to convert the adapt PyImarisWriter to a pure Python implementation.
"""
from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)

#: Imaris only understands these four sample types. Mirrors
#: ``ConvertToHDF5DataType`` (``bpWriterHDF5.cxx:30-53``), which maps anything
#: unrecognised to uint8; we would rather fail loudly, so unsupported dtypes are
#: promoted by :func:`resolve_imaris_dtype` instead.
IMARIS_DTYPES: Tuple[np.dtype, ...] = (
    np.dtype(np.uint8),
    np.dtype(np.uint16),
    np.dtype(np.uint32),
    np.dtype(np.float32),
)

#: ``bpMultiresolutionImsImage.cxx:708`` and ``:717`` both budget 1 MiB, then
#: divide by the sample size to get a voxel count. Used for the pyramid cutoff
#: and as the chunk budget.
_ONE_MIB = 1024 * 1024

#: Default per-channel display colours, matching the order Imaris itself uses
#: for the first few channels of a multi-channel image.
_DEFAULT_COLORS: Tuple[Tuple[float, float, float], ...] = (
    (1.0, 0.0, 0.0),
    (0.0, 1.0, 0.0),
    (0.0, 0.0, 1.0),
    (1.0, 1.0, 0.0),
    (0.0, 1.0, 1.0),
    (1.0, 0.0, 1.0),
    (1.0, 1.0, 1.0),
)


def resolve_imaris_dtype(dtype: np.dtype) -> np.dtype:
    """
    Return the Imaris sample type to store ``dtype`` as.

    Imaris supports only unsigned 8/16/32-bit integers and float32, so signed
    integers, float64 and bool have to be promoted. Each is widened to the
    narrowest supported type that cannot lose values, which keeps label images
    (often int32) and masks (bool) intact.
    """
    dtype = np.dtype(dtype)
    if dtype in IMARIS_DTYPES:
        return dtype
    if dtype == np.dtype(bool):
        return np.dtype(np.uint8)
    if dtype == np.dtype(np.float16):
        return np.dtype(np.float32)
    if np.issubdtype(dtype, np.floating):
        # float64 -> float32 is the only float Imaris has; precision is lost but
        # the alternative is not writing the image at all.
        return np.dtype(np.float32)
    if np.issubdtype(dtype, np.signedinteger):
        # A signed type needs the *next* unsigned size up to hold its negative
        # range once offset, but Imaris has no int types at all, so clipping at
        # zero is unavoidable. Keep the same width so IDs above 65535 survive.
        return np.dtype({1: np.uint8, 2: np.uint16, 4: np.uint32, 8: np.uint32}[dtype.itemsize])
    if np.issubdtype(dtype, np.unsignedinteger):
        # uint64 has no Imaris equivalent.
        return np.dtype(np.uint32)
    raise TypeError(f"Cannot store dtype {dtype} in an Imaris file")


def optimal_image_pyramid(
    shape_zyx: Tuple[int, int, int],
    itemsize: int,
    reduce_z: bool = True,
    max_bytes: int = _ONE_MIB,
) -> List[Tuple[int, int, int]]:
    """
    Resolution level sizes, as a faithful port of ``GetOptimalImagePyramid``
    (``writer/bpOptimalBlockLayout.cxx``).

    Levels are halved along whichever axes are still "long" relative to the
    others, so an anisotropic deskewed volume is reduced towards isotropy rather
    than blindly by 2 in every axis. Iteration stops once a level fits in
    ``max_bytes``, or once no axis is eligible.

    ``shape_zyx`` is in numpy order; the C++ works in (x, y, z), so the axes are
    reversed on the way in and out.
    """
    max_voxels = max(int(max_bytes // itemsize), 1)
    size = [int(shape_zyx[2]), int(shape_zyx[1]), int(shape_zyx[0])]  # x, y, z
    result = [tuple(size)]

    while size[0] * size[1] * size[2] > max_voxels:
        large_x, large_y = size[0], size[1]
        # When Z reduction is disabled the C++ substitutes 1 here, which also
        # makes the reduce_z test below fail, so Z is left alone.
        large_z = size[2] if reduce_z else 1

        # The factor of 10 biases towards keeping an axis until it is roughly an
        # order of magnitude smaller than the others.
        do_x = large_x > 1 and (10 * large_x) * (10 * large_x) > large_y * large_z
        do_y = large_y > 1 and (10 * large_y) * (10 * large_y) > large_x * large_z
        do_z = large_z > 1 and (10 * large_z) * (10 * large_z) > large_x * large_y
        if not (do_x or do_y or do_z):
            break
        if do_x:
            size[0] //= 2
        if do_y:
            size[1] //= 2
        if do_z:
            size[2] //= 2
        result.append(tuple(size))

    # Back to (z, y, x).
    return [(s[2], s[1], s[0]) for s in result]


def chunk_shape(shape_zyx: Tuple[int, int, int], itemsize: int, max_bytes: int = _ONE_MIB) -> Tuple[int, int, int]:
    """
    Pick an HDF5 chunk shape of at most ``max_bytes``.

    Deliberately *not* a port of ``GetOptimalBlockSize``, which searches
    power-of-two layouts against a three-part rendering cost model. Chunk shape
    only affects read performance, not whether Imaris can open the file, so the
    ~80 lines of cost model are not worth carrying. Instead grow the smallest
    axis repeatedly, which lands on the cube-ish, budget-filling shapes that
    cost model prefers anyway.
    """
    limit = [max(int(s), 1) for s in shape_zyx]
    budget = max(int(max_bytes // itemsize), 1)
    chunk = [1, 1, 1]

    while True:
        # Prefer growing X, then Y, then Z, so a chunk stays contiguous on disk
        # for the fastest-varying axis; among those, always grow the smallest.
        candidates = [
            axis for axis in (2, 1, 0)
            if chunk[axis] < limit[axis] and np.prod(chunk) * 2 <= budget
        ]
        if not candidates:
            break
        axis = min(candidates, key=lambda a: (chunk[a], -a))
        chunk[axis] *= 2

    return (min(chunk[0], limit[0]), min(chunk[1], limit[1]), min(chunk[2], limit[2]))


def _ims_string(value: str) -> np.ndarray:
    """
    Encode ``value`` the way Imaris stores string attributes: a 1-D array of
    single characters (``bpWriterHDF5.cxx:670``), not an HDF5 string.

    Empty strings become a single NUL, matching the zero-length special case at
    ``bpWriterHDF5.cxx:757``, because HDF5 rejects a zero-length dataspace.
    """
    encoded = value.encode("ascii", errors="replace")
    if not encoded:
        encoded = b"\0"
    return np.frombuffer(encoded, dtype="S1")


def _fmt_float(value: float, precision: int = 3) -> str:
    """``bpFloatToString`` (``bpWriterHDF5.cxx:82``): fixed-point, 3 dp default."""
    return f"{float(value):.{precision}f}"


def _fmt_time(when: datetime) -> str:
    """
    ``bpImsUtils::TimeInfoToString``. ``ToString`` defaults to
    ``decimals=2, leadingZeros=true`` (``bpImsUtils.h:36``) so the date and time
    parts are zero-padded, and milliseconds are appended only when non-zero
    (``bpImsUtils.cxx:117``).
    """
    stamp = f"{when.year:04d}-{when.month:02d}-{when.day:02d} {when.hour:02d}:{when.minute:02d}:{when.second:02d}"
    millisecond = when.microsecond // 1000
    if millisecond > 0:
        stamp += f".{millisecond:03d}"
    return stamp


def _downsample(volume: np.ndarray, factors: Tuple[int, int, int]) -> np.ndarray:
    """
    Reduce ``volume`` by integer ``factors`` using a block mean.

    Averaging (rather than striding) is what keeps a downsampled level a fair
    preview of the level above: subsampling a deskewed volume, which is mostly
    interpolated black space, aliases badly. Trailing voxels that do not fill a
    whole block are dropped, which matches the ``size // 2`` level sizes that
    :func:`optimal_image_pyramid` produces for odd inputs.
    """
    if factors == (1, 1, 1):
        return volume
    fz, fy, fx = factors
    trimmed = volume[
        : (volume.shape[0] // fz) * fz,
        : (volume.shape[1] // fy) * fy,
        : (volume.shape[2] // fx) * fx,
    ]
    # Accumulate in a wider type so summing blocks of uint8/uint16 cannot wrap.
    accum = np.float32 if trimmed.dtype == np.float32 else np.float64
    reshaped = trimmed.reshape(
        trimmed.shape[0] // fz, fz,
        trimmed.shape[1] // fy, fy,
        trimmed.shape[2] // fx, fx,
    )
    reduced = reshaped.mean(axis=(1, 3, 5), dtype=accum)
    if np.issubdtype(volume.dtype, np.integer):
        reduced = np.rint(reduced)
    return reduced.astype(volume.dtype, copy=False)


class ImsWriter:
    """
    Incremental Imaris 5.5 writer.

    Volumes are handed over one ``(t, c)`` at a time via :meth:`write_volume`,
    in any order, and each is immediately expanded into every resolution level.
    Nothing is buffered between calls, so peak memory is one input volume plus
    its first reduction (an eighth of the size).

    Metadata that depends on the data as a whole -- the display range taken from
    the accumulated histograms -- is written by :meth:`close`.
    """

    def __init__(
        self,
        path: Path,
        *,
        shape_tczyx: Tuple[int, int, int, int, int],
        dtype: np.dtype,
        voxel_size_zyx: Tuple[float, float, float],
        channel_names: Optional[Sequence[str]] = None,
        channel_colors: Optional[Sequence[Tuple[float, float, float]]] = None,
        unit: str = "um",
        compression: Optional[str] = "gzip",
        compression_opts: Optional[int] = 2,
        image_name: Optional[str] = None,
        recording_date: Optional[datetime] = None,
        application_name: str = "napari-lattice",
        application_version: str = "",
        reduce_z: bool = True,
    ) -> None:
        import h5py

        self.path = Path(path)
        self.n_t, self.n_c, n_z, n_y, n_x = (int(v) for v in shape_tczyx)
        self.shape_zyx = (n_z, n_y, n_x)
        self.dtype = resolve_imaris_dtype(dtype)
        self.voxel_size_zyx = tuple(float(v) for v in voxel_size_zyx)
        self.unit = unit
        self.compression = compression
        self.compression_opts = compression_opts
        self.image_name = image_name or self.path.stem
        self.recording_date = recording_date or datetime.now()
        self.application_name = application_name
        self.application_version = application_version

        self.channel_names = list(channel_names) if channel_names is not None else [f"Channel {c}" for c in range(self.n_c)]
        if len(self.channel_names) != self.n_c:
            raise ValueError(f"Expected {self.n_c} channel names, got {len(self.channel_names)}")
        if channel_colors is not None:
            self.channel_colors = [tuple(c) for c in channel_colors]
        else:
            self.channel_colors = [_DEFAULT_COLORS[c % len(_DEFAULT_COLORS)] for c in range(self.n_c)]

        self.levels = optimal_image_pyramid(self.shape_zyx, self.dtype.itemsize, reduce_z=reduce_z)
        self.chunks = [chunk_shape(level, self.dtype.itemsize) for level in self.levels]

        #: (channel, level) -> running (min, max) across timepoints, used for the
        #: display range that Imaris opens the image with.
        self._ranges: Dict[Tuple[int, int], Tuple[float, float]] = {}

        self.file = h5py.File(str(self.path), "w")
        self._write_root_attrs()

    # -- writing ---------------------------------------------------------

    def write_volume(self, time_index: int, channel_index: int, volume: np.ndarray) -> None:
        """Write one ``(Z, Y, X)`` volume and all of its reduced levels."""
        if not 0 <= time_index < self.n_t:
            raise IndexError(f"time_index {time_index} out of range for T={self.n_t}")
        if not 0 <= channel_index < self.n_c:
            raise IndexError(f"channel_index {channel_index} out of range for C={self.n_c}")

        volume = np.asanyarray(volume)
        if volume.ndim != 3:
            raise ValueError(f"Expected a (Z, Y, X) volume, got shape {volume.shape}")
        if volume.shape != self.shape_zyx:
            raise ValueError(f"Volume shape {volume.shape} does not match {self.shape_zyx}")

        current = self._as_imaris_dtype(volume)
        for level, level_shape in enumerate(self.levels):
            if level > 0:
                previous = self.levels[level - 1]
                # Levels only ever shrink by 1x or 2x per axis, so the ratio is
                # the reduction factor.
                factors = tuple(
                    2 if level_shape[axis] * 2 <= previous[axis] else 1
                    for axis in range(3)
                )
                current = _downsample(current, factors)
                # optimal_image_pyramid floor-divides, so an odd axis leaves one
                # extra plane/row that _downsample already dropped; trim any
                # residual so the stored array matches the declared level size.
                current = current[: level_shape[0], : level_shape[1], : level_shape[2]]
            self._write_level(level, time_index, channel_index, current)

    def _as_imaris_dtype(self, volume: np.ndarray) -> np.ndarray:
        """
        Cast to the file's sample type.

        Delegates to the shared :func:`lls_core.writers.to_output_dtype` so the
        Imaris path rounds, clips and reports NaNs exactly like the TIFF and Zarr
        writers do -- in particular it clips to bounds that are *exactly
        representable* in the source float type, which a plain
        ``clip(iinfo.min, iinfo.max)`` gets wrong for uint32 targets. Imported
        here rather than at module scope to keep this module importable without
        the rest of ``lls_core``.
        """
        if volume.dtype == self.dtype:
            return np.ascontiguousarray(volume)
        from lls_core.writers import to_output_dtype

        return np.ascontiguousarray(to_output_dtype(np.asarray(volume), self.dtype))

    def _write_level(self, level: int, time_index: int, channel_index: int, volume: np.ndarray) -> None:
        group_path = f"DataSet/ResolutionLevel {level}/TimePoint {time_index}/Channel {channel_index}"
        group = self.file.require_group(group_path)
        level_shape = self.levels[level]
        chunks = self.chunks[level]

        if "Data" in group:
            dataset = group["Data"]
        else:
            # Pad the dataspace up to whole chunks, as the reference writer does
            # (bpWriterHDF5.cxx:460-470). The real extent is recorded in the
            # ImageSize* attributes below.
            padded = tuple(
                -(-level_shape[axis] // chunks[axis]) * chunks[axis]
                for axis in range(3)
            )
            dataset = group.create_dataset(
                "Data",
                shape=padded,
                dtype=self.dtype,
                chunks=chunks,
                compression=self.compression,
                compression_opts=self.compression_opts if self.compression == "gzip" else None,
            )
            group.attrs["ImageSizeX"] = _ims_string(str(level_shape[2]))
            group.attrs["ImageSizeY"] = _ims_string(str(level_shape[1]))
            group.attrs["ImageSizeZ"] = _ims_string(str(level_shape[0]))

        dataset[: volume.shape[0], : volume.shape[1], : volume.shape[2]] = volume
        self._write_histogram(group, channel_index, level, volume)

    def _write_histogram(self, group, channel_index: int, level: int, volume: np.ndarray) -> None:
        """
        Write the 256-bin histogram Imaris uses to seed contrast settings
        (``bpWriterHDF5.cxx:216-278``).

        The reference writer also emits a 1024-bin ``Histogram1024`` when the
        source histogram has a different bin count; 256 bins is what every
        reader looks for, so only that is written here.
        """
        # Masking copies the whole volume, which for a multi-GB float deskew is
        # the difference between fitting in memory and not, so only pay for it
        # when there is actually something non-finite to exclude.
        finite = volume
        if np.issubdtype(volume.dtype, np.floating):
            mask = np.isfinite(volume)
            if not mask.all():
                finite = volume[mask]
            del mask
        if finite.size == 0:
            low, high = 0.0, 1.0
        else:
            low, high = float(finite.min()), float(finite.max())
        if high <= low:
            # A constant image would give a zero-width range, which Imaris shows
            # as fully saturated.
            high = low + 1.0

        counts, _ = np.histogram(finite, bins=256, range=(low, high))

        group.attrs["HistogramMin"] = _ims_string(_fmt_float(low))
        group.attrs["HistogramMax"] = _ims_string(_fmt_float(high))
        if "Histogram" in group:
            del group["Histogram"]
        group.create_dataset("Histogram", data=counts.astype(np.uint64))

        key = (channel_index, level)
        seen = self._ranges.get(key)
        self._ranges[key] = (low, high) if seen is None else (min(seen[0], low), max(seen[1], high))

    # -- metadata --------------------------------------------------------

    def _write_root_attrs(self) -> None:
        """``bpWriterHDF5.cxx:155-163``."""
        attrs = self.file.attrs
        attrs["ImarisDataSet"] = _ims_string("ImarisDataSet")
        attrs["ImarisVersion"] = _ims_string("5.5.0")
        attrs["DataSetInfoDirectoryName"] = _ims_string("DataSetInfo")
        attrs["ThumbnailDirectoryName"] = _ims_string("Thumbnail")
        attrs["DataSetDirectoryName"] = _ims_string("DataSet")
        attrs["NumberOfDataSets"] = np.array([1], dtype=np.uint32)

    def _write_metadata(self) -> None:
        """Populate ``/DataSetInfo`` (``bpWriterHDF5.cxx:566-660``)."""
        n_z, n_y, n_x = self.shape_zyx
        dz, dy, dx = self.voxel_size_zyx
        stamp = _fmt_time(self.recording_date)

        sections: Dict[str, Dict[str, str]] = {}

        sections["ImarisDataSet"] = {
            "NumberOfImages": "1",
            "Creator": self.application_name,
            "Version": self.application_version,
        }
        # An empty Log section still gets an Entries count.
        sections["Log"] = {"Entries": "0"}

        # Extents are the outer bounds of the voxel grid in physical units, so a
        # size-N axis spans N * pixel_size. AdaptExtents
        # (bpWriterHDF5.cxx:98-107) substitutes the voxel count when min == max,
        # which would silently give a 1 unit/voxel image, so never emit a
        # degenerate extent: fall back to a unit spacing instead.
        extents = []
        for size, spacing in ((n_x, dx), (n_y, dy), (n_z, dz)):
            span = float(size) * float(spacing)
            extents.append(span if span > 0 else float(size))

        sections["Image"] = {
            "Name": self.image_name,
            "Description": "(description not specified)",
            "X": str(n_x),
            "Y": str(n_y),
            "Z": str(n_z),
            "Unit": self.unit,
            "ExtMin0": _fmt_float(0.0, 9),
            "ExtMin1": _fmt_float(0.0, 9),
            "ExtMin2": _fmt_float(0.0, 9),
            "ExtMax0": _fmt_float(extents[0], 9),
            "ExtMax1": _fmt_float(extents[1], 9),
            "ExtMax2": _fmt_float(extents[2], 9),
            "ResampleDimensionX": "true",
            "ResampleDimensionY": "true",
            "ResampleDimensionZ": "true",
            "RecordingDate": stamp,
        }

        time_info = {
            "DatasetTimePoints": str(self.n_t),
            "FileTimePoints": str(self.n_t),
        }
        # TimePoint numbering in DataSetInfo is 1-based, unlike the group names
        # under /DataSet (bpWriterHDF5.cxx:638).
        for t in range(self.n_t):
            time_info[f"TimePoint{t + 1}"] = stamp
        sections["TimeInfo"] = time_info

        for c in range(self.n_c):
            red, green, blue = self.channel_colors[c]
            # Seed the display range from level 0 so the image opens with usable
            # contrast; this is what ImarisWriter's adjust_color_range does.
            low, high = self._ranges.get((c, 0), (0.0, 255.0))
            sections[f"Channel {c}"] = {
                "Name": self.channel_names[c],
                "Description": "(description not specified)",
                "ColorMode": "BaseColor",
                "Color": " ".join(_fmt_float(v) for v in (red, green, blue)),
                "ColorRange": f"{_fmt_float(low)} {_fmt_float(high)}",
                "ColorOpacity": _fmt_float(1.0),
                "GammaCorrection": _fmt_float(1.0),
            }

        info = self.file.require_group("DataSetInfo")
        for section_name, values in sections.items():
            # EncodeName (bpWriterHDF5.cxx:169-174): '%' then '/' are escaped so
            # a parameter name cannot create a nested group.
            group = info.require_group(section_name.replace("%", "%p").replace("/", "%s"))
            for key, value in values.items():
                group.attrs[key] = _ims_string(value)

    def close(self) -> None:
        if self.file is None:
            return
        try:
            self._write_metadata()
        finally:
            self.file.close()
            self.file = None

    def __enter__(self) -> "ImsWriter":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
