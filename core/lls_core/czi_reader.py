"""
Fast, lazy CZI reading shared by the napari plugin and the CLI.

bioio-czi builds its dask array eagerly, so merely reading ``BioImage.dims`` (or
``.shape`` / ``.dtype`` / ``.channel_names``, which all route through
``xarray_dask_data``) constructs a task per plane, and every slice then pays graph
work over the result. This module derives the dimensions from metadata and wraps a
lazy facade in ``da.from_array`` instead.

Reads go through pylibCZIrw, the library bioio-czi itself uses by default, so output
is byte-identical to ``xarray_dask_data``. Do not substitute another CZI library
here: aicspylibczi returns subblocks without applying their logical offset, which
silently misregisters files that record stage drift between timepoints.

Every entry point returns ``None`` rather than raising, so callers fall back to bioio.

This is a temporary workaround for an upstream bioio-czi issue. Once bioio-czi builds
its array lazily, delete this module and revert its call sites in ``lls_core.types``,
``lls_core.models.deskew`` and ``napari_lattice.reader``.
"""
from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from itertools import product
from logging import getLogger
from typing import Any, Optional

import dask.array as da
import numpy as np
from xarray import DataArray

from bioio import BioImage

logger = getLogger(__name__)

# Planes per chunk along Z. A dask-graph-size knob, not a read-volume one: dask pushes
# single-plane indexing down into ``from_array``, so a coarse chunk still reads exactly
# one plane. It matters because dask's per-slice cost scales with the number of chunks
# in the array, which otherwise dominates the read itself on large files.
_Z_CHUNK = 32

_pool_lock = threading.Lock()
_pool_instance: Optional[ThreadPoolExecutor] = None


def _pool() -> ThreadPoolExecutor:
    """
    Shared pool for whole-block reads. Module-level so that opening many files does
    not accumulate a pool per array.
    """
    global _pool_instance
    with _pool_lock:
        if _pool_instance is None:
            _pool_instance = ThreadPoolExecutor(max_workers=8, thread_name_prefix="lls-czi")
        return _pool_instance


def _decline(path: Any, reason: str) -> None:
    """Log why the fast path bowed out. Callers return this (None) to fall back."""
    logger.debug("CZI fast path declined for %s: %s", path, reason)
    return None


class CziPlanes:
    """
    A numpy-like facade over pylibCZIrw so dask can slice a CZI lazily: one ``read()``
    per plane, and several planes only when a whole block is genuinely requested.

    Readers are thread-local rather than shared behind a lock, so dask can read planes
    in parallel. Neither the reader nor the thread-local is picklable, so this must
    only be computed under an in-process threaded scheduler; parallel ROI workers
    re-open the file instead (see lls_core.models.lattice_data._dispatch_payload).
    """

    def __init__(self, path: str, shape, dtype, plane_dims, scene: Optional[int]):
        self.path = str(path)
        self.shape = tuple(int(s) for s in shape)
        self.dtype = np.dtype(dtype)
        self.ndim = len(self.shape)
        self._plane_dims = tuple(plane_dims)
        self._scene = scene
        self._local = threading.local()

    def _reader(self):
        # Keep the context manager alive next to the reader: letting it be garbage
        # collected closes the file underneath us, and the next read fails with
        # "CZIReader is not operational (must call 'Open' first)".
        held = getattr(self._local, "czi", None)
        if held is None:
            from pylibCZIrw import czi as pyczi

            ctx = pyczi.open_czi(self.path)
            held = self._local.czi = (ctx, ctx.__enter__())
        return held[1]

    def _plane(self, coords, y_key, x_key) -> np.ndarray:
        raw = np.asarray(self._reader().read(plane=coords, scene=self._scene))
        # pylibCZIrw returns (Y, X, samples); drop the trailing axis explicitly
        # rather than squeezing, which would also eat a genuine length-1 Y or X.
        if raw.ndim == 3 and raw.shape[-1] == 1:
            raw = raw[..., 0]
        return raw[y_key, x_key]

    def __getitem__(self, key):
        if not isinstance(key, tuple):
            key = (key,)
        # dask passes a plain integer for axes it has already fused away and expects
        # those axes dropped (numpy semantics); a slice means the axis is retained.
        # A block may span several indices along an axis (see _Z_CHUNK), so resolve
        # each leading axis to the list of indices it covers.
        axes = []
        for dim, part, size in zip(self._plane_dims, key, self.shape):
            if isinstance(part, slice):
                axes.append((dim, list(range(*part.indices(int(size)))), True))
            else:
                axes.append((dim, [int(part)], False))

        lead = len(self._plane_dims)
        y_key = key[lead] if len(key) > lead else slice(None)
        x_key = key[lead + 1] if len(key) > lead + 1 else slice(None)

        dims = [dim for dim, _, _ in axes]
        combos = list(product(*[indices for _, indices, _ in axes]))

        if len(combos) == 1:
            planes = [self._plane(dict(zip(dims, combos[0])), y_key, x_key)]
        else:
            # Only whole-block requests land here; a 2D slice never does. Pooling is
            # not worth its overhead for a single plane.
            planes = list(
                _pool().map(lambda c: self._plane(dict(zip(dims, c)), y_key, x_key), combos)
            )

        block = np.stack(planes).reshape(
            [len(indices) for _, indices, _ in axes] + list(planes[0].shape)
        )
        # Drop the axes dask indexed with an integer.
        out = block[tuple(slice(None) if keep else 0 for _, _, keep in axes)]
        return out


def czi_metadata(path: str, image: BioImage) -> Optional[dict]:
    """
    Derive dimension order, shape, dtype and channel names without touching
    ``image.dims`` / ``image.dask_data``, either of which would build bioio's graph.

    Channel names come from bioio-czi's own helper so naming matches it exactly.
    ``BioImage`` normalises to TCZYX, padding absent axes to length 1; the shape below
    reproduces that. The tests pin all of this against bioio for the bundled CZIs.

    Returns ``None`` for non-CZIs, mosaics, files carrying dimensions outside TCZYX,
    or anything unexpected.
    """
    if not str(path).lower().endswith(".czi"):
        return None
    try:
        from pylibCZIrw import czi as pyczi
        # Internal bioio-czi API, reused so channel naming and dtype mapping stay
        # identical to bioio's. A bioio-czi upgrade may move these; the metadata tests
        # are what catch it.
        from bioio_czi.bounding_box import size as _bbox_size
        from bioio_czi.channels import get_channel_names
        from bioio_czi.pylibczirw_reader.reader import PIXEL_DICT
    except Exception:
        logger.debug("bioio-czi internals unavailable; using bioio", exc_info=True)
        return None

    try:
        n_scenes = max(len(getattr(image, "scenes", None) or [None]), 1)
        scene_idx = int(getattr(image, "current_scene_index", 0) or 0)

        with pyczi.open_czi(str(path)) as czi:
            bbox = dict(czi.total_bounding_box)
            m_lo, m_hi = bbox.get("M", (0, 1))
            if int(m_hi) - int(m_lo) > 1:
                return _decline(path, "mosaic")
            # bioio prefers the scene rectangle for X/Y when one exists; it is what
            # carries the drift margin on offset CZIs.
            rect = czi.scenes_bounding_rectangle.get(scene_idx)
            if rect is None:
                rect = czi.total_bounding_rectangle
            pixel_type = czi.get_channel_pixel_type(0)

        dtype = np.dtype(PIXEL_DICT[str(pixel_type).lower()])
        names = get_channel_names(image.reader.metadata, scene_idx, bbox)
        n_channels = len(names) if names else max(_bbox_size(bbox, "C"), 1)
        shape = (
            max(_bbox_size(bbox, "T"), 1),
            int(n_channels),
            max(_bbox_size(bbox, "Z"), 1),
            int(rect.h),
            int(rect.w),
        )
    except Exception:
        logger.debug("CZI metadata unavailable for %s", path, exc_info=True)
        return None

    # CZI dimensions outside TCZYX (H, V, B, ...). We have never seen one, so we
    # decline rather than guess how it folds into the TCZYX bioio normalises to.
    extra_dims = sorted(d for d in bbox if d not in ("X", "Y", "Z", "T", "C", "M"))
    if extra_dims:
        return _decline(path, f"unsupported dimensions {extra_dims}")

    return {
        "order": "TCZYX",
        "shape": shape,
        "dtype": dtype,
        "channel_names": list(names) if names else [],
        "scene": scene_idx if n_scenes > 1 else None,
        "scene_index": scene_idx,
        "n_scenes": n_scenes,
    }


def czi_dask_array(path: str, image: BioImage, meta: Optional[dict] = None):
    """
    A lazily-read dask array for a CZI, chunked ``_Z_CHUNK`` planes deep along Z.
    Multi-scene files are supported: the caller iterates scenes and calls this once
    per scene, reading whichever one bioio has selected.

    Returns ``None`` whenever ``czi_metadata`` declines, or on any construction error.
    """
    if meta is None:
        meta = czi_metadata(path, image)
    if meta is None:
        return None
    try:
        from dask.base import tokenize
    except Exception:
        logger.debug("dask.base.tokenize unavailable", exc_info=True)
        return None

    order = meta["order"]
    dtype = meta["dtype"]
    scene = meta["scene"]
    shape = meta["shape"]
    sizes = dict(zip(order, shape))

    plane_dims = tuple(order[:-2])
    try:
        source = CziPlanes(str(path), shape, dtype, plane_dims, scene)
        arr = da.from_array(
            source,
            # Coarse along Z (see _Z_CHUNK), one index along every other leading axis.
            chunks=tuple(
                min(_Z_CHUNK, int(sizes[dim])) if dim == "Z" else 1 for dim in plane_dims
            ) + (int(sizes["Y"]), int(sizes["X"])),
            name="lls-czi-" + tokenize(str(path), meta["scene_index"], shape, str(dtype)),
            meta=np.empty((0,) * len(order), dtype),
        )
    except Exception:
        logger.debug("could not build CZI array for %s", path, exc_info=True)
        return None

    return arr



def czi_path_of(image: BioImage) -> Optional[str]:
    """The file a BioImage was opened from, or None. bioio has no public accessor."""
    path = getattr(getattr(image, "reader", None), "_path", None)
    return str(path) if path else None


def czi_xarray(path: str, image: BioImage, meta: Optional[dict] = None) -> Optional[DataArray]:
    """
    ``BioImage.xarray_dask_data`` for a CZI, without bioio's eager graph.

    Attaches ``dims`` but not coords: callers use ``.dims``, ``.sizes`` and ``.isel``,
    never coords, and reproducing bioio's coordinate handling would buy nothing.
    """
    if meta is None:
        meta = czi_metadata(path, image)
    if meta is None:
        return None
    arr = czi_dask_array(path, image, meta)
    if arr is None:
        return None
    try:
        return DataArray(arr, dims=tuple(meta["order"]))
    except Exception:
        logger.debug("could not wrap CZI array as DataArray for %s", path, exc_info=True)
        return None
