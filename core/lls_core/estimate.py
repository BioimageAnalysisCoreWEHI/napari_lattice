"""
Memory estimation for crop-deskew pipelines.

Computes per-ROI bounding boxes from the same affine math as `crop_volume_deskew`,
without touching pixel data, to estimate whether a requested worker count fits in
GPU and host memory before launching. Estimates are shape-and-dtype only, plus a
safety factor for OpenCL scratch buffers and driver overhead.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Iterable, List, Optional, Tuple

import numpy as np

from lls_core import DeskewDirection

if TYPE_CHECKING:
    from lls_core.models.lattice_data import LatticeData

logger = logging.getLogger(__name__)

# GPU-side buffers are float32 regardless of input dtype, because pyclesperanto
# promotes the input on transfer.
GPU_DTYPE_ITEMSIZE: int = 4

# With per-buffer accounting done explicitly, the safety factor only covers
# driver scratch and fragmentation. 1.5x is a conservative default.
DEFAULT_SAFETY_FACTOR: float = 1.5


@dataclass
class RoiEstimate:
    """Per-ROI memory estimate, in bytes.

    VRAM working set is gpu_input + gpu_intermediate; the output is a view into
    the intermediate and so is not double-counted. host_input_bytes uses the raw
    dtype, since the input is held at its natural dtype before GPU transfer.
    """

    roi_index: int
    input_bbox_zyx: Tuple[int, int, int]
    intermediate_zyx: Tuple[int, int, int]
    host_input_bytes: int
    gpu_input_bytes: int
    gpu_intermediate_bytes: int
    safety_factor: float

    @property
    def max_single_allocation(self) -> int:
        """Largest single OpenCL buffer for this ROI; if it exceeds
        CL_DEVICE_MAX_MEM_ALLOC_SIZE the ROI cannot be processed at any worker count."""
        return max(self.gpu_input_bytes, self.gpu_intermediate_bytes)

    @property
    def gpu_working_set(self) -> int:
        """Estimated peak VRAM for one worker on this ROI: input + intermediate
        buffer, times the safety factor."""
        return int((self.gpu_input_bytes + self.gpu_intermediate_bytes) * self.safety_factor)

    @property
    def host_working_set(self) -> int:
        """Estimated host-side RAM for one worker (raw subvolume copy)."""
        return int(self.host_input_bytes * self.safety_factor)


@dataclass
class MemoryEstimate:
    """Summary of a memory estimate across all ROIs."""

    rois: List[RoiEstimate]
    n_workers: int
    safety_factor: float
    gpu_global_bytes: Optional[int]
    gpu_max_alloc_bytes: Optional[int]
    gpu_reserve_bytes: int
    host_available_bytes: Optional[int]

    @property
    def worker_peak_bytes(self) -> int:
        """Peak per-worker GPU working set across all assigned ROIs."""
        if not self.rois:
            return 0
        return max(r.gpu_working_set for r in self.rois)

    @property
    def host_worker_peak_bytes(self) -> int:
        """Peak per-worker host working set across all assigned ROIs."""
        if not self.rois:
            return 0
        return max(r.host_working_set for r in self.rois)

    @property
    def total_gpu_bytes(self) -> int:
        """Total simultaneous VRAM usage with `n_workers` running in parallel."""
        return self.worker_peak_bytes * self.n_workers

    @property
    def total_host_bytes(self) -> int:
        return self.host_worker_peak_bytes * self.n_workers

    @property
    def recommended_workers(self) -> int:
        """Largest worker count that fits all caps (VRAM, host RAM, ROI count,
        SLURM CPUs). Returns 0 if any ROI violates the per-buffer cap, which no
        worker count can fix."""
        if not self.rois:
            return 1
        if self.per_buffer_violators:
            return 0
        ceiling = len(self.rois)  # never more workers than ROIs
        slurm_cpus = _slurm_cpu_cap()
        if slurm_cpus is not None:
            ceiling = min(ceiling, slurm_cpus)
        gpu_budget = self.gpu_budget_bytes
        host_budget = self.host_available_bytes
        n = 1
        peak_gpu = self.worker_peak_bytes
        peak_host = self.host_worker_peak_bytes
        while n + 1 <= ceiling:
            if gpu_budget is not None and (n + 1) * peak_gpu > gpu_budget:
                break
            if host_budget is not None and (n + 1) * peak_host > host_budget:
                break
            n += 1
        return n

    @property
    def gpu_budget_bytes(self) -> Optional[int]:
        if self.gpu_global_bytes is None:
            return None
        return max(0, self.gpu_global_bytes - self.gpu_reserve_bytes)

    @property
    def per_buffer_violators(self) -> List[RoiEstimate]:
        """ROIs whose single largest buffer exceeds CL_DEVICE_MAX_MEM_ALLOC_SIZE."""
        if self.gpu_max_alloc_bytes is None:
            return []
        return [r for r in self.rois if r.max_single_allocation > self.gpu_max_alloc_bytes]

    @property
    def fits_gpu(self) -> Optional[bool]:
        budget = self.gpu_budget_bytes
        if budget is None or self.gpu_max_alloc_bytes is None:
            return None
        if self.per_buffer_violators:
            return False
        return self.total_gpu_bytes <= budget

    @property
    def fits_host(self) -> Optional[bool]:
        if self.host_available_bytes is None:
            return None
        return self.total_host_bytes <= self.host_available_bytes

    def format_report(self) -> str:
        """Return a short, human-readable memory estimate for a log/console."""
        def fmt_bytes(n: Optional[int]) -> str:
            if n is None:
                return "unknown"
            for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
                if abs(n) < 1024:
                    return f"{n:.1f} {unit}"
                n /= 1024
            return f"{n:.1f} PiB"

        lines = [
            f"Memory estimate: {len(self.rois)} ROI(s), {self.n_workers} worker(s) "
            f"(recommended: {self.recommended_workers})",
            f"  VRAM (GPU)  : needs {fmt_bytes(self.total_gpu_bytes)} of {fmt_bytes(self.gpu_budget_bytes)} -> fits: {self.fits_gpu}",
            f"  RAM (host)  : needs {fmt_bytes(self.total_host_bytes)} of {fmt_bytes(self.host_available_bytes)} -> fits: {self.fits_host}",
        ]
        if self.per_buffer_violators:
            lines.append(
                f"  ERROR: {len(self.per_buffer_violators)} ROI(s) exceed the GPU's max single "
                f"allocation ({fmt_bytes(self.gpu_max_alloc_bytes)}); no worker count can fix this."
            )
        return "\n".join(lines)


# -- GPU / host detection -----------------------------------------------------

def get_max_allocation_size() -> Optional[int]:
    """
    Returns CL_DEVICE_MAX_MEM_ALLOC_SIZE for the currently-selected OpenCL
    device, or None if unavailable. Any single OpenCL buffer larger than this
    will fail to allocate even if total global memory has room.
    """
    try:
        import pyclesperanto_prototype as cle
        return cle.get_device().device.max_mem_alloc_size
    except Exception:
        logger.debug("Could not determine CL_DEVICE_MAX_MEM_ALLOC_SIZE", exc_info=True)
        return None


def get_global_mem_size() -> Optional[int]:
    """
    Returns global memory size for the currently-selected OpenCL device in
    bytes, or None if unavailable.
    """
    try:
        import pyclesperanto_prototype as cle
        return cle.get_device().device.global_mem_size
    except Exception:
        logger.debug("Could not determine global memory size", exc_info=True)
        return None


def get_host_available_bytes() -> Optional[int]:
    """
    Best-effort estimate of available host RAM. Honours SLURM cgroup limits
    when running inside a SLURM allocation, because the cgroup cap can be
    lower than the node's total memory.
    """
    slurm = _slurm_memory_limit_bytes()
    try:
        import psutil
        available = psutil.virtual_memory().available
    except Exception:
        logger.debug("psutil not available; cannot detect host RAM", exc_info=True)
        return slurm
    if slurm is not None:
        return min(slurm, available)
    return available


def _parse_slurm_mem_bytes(value: str) -> Optional[int]:
    """
    Parse a SLURM memory value to bytes. SLURM uses binary units and defaults to
    mebibytes when no suffix is given, but may also carry a K/M/G/T suffix (e.g.
    `16384`, `16384M`, `16G`, `2T`). Returns None for unparseable values and for
    `0`, which in SLURM means "all node memory" (i.e. no explicit cap).
    """
    text = value.strip().upper()
    if not text:
        return None
    multipliers = {"K": 1024, "M": 1024 ** 2, "G": 1024 ** 3, "T": 1024 ** 4}
    if text[-1] in multipliers:
        number, mult = text[:-1], multipliers[text[-1]]
    else:
        number, mult = text, multipliers["M"]  # bare number is in MiB
    try:
        result = int(float(number) * mult)
    except ValueError:
        return None
    return result if result > 0 else None


def _slurm_memory_limit_bytes() -> Optional[int]:
    """Read SLURM memory caps from environment if present. Returns bytes."""
    per_node = os.environ.get("SLURM_MEM_PER_NODE")
    if per_node:
        return _parse_slurm_mem_bytes(per_node)

    per_cpu = os.environ.get("SLURM_MEM_PER_CPU")
    cpus = _slurm_cpu_cap()
    if per_cpu and cpus is not None:
        mem = _parse_slurm_mem_bytes(per_cpu)
        if mem is not None:
            return mem * cpus
    return None


def _slurm_cpu_cap() -> Optional[int]:
    """Read SLURM_CPUS_PER_TASK; bounds the recommended worker count."""
    val = os.environ.get("SLURM_CPUS_PER_TASK")
    if val is None:
        return None
    try:
        return int(val)
    except ValueError:
        return None


# -- Per-ROI bbox math (pixel-free) ------------------------------------------

class _ShapeOnly:
    """Lightweight stand-in for a volume; only `.shape` is read by the deskew helpers."""
    def __init__(self, shape: Tuple[int, ...]) -> None:
        self.shape = shape


def _roi_to_shape_array(roi: Any) -> np.ndarray:
    """Coerce a Roi/np.ndarray/list-of-points into the ndarray shape that
    `calculate_crop_bbox` expects."""
    if isinstance(roi, np.ndarray):
        return roi
    # `Roi` is a NamedTuple-like sequence of (y, x) vertices; np.asarray handles both
    return np.asarray(list(roi))


def _roi_context(lattice: "LatticeData") -> Tuple[Tuple[int, int, int], Any, "DeskewDirection"]:
    """
    Compute the ROI-independent inputs shared by every ROI's bbox: the raw 3D shape,
    the reverse (deskewed->raw) affine, and the skew direction. Hoisting these out of
    the per-ROI loop avoids recomputing the affine and re-slicing for each ROI.
    """
    from lls_core.llsz_core import get_inverse_affine_transform

    raw_3d = lattice.get_3d_slice()
    raw_shape_zyx = tuple(int(s) for s in raw_3d.shape[-3:])
    skew_dir = lattice.skew if isinstance(lattice.skew, DeskewDirection) else DeskewDirection[str(lattice.skew)]
    reverse_aff, _excess, _deskew = get_inverse_affine_transform(
        _ShapeOnly(raw_shape_zyx), lattice.angle, lattice.dx, lattice.dy, lattice.dz, skew_dir,
    )
    return raw_shape_zyx, reverse_aff, skew_dir


def get_roi_bboxes(
    lattice: "LatticeData",
    roi_index: int,
    context: Optional[Tuple[Tuple[int, int, int], Any, "DeskewDirection"]] = None,
) -> Tuple[Tuple[int, int, int], Tuple[int, int, int], Tuple[int, int, int]]:
    """
    Returns (input_bbox_zyx, intermediate_zyx, output_crop_zyx) for one ROI
    without touching pixel data, mirroring the shape math in `crop_volume_deskew`:
    the raw subvolume read off disk, the deskewed subvolume the GPU produces
    (usually the largest allocation, as it grows along the shear axis), and the
    final crop written out.

    `context` is the ROI-independent `_roi_context(lattice)`; it is computed here when
    omitted, but `estimate_pipeline` passes it in once to avoid per-ROI recomputation.
    """
    from lls_core.utils import calculate_crop_bbox, get_deskewed_shape

    if lattice.crop is None:
        raise ValueError("get_roi_bboxes requires a LatticeData with cropping enabled")

    raw_shape_zyx, reverse_aff, skew_dir = context if context is not None else _roi_context(lattice)

    roi_shape = _roi_to_shape_array(lattice.crop.roi_list[roi_index])
    z_start, z_end = lattice.crop.z_range
    crop_bounding_box, crop_vol_shape = calculate_crop_bbox(roi_shape, z_start, z_end)
    crop_vol_shape_zyx: Tuple[int, int, int] = (
        int(crop_vol_shape[0]),
        int(crop_vol_shape[1]),
        int(crop_vol_shape[2]),
    )

    # Map ROI corners back to raw-volume xyz coordinates and take the clipped extents.
    bbox = np.asarray([reverse_aff._matrix @ v for v in crop_bounding_box])
    mn = np.around(bbox.min(axis=0)).astype(int)
    mx = np.around(bbox.max(axis=0)).astype(int)
    nx, ny, nz = raw_shape_zyx[2], raw_shape_zyx[1], raw_shape_zyx[0]
    x0, x1 = np.clip([mn[0], mx[0]], 0, nx)
    y0, y1 = np.clip([mn[1], mx[1]], 0, ny)
    z0, z1 = np.clip([mn[2], mx[2]], 0, nz)
    input_bbox_zyx: Tuple[int, int, int] = (
        int(max(0, z1 - z0)),
        int(max(0, y1 - y0)),
        int(max(0, x1 - x0)),
    )

    # The deskew kernel allocates a buffer sized to the deskewed shape of the
    # cropped subvolume. The final crop is sliced out of this, so it coexists
    # with the input on the GPU and is usually the largest single buffer.
    if all(d > 0 for d in input_bbox_zyx):
        intermediate_shape, _ = get_deskewed_shape(
            _ShapeOnly(input_bbox_zyx),
            lattice.angle,
            lattice.dx,
            lattice.dy,
            lattice.dz,
            skew_dir,
        )
        intermediate_zyx: Tuple[int, int, int] = (
            int(intermediate_shape[0]),
            int(intermediate_shape[1]),
            int(intermediate_shape[2]),
        )
    else:
        intermediate_zyx = (0, 0, 0)

    return input_bbox_zyx, intermediate_zyx, crop_vol_shape_zyx


def _input_dtype_itemsize(lattice: "LatticeData") -> int:
    try:
        return int(lattice.input_image.dtype.itemsize)
    except Exception:
        # `cle` deskew outputs are floats by default; pick a safe fallback
        return 4


def estimate_roi(
    lattice: "LatticeData",
    roi_index: int,
    safety_factor: float = DEFAULT_SAFETY_FACTOR,
    context: Optional[Tuple[Tuple[int, int, int], Any, "DeskewDirection"]] = None,
) -> RoiEstimate:
    """Compute the per-ROI memory estimate: the host subvolume copy (raw dtype)
    plus the float32 GPU input and intermediate deskewed buffer (usually the
    binding constraint). `context` is the ROI-independent `_roi_context(lattice)`.
    """
    input_bbox, intermediate, _output_bbox = get_roi_bboxes(lattice, roi_index, context=context)
    host_itemsize = _input_dtype_itemsize(lattice)
    host_input_bytes = int(np.prod(input_bbox)) * host_itemsize
    gpu_input_bytes = int(np.prod(input_bbox)) * GPU_DTYPE_ITEMSIZE
    gpu_intermediate_bytes = int(np.prod(intermediate)) * GPU_DTYPE_ITEMSIZE
    return RoiEstimate(
        roi_index=roi_index,
        input_bbox_zyx=input_bbox,
        intermediate_zyx=intermediate,
        host_input_bytes=host_input_bytes,
        gpu_input_bytes=gpu_input_bytes,
        gpu_intermediate_bytes=gpu_intermediate_bytes,
        safety_factor=safety_factor,
    )


def estimate_pipeline(
    lattice: "LatticeData",
    n_workers: int,
    safety_factor: float = DEFAULT_SAFETY_FACTOR,
    gpu_reserve_bytes: int = 512 * 1024 * 1024,
) -> MemoryEstimate:
    """
    Build a complete memory estimate for the configured pipeline.

    `gpu_reserve_bytes` is subtracted from total global memory to leave
    headroom for OpenCL runtime allocations the user can't see.
    """
    if lattice.crop is None or not lattice.cropping_enabled:
        return MemoryEstimate(
            rois=[],
            n_workers=n_workers,
            safety_factor=safety_factor,
            gpu_global_bytes=get_global_mem_size(),
            gpu_max_alloc_bytes=get_max_allocation_size(),
            gpu_reserve_bytes=gpu_reserve_bytes,
            host_available_bytes=get_host_available_bytes(),
        )
    # Compute the ROI-independent context (raw shape + reverse affine) once, not per ROI.
    context = _roi_context(lattice)
    rois = [estimate_roi(lattice, idx, safety_factor, context=context) for idx in lattice.crop.roi_subset]
    return MemoryEstimate(
        rois=rois,
        n_workers=max(1, n_workers),
        safety_factor=safety_factor,
        gpu_global_bytes=get_global_mem_size(),
        gpu_max_alloc_bytes=get_max_allocation_size(),
        gpu_reserve_bytes=gpu_reserve_bytes,
        host_available_bytes=get_host_available_bytes(),
    )


# -- Worker chunking ---------------------------------------------------------

def chunk_roi_subset(roi_subset: Iterable[int], n_workers: int) -> List[List[int]]:
    """
    Split a list of ROI indices into roughly equal-size chunks for parallel
    workers. Returns at most `n_workers` chunks; empty chunks are dropped.
    """
    rois = list(roi_subset)
    n = max(1, min(n_workers, len(rois)))
    chunks: List[List[int]] = [[] for _ in range(n)]
    for i, roi in enumerate(rois):
        chunks[i % n].append(roi)
    return [c for c in chunks if c]
