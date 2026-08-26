"""
Memory estimation for deskew pipelines.

Computes output bounding boxes from the same affine math as the deskew itself,
without touching pixel data, to estimate whether the work fits in GPU and host
memory before launching. Estimates are shape-and-dtype only, plus a safety factor
for OpenCL scratch buffers and driver overhead.

Two shapes of estimate live here:

* per-ROI (`estimate_roi`, `estimate_pipeline`), which sizes the crop-deskew path and
  answers "how many workers fit in parallel?";
* whole-volume (`estimate_deskew_volume`), which sizes the no-crop path, where the
  deskewed buffer is a single allocation whose size the user never chose.
"""
from __future__ import annotations

import logging
import math
import os
from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Generator, Iterable, List, Optional, Tuple

import numpy as np

from lls_core import DeskewDirection

if TYPE_CHECKING:
    from lls_core.models.deskew import DeskewParams
    from lls_core.models.lattice_data import LatticeData

logger = logging.getLogger(__name__)

# GPU-side buffers are float32 regardless of input dtype, because pyclesperanto
# promotes the input on transfer.
GPU_DTYPE_ITEMSIZE: int = 4

# With per-buffer accounting done explicitly, the safety factor only covers
# driver scratch and fragmentation. 1.5x is a conservative default.
DEFAULT_SAFETY_FACTOR: float = 1.5

# Headroom subtracted from total VRAM for OpenCL runtime allocations the user can't see.
DEFAULT_GPU_RESERVE_BYTES: int = 512 * 1024 * 1024

# Upstream report of the failure mode this module predicts, for maintainers:
# https://github.com/clEsperanto/pyclesperanto_prototype/issues/344


def _fmt_bytes(n: Optional[float]) -> str:
    """Format a byte count for a human-readable report; `None` renders as 'unknown'."""
    if n is None:
        return "unknown"
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(n) < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} PiB"


@dataclass
class DeviceLimits:
    """The device caps every estimate is judged against, in bytes.

    `gpu_max_alloc_bytes` is the per-buffer cap (CL_DEVICE_MAX_MEM_ALLOC_SIZE): a
    single allocation above it fails however much total VRAM is free. `None` means
    the device could not be queried, which is not evidence of a problem.
    """

    gpu_global_bytes: Optional[int]
    gpu_max_alloc_bytes: Optional[int]
    gpu_reserve_bytes: int
    host_available_bytes: Optional[int]

    @property
    def gpu_budget_bytes(self) -> Optional[int]:
        if self.gpu_global_bytes is None:
            return None
        return max(0, self.gpu_global_bytes - self.gpu_reserve_bytes)


def _query_device_limits(gpu_reserve_bytes: int = DEFAULT_GPU_RESERVE_BYTES) -> DeviceLimits:
    return DeviceLimits(
        gpu_global_bytes=get_global_mem_size(),
        gpu_max_alloc_bytes=get_max_allocation_size(),
        gpu_reserve_bytes=gpu_reserve_bytes,
        host_available_bytes=get_host_available_bytes(),
    )


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
class DeskewVolumeEstimate:
    """Memory estimate for deskewing one whole 3D volume, in bytes.

    This is the no-crop path, where the deskewed buffer is a single OpenCL
    allocation whose size the user never picked: it grows along the shear axis,
    so a modest raw stack can produce an output the GPU cannot allocate at all.
    That failure surfaces as an opaque OpenCL `OUT_OF_RESOURCES` which is not very clear
    (see `CLESPERANTO_MEMORY_ISSUE_URL`).
    """

    input_zyx: Tuple[int, int, int]
    output_zyx: Tuple[int, int, int]
    input_itemsize: int
    output_itemsize: int
    safety_factor: float
    device: DeviceLimits

    @property
    def voxels(self) -> int:
        return math.prod(self.output_zyx)

    @property
    def gpu_input_bytes(self) -> int:
        return math.prod(self.input_zyx) * GPU_DTYPE_ITEMSIZE

    @property
    def gpu_output_bytes(self) -> int:
        return self.voxels * GPU_DTYPE_ITEMSIZE

    @property
    def max_single_allocation(self) -> int:
        """Largest single OpenCL buffer. If this exceeds CL_DEVICE_MAX_MEM_ALLOC_SIZE
        the volume cannot be deskewed in one piece however much VRAM is free."""
        return max(self.gpu_input_bytes, self.gpu_output_bytes)

    @property
    def gpu_working_set(self) -> int:
        """Peak VRAM: the input and output buffers coexist for the duration of the kernel."""
        return int((self.gpu_input_bytes + self.gpu_output_bytes) * self.safety_factor)

    @property
    def host_peak_bytes(self) -> int:
        """Peak host RAM. The result is pulled back as float32 and then cast to the
        output dtype, so the raw input, the pulled array and the cast copy are all
        briefly alive at once."""
        raw = math.prod(self.input_zyx) * self.input_itemsize
        pulled = self.voxels * GPU_DTYPE_ITEMSIZE
        restored = 0 if self.output_itemsize == GPU_DTYPE_ITEMSIZE else self.voxels * self.output_itemsize
        return int((raw + pulled + restored) * self.safety_factor)

    @property
    def exceeds_max_alloc(self) -> Optional[bool]:
        """Whether a single buffer breaks the per-allocation cap. `None` if unknown."""
        if self.device.gpu_max_alloc_bytes is None:
            return None
        return self.max_single_allocation > self.device.gpu_max_alloc_bytes

    @property
    def fits_gpu(self) -> Optional[bool]:
        budget = self.device.gpu_budget_bytes
        if budget is None:
            return None
        if self.exceeds_max_alloc:
            return False
        return self.gpu_working_set <= budget

    @property
    def fits_host(self) -> Optional[bool]:
        if self.device.host_available_bytes is None:
            return None
        return self.host_peak_bytes <= self.device.host_available_bytes

    def describe_shape(self) -> str:
        """One line naming the deskewed size, always worth printing even when it fits."""
        z, y, x = self.output_zyx
        return (
            f"Deskewed volume is {z} x {y} x {x} (Z x Y x X), "
            f"{_fmt_bytes(self.gpu_output_bytes)} as float32 on the GPU and "
            f"{_fmt_bytes(self.voxels * self.output_itemsize)} once saved"
        )

    def summary_line(self) -> str:
        """A one-line version of `warnings()`, for a GUI panel where the full text is
        far too long to read. Says what is wrong and how big, and leaves the reasoning
        and remedies to the logged messages."""
        z, y, x = self.output_zyx
        size = f"Whole deskewed image: {z} x {y} x {x}, {_fmt_bytes(self.gpu_output_bytes)} on the GPU"
        # Cropping is the one fix reachable from here - it is the next tab along - and it
        # helps in all three cases, since each ROI is deskewed and pulled back on its own.
        may_fail = "Processing may fail; try cropping, or see terminal."
        if self.exceeds_max_alloc:
            return (f"{size} - larger than what this GPU can handle at once "
                    f"({_fmt_bytes(self.device.gpu_max_alloc_bytes)}). {may_fail}")
        if self.fits_gpu is False:
            return (f"{size} - needs {_fmt_bytes(self.gpu_working_set)} of the "
                    f"{_fmt_bytes(self.device.gpu_budget_bytes)} free. {may_fail}")
        if self.fits_host is False:
            return (f"{size} - needs {_fmt_bytes(self.host_peak_bytes)} of RAM, "
                    f"{_fmt_bytes(self.device.host_available_bytes)} free. {may_fail}")
        return f"{size}, {_fmt_bytes(self.voxels * self.output_itemsize)} once saved"

    def warnings(self) -> List[str]:
        """Reasons this deskew is expected to fail or thrash, most severe first.
        Empty when it fits, or when the device could not be queried.

        Each message names the deskewed size, because they are logged individually and
        a bare "not enough memory" line tells the user nothing they can act on.
        """
        avoid = ("Cropping to a region of interest avoids this, since each ROI is "
                 "deskewed on its own")
        messages: List[str] = []
        if self.exceeds_max_alloc:
            messages.append(
                f"{self.describe_shape()}. This GPU can only handle "
                f"{_fmt_bytes(self.device.gpu_max_alloc_bytes)} at once, so deskewing the "
                f"whole volume is likely to fail. {avoid}; otherwise use a GPU with more memory."
            )
        elif self.fits_gpu is False:
            messages.append(
                f"{self.describe_shape()}. With the input image that needs about "
                f"{_fmt_bytes(self.gpu_working_set)} of GPU memory, but only "
                f"{_fmt_bytes(self.device.gpu_budget_bytes)} is free. "
                f"{avoid}; otherwise use a GPU with more memory."
            )
        if self.fits_host is False:
            messages.append(
                f"{self.describe_shape()}. Holding that result in RAM needs about "
                f"{_fmt_bytes(self.host_peak_bytes)}, but only "
                f"{_fmt_bytes(self.device.host_available_bytes)} is available. Processing may "
                "swap heavily or be killed by the operating system."
            )
        return messages

    def format_report(self) -> str:
        """Return a short, human-readable estimate for a log/console."""
        lines = [
            f"Deskew estimate (no cropping): {self.input_zyx} -> {self.output_zyx}",
            f"  Deskewed image: {_fmt_bytes(self.gpu_output_bytes)}, of the {_fmt_bytes(self.device.gpu_max_alloc_bytes)} this GPU can handle at once",
            f"  GPU memory    : needs {_fmt_bytes(self.gpu_working_set)} of {_fmt_bytes(self.device.gpu_budget_bytes)} -> fits: {self.fits_gpu}",
            f"  Host RAM      : needs {_fmt_bytes(self.host_peak_bytes)} of {_fmt_bytes(self.device.host_available_bytes)} -> fits: {self.fits_host}",
        ]
        lines.extend(f"  WARNING: {message}" for message in self.warnings())
        return "\n".join(lines)


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
    # Set instead of `rois` when cropping is disabled: there are no ROIs to size,
    # but the whole-volume deskew still has to fit.
    deskew_volume: Optional[DeskewVolumeEstimate] = None

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
        CPUs). Returns 0 if any ROI violates the per-buffer cap, which no worker
        count can fix."""
        if not self.rois:
            return 1
        if self.per_buffer_violators:
            return 0
        ceiling = len(self.rois)  # never more workers than ROIs
        for cpus in (_slurm_cpu_cap(), _local_cpu_cap()):
            if cpus is not None:
                ceiling = min(ceiling, cpus)
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
        if not self.rois and self.deskew_volume is not None:
            # No cropping: the per-ROI/worker summary would be all zeroes and would
            # read as "nothing to worry about" for exactly the volumes that fail.
            return self.deskew_volume.format_report()

        lines = [
            f"Memory estimate: {len(self.rois)} ROI(s), {self.n_workers} worker(s) "
            f"(recommended: {self.recommended_workers})",
            f"  VRAM (GPU)  : needs {_fmt_bytes(self.total_gpu_bytes)} of {_fmt_bytes(self.gpu_budget_bytes)} -> fits: {self.fits_gpu}",
            f"  RAM (host)  : needs {_fmt_bytes(self.total_host_bytes)} of {_fmt_bytes(self.host_available_bytes)} -> fits: {self.fits_host}",
        ]
        if self.per_buffer_violators:
            lines.append(
                f"  ERROR: {len(self.per_buffer_violators)} ROI(s) exceed the GPU's max single "
                f"allocation ({_fmt_bytes(self.gpu_max_alloc_bytes)}); no worker count can fix this."
            )
        return "\n".join(lines)


# -- GPU / host detection -----------------------------------------------------

def _parse_device_info_mb(label: str) -> Optional[int]:
    """
    Extracts a `"<label>:  <N> MB"` field from `pyclesperanto.get_device().info` and
    returns it in bytes. Unlike the old pyclesperanto_prototype, the new pyclesperanto
    Device doesn't expose numeric byte counts directly (no `.device.max_mem_alloc_size`
    / `.global_mem_size`) - only this human-readable text blob - so parsing it is the
    only generic (backend-agnostic) way to get these figures. Returns None, MB-rounded,
    on any parse failure.
    """
    import re
    import pyclesperanto as cle
    match = re.search(rf"{re.escape(label)}:\s*(\d+)\s*MB", cle.get_device().info)
    if match is None:
        return None
    return int(match.group(1)) * 1024 * 1024


def get_max_allocation_size() -> Optional[int]:
    """
    Returns the current device's maximum single-buffer allocation size (in bytes),
    or None if unavailable. Any single OpenCL buffer larger than this will fail to
    allocate even if total global memory has room.
    """
    try:
        return _parse_device_info_mb("Maximum Buffer Size")
    except Exception:
        logger.debug("Could not determine max allocation size", exc_info=True)
        return None


def get_global_mem_size() -> Optional[int]:
    """
    Returns global memory size for the currently-selected device in bytes, or None
    if unavailable.
    """
    try:
        return _parse_device_info_mb("Global Memory Size")
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


def _local_cpu_cap() -> Optional[int]:
    """
    Bound the recommended worker count by the machine's cores. Each worker is a whole
    process doing GPU work, so more of them than cores just adds contention - and
    memory alone permits one per ROI, which on many small ROIs is far too many.
    """
    return os.cpu_count()


# -- Per-ROI bbox math (pixel-free) ------------------------------------------

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
    the deskew transforms, and the skew direction. Hoisting these out of the per-ROI
    loop avoids recomputing the affine and re-slicing for each ROI.
    """
    from lls_core.llsz_core import objective_crop_transforms

    raw_3d = lattice.get_3d_slice()
    raw_shape_zyx = tuple(int(s) for s in raw_3d.shape[-3:])
    skew_dir = lattice.skew if isinstance(lattice.skew, DeskewDirection) else DeskewDirection[str(lattice.skew)]
    transforms = objective_crop_transforms(
        raw_shape_zyx, lattice.angle, lattice.dx, lattice.dy, lattice.dz, skew_dir,
    )
    return raw_shape_zyx, transforms, skew_dir


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
    from lls_core.llsz_core import objective_crop_geometry
    from lls_core.utils import ShapeOnly, get_deskewed_shape

    if lattice.crop is None:
        raise ValueError("get_roi_bboxes requires a LatticeData with cropping enabled")

    raw_shape_zyx, transforms, skew_dir = context if context is not None else _roi_context(lattice)

    roi_shape = _roi_to_shape_array(lattice.crop.roi_list[roi_index])
    z_start, z_end = lattice.crop.z_range

    # Same helper `crop_volume_deskew` uses, so the estimate cannot describe a
    # differently-sized sub-block than the one actually read.
    geometry = objective_crop_geometry(
        raw_shape_zyx=raw_shape_zyx,
        roi_shape=roi_shape,
        z_start=z_start,
        z_end=z_end,
        angle_in_degrees=lattice.angle,
        voxel_size_x=lattice.dx,
        voxel_size_y=lattice.dy,
        voxel_size_z=lattice.dz,
        skew_dir=skew_dir,
        transforms=transforms,
    )
    crop_vol_shape_zyx: Tuple[int, int, int] = geometry.crop_vol_shape
    x0, x1 = geometry.raw_x
    y0, y1 = geometry.raw_y
    z0, z1 = geometry.raw_z
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
            ShapeOnly(input_bbox_zyx),
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
    # math.prod, not np.prod: numpy's default integer is 32-bit on Windows, so a
    # volume past 2**31 voxels - exactly the size that motivates this estimate -
    # silently overflows to a negative byte count.
    host_input_bytes = math.prod(input_bbox) * host_itemsize
    gpu_input_bytes = math.prod(input_bbox) * GPU_DTYPE_ITEMSIZE
    gpu_intermediate_bytes = math.prod(intermediate) * GPU_DTYPE_ITEMSIZE
    return RoiEstimate(
        roi_index=roi_index,
        input_bbox_zyx=input_bbox,
        intermediate_zyx=intermediate,
        host_input_bytes=host_input_bytes,
        gpu_input_bytes=gpu_input_bytes,
        gpu_intermediate_bytes=gpu_intermediate_bytes,
        safety_factor=safety_factor,
    )


def estimate_deskew_shapes(
    input_shape_zyx: Tuple[int, int, int],
    output_shape_zyx: Tuple[int, int, int],
    input_dtype: Any,
    deconvolved: bool = False,
    safety_factor: float = DEFAULT_SAFETY_FACTOR,
) -> DeskewVolumeEstimate:
    """
    Build a whole-volume estimate from shapes alone, querying the device for its limits.

    Kept separate from `estimate_deskew_volume` so it can be called from the
    `LatticeData` root validator, which has a dict of values rather than a model.
    """
    from lls_core.writers import resolve_output_dtype

    input_itemsize = int(np.dtype(input_dtype).itemsize)
    if deconvolved:
        # Deconvolved output is kept as float32 rather than cast back to the input dtype.
        output_itemsize = GPU_DTYPE_ITEMSIZE
    else:
        try:
            output_itemsize = int(resolve_output_dtype(input_dtype).itemsize)
        except Exception:
            output_itemsize = input_itemsize

    return DeskewVolumeEstimate(
        input_zyx=tuple(int(s) for s in input_shape_zyx),  # type: ignore[arg-type]
        output_zyx=tuple(int(s) for s in output_shape_zyx),  # type: ignore[arg-type]
        input_itemsize=input_itemsize,
        output_itemsize=output_itemsize,
        safety_factor=safety_factor,
        device=_query_device_limits(),
    )


def estimate_deskew_volume(
    deskew: "DeskewParams",
    safety_factor: float = DEFAULT_SAFETY_FACTOR,
) -> DeskewVolumeEstimate:
    """
    Estimate the memory needed to deskew one whole 3D volume (the no-crop path).

    Takes a `DeskewParams` rather than a full `LatticeData` so the GUI can size the
    deskew from the Deskew tab alone, before any output settings exist.

    The deskewed shape is read from the already-derived `deskew_vol_shape` rather
    than recomputed, so this reports the buffer the pipeline will actually allocate
    for either geometry (standard deskew or shear-only).
    """
    return estimate_deskew_shapes(
        input_shape_zyx=deskew.input_image.shape[-3:],
        output_shape_zyx=deskew.derived.deskew_vol_shape[-3:],
        input_dtype=deskew.input_image.dtype,
        # `deconv_enabled` only exists on LatticeData; a bare DeskewParams never deconvolves.
        deconvolved=bool(getattr(deskew, "deconv_enabled", False)),
        safety_factor=safety_factor,
    )


# Deskewed shapes already warned about. Model construction repeats for every sublattice
# and every ROI worker, all with the same geometry, so without this the same warning
# would be printed once per timepoint. Keyed on the shape rather than the message text:
# the host-RAM message quotes currently-available RAM, which drifts between timepoints
# and would defeat the dedupe exactly when the log is longest.
_warned_shapes: set = set()


def reset_warning_history() -> None:
    """Forget which sizes have been warned about, so they are reported again. For tests."""
    _warned_shapes.clear()


def warn_once(estimate: DeskewVolumeEstimate) -> List[str]:
    """Log this estimate's warnings in full, at most once per deskewed shape. Returns
    the messages logged, empty if this shape has already been reported."""
    key = tuple(estimate.output_zyx)
    if key in _warned_shapes:
        return []
    _warned_shapes.add(key)
    messages = estimate.warnings()
    for message in messages:
        logger.warning(message)
    return messages


def warn_if_deskew_may_not_fit(
    input_shape_zyx: Tuple[int, int, int],
    output_shape_zyx: Tuple[int, int, int],
    input_dtype: Any,
    deconvolved: bool = False,
    safety_factor: float = DEFAULT_SAFETY_FACTOR,
) -> List[str]:
    """
    Warn, at most once per deskewed shape, when a deskew of this size is unlikely to
    fit on the GPU. Returns the messages logged.

    Advisory only: never raises and never blocks, because the estimate is a prediction,
    the device query can be wrong or unavailable, and a user who knows their hardware
    should still be able to try.
    """
    if tuple(int(s) for s in output_shape_zyx) in _warned_shapes:
        return []
    try:
        estimate = estimate_deskew_shapes(
            input_shape_zyx, output_shape_zyx, input_dtype,
            deconvolved=deconvolved, safety_factor=safety_factor,
        )
    except Exception:
        logger.debug("Could not estimate deskew memory usage", exc_info=True)
        return []
    return warn_once(estimate)


# -- Translating opaque OpenCL failures --------------------------------------

# Substrings pyopencl puts in the message when an allocation or transfer runs out of
# room. They arrive as bare RuntimeErrors naming an API call ("clEnqueueWriteBuffer
# failed: OUT_OF_RESOURCES") with nothing about image size, which is what makes the
# real cause so hard to guess.
_OPENCL_MEMORY_MARKERS: Tuple[str, ...] = (
    "OUT_OF_RESOURCES",
    "OUT_OF_HOST_MEMORY",
    "MEM_OBJECT_ALLOCATION_FAILURE",
    "INVALID_BUFFER_SIZE",
    "INVALID_IMAGE_SIZE",
)


class DeskewMemoryError(RuntimeError):
    """A deskew failed because the image did not fit in GPU or host memory.

    Raised in place of the original OpenCL error, which it chains as `__cause__`,
    so the traceback still shows the underlying call that failed.
    """


def is_memory_error(exc: BaseException) -> bool:
    """Whether an exception is a GPU/host out-of-memory failure in disguise.

    pyopencl's own `MemoryError` does not inherit the builtin, so OpenCL failures are
    matched on the status name in the message; the builtin still catches host-side
    allocation failures from numpy.
    """
    if isinstance(exc, (MemoryError, DeskewMemoryError)):
        return True
    text = f"{type(exc).__name__}: {exc}".upper()
    return any(marker in text for marker in _OPENCL_MEMORY_MARKERS)


def _failed_buffer_detail(lattice: "LatticeData", roi_index: Optional[int]) -> List[str]:
    """Describe what just failed: the whole deskewed volume, or one ROI's deskewed
    subvolume when cropping. Quoting the whole-volume size for a failed ROI would name
    an image that was never actually created."""
    device = _query_device_limits()
    if roi_index is None:
        estimate = estimate_deskew_volume(lattice)
        shape, needed = estimate.output_zyx, estimate.gpu_working_set
        what = "The deskewed volume"
    else:
        roi = estimate_roi(lattice, roi_index)
        shape, needed = roi.intermediate_zyx, roi.gpu_working_set
        what = f"Deskewed ROI {roi_index}"
    z, y, x = shape
    return [
        f"  {what} is {z} x {y} x {x} (Z x Y x X) and needs about "
        f"{_fmt_bytes(needed)} of GPU memory.",
        f"  This GPU has {_fmt_bytes(device.gpu_global_bytes)} and can only handle "
        f"{_fmt_bytes(device.gpu_max_alloc_bytes)} at once.",
    ]


@contextmanager
def memory_errors_explained(
    lattice: Optional["LatticeData"] = None,
    operation: str = "Deskewing",
    roi_index: Optional[int] = None,
) -> Generator[None, None, None]:
    """
    Re-raise out-of-memory failures from the wrapped deskew as a `DeskewMemoryError`
    naming the size of the buffer that failed, and let every other exception through
    untouched.

    The size is only computed once a failure has happened, so this costs nothing on the
    successful path.
    """
    try:
        yield
    except Exception as exc:
        if not is_memory_error(exc):
            raise
        lines = [f"{operation} ran out of GPU or host memory."]
        if lattice is not None:
            try:
                lines.extend(_failed_buffer_detail(lattice, roi_index))
            except Exception:
                logger.debug("Could not size the image for the memory error report", exc_info=True)
        lines.append(
            "  Crop to a smaller region of interest, or use a GPU with more memory."
            if roi_index is not None else
            "  Crop to a region of interest, or use a GPU with more memory."
        )
        lines.append(f"  Underlying error: {exc}")
        raise DeskewMemoryError("\n".join(lines)) from exc


def estimate_pipeline(
    lattice: "LatticeData",
    n_workers: int,
    safety_factor: float = DEFAULT_SAFETY_FACTOR,
    gpu_reserve_bytes: int = DEFAULT_GPU_RESERVE_BYTES,
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
            deskew_volume=estimate_deskew_volume(lattice, safety_factor=safety_factor),
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
