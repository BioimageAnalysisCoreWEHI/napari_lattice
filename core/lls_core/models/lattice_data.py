from __future__ import annotations
from pathlib import Path
from typing import Tuple, cast
from dask.array.core import Array as DaskArray

from typing_extensions import Any, Iterable, Optional, TYPE_CHECKING, Type
from lls_core.deconvolution import pycuda_decon, skimage_decon, DeconvolutionChoice
from lls_core.estimate import memory_errors_explained, warn_if_deskew_may_not_fit
from lls_core.llsz_core import crop_volume_deskew
from lls_core.models.crop import CropParams
from lls_core.models.deconvolution import DeconvolutionParams
from lls_core.models.deskew import DeskewParams
from lls_core.models.output import OutputParams, SaveFileType
from napari_workflows import Workflow
from pydantic import Field, ValidationInfo, field_validator, model_validator

if TYPE_CHECKING:
    from lls_core.models.results import ImageSlice, ImageSlices, ProcessedSlice
    from lls_core.writers import Writer
    from xarray import DataArray
    from lls_core.workflow import RawWorkflowOutput
    from lls_core.types import ArrayLike
    from lls_core.models.results import WorkflowSlices
    from lls_core.estimate import MemoryEstimate

import logging

logger = logging.getLogger(__name__)


def _run_roi_chunk(lattice: "LatticeData", roi_indices: list) -> None:
    """
    Worker entry point for parallel ROI processing; module-level so it is
    picklable by `ProcessPoolExecutor`. Restricts the lattice to `roi_indices`,
    disables further parallelism, and runs the serial save path.

    When the parent stripped `input_image` before dispatch (because a file-backed
    lazy image is not picklable), re-open the file here so this worker reads only
    its own ROI crops from disk rather than the whole volume.

    Uses non-validating `.copy()` so the child does not re-run validators like
    `add_save_suffix` (which would turn `test_deskewed` into `test_deskewed_deskewed`).
    """
    if lattice.crop is None:
        raise RuntimeError("ROI worker invoked without crop configured")
    image = lattice.input_image
    if image is None:
        from lls_core.models.deskew import load_image_lazy
        if lattice.input_image_path is None:
            raise RuntimeError(
                "Parallel ROI worker received no input image and no path to re-open it from"
            )
        image = load_image_lazy(lattice.input_image_path)
    sub_crop = lattice.crop.model_copy(update={"roi_subset": list(roi_indices)})
    sub_lattice = lattice.model_copy(update={"crop": sub_crop, "process_parallel": 1, "input_image": image})
    sub_lattice.save()


def _is_lazy(image: Any) -> bool:
    """Whether an image is dask-backed, i.e. its pixels have not been read yet."""
    import dask.array as da
    return isinstance(getattr(image, "data", None), da.Array)


def _materialized_image(image: Any) -> Any:
    """
    Compute a lazy image so it can be pickled to workers; pass numpy through.

    Only for small images (PSFs) - the input image is never materialized, see
    `_input_reaches_workers`.
    """
    if _is_lazy(image):
        return image.copy(data=image.data.compute())
    return image


def _run_chunk_isolated(lattice: "LatticeData", roi_indices: list) -> None:
    """
    Run one ROI chunk in its own single-worker process pool, so that a hard worker
    death (OOM kill, segfault) breaks only this pool. Sibling chunks live in
    separate pools and are unaffected, instead of all failing together via a shared
    pool's `BrokenProcessPool`.
    """
    from concurrent.futures import ProcessPoolExecutor
    from multiprocessing import get_context

    # `spawn`, not fork: forking after pyclesperanto has created an OpenCL context
    # in the parent deadlocks the workers.
    with ProcessPoolExecutor(max_workers=1, mp_context=get_context("spawn")) as pool:
        pool.submit(_run_roi_chunk, lattice, roi_indices).result()


class LatticeData(OutputParams, DeskewParams):
    """
    Parameters for the entire deskewing process, including outputs and optional steps such as deconvolution.
    This is the recommended entry point for Python users: construct an instance of this class, and then perform the processing using methods.

    Note that none of this class's methods have any parameters: all parameters are class fields for validation purposes.
    """

    # Note: originally the save-related fields were included via composition and not inheritance
    # (similar to how `crop` and `workflow` are handled), but this was impractical for implementing validations

    deconvolution: Optional[DeconvolutionParams] = Field(
        default=None,
        description="Parameters associated with the deconvolution. If this is None, then deconvolution is disabled"
    )

    crop: Optional[CropParams] = Field(
        default=None,
        description="Cropping parameters. If this is None, then cropping is disabled"
    )
 
    workflow: Optional[Workflow] = Field(
        default=None,
        description="If defined, this is a workflow to add lightsheet processing onto",
        cli_description="Path to a JSON file specifying a napari_workflow-compatible workflow to add lightsheet processing onto"
    )

    workflow_path: Optional[Path] = Field(
        default=None,
        cli_hide=True,
        description="Internal: the filesystem path the workflow was loaded from, if any. "
                    "A `Workflow` object cannot be serialised back to a path, so output "
                    "metadata records this instead. Mirrors `input_image_path`; "
                    "not a user-facing parameter."
    )

    progress_bar: bool = Field(
        default = True,
        description = "If true, show progress bars"
    )

    # Redeclared from OutputParams with validate_default=True added. Under pydantic v1,
    # the parse_time_range/parse_channel_range validators below used always=True.
    # Pydantic v2 has no per-validator equivalent, only a field-scoped one, so
    # redeclaring the fields here keeps that forced validation scoped to LatticeData,
    # matching the old v1 behaviour.
    time_range: range = Field(
        default=None,
        description="The range of times to process. This defaults to all time points in the image array.",
        cli_description="The range of times to process, as an array with two items: the first and last time index. This defaults to all time points in the image array.",
        validate_default=True
    )
    channel_range: range = Field(
        default=None,
        description="The range of channels to process. This defaults to all time points in the image array.",
        cli_description="The range of channels to process, as an array with two items: the first and last channel index. This defaults to all channels in the image array.",
        validate_default=True
    )

    @model_validator(mode="before")
    @classmethod
    def read_image(cls, values: dict):
        from lls_core.types import is_pathlike
        input_image = values.get("input_image")
        logger.info(f"Processing File {input_image}") # this is handy for debugging
        if is_pathlike(input_image):
            if values.get("save_name") is None:
                values["save_name"] = Path(values["input_image"]).stem

            save_dir = values.get("save_dir")
            if save_dir is None:
                # By default, make the save dir be the same dir as the input
                values["save_dir"] = Path(input_image).parent
            elif is_pathlike(save_dir):
                # Convert a string path to a Path object
                values["save_dir"] = Path(save_dir)

        # A Workflow object does not remember where it was read from, so capture the
        # path now - it is the only chance - for output metadata to record.
        workflow = values.get("workflow")
        if is_pathlike(workflow):
            values["workflow_path"] = Path(workflow)

        # Use the Deskew version of this validator, to do the actual image loading
        return super().read_image(values)

    @field_validator("input_image")
    @classmethod
    def incomplete_final_frame(cls, v: DataArray) -> Any:
        """
        Check final frame, if acquisition is stopped halfway through it causes failures
        This validator will remove a bad final frame
        """
        final_frame = v.isel(T=-1,C=-1, drop=True)
        try:
            final_frame.compute()
        except (ValueError,RuntimeError):
            logger.warning("Final frame is borked. Acquisition probably stopped prematurely. Removing final frame.")
            v = v.drop_isel(T=-1)
        return v

    @model_validator(mode="after")
    def warn_deskew_size(self) -> "LatticeData":
        """
        Report the deskewed volume size at construction, and warn if it looks too large
        for this GPU.

        Deskewing shears the volume, so the output is substantially larger than the input
        along one axis: an image that loaded fine can still produce a buffer the device
        cannot allocate, and the resulting OpenCL error names no dimensions. Warning here
        means the user is told when they build the model, not partway through a long run.

        Skipped for the two paths that never allocate the whole deskewed volume, where the
        warning would be a false alarm: cropping deskews each ROI's bounding box, and MIP
        output projects straight from the raw data.
        """
        if self.crop is not None or self.save_mip:
            return self
        data = self.input_image
        derived = self.derived
        if data is None or derived is None or derived.deskew_vol_shape is None:
            return self
        warn_if_deskew_may_not_fit(
            input_shape_zyx=data.shape[-3:],
            output_shape_zyx=derived.deskew_vol_shape[-3:],
            input_dtype=data.dtype,
            deconvolved=self.deconvolution is not None,
            safety_factor=self.memory_safety_factor,
        )
        return self


    @field_validator("workflow", mode="before")
    @classmethod
    def parse_workflow(cls, v: Any):
        # Load the workflow from disk if it was provided as a path
        from lls_core.types import is_pathlike
        from lls_core.workflow import workflow_from_path
        from pathlib import Path

        if is_pathlike(v):
            return workflow_from_path(Path(v))
        return v

    @field_validator("workflow")
    @classmethod
    def validate_workflow(cls, v: Optional[Workflow]):
        from lls_core.workflow import get_workflow_output_name
        if v is not None:
            if not "deskewed_image" in v.roots():
                raise ValueError("The workflow has no deskewed_image parameter, so is not compatible with the lls processing.")
            try:
                get_workflow_output_name(v)
            except:
                raise ValueError("The workflow has multiple output tasks. Only one is currently supported.")
        return v

    @field_validator("crop")
    @classmethod
    def reject_cpu_engine_crop(cls, v: Optional[CropParams], info: ValidationInfo) -> Optional[CropParams]:
        """
        ROI cropping is only implemented for the GPU deskew engine (`crop_volume_deskew`
        is a GPU/pyclesperanto code path with no CPU counterpart yet).
        """
        from lls_core.models.utils import ignore_keyerror
        from lls_core import DeskewEngine
        values = info.data

        if v is None:
            return v
        with ignore_keyerror():
            if values["engine"] == DeskewEngine.CPU:
                raise ValueError(
                    "ROI cropping is not supported with the CPU deskew engine. Switch the engine to GPU, "
                    "or remove the crop/ROI configuration."
                )
        return v

    @field_validator("crop")
    @classmethod
    def convert_roi_units(cls, v: Optional[CropParams], info: ValidationInfo) -> Optional[CropParams]:
        """
        Bring `roi_list` into deskewed-image pixels, the unit everything downstream
        assumes. Only possible here, since the pixel size may come from the image
        metadata and so is not known when `CropParams` alone is built.
        """
        from lls_core.cropping import RoiUnits, scale_rois
        from lls_core.models.utils import ignore_keyerror
        values = info.data

        if v is None or v.roi_units == RoiUnits.Pixels:
            return v
        with ignore_keyerror():
            # dy for both axes, matching the plugin's own shape-to-ROI conversion.
            v.roi_list = scale_rois(v.roi_list, 1 / values["physical_pixel_sizes"].Y)
            # Mark the conversion done, so re-validating a copy cannot repeat it.
            v.roi_units = RoiUnits.Pixels
        return v

    @field_validator("crop")
    @classmethod
    def warn_rois_outside_image(cls, v: Optional[CropParams], info: ValidationInfo) -> Optional[CropParams]:
        """
        Say so when an ROI lies outside the deskewed image. Usually it means the units
        were wrong, and the alternative is a crop failing later inside the writer with
        an unrelated-looking message.
        """
        from lls_core.models.utils import ignore_keyerror
        values = info.data

        if v is None or not v.roi_list:
            return v
        with ignore_keyerror():
            height, width = values["derived"].deskew_vol_shape[1:]
            worst_y = max(y for roi in v.roi_list for y, _ in roi)
            worst_x = max(x for roi in v.roi_list for _, x in roi)
            if worst_y > height or worst_x > width:
                logger.warning(
                    "ROIs extend to (%.0f, %.0f) but the deskewed image is only "
                    "(%d, %d) pixels. Check roi_units: coordinates in the wrong unit "
                    "are out by the pixel size.", worst_y, worst_x, height, width
                )
        return v

    @field_validator("crop")
    @classmethod
    def default_z_range(cls, v: Optional[CropParams], info: ValidationInfo) -> Optional[CropParams]:
        from lls_core.models.utils import ignore_keyerror
        values = info.data
        if v is None:
            return v
        with ignore_keyerror():
            # Fill in missing parts of the z range
            # The max allowed value is the length of the deskew Z axis
            default_start = 0
            default_end = values["derived"].deskew_vol_shape[0]

            # Set defaults
            if v.z_range is None:
                v.z_range = (default_start, default_end)
            if v.z_range[0] is None:
                v.z_range[0] = default_start
            if v.z_range[1] is None:
                v.z_range[1] = default_end

            # Validate
            if v.z_range[1] > default_end:
                raise ValueError(f"The z-index endpoint of {v.z_range[1]} is outside the size of the z-axis ({default_end})")
            if v.z_range[0] < default_start:
                raise ValueError(f"The z-index start of {v.z_range[0]} is outside the size of the z-axis")

        return v

    @field_validator("time_range", mode="before")
    @classmethod
    def parse_time_range(cls, v: Any, info: ValidationInfo) -> Any:
        """
        Sets the default time range if undefined
        """
        from lls_core.models.utils import ignore_keyerror
        # This skips the conversion if no image was provided, to ensure a more
        # user-friendly error is provided, namely "image was missing"
        from collections.abc import Sequence
        values = info.data
        with ignore_keyerror():
            default_start = 0
            default_end = values["input_image"].sizes["T"]
            if v is None:
                return range(default_start, default_end)
            elif not isinstance(v, range) and isinstance(v, Sequence) and len(v) == 2:
                # Allow 2-tuples to be used as input for this field
                return range(v[0] or default_start, v[1] or default_end)
        return v

    @field_validator("channel_range", mode="before")
    @classmethod
    def parse_channel_range(cls, v: Any, info: ValidationInfo) -> Any:
        """
        Sets the default channel range if undefined
        """
        from lls_core.models.utils import ignore_keyerror
        from collections.abc import Sequence
        values = info.data

        with ignore_keyerror():
            default_start = 0
            default_end = values["input_image"].sizes["C"]
            if v is None:
                return range(default_start, default_end)
            elif not isinstance(v, range) and isinstance(v, Sequence) and len(v) == 2:
                # Allow 2-tuples to be used as input for this field
                return range(v[0] or default_start, v[1] or default_end)
        return v

    @field_validator("time_range")
    @classmethod
    def disjoint_time_range(cls, v: range, info: ValidationInfo):
        """
        Validates that the time range is within the range of channels in our array
        """
        from lls_core.models.utils import ignore_keyerror
        values = info.data
        with ignore_keyerror():
            max_time = values["input_image"].sizes["T"]
            if v.start < 0:
                raise ValueError("The lowest valid start value is 0")
            if v.stop > max_time:
                raise ValueError(f"The highest valid time value is the length of the time axis, which is {max_time}")

        return v

    @field_validator("channel_range")
    @classmethod
    def disjoint_channel_range(cls, v: range, info: ValidationInfo):
        """
        Validates that the channel range is within the range of channels in our array
        """
        from lls_core.models.utils import ignore_keyerror
        values = info.data
        with ignore_keyerror():
            max_channel = values["input_image"].sizes["C"]
            if v.start < 0:
                raise ValueError("The lowest valid start value is 0")
            if v.stop > max_channel:
                raise ValueError(f"The highest valid channel value is the length of the channel axis, which is {max_channel}")
        return v

    @field_validator("channel_range")
    @classmethod
    def channel_range_subset(cls, v: Optional[range], info: ValidationInfo):
        from lls_core.models.utils import ignore_keyerror
        values = info.data
        with ignore_keyerror():
            if v is not None and (min(v) < 0 or max(v) > values["input_image"].sizes["C"]):
                raise ValueError("The output channel range must be a subset of the total available channels")
        return v

    @field_validator("time_range")
    @classmethod
    def time_range_subset(cls, v: Optional[range], info: ValidationInfo):
        values = info.data
        if v is not None and (min(v) < 0 or max(v) > values["input_image"].sizes["T"]):
            raise ValueError("The output time range must be a subset of the total available time points")
        return v

    @field_validator("deconvolution")
    @classmethod
    def check_psfs(cls, v: Optional[DeconvolutionParams], info: ValidationInfo):
        from lls_core.models.utils import ignore_keyerror
        values = info.data
        if v is None:
            return v
        with ignore_keyerror():
            channels = values["input_image"].sizes["C"]
            psfs = len(v.psf)
            if psfs != channels:
                raise ValueError(f"There should be one PSF per channel, but there are {psfs} PSFs and {channels} channels.")
        return v

    @property
    def cropping_enabled(self) -> bool:
        "True if cropping should be performed"
        return self.crop is not None

    @property
    def deconv_enabled(self) -> bool:
        "True if deconvolution should be performed"
        return self.deconvolution is not None

    def __post_init__(self):
        logger.info(f"Channels: {self.channels}, Time: {self.time}")
        logger.info("If channel and time need to be swapped, you can enforce this by choosing 'Last dimension is channel' when initialising the plugin")

    def slice_data(self, time: int, channel: int) -> DataArray:
        if time > self.time:
            raise ValueError("time is out of range")
        if channel > self.channels:
            raise ValueError("channel is out of range")

        return self.apply_scan_flip(self.input_image.isel(T=time, C=channel))

    def iter_roi_indices(self) -> Iterable[Optional[int]]:
        """
        Yields region of interest indices, with a progress bar.
        This yields `None` exactly once if cropping is disabled, for compatibility.
        """
        from tqdm import tqdm
        if self.cropping_enabled and self.crop is not None:
            for index in tqdm(self.crop.roi_subset, desc="ROI", position=0, disable=not self.progress_bar):
                yield index
        else:
            yield None

    def iter_slices(self) -> Iterable[ProcessedSlice[ArrayLike]]:
        """
        Yields 3D array slices for each time, channel and region of interest.
        These are guaranteed to iterate in the following order: ROI (slowest), timepoint, channel (fastest)
        """
        from lls_core.models.results import ProcessedSlice
        from tqdm import tqdm

        for roi_index in self.iter_roi_indices():
            for time_idx, time in tqdm(enumerate(self.time_range), desc="Timepoints", total=len(self.time_range), disable=not self.progress_bar, leave=not self.cropping_enabled, position=1 if self.cropping_enabled else 0):
                for ch_idx, ch in tqdm(enumerate(self.channel_range), desc="Channels", total=len(self.channel_range), leave=False, disable=not self.progress_bar, position=2 if self.cropping_enabled else 1):
                    yield ProcessedSlice(
                        data=self.slice_data(time=time, channel=ch),
                        roi_index=roi_index,
                        time_index=time_idx,
                        time=time,
                        channel_index=ch_idx,
                        channel=ch,
                    ) 

    @property
    def n_slices(self) -> int:
        """
        Returns the number of slices that will be returned by the `iter_*` methods.
        """
        return len(self.time_range) * len(self.channel_range)

    def iter_sublattices(self, update_with: dict = {}) -> Iterable[ProcessedSlice[LatticeData]]:
        """
        Yields copies of the current LatticeData, one for each slice.
        These copies can then be processed separately.
        Args:
            update_with: dictionary of arguments to update the generated lattices with
        """
        for subarray in self.iter_slices():

            if subarray.roi_index is not None and self.crop is not None:
                crop = self.crop.copy_validate(update = {
                    "roi_subset": [subarray.roi_index]
                })
            else:
                crop = None
            new_lattice = self.copy_validate(update={
                "input_image": subarray.data,
                # The scan flip is already baked into subarray.data by slice_data, so
                # disable it here to avoid flipping the volume a second time.
                "invert_scan_direction": False,
                "time_range": range(1),
                "channel_range": range(1),
                "crop": crop,
                **update_with
            })
            yield subarray.copy_with_data(new_lattice)

    def generate_workflows(
        self,
    ) -> Iterable[ProcessedSlice[Workflow]]:
        """
        Yields copies of the input workflow, modified with the addition of deskewing and optionally,
        cropping and deconvolution
        """
        from lls_core.workflow import workflow_set
        
        if self.workflow is None:
            return

        from copy import copy
        # We make a copy of the lattice for each slice, each of which has no associated workflow
        # Also hide the progress bar for each sublattice, because we already have a global progress bar at this point
        for lattice_slice in self.iter_sublattices(update_with={"workflow": None, "progress_bar": False}):
            user_workflow = copy(self.workflow)   
            # We add a step whose result is called "input_img" that outputs a 2D image slice
            user_workflow.set(
                "deskewed_image",
                LatticeData.process_into_image,
                lattice_slice.data
            )
            # Also add channel metadata to the workflow
            for key in {"channel", "channel_index", "time", "time_index", "roi_index"}:
                workflow_set(
                    user_workflow,
                    key,
                    getattr(lattice_slice, key)
                )
            # The user can use any of these arguments as inputs to their tasks
            yield lattice_slice.copy_with_data(user_workflow)

    @property
    def deskewed_volume(self) -> DaskArray:
        from dask.array import zeros
        return zeros(self.derived.deskew_vol_shape)

    def _restore_input_dtype(self, data: ArrayLike) -> ArrayLike:
        """
        Return deskewed data to the dtype of the input image.

        As Deskewing is on GPU, data is in float 32. 
        Convert image back into input image type. 
        Deconvolved data is kept as float 32. 
        
        Workflow outputs keep whatever dtype workflow produced.
        """
        import numpy as np
        from lls_core.writers import resolve_output_dtype, to_output_dtype
        if self.deconv_enabled:
            return data
        return to_output_dtype(
            np.asarray(data), resolve_output_dtype(self.input_image.dtype)
        )

    def _process_crop(self) -> Iterable[ImageSlice]:
        """
        Yields processed image slices with cropping enabled
        """
        if self.crop is None:
            raise Exception("This function can only be called when crop is set")
        
        for slice in self.iter_slices():
            roi_index = cast(int, slice.roi_index)
            roi = self.crop.roi_list[roi_index]
            deconv_args: dict[Any, Any] = {}
            if self.deconvolution is not None:
                deconv_args = dict(
                    num_iter = self.deconvolution.decon_num_iter,
                    psf = self.deconvolution.psf[slice.channel].to_numpy(),
                    decon_processing=self.deconvolution.decon_processing
                )

            with memory_errors_explained(self, f"Deskewing ROI {roi_index}", roi_index=roi_index):
                cropped = self._restore_input_dtype(crop_volume_deskew(
                    original_volume=slice.data,
                    deconvolution=self.deconv_enabled,
                    get_deskew_and_decon=False,
                    debug=False,
                    roi_shape=list(roi),
                    linear_interpolation=True,
                    voxel_size_x=self.dx,
                    voxel_size_y=self.dy,
                    voxel_size_z=self.dz,
                    angle_in_degrees=self.angle,
                    deskewed_volume=self.deskewed_volume,
                    z_start=self.crop.z_range[0],
                    z_end=self.crop.z_range[1],
                    skew_dir=self.skew_dir,
                    coverslip_rotation=self.coverslip_rotation,
                    **deconv_args
                ))
            yield slice.copy(update={"data": cropped, "roi_index": roi_index})

    def _process_non_crop(self) -> Iterable[ImageSlice]:
        """
        Yields processed image slices without cropping
        """
        import pyclesperanto as cle

        for slice in self.iter_slices():
            data: ArrayLike = slice.data
            if isinstance(slice.data, DaskArray):
                data = slice.data.compute()
            if self.deconvolution is not None:
                if self.deconvolution.decon_processing == DeconvolutionChoice.cuda_gpu:
                    data = pycuda_decon(
                        image=data,
                        psf=self.deconvolution.psf[slice.channel].to_numpy(),
                        background=self.deconvolution.background,
                        dzdata=self.dz,
                        dxdata=self.dx,
                        dzpsf=self.dz,
                        dxpsf=self.dx,
                        num_iter=self.deconvolution.decon_num_iter
                    )
                else:
                    data = skimage_decon(
                        vol_zyx=data,
                        psf=self.deconvolution.psf[slice.channel].to_numpy(),
                        num_iter=self.deconvolution.decon_num_iter,
                        clip=False,
                        filter_epsilon=0,
                        boundary='nearest'
                    )

            # The deskewed buffer and the pull back to the host are where an oversized
            # volume actually fails, with an OpenCL error that names no dimensions.
            with memory_errors_explained(self, "Deskewing this image"):
                deskewed = self._restore_input_dtype(cle.pull(self.deskew_func(
                    input_image=data,
                    angle=self.angle,
                    voxel_size_x=self.dx,
                    voxel_size_y=self.dy,
                    voxel_size_z=self.dz
                )))
            yield slice.copy_with_data(deskewed)

    def process_workflow(self) -> WorkflowSlices:
        """
        Runs the workflow on each slice and returns the workflow results
        """
        import dask
        from lls_core.workflow import get_workflow_output_name
        from lls_core.models.results import WorkflowSlices
        from lls_core.models.utils import as_tuple

        WorkflowSlices.model_rebuild(force=True, _types_namespace={"LatticeData": LatticeData})

        def _generator() -> Iterable[ProcessedSlice[Tuple[RawWorkflowOutput, ...]]]:
            for workflow in self.generate_workflows():
                # Evaluates the workflow here. `Workflow.get()` hard-codes dask's
                # threaded scheduler, which runs GPU (pyclesperanto) steps on a
                # worker thread. pyclesperanto's OpenCL context isn't safe to use
                # across threads: once a workflow has run its GPU steps off-thread,
                # later pyclesperanto calls on the main thread silently return
                # zeroed/wrong data. Run the same task graph with dask's
                # synchronous scheduler instead, which never leaves this thread.
                result = dask.get(workflow.data._tasks, get_workflow_output_name(workflow.data))
                yield workflow.copy_with_data(as_tuple(result))

        return WorkflowSlices(
            slices=_generator(),
            lattice_data=self
        )

    def _process_mip(self) -> Iterable[ImageSlice]:
        """
        Yields a 2D deskewed maximum-intensity projection per timepoint and
        channel, as a singleton-Z `(1, Y, X)` slice so it flows through the
        existing 3D writers unchanged. The MIP is computed directly from the raw
        data (no full deskew), grid-pinned to the deskewed shape. Cropping and
        deconvolution are intentionally ignored for MIP output.
        """
        import numpy as np
        from lls_core.mip import deskew_mip
        from lls_core.models.results import ProcessedSlice

        # MIP is whole-FOV, so iterate time/channel directly (no ROI axis).
        target_shape = self.derived.deskew_vol_shape[1:]
        for time_idx, time in enumerate(self.time_range):
            for ch_idx, ch in enumerate(self.channel_range):
                raw_3d = self.slice_data(time=time, channel=ch)
                mip = deskew_mip(
                    raw_3d.data,
                    angle_in_degrees=self.angle,
                    voxel_size_z=self.dz,
                    voxel_size_y=self.dy,
                    voxel_size_x=self.dx,
                    skew=self.skew,
                    interpolation=self.mip_interpolation,
                    target_shape=target_shape,
                    frame="shear_only" if not self.coverslip_rotation else "objective",
                )
                yield ProcessedSlice(
                    data=mip[np.newaxis, :, :],  # (1, Y, X): singleton Z for the 3D writers
                    roi_index=None,
                    time_index=time_idx,
                    time=time,
                    channel_index=ch_idx,
                    channel=ch,
                )

    def process(self) -> ImageSlices:
        """
        Execute the processing and return the result.
        This will not execute the attached workflow.
        """
        from lls_core.models.results import ImageSlices
        ImageSlices.model_rebuild(force=True, _types_namespace={"LatticeData": LatticeData})

        if self.save_mip:
            if self.deconvolution is not None or self.workflow is not None or self.cropping_enabled:
                logger.warning(
                    "save_mip is enabled: the deskewed MIP is a whole-FOV projection, so the "
                    "attached cropping/deconvolution/workflow will be ignored for this output."
                )
            return ImageSlices(
                lattice_data=self,
                slices=self._process_mip()
            )
        elif self.cropping_enabled:
            return ImageSlices(
                lattice_data=self,
                slices=self._process_crop()
            )
        else:
            return ImageSlices(
                lattice_data=self,
                slices=self._process_non_crop()
            )

    def save(self) -> None:
        """
        Apply the processing, and saves the results to disk.
        Results can be found in `save_dir`.

        When `process_parallel > 1` and cropping is enabled, ROIs are distributed
        across worker processes; otherwise the original serial path runs.
        """
        # MIP output is a whole-FOV projection: it bypasses cropping, parallel-ROI
        # dispatch and workflows.
        if self.save_mip:
            self.process().save_image()
            return
        if self._use_parallel_roi_processing():
            return self._save_parallel_rois()
        if self.workflow:
            list(self.process_workflow().save())
        else:
            self.process().save_image()

    def _resolve_worker_count(self, estimate: Optional["MemoryEstimate"] = None) -> int:
        """
        Resolve the effective worker count. `process_parallel >= 1` is used as-is;
        `0` means 'auto': derive a memory-safe count from the memory estimate.

        The estimate only models the crop->deskew buffers, so 'auto' falls back to
        serial (1) when deconvolution or a workflow is attached, since their extra
        memory cannot be sized. An explicit `process_parallel` overrides this. Pass a
        precomputed `estimate` to avoid recomputing it for the auto case.
        """
        if self.process_parallel != 0:
            return self.process_parallel
        if self.deconvolution is not None or self.workflow is not None:
            logger.warning(
                "process_parallel=auto cannot size deconvolution/workflow memory; "
                "running serially. Pass an explicit process_parallel to override."
            )
            return 1
        try:
            if estimate is None:
                from lls_core.estimate import estimate_pipeline
                estimate = estimate_pipeline(self, n_workers=1, safety_factor=self.memory_safety_factor)
            return max(1, estimate.recommended_workers)
        except Exception:
            logger.warning(
                "Could not estimate a memory-safe worker count; running serially. "
                "Pass an explicit process_parallel to override.", exc_info=True
            )
            return 1

    def _use_parallel_roi_processing(self) -> bool:
        """
        Return True when the parallel-ROI save path should be used. Every route back
        to serial logs why: silence reads as "parallel ran and didn't help".
        """
        if not self.cropping_enabled or self.crop is None:
            return False
        if len(self.crop.roi_subset) <= 1:
            return False
        if self._resolve_worker_count() <= 1:
            if self.process_parallel != 1:
                # Not what was asked for: 'auto' sized itself down to one worker.
                logger.info("Running ROIs serially: the resolved worker count is 1.")
            return False
        if self.workflow is not None and not self._workflow_is_picklable():
            # Workers run in spawned processes, so the workflow must pickle.
            # Lambdas and custom-module workflows don't; run those serially.
            logger.warning(
                "process_parallel was set but the attached workflow is not "
                "picklable (e.g. lambdas or custom modules); falling back to "
                "serial ROI processing."
            )
            return False
        if not self._input_reaches_workers():
            return False
        return True

    def _input_reaches_workers(self) -> bool:
        """
        Whether the input image can reach worker processes at all: in-memory images
        are pickled, lazy ones are re-opened from their file.

        Computing a lazy image to pickle it instead would read the whole volume into
        the parent and copy it to every worker - on a 300 GB file that never finishes.
        """
        if not _is_lazy(self.input_image):
            return True
        if self._reload_reproduces_input():
            return True
        logger.warning(
            "Parallel ROI processing needs each worker to re-open the source file, "
            "which is not possible for this image - it has no single source file, or "
            "re-opening it does not reproduce the loaded image. Falling back to serial "
            "ROI processing."
        )
        return False

    def _reload_reproduces_input(self) -> bool:
        """
        Whether re-opening `input_image_path` yields the array this lattice holds.
        Workers get the path, not the pixels, so a reload that differs is silently
        wrong data.

        Matching axes and dtype are not enough: the plugin concatenates one layer per
        channel in layer-list order, and multi-scene files give several layers the same
        path. Both can match in shape and differ in content, so sample pixels too.
        """
        import numpy as np

        from lls_core.models.deskew import load_image_lazy

        if self.input_image_path is None:
            return False
        try:
            reopened = load_image_lazy(self.input_image_path)
        except Exception:
            logger.warning(
                "Could not re-open %s to check it against the loaded image",
                self.input_image_path, exc_info=True
            )
            return False

        mine = self.input_image
        # By named axis, not position: the plugin builds CTZYX where a reload is TCZYX,
        # and the pipeline addresses axes by name, so that is not a mismatch.
        if dict(reopened.sizes) != dict(mine.sizes) or reopened.dtype != mine.dtype:
            logger.info(
                "Re-opening %s gives %s (%s), but the loaded image is %s (%s)",
                self.input_image_path, dict(reopened.sizes), reopened.dtype,
                dict(mine.sizes), mine.dtype
            )
            return False

        index = {}
        if "T" in mine.dims:
            index["T"] = 0
        if "Z" in mine.dims:
            index["Z"] = mine.sizes["Z"] // 2
        channels = range(mine.sizes["C"]) if "C" in mine.dims else [None]

        def sampled(image, at: dict):
            # Sorted dims, so both planes come out laid out the same way.
            plane = image.isel(**at)
            return np.asarray(plane.transpose(*sorted(plane.dims)))

        varies = False
        for channel in channels:
            plane = index if channel is None else {**index, "C": channel}
            try:
                mine_plane = sampled(mine, plane)
                reopened_plane = sampled(reopened, plane)
            except Exception:
                logger.warning(
                    "Could not read a plane to compare %s against the loaded image",
                    self.input_image_path, exc_info=True
                )
                return False
            if not np.array_equal(mine_plane, reopened_plane):
                logger.info(
                    "Re-opening %s gives different pixels than the loaded image "
                    "(channel %s)", self.input_image_path, channel
                )
                return False
            varies = varies or bool(mine_plane.min() != mine_plane.max())

        if not varies:
            # Flat planes match anything, so their equality proves nothing.
            logger.info(
                "The planes sampled from %s are uniform, so they cannot confirm the "
                "channels match", self.input_image_path
            )
            return False
        return True

    def _workflow_is_picklable(self) -> bool:
        import pickle
        try:
            pickle.dumps(self.workflow)
            return True
        except Exception:
            return False

    def _dispatch_payload(self) -> "LatticeData":
        """
        Return a picklable copy of this lattice to hand to worker processes.

        A lazy `input_image` is stripped so each worker re-opens the file and reads
        only its own crops; `_input_reaches_workers` has already verified that reload.
        An in-memory image is pickled as-is. PSFs are small, so materialize those.
        """
        payload = self.model_copy(update={"input_image": None}) if _is_lazy(self.input_image) else self.model_copy()

        if payload.deconvolution is not None:
            payload = payload.model_copy(update={
                "deconvolution": payload.deconvolution.model_copy(update={
                    "psf": [_materialized_image(p) for p in payload.deconvolution.psf]
                })
            })
        return payload

    def _save_parallel_rois(self) -> None:
        """
        Dispatch ROI processing across worker processes: each worker runs the
        serial save() path on a chunk of `roi_subset`. Every chunk is attempted;
        if any fail, the partial output is kept and a RuntimeError is raised so
        the run fails loudly instead of being mistaken for success.
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed
        from dataclasses import replace

        from lls_core.estimate import chunk_roi_subset, estimate_pipeline

        assert self.crop is not None  # for type-checkers; gated by _use_parallel_roi_processing

        # Compute the memory estimate once and reuse it for both the worker-count
        # decision and the report. The estimator only models crop->deskew buffers, so
        # it is skipped for workflows (their extra steps aren't covered).
        estimate = None
        if self.workflow is None:
            try:
                estimate = estimate_pipeline(self, n_workers=1, safety_factor=self.memory_safety_factor)
            except Exception:
                logger.debug("Memory estimate failed; continuing without it", exc_info=True)

        chunks = chunk_roi_subset(self.crop.roi_subset, self._resolve_worker_count(estimate=estimate))
        n_workers = len(chunks)

        # Warn-only report; the user knows their hardware best.
        if self.workflow is not None:
            logger.info("Skipping memory estimate: covers deskew/crop only, not workflow steps.")
        elif estimate is not None:
            report = replace(estimate, n_workers=n_workers)
            logger.info("\n" + report.format_report())
            if report.fits_gpu is False:
                logger.warning(
                    "Memory estimate suggests the requested concurrency "
                    "may exceed available GPU memory. Proceeding anyway."
                )
            if report.fits_host is False:
                logger.warning(
                    "Memory estimate suggests the requested concurrency "
                    "may exceed available host memory. Proceeding anyway."
                )

        payload = self._dispatch_payload()

        failures: list[tuple[list[int], str]] = []
        # Each chunk runs in its own single-worker process pool (see
        # `_run_chunk_isolated`), so a hard worker death only fails that chunk. A
        # thread per chunk just waits on its child process, so a ThreadPoolExecutor
        # gives `n_workers` concurrent jobs without the GIL mattering.
        with ThreadPoolExecutor(max_workers=n_workers) as driver:
            future_to_chunk = {
                driver.submit(_run_chunk_isolated, payload, chunk): chunk for chunk in chunks
            }
            for future in as_completed(future_to_chunk):
                chunk = future_to_chunk[future]
                try:
                    future.result()
                except Exception as e:  # continue-on-error: log and move on
                    logger.exception("ROI chunk %s failed", chunk)
                    failures.append((chunk, f"{type(e).__name__}: {e}"))

        if failures:
            summary = "; ".join(f"ROIs {c} -> {msg}" for c, msg in failures)
            logger.warning(
                "Parallel ROI processing finished with %d of %d failed chunk(s): %s",
                len(failures),
                len(chunks),
                summary,
            )
            # Partial output is kept, but raise so the run fails loudly, matching
            # the serial path where an ROI error aborts the run.
            raise RuntimeError(
                f"Parallel ROI processing failed for {len(failures)} of "
                f"{len(chunks)} chunk(s): {summary}"
            )

    def process_into_image(self) -> ArrayLike:
        """
        Shortcut method for calling process, then extracting one image layer.
        This is mostly here to simplify the Workflow integration
        """
        for slice in self.process().slices:
            return slice.data
        raise Exception("No slices produced!")

    def get_writer(self) -> Type[Writer]:
        from lls_core.writers import BdvWriter, TiffWriter, OMEZarrWriter
        if self.save_type == SaveFileType.h5:
            return BdvWriter
        elif self.save_type == SaveFileType.tiff:
            return TiffWriter
        elif self.save_type == SaveFileType.omezarr:
            return OMEZarrWriter
        raise Exception("Unknown output type")
