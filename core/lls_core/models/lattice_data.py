from __future__ import annotations
from typing import Tuple, cast
from pydantic.v1 import Field, root_validator, validator
from dask.array.core import Array as DaskArray

from typing_extensions import Any, Iterable, Optional, TYPE_CHECKING, Type
from lls_core.deconvolution import pycuda_decon, skimage_decon, DeconvolutionChoice
from lls_core.llsz_core import crop_volume_deskew
from lls_core.models.crop import CropParams
from lls_core.models.deconvolution import DeconvolutionParams
from lls_core.models.deskew import DeskewParams
from lls_core.models.output import OutputParams, SaveFileType
from napari_workflows import Workflow

if TYPE_CHECKING:
    from lls_core.models.results import ImageSlice, ImageSlices, ProcessedSlice
    from lls_core.writers import Writer
    from xarray import DataArray
    from lls_core.workflow import RawWorkflowOutput
    from lls_core.types import ArrayLike
    from lls_core.models.results import WorkflowSlices

import logging

logger = logging.getLogger(__name__)

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

    progress_bar: bool = Field(
        default = True,
        description = "If true, show progress bars"
    )

    @root_validator(pre=True)
    def read_image(cls, values: dict):
        from lls_core.types import is_pathlike
        from pathlib import Path
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

        # Use the Deskew version of this validator, to do the actual image loading
        return super().read_image(values)

    @validator("input_image", pre=True, always=True)
    def incomplete_final_frame(cls, v: DataArray) -> Any:
        """
        Check final frame, if acquisition is stopped halfway through it causes failures
        This validator will remove a bad final frame
        """
        final_frame = v.isel(T=-1,C=-1, drop=True)
        try:
            final_frame.compute()
        except ValueError:
            logger.warning("Final frame is borked. Acquisition probably stopped prematurely. Removing final frame.")
            v = v.drop_isel(T=-1)
        return v
        

    @validator("workflow", pre=True)
    def parse_workflow(cls, v: Any):
        # Load the workflow from disk if it was provided as a path
        from lls_core.types import is_pathlike
        from lls_core.workflow import workflow_from_path
        from pathlib import Path

        if is_pathlike(v):
            return workflow_from_path(Path(v))
        return v

    @validator("workflow", pre=False)
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

    @validator("crop")
    def default_z_range(cls, v: Optional[CropParams], values: dict) -> Optional[CropParams]:
        from lls_core.models.utils import ignore_keyerror
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

    @validator("time_range", pre=True, always=True)
    def parse_time_range(cls, v: Any, values: dict) -> Any:
        """
        Sets the default time range if undefined
        """
        from lls_core.models.utils import ignore_keyerror
        # This skips the conversion if no image was provided, to ensure a more 
        # user-friendly error is provided, namely "image was missing"
        from collections.abc import Sequence
        with ignore_keyerror():
            default_start = 0
            default_end = values["input_image"].sizes["T"]
            if v is None:
                return range(default_start, default_end)
            elif isinstance(v, Sequence) and len(v) == 2:
                # Allow 2-tuples to be used as input for this field
                return range(v[0] or default_start, v[1] or default_end)
        return v

    @validator("channel_range", pre=True, always=True)
    def parse_channel_range(cls, v: Any, values: dict) -> Any:
        """
        Sets the default channel range if undefined
        """
        from lls_core.models.utils import ignore_keyerror
        from collections.abc import Sequence

        with ignore_keyerror():
            default_start = 0
            default_end = values["input_image"].sizes["C"]
            if v is None:
                return range(default_start, default_end)
            elif isinstance(v, Sequence) and len(v) == 2:
                # Allow 2-tuples to be used as input for this field
                return range(v[0] or default_start, v[1] or default_end)
        return v

    @validator("time_range")
    def disjoint_time_range(cls, v: range, values: dict):
        """
        Validates that the time range is within the range of channels in our array
        """
        from lls_core.models.utils import ignore_keyerror
        with ignore_keyerror():
            max_time = values["input_image"].sizes["T"]
            if v.start < 0:
                raise ValueError("The lowest valid start value is 0")
            if v.stop > max_time:
                raise ValueError(f"The highest valid time value is the length of the time axis, which is {max_time}")

        return v

    @validator("channel_range")
    def disjoint_channel_range(cls, v: range, values: dict):
        """
        Validates that the channel range is within the range of channels in our array
        """
        from lls_core.models.utils import ignore_keyerror
        with ignore_keyerror():
            max_channel = values["input_image"].sizes["C"]
            if v.start < 0:
                raise ValueError("The lowest valid start value is 0")
            if v.stop > max_channel:
                raise ValueError(f"The highest valid channel value is the length of the channel axis, which is {max_channel}")
        return v

    @validator("channel_range")
    def channel_range_subset(cls, v: Optional[range], values: dict):
        from lls_core.models.utils import ignore_keyerror
        with ignore_keyerror():
            if v is not None and (min(v) < 0 or max(v) > values["input_image"].sizes["C"]):
                raise ValueError("The output channel range must be a subset of the total available channels")
        return v

    @validator("time_range")
    def time_range_subset(cls, v: Optional[range], values: dict):
        if v is not None and (min(v) < 0 or max(v) > values["input_image"].sizes["T"]):
            raise ValueError("The output time range must be a subset of the total available time points")
        return v

    @validator("deconvolution")
    def check_psfs(cls, v: Optional[DeconvolutionParams], values: dict):
        from lls_core.models.utils import ignore_keyerror
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

        return self.input_image.isel(T=time, C=channel)

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

            yield slice.copy(update={
                "data": crop_volume_deskew(
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
                    **deconv_args
                ),
                "roi_index": roi_index
            })
            
    def _process_non_crop(self) -> Iterable[ImageSlice]:
        """
        Yields processed image slices without cropping
        """
        import pyclesperanto_prototype as cle

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

            yield slice.copy_with_data(
                cle.pull_zyx(self.deskew_func(
                    input_image=data,
                    angle_in_degrees=self.angle,
                    linear_interpolation=True,
                    voxel_size_x=self.dx,
                    voxel_size_y=self.dy,
                    voxel_size_z=self.dz
                ))
            )

    def process_workflow(self) -> WorkflowSlices:
        """
        Runs the workflow on each slice and returns the workflow results
        """
        from lls_core.workflow import get_workflow_output_name
        from lls_core.models.results import WorkflowSlices
        from lls_core.models.utils import as_tuple

        WorkflowSlices.update_forward_refs(LatticeData=LatticeData)

        def _generator() -> Iterable[ProcessedSlice[Tuple[RawWorkflowOutput, ...]]]:
            for workflow in self.generate_workflows():
                # Evaluates the workflow here.
                result = workflow.data.get(get_workflow_output_name(workflow.data))
                yield workflow.copy_with_data(as_tuple(result))

        return WorkflowSlices(
            slices=_generator(),
            lattice_data=self
        )

    def process(self) -> ImageSlices:
        """
        Execute the processing and return the result.
        This will not execute the attached workflow.
        """
        from lls_core.models.results import ImageSlices
        ImageSlices.update_forward_refs(LatticeData=LatticeData)

        if self.cropping_enabled:
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
        """
        if self.workflow:
            list(self.process_workflow().save())
        else:
            self.process().save_image()

    def process_into_image(self) -> ArrayLike:
        """
        Shortcut method for calling process, then extracting one image layer.
        This is mostly here to simplify the Workflow integration
        """
        for slice in self.process().slices:
            return slice.data
        raise Exception("No slices produced!")

    def get_writer(self) -> Type[Writer]:
        from lls_core.writers import BdvWriter, TiffWriter
        if self.save_type == SaveFileType.h5:
            return BdvWriter
        elif self.save_type == SaveFileType.tiff:
            return TiffWriter
        raise Exception("Unknown output type")
