from __future__ import annotations
# class for initializing lattice data and setting metadata
# TODO: handle scenes
from pydantic import BaseModel, DirectoryPath, Field, NonNegativeInt, root_validator, validator
from dask.array.core import Array as DaskArray
from itertools import groupby
import tifffile

from typing import Any, Iterable, List, Optional, TYPE_CHECKING
from typing_extensions import TypedDict, NotRequired, Generic, TypeVar

import pyclesperanto_prototype as cle
from tqdm import tqdm

from lls_core import DeskewDirection, DeconvolutionChoice
from lls_core.deconvolution import pycuda_decon, skimage_decon
from lls_core.llsz_core import crop_volume_deskew
from lls_core.models.crop import CropParams
from lls_core.models.deconvolution import DeconvolutionParams
from lls_core.models.output import OutputParams, SaveFileType
from lls_core.models.utils import ignore_keyerror
from lls_core.types import ArrayLike
from lls_core.models.deskew import DeskewParams
from napari_workflows import Workflow
from xarray import DataArray
from pathlib import Path

if TYPE_CHECKING:
    import pyclesperanto_prototype as cle
    from lls_core.models.deskew import DefinedPixelSizes
    from numpy.typing import NDArray

import logging

logger = logging.getLogger(__name__)

def make_filename_prefix(prefix: Optional[str] = None, roi_index: Optional[str] = None, channel: Optional[str] = None, time: Optional[str] = None) -> str:
    """
    Generates a filename for this result
    """
    components: List[str] = []
    if prefix is not None:
        components.append(prefix)
    if roi_index is not None:
        components.append(f"ROI_{roi_index}")
    if channel is not None:
        components.append(f"C{channel}")
    if time is not None:
        components.append(f"T{time}")
    return "_".join(components)

T = TypeVar("T")
S = TypeVar("S")
class SlicedData(BaseModel, Generic[T], arbitrary_types_allowed=True):
    data: T
    time_index: NonNegativeInt
    time: NonNegativeInt
    channel_index: NonNegativeInt
    channel: NonNegativeInt
    roi_index: Optional[NonNegativeInt] = None

    def copy_with_data(self, data: S) -> SlicedData[S]:
        """
        Return a modified version of this with new inner data
        """
        from typing_extensions import cast
        return cast(
            SlicedData[S],
            self.copy(update={
                "data": data
            })
        ) 
ProcessedVolume = SlicedData[ArrayLike]

class ProcessedSlices(BaseModel, arbitrary_types_allowed=True):
    #: Iterable of result slices.
    #: Note that this is a finite iterator that can only be iterated once
    slices: Iterable[ProcessedVolume]

    #: The "parent" LatticeData that was used to create this result
    lattice_data: LatticeData

    def save_image(self):
        """
        Saves result slices to disk
        """
        # TODO: refactor this into a class system, one for each format
        import numpy as np
        import npy2bdv

        for roi, roi_results in groupby(self.slices, key=lambda it: it.roi_index):
            if self.lattice_data.save_type == SaveFileType.h5:
                bdv_writer = npy2bdv.BdvWriter(
                    filename=str(self.lattice_data.make_filepath(make_filename_prefix(roi_index=roi))),
                    compression='gzip',
                    nchannels=len(self.lattice_data.channel_range),
                    subsamp=((1, 1, 1), (1, 2, 2), (2, 4, 4)),
                    overwrite=False
                )
                for result in roi_results:
                    bdv_writer.append_view(
                        np.array(result.data),
                        time=result.time,
                        channel=result.channel,
                        voxel_size_xyz=(self.lattice_data.dx, self.lattice_data.dy, self.lattice_data.new_dz),
                        voxel_units='um'
                    )
                bdv_writer.write_xml()
                bdv_writer.close()
            elif self.lattice_data.save_type == SaveFileType.tiff:
                # For each time point, we write a separate TIFF
                for time, results in groupby(roi_results, key=lambda it: it.time):
                    result_list = list(results)
                    first_result = result_list[0]
                    images_array = np.swapaxes(np.expand_dims([result.data for result in result_list], axis=0), 1, 2)
                    tifffile.imwrite(
                        str(self.lattice_data.make_filepath(make_filename_prefix(channel=first_result.channel, time=time, roi_index=roi))),
                        data = images_array,
                        bigtiff=True,
                        resolution=(1./self.lattice_data.dx, 1./self.lattice_data.dy, "MICROMETER"),
                        metadata={'spacing': self.lattice_data.new_dz, 'unit': 'um', 'axes': 'TZCYX'},
                        imagej=True
                    )

workflow: Optional[Workflow] = Field(
    default=None,
    description="If defined, this is a workflow to add lightsheet processing onto"
)

class CommonOutputArgs(TypedDict):
    # Arguments
    save_dir: NotRequired[DirectoryPath]
    save_type: SaveFileType
    time_range: range
    channel_range: range

class CommonDeskewArgs(TypedDict):
    skew: DeskewDirection
    angle: float

class CommonLatticeArgs(CommonDeskewArgs, CommonOutputArgs):
    deconvolution: Optional[DeconvolutionParams]
    crop: Optional[CropParams]
    workflow: Optional[Workflow]

class LatticeData(OutputParams, DeskewParams):
    """
    Holds data and metadata for a given image in a consistent format
    """

    # Note: originally the save-related fields were included via composition and not inheritance
    # (similar to how `crop` and `workflow` are handled), but this was impractical for implementing validations

    #: If this is None, then deconvolution is disabled
    deconvolution: Optional[DeconvolutionParams] = None

    #: If this is None, then cropping is disabled
    crop: Optional[CropParams] = None
 
    workflow: Optional[Workflow] = workflow

    @root_validator(pre=True)
    def default_save_name(cls, values: dict):
        # This needs to be a root validator to ensure it runs before the 
        # reshaping validator. We can't override that either since it's 
        # a field validator and can't modify save_name
        from lls_core.types import is_pathlike
        if values.get("save_name", None) is None and is_pathlike(values.get("input_image")):
            values["save_name"] = Path(values["input_image"]).stem
        return values

    @validator("workflow", pre=True)
    def parse_workflow(cls, v: Any):
        # Load the workflow from disk if it was provided as a path
        from lls_core.types import is_pathlike
        from lls_core.workflow import workflow_from_path
        from pathlib import Path

        if is_pathlike(v):
            return workflow_from_path(Path(v))
        return v

    @validator("crop")
    def default_z_range(cls, v: CropParams, values: dict):
        # If any part of the z range is missing, assume the user wants all Z indices
        if v is None:
            return v
        with ignore_keyerror():
            default_start = 0
            default_end = values["input_image"].sizes["Z"]
            if v.z_range is None:
                v.z_range = (default_start, default_end)
            if v.z_range[0] is None:
                v.z_range[0] = default_start
            if v.z_range[1] is None:
                v.z_range[1] = default_end
            return v

    @validator("time_range", pre=True, always=True)
    def parse_time_range(cls, v: Any, values: dict) -> Any:
        """
        Sets the default time range if undefined
        """
        # This skips the conversion if no image was provided, to ensure a more 
        # user-friendly error is provided, namely "image was missing"
        with ignore_keyerror():
            default_start = 0
            default_end = values["input_image"].sizes["T"]
            if v is None:
                return range(default_start, default_end)
            elif isinstance(v, tuple) and len(v) == 2:
                # Allow 2-tuples to be used as input for this field
                return range(v[0] or default_start, v[1] or default_end)
        return v

    @validator("channel_range", pre=True, always=True)
    def parse_channel_range(cls, v: Any, values: dict) -> Any:
        """
        Sets the default channel range if undefined
        """
        with ignore_keyerror():
            default_start = 0
            default_end = values["input_image"].sizes["C"]
            if v is None:
                return range(default_start, default_end)
            elif isinstance(v, tuple) and len(v) == 2:
                # Allow 2-tuples to be used as input for this field
                return range(v[0] or default_start, v[1] or default_end)
        return v

    @validator("time_range")
    def disjoint_time_range(cls, v: range, values: dict):
        """
        Validates that the time range is within the range of channels in our array
        """
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
        with ignore_keyerror():
            max_channel = values["input_image"].sizes["C"]
            if v.start < 0:
                raise ValueError("The lowest valid start value is 0")
            if v.stop > max_channel:
                raise ValueError(f"The highest valid channel value is the length of the channel axis, which is {max_channel}")
        return v

    @validator("channel_range")
    def channel_range_subset(cls, v: Optional[range], values: dict):
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

    def iter_slices(self) -> Iterable[SlicedData[ArrayLike]]:
        """
        Yields array slices for each time and channel of interest.

        Returns:
            An iterable of tuples. Each tuple contains (time_index, time, channel_index, channel, slice)
        """
        for time_idx, time in enumerate(self.time_range):
            for ch_idx, ch in enumerate(self.channel_range):
                yield SlicedData(
                    data=self.slice_data(time=time, channel=ch),
                    time_index=time_idx,
                    time= time,
                    channel_index=ch_idx,
                    channel=ch,
                ) 

    def iter_sublattices(self, update_with: dict = {}) -> Iterable[SlicedData[LatticeData]]:
        """
        Yields copies of the current LatticeData, one for each slice.
        These copies can then be processed separately.
        Args:
            update_with: dictionary of arguments to update the generated lattices with
        """
        for subarray in self.iter_slices():
            # Copy doesn't manu
            new_lattice = self.copy_validate(update={
                "input_image": subarray.data,
                **update_with
            })
            yield subarray.copy_with_data( new_lattice)

    def generate_workflows(
        self,
    ) -> Iterable[SlicedData[Workflow]]:
            """
            Yields copies of the input workflow, modified with the addition of deskewing and optionally,
            cropping and deconvolution
            """
            if self.workflow is None:
                return

            from copy import copy
            from lls_core.workflow import get_workflow_inputs, update_workflow
            # We make a copy of the lattice for each slice, each of which has no associated workflow
            for lattice_slice in self.iter_sublattices(update_with={"workflow": None}):
                user_workflow = copy(self.workflow)   
                user_workflow.set(
                    "deskew_image",
                    LatticeData.process_into_image,
                    lattice_slice.data
                )
                for task_name, arg_index, arg_name in get_workflow_inputs(user_workflow):
                    update_workflow(
                        user_workflow,
                        task_name,
                        arg_index,
                        "deskew_image"
                    )
                yield lattice_slice.copy_with_data(user_workflow)

    def check_incomplete_acquisition(self, volume: ArrayLike, time_point: int, channel: int):
        """
        Checks for a slice with incomplete data, caused by incomplete acquisition
        """
        import numpy as np
        if not isinstance(volume, DaskArray):
            return volume
        orig_shape = volume.shape
        raw_vol = volume.compute()
        if raw_vol.shape != orig_shape:
            logger.warn(f"Time {time_point}, channel {channel} is incomplete. Actual shape {orig_shape}, got {raw_vol.shape}")
            z_diff, y_diff, x_diff = np.subtract(orig_shape, raw_vol.shape)
            logger.info(f"Padding with{z_diff,y_diff,x_diff}")
            raw_vol = np.pad(raw_vol, ((0, z_diff), (0, y_diff), (0, x_diff)))
            if raw_vol.shape != orig_shape:
                raise Exception(f"Shape of last timepoint still doesn't match. Got {raw_vol.shape}")
            return raw_vol

    @property
    def deskewed_volume(self) -> DaskArray:
        from dask.array import zeros
        return zeros(self.deskew_vol_shape)

    def _process_crop(self) -> Iterable[ProcessedVolume]:
        """
        Yields processed image slices with cropping enabled
        """
        if self.crop is None:
            raise Exception("This function can only be called when crop is set")
            
        # We have an extra level of iteration for the crop path: iterating over each ROI
        for roi_index, roi in enumerate(tqdm(self.crop.roi_list, desc="ROI:", position=0)):
            # pass arguments for save tiff, callable and function arguments
            logger.info(f"Processing ROI {roi_index}")
            
            deconv_args: dict[Any, Any] = {}
            if self.deconvolution is not None:
                deconv_args = dict(
                    num_iter = self.deconvolution.psf_num_iter,
                    psf = self.deconvolution.psf,
                    decon_processing=self.deconvolution.decon_processing
                )

            for slice in self.iter_slices():
                yield slice.copy_with_data(
                    crop_volume_deskew(
                        original_volume=slice.data,
                        deconvolution=self.deconv_enabled,
                        get_deskew_and_decon=False,
                        debug=False,
                        roi_shape=list(roi),
                        linear_interpolation=True,
                        voxel_size_x=self.dx,
                        voxel_size_y=self.dy,
                        voxel_size_z=self.dy,
                        angle_in_degrees=self.angle,
                        deskewed_volume=self.deskewed_volume,
                        z_start=self.crop.z_range[0],
                        z_end=self.crop.z_range[1],
                        **deconv_args
                    )
                )
                
    def _process_non_crop(self) -> Iterable[ProcessedVolume]:
        """
        Yields processed image slices without cropping
        """
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
                        num_iter=self.deconvolution.psf_num_iter
                    )
                else:
                    data = skimage_decon(
                        vol_zyx=data,
                        psf=self.deconvolution.psf[slice.channel].to_numpy(),
                        num_iter=self.deconvolution.psf_num_iter,
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

    def _process_workflow(self) -> ProcessedSlices:
        outputs = []
        for workflow in self.generate_workflows():
            for leaf in workflow.data.leafs():
                outputs.append(
                    workflow.copy_with_data(
                        workflow.data.get(leaf)
                    )
                )

        return ProcessedSlices(
            slices = outputs,
            lattice_data=self
        )

    def process(self) -> ProcessedSlices:
        """
        Execute the processing and return the result.
        This is the main public API for processing
        """
        ProcessedSlices.update_forward_refs()

        if self.workflow is not None:
            return self._process_workflow()
        elif self.cropping_enabled:
            return ProcessedSlices(
                lattice_data=self,
                slices=self._process_crop()
            )
        else:
            return ProcessedSlices(
                lattice_data=self,
                slices=self._process_non_crop()
            )

    def process_into_image(self) -> ArrayLike:
        """
        Shortcut method for calling process, then extracting one image layer.
        This is mostly here to simplify the Workflow integration
        """
        for slice in self.process().slices:
            return slice.data
        raise Exception("No slices produced!")
