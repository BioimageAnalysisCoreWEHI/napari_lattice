from __future__ import annotations
# class for initializing lattice data and setting metadata
# TODO: handle scenes
from pydantic import BaseModel, Field, NonNegativeInt, NonNegativeFloat, root_validator, validator
from aicsimageio.aics_image import AICSImage
from aicsimageio.dimensions import Dimensions
# from numpy.typing import NDArray
import math
from dask.array.core import Array as DaskArray
import dask as da
from itertools import groupby
import tifffile
from pydantic_numpy import NDArray

from typing import Any, Iterable, List, Literal, Optional, TYPE_CHECKING, Tuple, TypeVar, Union
from typing_extensions import TypedDict
from pathlib import Path

from aicsimageio.types import PhysicalPixelSizes
import pyclesperanto_prototype as cle
from tqdm import tqdm

from lls_core import DeskewDirection, DeconvolutionChoice, SaveFileType
from lls_core.deconvolution import pycuda_decon, skimage_decon
from lls_core.llsz_core import crop_volume_deskew
from lls_core.utils import get_deskewed_shape
from lls_core.types import ArrayLike
from napari_workflows import Workflow

if TYPE_CHECKING:
    import pyclesperanto_prototype as cle
    from napari.types import ShapesData

import logging

logger = logging.getLogger(__name__)

T = TypeVar("T")
def raise_if_none(obj: Optional[T], message: str) -> T:
    if obj is None:
        raise TypeError(message)
    return obj

class DefaultMixin(BaseModel):
    """
    Adds a method for retrieving default values from a BaseModel
    """

    @classmethod
    def get_default(cls, field_name: str):
        return cls.__fields__[field_name].get_default()

class ProcessedVolume(BaseModel, arbitrary_types_allowed=True):
    """
    A slice of the image processing result
    """
    time_index: NonNegativeInt
    time: NonNegativeInt
    channel_index: NonNegativeInt
    channel: NonNegativeInt
    data: ArrayLike
    roi_index: Optional[NonNegativeInt] = None

class ProcessedSlices(BaseModel):
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
        from pathlib import Path
        import npy2bdv

        for roi, roi_results in groupby(self.slices, key=lambda it: it.roi_index):
            if self.lattice_data.save_type == SaveFileType.h5:
                bdv_writer = npy2bdv.BdvWriter(
                    make_filename_prefix(prefix=self.lattice_data.save_name, roi_index=roi),
                    compression='gzip',
                    nchannels=len(self.lattice_data.channel_range),
                    subsamp=((1, 1, 1), (1, 2, 2), (2, 4, 4)),
                    overwrite=False
                )
                for result in roi_results:
                    bdv_writer.append_view(
                        result.data,
                        time=result.time,
                        channel=result.channel,
                        voxel_size_xyz=(self.lattice_data.dx, self.lattice_data.dy, self.lattice_data.new_dz),
                        voxel_units='um'
                    )
            elif self.lattice_data.save_type == SaveFileType.tiff:
                # For each time point, we write a separate TIFF
                for time, results in groupby(roi_results, key=lambda it: it.time):
                    result_list = list(results)
                    first_result = result_list[0]
                    images_array = np.swapaxes(np.expand_dims([result.data for result in result_list], axis=0), 1, 2)
                    tifffile.imwrite(
                        file = Path(make_filename_prefix(channel=first_result.channel, time=time, roi_index=roi)).with_suffix("tiff"),
                        data = images_array,
                        bigtiff=True,
                        resolution=(1./self.lattice_data.dx, 1./self.lattice_data.dy, "MICROMETER"),
                        metadata={'spacing': self.lattice_data.new_dz, 'unit': 'um', 'axes': 'TZCYX'},
                        imagej=True
                    )

def make_filename_prefix(prefix: Optional[str] = None, roi_index: Optional[int] = None, channel: Optional[int] = None, time: Optional[int] = None) -> str:
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

class DefinedPixelSizes(DefaultMixin):
    """
    Like PhysicalPixelSizes, but it's a dataclass, and
    none of its fields are None
    """
    X: NonNegativeFloat = 0.14
    Y: NonNegativeFloat = 0.14
    Z: NonNegativeFloat = 0.3

class DeconvolutionParams(BaseModel, arbitrary_types_allowed=True):
    """
    Parameters for the optional deconvolution step
    """
    decon_processing: DeconvolutionChoice = DeconvolutionChoice.cpu
    psf: List[NDArray] = []
    psf_num_iter: NonNegativeInt = 10
    # TODO: handle this
    # otf_path: List = []
    # Background value to subtract
    background: Union[float, Literal["auto", "second_last"]] = 0 

class CropParams(BaseModel, arbitrary_types_allowed=True):
    """
    Parameters for the optional cropping step
    """
    roi_layer_list: ShapesData
    z_start: NonNegativeInt = 0
    z_end: NonNegativeInt = 1

class OutputParams(DefaultMixin, arbitrary_types_allowed=True):
    #: The directory where this data will be saved
    save_dir: Path

    #: The file name to save this as, without the directory name or file extension
    save_name: str

    #: The range of times to process
    time_range: range

    #: The range of channels to process
    channel_range: range

    #: The data type to save the result as
    save_type: SaveFileType = SaveFileType.h5

    @validator("time_range")
    def default_time_range(cls, v: Any, values: dict) -> range:
        """
        Sets the default time range if undefined
        """
        if v is None:
            return range(values["dims"].T + 1)
        return v

    @validator("channel_range")
    def default_channel_range(cls, v: Any, values: dict) -> range:
        """
        Sets the default channel range if undefined
        """
        if v is None:
            return range(values["dims"].C + 1)
        return v

class DeskewParams(DefaultMixin, arbitrary_types_allowed=True):
    #: A 3-5D array containing the image data
    data: ArrayLike

    #: Dimensions of `data`
    dims: Dimensions

    #: Dimensions of the deskewed output
    deskew_vol_shape: Tuple[int, ...] = Field(init_var=False)

    deskew_affine_transform: cle.AffineTransform3D = Field(init_var=False)

    #: Geometry of the light path
    skew: DeskewDirection = DeskewDirection.Y
    angle: float = 30.0

    #: Pixel size in microns
    physical_pixel_sizes: DefinedPixelSizes = Field(default_factory=DefinedPixelSizes)

    @root_validator(pre=True)
    def set_deskew(cls, values: dict) -> dict:
        """
        Sets the default deskew shape values if the user has not provided them
        """
        # process the file to get shape of final deskewed image
        if values.get('deskew_vol_shape') is None:
            if values.get('deskew_affine_transform') is None:
                # If neither has been set, calculate them ourselves
                values["deskew_vol_shape"], values["deskew_affine_transform"] = get_deskewed_shape(values["data"], values["angle"], values["physical_pixel_sizes"].X, values["physical_pixel_sizes"].Y, values["physical_pixel_sizes"].Z, values["skew"])
            else:
                raise ValueError("deskew_vol_shape and deskew_affine_transform must be either both specified or neither specified")
        return values


class LatticeData(OutputParams, DeskewParams, arbitrary_types_allowed=True):
    """
    Holds data and metadata for a given image in a consistent format
    """

    # Note: originally the save-related fields were included via composition and not inheritance
    # (similar to how `crop` and `workflow` are handled), but this was impractical for implementing validations

    #: If this is None, then deconvolution is disabled
    deconvolution: Optional[DeconvolutionParams] = None

    #: If this is None, then cropping is disabled
    crop: Optional[CropParams] = None
 
    #: If defined, this is a workflow to add lightsheet processing onto
    workflow: Optional[Workflow] = None

    @validator("time_range")
    def disjoint_time_range(cls, v: range, values: dict):
        """
        Validates that the time range is within the range of channels in our array
        """
        max_time = values["dims"].T
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
        max_channel = values["dims"].T
        if v.start < 0:
            raise ValueError("The lowest valid start value is 0")
        if v.stop > max_channel:
            raise ValueError(f"The highest valid channel value is the length of the channel axis, which is {max_channel}")
        return v

    @validator("channel_range")
    def channel_range_subset(cls, v: range, values: dict):
        if min(v) < 0 or max(v) > values["dims"].C:
            raise ValueError("The output channel range must be a subset of the total available channels")

    @validator("time_range")
    def time_range_subset(cls, v: range, values: dict):
            if min(v) < 0 or max(v) > values["dims"].T:
                raise ValueError("The output time range must be a subset of the total available time points")

    # Hack to ensure that .skew_dir behaves identically to .skew
    @property
    def skew_dir(self) -> DeskewDirection:
        return self.skew

    @skew_dir.setter
    def skew_dir(self, value: DeskewDirection):
        self.skew = value

    @property
    def deskew_func(self):
        # Chance deskew function absed on skew direction
        if self.skew == DeskewDirection.Y:
            return cle.deskew_y
        elif self.skew == DeskewDirection.X:
            return cle.deskew_x
        else:
            raise ValueError()

    @property
    def dx(self) -> float:
        return self.physical_pixel_sizes.X

    @dx.setter
    def dx(self, value: float):
        self.physical_pixel_sizes.X = value

    @property
    def dy(self) -> float:
        return self.physical_pixel_sizes.Y

    @dy.setter
    def dy(self, value: float):
        self.physical_pixel_sizes.Y = value

    @property
    def dz(self) -> float:
        return self.physical_pixel_sizes.Z

    @dz.setter
    def dz(self, value: float):
        self.physical_pixel_sizes.Z = value

    def get_angle(self) -> float:
        return self.angle

    def set_angle(self, angle: float) -> None:
        self.angle = angle

    def set_skew(self, skew: DeskewDirection) -> None:
        self.skew = skew

    @property
    def cropping_enabled(self) -> bool:
        "True if cropping should be performed"
        return self.crop is not None

    @property
    def deconv_enabled(self) -> bool:
        "True if deconvolution should be performed"
        return self.deconvolution is not None

    @property
    def time(self) -> int:
        """Number of time points"""
        return self.dims.T

    @property
    def channels(self) -> int:
        """Number of channels"""
        return self.dims.C

    @property
    def new_dz(self):
        return math.sin(self.angle * math.pi / 180.0) * self.dz

    def __post_init__(self):
        logger.info(f"Channels: {self.channels}, Time: {self.time}")
        logger.info("If channel and time need to be swapped, you can enforce this by choosing 'Last dimension is channel' when initialising the plugin")

    def slice_data(self, time: int, channel: int) -> ArrayLike:
        if time > self.time:
            raise ValueError("time is out of range")
        if channel > self.channels:
            raise ValueError("channel is out of range")

        if len(self.dims.shape) == 3:
            return self.data
        elif len(self.dims.shape) == 4:
            return self.data[time, :, :, :]
        elif len(self.dims.shape) == 5:
            return self.data[time, channel, :, :, :]

        raise Exception("Lattice data must be 3-5 dimensions")

    def iter_slices(self) -> Iterable[Tuple[int, int, int, int, ArrayLike]]:
        """
        Yields array slices for each time and channel of interest.

        Returns:
            An iterable of tuples. Each tuple contains (time_index, time, channel_index, channel, slice)
        """
        for time_idx, time in enumerate(self.time_range):
            for ch_idx, ch in enumerate(self.channel_range):
                yield time_idx, time, ch_idx, ch, self.slice_data(time=time, channel=ch)

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
        return da.zeros(self.deskew_vol_shape)

    def _process_crop(self) -> Iterable[ProcessedVolume]:
        """
        Yields processed image slices with cropping enabled
        """
        if self.crop is None:
            raise Exception("This function can only be called when crop is set")
            
        # We have an extra level of iteration for the crop path: iterating over each ROI
        for roi_index, roi in enumerate(tqdm(self.crop.roi_layer_list, desc="ROI:", position=0)):
            # pass arguments for save tiff, callable and function arguments
            logger.info("Processing ROI ", roi_index)
            
            deconv_args: dict[Any, Any] = {}
            if self.deconvolution is not None:
                deconv_args = dict(
                    num_iter = self.deconvolution.psf_num_iter,
                    psf = self.deconvolution.psf,
                    decon_processing=self.deconvolution.decon_processing
                )

            for time_idx, time, ch_idx, ch, data in self.iter_slices():
                yield ProcessedVolume(
                    data = crop_volume_deskew(
                        original_volume=data,
                        deconvolution=self.deconv_enabled,
                        get_deskew_and_decon=False,
                        debug=False,
                        roi_shape=roi,
                        linear_interpolation=True,
                        voxel_size_x=self.dx,
                        voxel_size_y=self.dy,
                        voxel_size_z=self.dy,
                        angle_in_degrees=self.angle,
                        deskewed_volume=self.deskewed_volume,
                        z_start=self.crop.z_start,
                        z_end=self.crop.z_end,
                        **deconv_args
                    ),
                    channel=ch,
                    channel_index=ch_idx,
                    time=time,
                    time_index=time_idx,
                    roi_index=roi_index
                ) 
    def _process_non_crop(self) -> Iterable[ProcessedVolume]:
        """
        Yields processed image slices without cropping
        """
        for time_idx, time, ch_idx, ch, data in self.iter_slices():
            if isinstance(data, DaskArray):
                data = data.compute()
            if self.deconvolution is not None:
                if self.deconvolution.decon_processing == DeconvolutionChoice.cuda_gpu:
                    data= pycuda_decon(
                        image=data,
                        psf=self.deconvolution.psf[ch],
                        dzdata=self.dz,
                        dxdata=self.dx,
                        dzpsf=self.dz,
                        dxpsf=self.dx,
                        num_iter=self.deconvolution.psf_num_iter
                    )
                else:
                    data= skimage_decon(
                            vol_zyx=data,
                            psf=self.deconvolution.psf[ch],
                            num_iter=self.deconvolution.psf_num_iter,
                            clip=False,
                            filter_epsilon=0,
                            boundary='nearest'
                        )

            yield ProcessedVolume(
                data = cle.pull_zyx(self.deskew_func(
                    input_image=data,
                    angle_in_degrees=self.angle,
                    linear_interpolation=True,
                    voxel_size_x=self.dx,
                    voxel_size_y=self.dy,
                    voxel_size_z=self.dz
                )),
                channel=ch,
                channel_index=ch_idx,
                time=time,
                time_index=time_idx
            )
    
    def process(self) -> ProcessedSlices:
        """
        Execute the processing and return the result.
        This is the main public API for processing
        """
        ProcessedSlices.update_forward_refs()
        if self.cropping_enabled:
            return ProcessedSlices(
                lattice_data=self,
                slices=self._process_crop()
            )
        else:
            return ProcessedSlices(
                lattice_data=self,
                slices=self._process_non_crop()
            )

class AicsLatticeParams(TypedDict):
    data: DaskArray
    dims: Dimensions
    physical_pixel_sizes: DefinedPixelSizes

def lattice_params_from_aics(img: AICSImage, physical_pixel_sizes: PhysicalPixelSizes = PhysicalPixelSizes(None, None, None)) -> AicsLatticeParams:
    # Note: The reason we copy all of these fields rather than just storing the AICSImage is because that class is mostly immutable and so not suitable

    pixel_sizes = DefinedPixelSizes(
        X = physical_pixel_sizes[0] or img.physical_pixel_sizes.X or LatticeData.physical_pixel_sizes.X,
        Y = physical_pixel_sizes[1] or img.physical_pixel_sizes.Y or LatticeData.physical_pixel_sizes.Y, 
        Z = physical_pixel_sizes[2] or img.physical_pixel_sizes.Z or LatticeData.physical_pixel_sizes.Z 
    )

    return AicsLatticeParams(
        data = img.dask_data,
        dims = img.dims,
        physical_pixel_sizes = pixel_sizes,
    )

def img_from_array(arr: ArrayLike, last_dimension: Optional[Literal["channel", "time"]] = None, **kwargs: Any) -> AICSImage:
    """
    Creates an AICSImage from an array without metadata

    Args:
        arr (ArrayLike): An array
        last_dimension: How to handle the dimension order
        kwargs: Additional arguments to pass to the AICSImage constructor
    """    
    dim_order: str

    if len(arr.shape) < 3 or len(arr.shape) > 5:
        raise ValueError("Array dimensions must be in the range [3, 5]")

    # if aicsimageio tiffreader assigns last dim as time when it should be channel, user can override this
    if len(arr.shape) == 3:
        dim_order="ZYX"
    else:
        if last_dimension not in ["channel", "time"]:
            raise ValueError("last_dimension must be either channel or time")
        if len(arr.shape) == 4:
            if last_dimension == "channel":
                dim_order = "CZYX"
            elif last_dimension == "time":
                dim_order = "TZYX"
        elif len(arr.shape) == 5:
            if last_dimension == "channel":
                dim_order = "CTZYX"
            elif last_dimension == "time":
                dim_order = "TCZYX"
        else:
            raise ValueError()

    img = AICSImage(image=arr, dim_order=dim_order, **kwargs)

    # if last axes of "aicsimage data" shape is not equal to time, then swap channel and time
    if img.data.shape[0] != img.dims.T or img.data.shape[1] != img.dims.C:
        arr = np.swapaxes(arr, 0, 1)
    return AICSImage(image=arr, dim_order=dim_order, **kwargs)

def lattice_fom_array(arr: ArrayLike, last_dimension: Optional[Literal["channel", "time"]] = None, **kwargs: Any) -> AicsLatticeParams:
    """
    Creates a `LatticeData` from an array

    Args:
        arr: Array to use as the data source
        last_dimension: See img_from_array
    """   
    aics = img_from_array(arr, last_dimension)
    return lattice_params_from_aics(aics, **kwargs)
