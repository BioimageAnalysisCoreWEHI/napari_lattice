from __future__ import annotations
# class for initializing lattice data and setting metadata
# TODO: handle scenes

from typing_extensions import Self, TYPE_CHECKING, Any, Optional, Tuple

from pathlib import Path

import pyclesperanto_prototype as cle

from lls_core import DeskewDirection
from xarray import DataArray

from lls_core.models.utils import FieldAccessModel, enum_choices
from lls_core.types import is_arraylike, is_pathlike
from lls_core.utils import get_deskewed_shape,convert_xyz_to_zyx_order
import numpy as np
import logging
from pydantic import Field, NonNegativeFloat, ValidationInfo, field_validator, model_validator

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from bioio import PhysicalPixelSizes


def load_image_lazy(path: Path) -> DataArray:
    """
    Re-open an image file as a lazy DataArray. Used by parallel workers, since the
    bioio reader backing a file-loaded `input_image` is not picklable. bioio always
    returns standard TCZYX dims, so this matches the parent's `input_image` directly
    (no reshaping needed); each worker then reads only its own ROI crops from disk.
    """
    from bioio import BioImage
    from os import fspath
    from lls_core.czi_reader import czi_xarray
    resolved = fspath(path)
    image = BioImage(resolved)
    # The facade czi_xarray builds is created here, inside the worker, so the
    # non-picklable reader it holds never has to cross a process boundary.
    fast = czi_xarray(resolved, image)
    return fast if fast is not None else image.xarray_dask_data

class DefinedPixelSizes(FieldAccessModel):
    """
    Like PhysicalPixelSizes, but it's a dataclass, and
    none of its fields are None
    """
    X: NonNegativeFloat = Field(default=0.1499219272808386, description="Size of the X dimension of the microscope pixels, in microns.")
    Y: NonNegativeFloat = Field(default=0.1499219272808386, description="Size of the Y dimension of the microscope pixels, in microns.")
    Z: NonNegativeFloat = Field(default=0.3, description="Size of the Z dimension of the microscope pixels, in microns.")

    @classmethod
    def from_physical(cls, pixels: PhysicalPixelSizes) -> Self:
        from lls_core.utils import raise_if_none

        return DefinedPixelSizes(
            X=raise_if_none(pixels.X, "All pixels must be defined"),
            Y=raise_if_none(pixels.Y, "All pixels must be defined"),
            Z=raise_if_none(pixels.Z, "All pixels must be defined"),
        )

class DerivedDeskewFields(FieldAccessModel):
    """
    Fields that are automatically calculated based on other fields in DeskewParams.
    Grouping these together into one model makes validation simpler.
    """
    deskew_vol_shape: Tuple[int, ...] = Field(
        init_var=False,
        default=None,
        description="Dimensions of the deskewed output. This is set automatically based on other input parameters, and doesn't need to be provided by the user."
    )

    deskew_affine_transform: cle.AffineTransform3D = Field(init_var=False, default=None, description="Deskewing affine transformation matrix (in xyz order for OpenCL). This is set automatically based on other input parameters, and doesn't need to be provided by the user.")
    deskew_affine_transform_zyx: np.ndarray = Field(init_var=False, default=None, description="Deskewing affine transformation matrix (zyx order). This is set automatically based on other input parameters, and doesn't need to be provided by the user.")


class DeskewParams(FieldAccessModel):
    input_image: DataArray = Field(
        description="A 3-5D array containing the image data. Can be anything convertible to an Xarray, including a `dask.array` or `numpy.ndarray`. Can also be provided as a `str`, in which case it must indicate the path to an image to load from disk.",
        cli_description="A path to any standard image file (TIFF, H5 etc) containing a 3-5D array to process."
    )
    input_image_path: Optional[Path] = Field(
        default=None,
        cli_hide=True,
        description="Internal: the filesystem path the input image was loaded from, if any. "
                    "Set automatically when `input_image` is provided as a path, and used to re-open "
                    "the image lazily inside parallel worker processes. Not a user-facing parameter."
    )
    skew: DeskewDirection = Field(
        default=DeskewDirection.Y,
        description=f"Axis along which to deskew the image. Choices: `{enum_choices(DeskewDirection)}`. These can be provided as `str`."
    )
    angle: float = Field(
        default=30.0,
        description="Angle of deskewing, in degrees, as a float."
    )
    physical_pixel_sizes: DefinedPixelSizes = Field(
        # No default, because we need to distinguish between user provided arguments and defaults
        description="Pixel size of the microscope, in microns. This can alternatively be provided as a `tuple[float]` of `(Z, Y, X)`",
        default=None,
        validate_default=True
    )
    invert_scan_direction: bool = Field(
        default=False,
        description="If `True`, reverse the order of planes along the scan (Z) axis before deskewing. "
                    "This is required for oblique plane microscopes whose stage/galvo scans can be in opposite directions. "
                    "Leaving this `False` preserves the original behaviour compatible with Zeiss LLS."
    )
    coverslip_rotation: bool = Field(
        default=True,
        description="Apply the coverslip rotation (rotate the deskewed volume by the deskew angle). "
                    "Effect is acquisition-geometry dependent: True (default) uses the standard deskew "
                    "(cle.deskew_y/x) and is coverslip-level for Zeiss LLS7; False skips the rotation and "
                    "is coverslip-level for some OPM/SOPi."
    )
    derived: DerivedDeskewFields = Field(
        init_var=False,
        default=None,
        description="Refer to the `DerivedDeskewFields` docstring",
        cli_hide=True,
        validate_default=True
    )
    # Hack to ensure that .skew_dir behaves identically to .skew
    @property
    def skew_dir(self) -> DeskewDirection:
        return self.skew

    @skew_dir.setter
    def skew_dir(self, value: DeskewDirection):
        self.skew = value

    @property
    def deskew_func(self):
        if not self.coverslip_rotation:
            # OPM/SOPi (shear-only) branch
            from lls_core.shear_only_deskew import shear_only_deskew
            skew_name = "Y" if self.skew == DeskewDirection.Y else "X"
            # Adapt to the deskew_func call convention used in _process_non_crop
            def _cover(input_image, angle_in_degrees, linear_interpolation,
                       voxel_size_x, voxel_size_y, voxel_size_z):
                return shear_only_deskew(input_image, angle_in_degrees, voxel_size_z,
                                        voxel_size_y, voxel_size_x, skew=skew_name)
            return _cover
        # Choose deskew function based on skew direction
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
    def dy(self, value: float) -> None:
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
    def dims(self):
        return self.input_image.dims

    @property
    def time(self) -> int:
        """Number of time points"""
        return self.input_image.sizes["T"]

    @property
    def channels(self) -> int:
        """Number of channels"""
        return self.input_image.sizes["C"]

    @property
    def nslices(self) -> int:
        """The number of 3D slices within the image"""
        return self.time * self.channels

    @property
    def new_dz(self):
        import math
        if not self.coverslip_rotation:
            # Coverslip output Z axis sits on the lateral-pixel grid: one zc index = d_lateral
            return self.dy if self.skew == DeskewDirection.Y else self.dx
        return math.sin(self.angle * math.pi / 180.0) * self.dz

    def apply_scan_flip(self, volume: DataArray) -> DataArray:
        """
        Reverse the plane/scan (Z) axis of a volume if `invert_scan_direction` is set.
        This is a lazy operation (a view for numpy-backed arrays, lazy for dask), so it
        adds no measurable cost and does not introduce a second resampling pass.
        The scan axis is always Z (the plane-stacking axis), regardless of skew direction.
        """
        if self.invert_scan_direction:
            return volume.isel(Z=slice(None, None, -1))
        return volume

    @field_validator("skew", mode="before")
    @classmethod
    def convert_skew(cls, v: Any):
        # Allow skew to be provided as a string
        if isinstance(v, str):
            return DeskewDirection[v]
        return v

    @field_validator("physical_pixel_sizes", mode="before")
    @classmethod
    def convert_pixels(cls, v: Any):
        from bioio import PhysicalPixelSizes
        if isinstance(v, PhysicalPixelSizes):
            v = DefinedPixelSizes.from_physical(v)
        elif isinstance(v, tuple) and len(v) == 3:
            # Allow the pixel sizes to be specified as a tuple
            v = DefinedPixelSizes(Z=v[0], Y=v[1], X=v[2])
        elif v is None:
            # At this point, we have exhausted all other methods of obtaining pixel sizes:
            # User defined and image metadata. So we just use the defaults
            defaults = DefinedPixelSizes()
            logger.warning(
                "No pixel sizes were provided or found in the image metadata; falling back to "
                f"default (Zeiss) values Z={defaults.Z}, Y={defaults.Y}, X={defaults.X} microns. "
                "For non-Zeiss data, specify physical_pixel_sizes (deskew/MIP geometry depends on it)."
            )
            return defaults

        return v

    @model_validator(mode="before")
    @classmethod
    def read_image(cls, values: dict):
        from bioio import BioImage
        from os import fspath

        # Warn when geometry-defining parameters are left at their defaults, since
        # the deskew/MIP geometry depends entirely on them. `values` here is the
        # raw user input (pre-validation), so a missing key means "not provided".
        # (Internal copies pass all fields explicitly, so they don't trigger this.)
        if "angle" not in values:
            logger.warning(
                f"No deskew angle was provided; using the default of {cls.model_fields['angle'].default} degrees. "
                "Specify the angle if your microscope differs."
            )
        if "skew" not in values:
            logger.warning(
                f"No skew direction was provided; using the default '{cls.model_fields['skew'].default.name}'."
            )

        img = values["input_image"]

        aics: BioImage | None = None
        if is_pathlike(img):
            aics = BioImage(fspath(img))
            # Remember the source path so parallel workers can re-open the file lazily
            values["input_image_path"] = Path(fspath(img)).resolve()
        elif isinstance(img, BioImage):
            aics = img
        elif isinstance(img, DataArray):
            if set(img.dims) >= {"Z", "Y", "X"}:
                # If it's already a DataArray with the right dimensions, we're done
                return values
            else:
                raise ValueError("If passing a DataArray, it should at least have Z Y and X dimensions, appropriately labelled.")
        elif is_arraylike(img):
            if len(img.shape) == 3:
                values["input_image"] = DataArray(img, dims=["Z", "Y", "X"])
            else:
                raise ValueError("Only 3D numpy arrays are currently supported. If you have a different shape, please use a DataArray and name your dimensions C, T, Z, Y and/or Z.")
        else:
            raise ValueError("Value of input_image was neither a path, a BioImage, or array-like.")

        # If the image was convertible to BioImage, we should use the metadata from there
        if aics:
            from lls_core.czi_reader import czi_path_of, czi_xarray
            czi_path = fspath(img) if is_pathlike(img) else czi_path_of(aics)
            fast = czi_xarray(czi_path, aics) if czi_path is not None else None
            values["input_image"] = fast if fast is not None else aics.xarray_dask_data
            # Take pixel sizes from the image metadata, but only if they're defined
            # and only if we don't already have them
            if all(size is not None for size in aics.physical_pixel_sizes) and values.get("physical_pixel_sizes") is None:
                values["physical_pixel_sizes"] = DefinedPixelSizes.from_physical(aics.physical_pixel_sizes)

        # In all cases, input_image will be a DataArray (XArray) at this point

        return values

    @field_validator("input_image", mode="before")
    @classmethod
    def reshaping(cls, v: DataArray):
        # This allows a user to pass in any array-like object and have it
        # converted and reshaped appropriately
        array = v
        if not set(array.dims).issuperset({"X", "Y", "Z"}):
            raise ValueError("The input array must at least have XYZ coordinates")
        if "T" not in array.dims:
            array = array.expand_dims("T")
        if "C" not in array.dims:
            array = array.expand_dims("C")
        return array.transpose("T", "C", "Z", "Y", "X")

    def get_3d_slice(self) -> DataArray:
        return self.apply_scan_flip(self.input_image.isel(C=0, T=0))

    @field_validator("derived", mode="before")
    @classmethod
    def calculate_derived(cls, v: Any, info: ValidationInfo) -> DerivedDeskewFields:
        """
        Sets the default deskew shape values if the user has not provided them
        """
        values = info.data
        data: DataArray = values["input_image"]
        if isinstance(v, DerivedDeskewFields):
            return v
        elif isinstance(v, dict):
            # Catches the edge case where a plain dict is passed, converting it to a DerivedDeskewFields object.
            return DerivedDeskewFields(**v)
        elif v is None:
            if not values.get("coverslip_rotation", True):
                # OPM/SOPi (shear-only) branch
                from lls_core.shear_only_geometry import shear_only_output_shape, shear_only_display_affine_zyx
                raw3d = data.isel(C=0, T=0)
                raw_shape_zyx = tuple(int(s) for s in raw3d.shape)
                skew_name = "Y" if values["skew"] == DeskewDirection.Y else "X"
                cover_shape = shear_only_output_shape(
                    raw_shape_zyx,
                    values["angle"],
                    values["physical_pixel_sizes"].Z,
                    values["physical_pixel_sizes"].Y,
                    values["physical_pixel_sizes"].X,
                    skew_name,
                )
                # deskew_affine_transform (xyz/CLE) is only used by the objective display/crop
                # path; keep the objective CLE transform so that path (if ever reached) still works.
                _, deskew_affine_transform = get_deskewed_shape(
                    raw3d, values["angle"],
                    values["physical_pixel_sizes"].X, values["physical_pixel_sizes"].Y,
                    values["physical_pixel_sizes"].Z, values["skew"])
                # The _zyx display affine is the only one consumed by Quick Deskew preview.
                # Use the coverslip shear-only affine so toggling "Coverslip Rotation" OFF
                # shows the OPM/shear-only orientation in the napari canvas.
                deskew_affine_transform_zyx = shear_only_display_affine_zyx(
                    raw_shape_zyx,
                    values["angle"],
                    values["physical_pixel_sizes"].Z,
                    values["physical_pixel_sizes"].Y,
                    values["physical_pixel_sizes"].X,
                    skew_name,
                    invert_scan_direction=values.get("invert_scan_direction", False),
                )
                return DerivedDeskewFields(
                    deskew_affine_transform=deskew_affine_transform,
                    deskew_vol_shape=cover_shape,
                    deskew_affine_transform_zyx=deskew_affine_transform_zyx,
                )
            deskew_vol_shape, deskew_affine_transform = get_deskewed_shape(
                data.isel(C=0, T=0),
                values["angle"],
                values["physical_pixel_sizes"].X,
                values["physical_pixel_sizes"].Y,
                values["physical_pixel_sizes"].Z,
                values["skew"]
            )
            deskew_affine_transform_zyx = convert_xyz_to_zyx_order(deskew_affine_transform)
            if values.get("invert_scan_direction"):
                # The actual deskew/crop output flips the scan (Z) axis of the data (see
                # `apply_scan_flip`). The deskew transform itself is shape-derived and so
                # is identical regardless of the flip, which means the in-canvas "Quick
                # Deskew" display (the only consumer of this matrix) would otherwise ignore
                # the flag. Fold the Z reflection into the *display* transform so that
                # applying it to the unflipped layer reproduces the flipped-then-deskewed
                # geometry, matching the processed/preview output.
                nz = int(data.sizes["Z"])
                flip_zyx = np.eye(4)
                flip_zyx[0, 0] = -1
                flip_zyx[0, 3] = nz - 1
                deskew_affine_transform_zyx = deskew_affine_transform_zyx @ flip_zyx
            return DerivedDeskewFields(
                deskew_affine_transform=deskew_affine_transform,
                deskew_vol_shape=deskew_vol_shape,
                deskew_affine_transform_zyx=deskew_affine_transform_zyx
            )
        else:
            raise ValueError("Invalid derived fields")
