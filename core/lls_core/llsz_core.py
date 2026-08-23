from __future__ import annotations

import numpy as np
import pyclesperanto as cle
from dask.array.core import Array as DaskArray
import dask.array as da
from resource_backed_dask_array import ResourceBackedDaskArray
from typing import Any, NamedTuple, Optional, Union, TYPE_CHECKING, overload, Literal, Tuple
from typing_extensions import Unpack, TypedDict, Required
from lls_core.affine import AffineTransform3D, determine_translation_and_bounding_box
from lls_core.affine_transform_deskew import affine_transform_deskew_3d
from numpy.typing import NDArray 
from lls_core.utils import calculate_crop_bbox, ShapeOnly
from lls_core import config, DeskewDirection
from lls_core.types import ArrayLike
from lls_core.deconvolution import pycuda_decon, skimage_decon, DeconvolutionChoice

# Enable Logging
import logging

logger = logging.getLogger(__name__)
logger.setLevel(config.log_level)
# pass shapes data from single ROI to crop the volume from original data

if TYPE_CHECKING:
    from napari.layers.shapes import Shapes

Psf = Union[
        NDArray,
        DaskArray,
        ResourceBackedDaskArray,
        cle.Array,
]

class ObjectiveCropGeometry(NamedTuple):
    """
    Where an objective-frame (`coverslip_rotation=True`) deskewed crop sits, and which
    raw sub-block produces it. Pure geometry: no pixels are read.

    `origin_zyx` is the position of the crop's voxel `(0, 0, 0)` within the FULL
    deskewed volume, in deskewed voxels. It is **not** simply the ROI origin:
    `crop_volume_deskew` only trims the *skew* axis to the ROI (via `crop_excess`),
    so on the other two axes the output keeps whatever bounds the clipped raw
    sub-block happened to deskew into. In particular the Z origin is the sub-block's
    minimum deskewed Z, **not** `z_start` - the output is never sliced on Z.

    Deriving this anywhere other than here would drift from what the crop actually
    does; `crop_volume_deskew`, `get_roi_bboxes` and the output metadata all read it
    from this one function.
    """
    #: Raw-volume slice bounds, each `(start, stop)`.
    raw_x: Tuple[int, int]
    raw_y: Tuple[int, int]
    raw_z: Tuple[int, int]
    #: Offset along the skew axis from the sub-block's deskewed origin to the ROI's.
    crop_excess: int
    #: Final crop shape `(z, y, x)`, i.e. the ROI extent.
    crop_vol_shape: Tuple[int, int, int]
    #: Crop voxel (0,0,0) in full-deskewed-volume voxels, `(z, y, x)`.
    origin_zyx: Tuple[float, float, float]
    #: Raw -> deskewed affine for this acquisition.
    deskew_transform: Any


def objective_crop_transforms(
    raw_shape_zyx: Tuple[int, ...],
    angle_in_degrees: float,
    voxel_size_x: float,
    voxel_size_y: float,
    voxel_size_z: float,
    skew_dir: DeskewDirection = DeskewDirection.Y,
) -> Tuple[Any, Any]:
    """
    The ROI-independent `(reverse_aff, deskew_transform)` pair, hoisted so a caller
    looping over many ROIs computes the affine once rather than per ROI.
    """
    reverse_aff, _excess_bounds, deskew_transform = get_inverse_affine_transform(
        ShapeOnly(tuple(int(s) for s in raw_shape_zyx)),
        angle_in_degrees,
        voxel_size_x,
        voxel_size_y,
        voxel_size_z,
        skew_dir,
    )
    return reverse_aff, deskew_transform


def objective_crop_geometry(
    raw_shape_zyx: Tuple[int, ...],
    roi_shape: Union[list, NDArray],
    z_start: int,
    z_end: int,
    angle_in_degrees: float,
    voxel_size_x: float,
    voxel_size_y: float,
    voxel_size_z: float,
    skew_dir: DeskewDirection = DeskewDirection.Y,
    transforms: Optional[Tuple[Any, Any]] = None,
) -> ObjectiveCropGeometry:
    """
    Resolve one ROI into the raw sub-block to read and the deskewed position of the
    resulting crop. See `ObjectiveCropGeometry` for what `origin_zyx` means.

    Pass `transforms` from `objective_crop_transforms` to avoid recomputing the affine
    once per ROI.
    """
    from itertools import product

    crop_bounding_box, crop_vol_shape = calculate_crop_bbox(roi_shape, z_start, z_end)

    if transforms is None:
        transforms = objective_crop_transforms(
            raw_shape_zyx, angle_in_degrees, voxel_size_x, voxel_size_y, voxel_size_z, skew_dir
        )
    reverse_aff, deskew_transform = transforms

    # Apply the reverse transform to get the corresponding bounding box in the raw volume
    crop_transform_bbox = np.asarray(
        [reverse_aff._matrix @ corner for corner in crop_bounding_box]
    )

    # Raw shape in xyz, to match the transform's coordinate order
    orig_img_shape = tuple(int(s) for s in raw_shape_zyx)[::-1]

    # Take min and max of the transformed bounding box, clipped so we never index
    # outside the raw volume
    min_coordinate = np.around(crop_transform_bbox.min(axis=0))
    max_coordinate = np.around(crop_transform_bbox.max(axis=0))

    x_start = int(np.clip(min_coordinate[0].astype(int), 0, orig_img_shape[0]))
    x_end = int(np.clip(max_coordinate[0].astype(int), 0, orig_img_shape[0]))
    y_start = int(np.clip(min_coordinate[1].astype(int), 0, orig_img_shape[1]))
    y_end = int(np.clip(max_coordinate[1].astype(int), 0, orig_img_shape[1]))
    z_start_vol = int(np.clip(min_coordinate[2].astype(int), 0, orig_img_shape[2]))
    z_end_vol = int(np.clip(max_coordinate[2].astype(int), 0, orig_img_shape[2]))

    # make sure z_start < z_end
    if z_start_vol > z_end_vol:
        z_start_vol, z_end_vol = z_end_vol, z_start_vol

    # The deskewed sub-block is larger than the crop (the shear adds empty wedges), and
    # its row/col 0 is the MINIMUM deskewed coordinate of the clipped sub-block, so
    # crop_excess = round(ROI_origin - that_minimum). Old code assumed the ROI was
    # centred ((deskewed_dim - crop_dim) / 2 + an out-of-bounds fudge), which only holds
    # at scan-centre and mis-placed off-centre ROIs. This was not the case with larger
    # datasets where ROIs were far away from scan centre. Projecting the clipped corners
    # through the raw->deskewed transform is exact.
    full_deskew_matrix = np.linalg.inv(reverse_aff._matrix)  # raw->deskewed, incl. translation
    sub_corners = np.asarray([
        [x, y, z, 1]
        for x, y, z in product((x_start, x_end), (y_start, y_end), (z_start_vol, z_end_vol))
    ])
    prelim_corners = (full_deskew_matrix @ sub_corners.T).T  # deskewed xyz of sub-block corners
    roi_origin = np.asarray(crop_bounding_box).min(axis=0)   # [x, y, z, 1]

    # Deskewed position of the sub-block's own voxel (0, 0, 0)
    prelim_origin_x = float(prelim_corners[:, 0].min())
    prelim_origin_y = float(prelim_corners[:, 1].min())
    prelim_origin_z = float(prelim_corners[:, 2].min())

    if skew_dir == DeskewDirection.Y:
        crop_excess = max(int(round(float(roi_origin[1]) - prelim_origin_y)), 0)
        origin_zyx = (prelim_origin_z, prelim_origin_y + crop_excess, prelim_origin_x)
    elif skew_dir == DeskewDirection.X:
        crop_excess = max(int(round(float(roi_origin[0]) - prelim_origin_x)), 0)
        origin_zyx = (prelim_origin_z, prelim_origin_y, prelim_origin_x + crop_excess)
    else:
        raise ValueError(f"Unknown skew direction {skew_dir!r}")

    return ObjectiveCropGeometry(
        raw_x=(x_start, x_end),
        raw_y=(y_start, y_end),
        raw_z=(z_start_vol, z_end_vol),
        crop_excess=crop_excess,
        crop_vol_shape=(int(crop_vol_shape[0]), int(crop_vol_shape[1]), int(crop_vol_shape[2])),
        origin_zyx=origin_zyx,
        deskew_transform=deskew_transform,
    )


class CommonArgs(TypedDict, total=False):
    original_volume: Required[ArrayLike]
    deskewed_volume: Union[ ArrayLike, None ]
    roi_shape: Union[list, NDArray, None]
    angle_in_degrees: float
    voxel_size_x: float
    voxel_size_y: float
    voxel_size_z: float
    z_start: int
    z_end: int
    deconvolution: bool
    decon_processing: Optional[DeconvolutionChoice]
    psf: Union[Psf, None]
    num_iter: int
    linear_interpolation: bool
    skew_dir: DeskewDirection
    coverslip_rotation: bool

@overload
def crop_volume_deskew(*, debug: Literal[True], get_deskew_and_decon: bool = False, **kwargs: Unpack[CommonArgs]) -> Tuple[NDArray, NDArray]:
    ...
@overload
def crop_volume_deskew(*, debug: Literal[False] = False, get_deskew_and_decon: Literal[True], **kwargs: Unpack[CommonArgs]) -> Tuple[NDArray, NDArray]:
    ...
@overload
def crop_volume_deskew(*, debug: Literal[False] = False, get_deskew_and_decon: Literal[False] = False, **kwargs: Unpack[CommonArgs]) -> NDArray:
    ...
def crop_volume_deskew(
    original_volume: ArrayLike,
    deskewed_volume: Union[ ArrayLike, None ] = None,
    roi_shape: Union[list, NDArray, None] = None,
    angle_in_degrees: float = 30,
    voxel_size_x: float = 1,
    voxel_size_y: float = 1,
    voxel_size_z: float = 1,
    z_start: int = 0,
    z_end: int = 1,
    debug: bool = False,
    deconvolution: bool = False,
    decon_processing: Optional[DeconvolutionChoice]=None,
    psf: Union[Psf, None]=None,
    num_iter: int = 10,
    linear_interpolation: bool=True,
    skew_dir: DeskewDirection=DeskewDirection.Y,
    get_deskew_and_decon: bool = False,
    coverslip_rotation: bool = True,
):
    """Crop the volume from original data and deskew the cropped volume
    Args:
        original_volume (Union[da.core.Array,np.ndarray,cle.Array,resource_backed_dask_array.ResourceBackedDaskArray]): Volume to deskew (zyx)
        deskewed_volume:DEPRECATED
        roi_shape (Union[shapes.Shapes,list,np.array]): shapes layer or rois
        angle_in_degrees (float, optional): deskewing angle in degrees. Defaults to 30.
        voxel_size_x (float, optional): microns. Defaults to 1.
        voxel_size_y (float, optional): microns.  Defaults to 1.
        voxel_size_z (float, optional): microns.  Defaults to 1.
        z_start (int, optional): Currently not used, but can be used to crop the volume in z. Defaults to 0.
        z_end (int, optional): _description_. Currently not used, but can be used to crop the volume in z. Defaults to 1.
        debug (bool, optional): If True, returns the cropped volume and the cropped volume with extra bounds. Defaults to False.
        deconvolution (bool, optional): Perform deconvolution. Defaults to False.
        decon_processing (str, optional): Choose decon option, cuda_gpu or cpu. Defaults to None.
        psf (_type_, optional): Pass a psf array for deconvolution. Defaults to None.
        num_iter (int, optional): Number of Iterations for Richardson Lucy deconvolution. Defaults to 10.
        linear_interpolation (bool, optional): Linear Interpolation after deskewing. Defaults to True.
        skew_direct (DeskewDirection, optional): Deskew direction. Defaults to DeskewDirection.Y.
        get_deskew_no_decon (bool, optional): Return both deconvolved data and deskewed data with no deconvolution. Defaults to False.

    Returns:
        _type_: _description_
    """

    assert len(original_volume.shape) == 3, print(
        "Shape of original volume must be 3"
    )
    # assert len(deskewed_volume.shape) == 3, print("Shape of deskewed volume must be 3")
    # assert len(shape) == 4, print("Shape must be an array of shape 4 ")
    shape = None

    # if shapes layer, get first one
    # TODO: test this
    # if is_napari_shape(roi_shape):
    #     shape = roi_shape.data[0]
    # if its a list and each element has a shape of 4, its a list of rois
    if isinstance(roi_shape, list) and len(roi_shape[0]) == 4:
        # TODO:change to accept any roi by passing index
        shape = roi_shape[0]
        # len(roi_shape) >= 1:
        # if its a array or list with shape of 4, its a single ROI
    elif len(roi_shape) == 4 and isinstance(roi_shape, (np.ndarray, list)):
        shape = roi_shape

    assert len(shape) == 4, print("Shape must be an array of shape 4")

    if not coverslip_rotation:
        # OPM/SOPi: ROI corners are in the coverslip frame → map to raw via the
        # frozen coverslip inverse map (shear_only_inverse_map), not the objective affine.
        return _crop_volume_deskew_shear_only(
            original_volume=original_volume,
            roi_shape=shape,
            z_start=z_start,
            z_end=z_end,
            angle_in_degrees=angle_in_degrees,
            voxel_size_x=voxel_size_x,
            voxel_size_y=voxel_size_y,
            voxel_size_z=voxel_size_z,
            skew_dir=skew_dir,
            debug=debug,
            deconvolution=deconvolution,
            decon_processing=decon_processing,
            psf=psf,
            num_iter=num_iter,
            get_deskew_and_decon=get_deskew_and_decon,
        )

    geometry = objective_crop_geometry(
        raw_shape_zyx=original_volume.shape,
        roi_shape=shape,
        z_start=z_start,
        z_end=z_end,
        angle_in_degrees=angle_in_degrees,
        voxel_size_x=voxel_size_x,
        voxel_size_y=voxel_size_y,
        voxel_size_z=voxel_size_z,
        skew_dir=skew_dir,
    )
    x_start, x_end = geometry.raw_x
    y_start, y_end = geometry.raw_y
    z_start_vol, z_end_vol = geometry.raw_z
    deskew_transform = geometry.deskew_transform

    # Guard against a degenerate (zero- or one-voxel-wide) crop: if the projected
    # ROI bounding box falls entirely outside the raw volume along an axis,
    # clipping collapses both edges to the same boundary. A single-voxel extent
    # is not enough either: pushing an array with a size-1 leading (Z) dimension
    # makes some pyclesperanto backends generate 2D buffer-read macros instead of
    # 3D ones, which the (always-3D, int4-indexed) vendored kernels can't compile
    # against. Expand to at least 2 voxels so the GPU push below never receives a
    # degenerate array; the crop_excess trimming/padding further down already
    # handles the resulting short crop correctly. Mirrors the equivalent guard in
    # _crop_volume_deskew_shear_only (which requires >=2 scan planes for the same
    # reason).
    if x_end - x_start < 2:
        x_end = min(x_start + 2, orig_img_shape[0])
        x_start = max(x_end - 2, 0)
    if y_end - y_start < 2:
        y_end = min(y_start + 2, orig_img_shape[1])
        y_start = max(y_end - 2, 0)
    if z_end_vol - z_start_vol < 2:
        z_end_vol = min(z_start_vol + 2, orig_img_shape[2])
        z_start_vol = max(z_end_vol - 2, 0)

    # After getting the coordinates, crop from original volume and deskew only the cropped volume

    if isinstance(original_volume, (
        DaskArray,
        ResourceBackedDaskArray,
    )):
        # If using dask, use .map_blocks(np.copy) to copy subset (faster)
        crop_volume = (
            original_volume[
                z_start_vol:z_end_vol, y_start:y_end, x_start:x_end
            ]
            .map_blocks(np.copy)
            .squeeze()
        )
    else:
        crop_volume = original_volume[
            z_start_vol:z_end_vol, y_start:y_end, x_start:x_end
        ]

    # check if deconvolution is checked
    if deconvolution:
        if decon_processing == DeconvolutionChoice.cuda_gpu:
            crop_volume_processed = pycuda_decon(
                image=crop_volume,
                psf=psf,
                dzdata=voxel_size_z,
                dxdata=voxel_size_x,
                dzpsf=voxel_size_z,
                dxpsf=voxel_size_x,
                num_iter=num_iter,
                cropping=True,
            )
        else:
            crop_volume_processed = skimage_decon(
                vol_zyx=crop_volume,
                psf=psf,
                num_iter=num_iter,
                clip=False,
                filter_epsilon=0,
                boundary="nearest",
            )

        deskewed_prelim = affine_transform_deskew_3d(
            crop_volume_processed,
            transform=deskew_transform,
            deskewing_angle_in_degrees=angle_in_degrees,
            voxel_size_x=voxel_size_x,
            voxel_size_y=voxel_size_y,
            voxel_size_z=voxel_size_z,
            deskew_direction=skew_dir,
        )
        if get_deskew_and_decon:
            deskewed_no_decon = affine_transform_deskew_3d(
                crop_volume,
                transform=deskew_transform,
                deskewing_angle_in_degrees=angle_in_degrees,
                voxel_size_x=voxel_size_x,
                voxel_size_y=voxel_size_y,
                voxel_size_z=voxel_size_z,
                deskew_direction=skew_dir,
            )
    else:
        deskewed_prelim = affine_transform_deskew_3d(
            crop_volume,
            transform=deskew_transform,
            deskewing_angle_in_degrees=angle_in_degrees,
            voxel_size_x=voxel_size_x,
            voxel_size_y=voxel_size_y,
            voxel_size_z=voxel_size_z,
            deskew_direction=skew_dir,
        )

    # Only the skew axis is trimmed to the ROI; see `ObjectiveCropGeometry`.
    deskewed_prelim = np.asarray(deskewed_prelim)
    crop_excess = geometry.crop_excess

    if skew_dir == DeskewDirection.Y:
        crop_height = geometry.crop_vol_shape[1]
        deskewed_crop = deskewed_prelim[:, crop_excess : crop_height + crop_excess, :]
        # Pad the skew axis if the prelim is short near the far field edge
        if deskewed_crop.shape[1] < crop_height:
            pad = crop_height - deskewed_crop.shape[1]
            deskewed_crop = np.pad(deskewed_crop, ((0, 0), (0, pad), (0, 0)))
    elif skew_dir == DeskewDirection.X:
        crop_width = geometry.crop_vol_shape[2]
        deskewed_crop = deskewed_prelim[:, :, crop_excess : crop_width + crop_excess]
        if deskewed_crop.shape[2] < crop_width:
            pad = crop_width - deskewed_crop.shape[2]
            deskewed_crop = np.pad(deskewed_crop, ((0, 0), (0, 0), (0, pad)))

    # For debugging, ,deskewed_prelim will also be returned which is the uncropped volume
    if debug:
        return deskewed_crop, deskewed_prelim
    elif get_deskew_and_decon:
        if skew_dir == DeskewDirection.Y:
            deskewed_crop_no_decon = deskewed_no_decon[
                :, crop_excess : crop_height + crop_excess, :
            ]
        elif skew_dir == DeskewDirection.X:
            deskewed_crop_no_decon = deskewed_no_decon[
                :, :, crop_excess : crop_width + crop_excess
            ]
        return deskewed_crop, deskewed_crop_no_decon
    else:
        return deskewed_crop


def _crop_volume_deskew_shear_only(
    original_volume,
    roi_shape,
    z_start: int,
    z_end: int,
    angle_in_degrees: float,
    voxel_size_x: float,
    voxel_size_y: float,
    voxel_size_z: float,
    skew_dir: DeskewDirection,
    debug: bool = False,
    deconvolution: bool = False,
    decon_processing: Optional[DeconvolutionChoice] = None,
    psf: Union[Psf, None] = None,
    num_iter: int = 10,
    get_deskew_and_decon: bool = False,
):
    """Crop + deskew for the coverslip frame (see caller for the rationale).

    The crop ROI (coverslip-frame corners) is mapped to raw scan/Y/X via the frozen
    coverslip inverse map; the raw sub-volume is extracted (with a 1px halo for
    interpolation), deskewed into the shear-only (coverslip) frame with
    ``shear_only_deskew``, then trimmed to the ROI's shear-only extent using the
    sub-block shear-only offset (a pure translation).
    """
    from lls_core.shear_only_deskew import shear_only_deskew
    from lls_core.shear_only_geometry import (
        shear_only_inverse_map,
        shear_only_subblock_offset,
    )

    skew_name = "Y" if skew_dir == DeskewDirection.Y else "X"
    crop_bounding_box, crop_vol_shape = calculate_crop_bbox(roi_shape, z_start, z_end)
    Nz, Ny, Nx = (int(s) for s in original_volume.shape)

    # Map shear-only ROI corners -> raw (scan p, raw_y, raw_x) via the frozen inverse
    raw_pts = np.asarray([
        shear_only_inverse_map(
            zc, yc, xc, angle_in_degrees,
            voxel_size_z, voxel_size_y, voxel_size_x, skew_name,
        )
        for (xc, yc, zc, _one) in crop_bounding_box
    ])

    halo = 1  # extra margin so bilinear interpolation has neighbours
    scan_start = int(np.clip(np.floor(raw_pts[:, 0].min()) - halo, 0, Nz))
    scan_end = int(np.clip(np.ceil(raw_pts[:, 0].max()) + halo, 0, Nz))
    rawy_start = int(np.clip(np.floor(raw_pts[:, 1].min()) - halo, 0, Ny))
    rawy_end = int(np.clip(np.ceil(raw_pts[:, 1].max()) + halo, 0, Ny))
    rawx_start = int(np.clip(np.floor(raw_pts[:, 2].min()) - halo, 0, Nx))
    rawx_end = int(np.clip(np.ceil(raw_pts[:, 2].max()) + halo, 0, Nx))

    # shear_only_deskew needs at least two scan planes (bilinear over plane, plane+1)
    if scan_end - scan_start < 2:
        scan_end = min(scan_start + 2, Nz)
        scan_start = max(scan_end - 2, 0)
    # guard degenerate lateral extents
    if rawy_end <= rawy_start:
        rawy_end = min(rawy_start + 1, Ny)
    if rawx_end <= rawx_start:
        rawx_end = min(rawx_start + 1, Nx)

    if isinstance(original_volume, (DaskArray, ResourceBackedDaskArray)):
        crop_volume = (
            original_volume[scan_start:scan_end, rawy_start:rawy_end, rawx_start:rawx_end]
            .map_blocks(np.copy)
            .squeeze()
        )
    else:
        crop_volume = original_volume[
            scan_start:scan_end, rawy_start:rawy_end, rawx_start:rawx_end
        ]

    crop_volume_no_decon = crop_volume
    if deconvolution:
        if decon_processing == DeconvolutionChoice.cuda_gpu:
            crop_volume = pycuda_decon(
                image=crop_volume,
                psf=psf,
                dzdata=voxel_size_z,
                dxdata=voxel_size_x,
                dzpsf=voxel_size_z,
                dxpsf=voxel_size_x,
                num_iter=num_iter,
                cropping=True,
            )
        else:
            crop_volume = skimage_decon(
                vol_zyx=crop_volume,
                psf=psf,
                num_iter=num_iter,
                clip=False,
                filter_epsilon=0,
                boundary="nearest",
            )

    # Deskew the raw sub-volume into the shear-only (coverslip) frame
    def _deskew(vol):
        return np.asarray(cle.pull(shear_only_deskew(
            vol, angle_in_degrees, voxel_size_z, voxel_size_y, voxel_size_x,
            skew=skew_name,
        )))

    deskewed_prelim = _deskew(crop_volume)

    # Trim the sub-block coverslip output to the ROI's coverslip extent.
    # The sub-block deskews into its OWN coverslip frame, offset from the global
    # shear-only frame by a pure translation (shear_only_subblock_offset).
    off_zc, off_yc, off_xc = shear_only_subblock_offset(
        scan_start, rawy_start, rawx_start, angle_in_degrees,
        voxel_size_z, voxel_size_y, voxel_size_x, skew_name,
    )
    roi_origin = np.asarray(crop_bounding_box).min(axis=0)  # [x0, y0, z0, 1]
    x0, y0, z0 = float(roi_origin[0]), float(roi_origin[1]), float(roi_origin[2])
    dz_roi, dy_roi, dx_roi = (int(v) for v in crop_vol_shape)  # (z, y, x) extents

    def _trim(vol):
        # Negative start ⇒ zero-pad the leading edge to keep ROI-origin alignment.
        zc_start_raw = int(round(z0 - off_zc))
        yc_start_raw = int(round(y0 - off_yc))
        xc_start_raw = int(round(x0 - off_xc))

        pre_z = max(-zc_start_raw, 0)
        pre_y = max(-yc_start_raw, 0)
        pre_x = max(-xc_start_raw, 0)

        zc_start = max(zc_start_raw, 0)
        yc_start = max(yc_start_raw, 0)
        xc_start = max(xc_start_raw, 0)

        # How many voxels to take from the sub-block after accounting for leading pad
        take_z = dz_roi - pre_z
        take_y = dy_roi - pre_y
        take_x = dx_roi - pre_x

        out = vol[
            zc_start:zc_start + take_z,
            yc_start:yc_start + take_y,
            xc_start:xc_start + take_x,
        ]
        # Prepend zero-padding on leading edges that extend before the sub-block origin
        if pre_z or pre_y or pre_x:
            out = np.pad(out, ((pre_z, 0), (pre_y, 0), (pre_x, 0)))
        # Append zero-padding on trailing edges if sub-block was too short
        pad_z = max(dz_roi - out.shape[0], 0)
        pad_y = max(dy_roi - out.shape[1], 0)
        pad_x = max(dx_roi - out.shape[2], 0)
        if pad_z or pad_y or pad_x:
            out = np.pad(out, ((0, pad_z), (0, pad_y), (0, pad_x)))
        return out

    deskewed_crop = _trim(deskewed_prelim)

    if debug:
        return deskewed_crop, deskewed_prelim
    elif get_deskew_and_decon:
        deskewed_crop_no_decon = _trim(_deskew(crop_volume_no_decon))
        return deskewed_crop, deskewed_crop_no_decon
    else:
        return deskewed_crop


# Get reverse affine transform by rotating around a user-specified volume


def get_inverse_affine_transform(
    original_volume,
    angle_in_degrees,
    voxel_x,
    voxel_y,
    voxel_z,
    skew_dir=DeskewDirection.Y,
):
    """
    Calculate the inverse deskew transform and the excess z_bounds

    Args:
        original_volume (_type_): unprocessed volume
        angle_in_degrees (_type_): _description_
        voxel_x (_type_): _description_
        voxel_y (_type_): _description_
        voxel_z (_type_): _description_
        skew_dir: Direction of skew

    Returns:
        Inverse Affine transform (AffineTransform3D), int: Excess z slices, Deskew transform (AffineTransform3D)
    """
    # calculate the deskew transform for specified volume
    if skew_dir == DeskewDirection.Y:
        deskew_transform = _deskew_y_vol_transform(
            original_volume, angle_in_degrees, voxel_x, voxel_y, voxel_z
        )
    elif skew_dir == DeskewDirection.X:
        deskew_transform = _deskew_x_vol_transform(
            original_volume, angle_in_degrees, voxel_x, voxel_y, voxel_z
        )

    # Get the deskew transform after bringing the volume into bounds
    (
        deskewed_shape,
        new_deskew_transform,
        _,
    ) = determine_translation_and_bounding_box(
        original_volume, deskew_transform
    )

    # Get the inverse of adjusted deskew transform
    deskew_inverse = new_deskew_transform.inverse()

    # We use the shape of deskewed volume to get the new vertices of deskewed volume in x,y and z
    from itertools import product

    nz, ny, nx = deskewed_shape
    deskewed_bounding_box = [
        list(x) + [1] for x in product((0, nx), (0, ny), (0, nz))
    ]

    # transform the corners of deskewed volume using the reverse affine transform
    undeskew_bounding_box = np.asarray(
        list(map(lambda x: deskew_inverse._matrix @ x, deskewed_bounding_box))
    )

    # Get the maximum z value and subtract it from shape of original volume to get excess bounds of bounding box
    max_bounds = undeskew_bounding_box.max(axis=0).astype(int)
    rev_deskew_z = max_bounds[2]
    extra_bounds = int((rev_deskew_z - original_volume.shape[0]))

    return deskew_inverse, extra_bounds, deskew_transform


# Get deskew transform where rotation is around centre of "original_volume"


def _deskew_y_vol_transform(
    original_volume,
    angle_in_degrees: float = 30,
    voxel_size_x: float = 1,
    voxel_size_y: float = 1,
    voxel_size_z: float = 1,
    scale_factor: float = 1,
):
    """Return deskew transform for specified volume when skew direction is Y
       Rotation is performed around centre of "original_volume"
    Args:
        crop ([type]): Volume to deskew (zyx)
        original_volume ([type]): Reference volume around with to perform rotation (zyx)
        angle_in_degrees (float): Deskewing angle
        voxel_size_x (float, optional): [description]. Defaults to 1.
        voxel_size_y (float, optional): [description]. Defaults to 1.
        voxel_size_z (float, optional): [description]. Defaults to 1.
        scale_factor (float, optional): [description]. Defaults to 1.

    Returns:
        AffineTransform3D
    """
    import math

    transform = AffineTransform3D()

    # shear factor for deskewing
    shear_factor = math.sin((90 - angle_in_degrees) * math.pi / 180.0) * (
        voxel_size_z / voxel_size_y
    )
    transform._matrix[1, 2] = shear_factor

    # make voxels isotropic, calculate the new scaling factor for Z after shearing
    # https://github.com/tlamberimage3/napari-ndtiffs/blob/092acbd92bfdbf3ecb1eb9c7fc146411ad9e6aae/napari_ndtiffs/affine.py#L57
    new_dz = math.sin(angle_in_degrees * math.pi / 180.0) * voxel_size_z
    scale_factor_z = (new_dz / voxel_size_y) * scale_factor
    transform.scale(
        scale_x=scale_factor, scale_y=scale_factor, scale_z=scale_factor_z
    )

    # rotation around centre of ref_vol
    # transform._concatenate(rotate_around_vol_mat(original_volume, (0-angle_in_degrees)))
    transform.rotate(angle_in_degrees=0 - angle_in_degrees, axis=0)
    # correct orientation so that the new Z-plane goes proximal-distal from the objective.

    return transform


# Get deskew transform where rotation is around centre of "original_volume"
def _deskew_x_vol_transform(
    original_volume,
    angle_in_degrees: float = 30,
    voxel_size_x: float = 1,
    voxel_size_y: float = 1,
    voxel_size_z: float = 1,
    scale_factor: float = 1,
):
    """Return deskew transform for specified volume when skew direction is X
       Rotation is performed around centre of "original_volume"
    Args:
        crop ([type]): Volume to deskew (zyx)
        original_volume ([type]): Reference volume around with to perform rotation (zyx)
        angle_in_degrees (float): Deskewing angle
        voxel_size_x (float, optional): [description]. Defaults to 1.
        voxel_size_y (float, optional): [description]. Defaults to 1.
        voxel_size_z (float, optional): [description]. Defaults to 1.
        scale_factor (float, optional): [description]. Defaults to 1.

    Returns:
        AffineTransform3D
    """
    import math

    transform = AffineTransform3D()

    # shear factor for deskewing
    shear_factor = math.sin((90 - angle_in_degrees) * math.pi / 180.0) * (
        voxel_size_z / voxel_size_x
    )
    transform._matrix[0, 2] = shear_factor

    # make voxels isotropic, calculate the new scaling factor for Z after shearing
    # https://github.com/tlamberimage3/napari-ndtiffs/blob/092acbd92bfdbf3ecb1eb9c7fc146411ad9e6aae/napari_ndtiffs/affine.py#L57
    new_dz = math.sin(angle_in_degrees * math.pi / 180.0) * voxel_size_z
    scale_factor_z = (new_dz / voxel_size_x) * scale_factor

    transform.scale(
        scale_x=scale_factor, scale_y=scale_factor, scale_z=scale_factor_z
    )

    # rotation around centre of ref_vol
    transform.rotate(angle_in_degrees=angle_in_degrees, axis=1)
    # correct orientation so that the new Z-plane goes proximal-distal from the objective.

    return transform


# deprecated
# Calculate rotation transform around a volume
def rotate_around_vol_mat(ref_vol, angle_in_degrees: float = 30.0):
    """Return the rotation matrix , so its rotated around centre of ref_vol

    Args:
        ref_vol (tuple): Shape of the ref volume (zyx)
        angle_in_degrees (float, optional): [description]. Defaults to 30.0.

    Returns:
        Rotation matrix: Will be returned in the form xyz for clesperanto affine transforms
    """
    angle_in_rad = angle_in_degrees * np.pi / 180.0
    # rotate_transform = cle.AffineTransform3D()
    # rotate_transform._matrix
    # first translate the middle of the image to the origin
    nz, ny, nx = ref_vol.shape
    T1 = np.array(
        [[1, 0, 0, nx / 2], [0, 1, 0, ny / 2], [0, 0, 1, nz / 2], [0, 0, 0, 1]]
    )

    R = np.array(
        [
            [1, 0, 0, 0],
            [0, np.cos(angle_in_rad), np.sin(angle_in_rad), 0],
            [0, -np.sin(angle_in_rad), np.cos(angle_in_rad), 0],
            [0, 0, 0, 1],
        ]
    )

    T2 = np.array(
        [
            [1, 0, 0, -nx / 2],
            [0, 1, 0, -ny / 2],
            [0, 0, 1, -nz / 2],
            [0, 0, 0, 1],
        ]
    )
    T = np.eye(4)
    rotate_mat = np.dot(np.dot(np.dot(T, T1), R), T2)
    # print(rotate_mat)
    return rotate_mat


def _yield_arr_slice(img):
    """
    Create an array generator that yields each z slice
    """
    img = np.squeeze(img)
    assert img.ndim == 3, f"Image needs to be 3D. Got {img.ndim}"

    for slice in img:
        yield slice
