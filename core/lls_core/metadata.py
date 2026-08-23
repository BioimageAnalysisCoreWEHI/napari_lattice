"""
Coordinate-transform and provenance metadata for output files.

Every image a `Writer` produces gets a sibling `<name>.lattice.json` recording the deskew
geometry that was applied and where the output sits in the full deskewed volume. See
`docs/miscellaneous/output_metadata.md` for the document layout and how to consume it.
"""
from __future__ import annotations

import json
from enum import Enum
from pathlib import Path, PurePath
from typing import TYPE_CHECKING, Any, Optional, Tuple

import numpy as np

if TYPE_CHECKING:
    from lls_core.models.lattice_data import LatticeData

#: Bumped when the document layout changes in a way consumers must notice.
SCHEMA_VERSION = "1.0"

SIDECAR_SUFFIX = ".lattice.json"

#: Stripped when naming a sidecar. Listed explicitly rather than removing every suffix,
#: so a dot in the base name (`sample.v2_deskewed.ome.tif`) survives.
_OUTPUT_SUFFIXES = frozenset({".ome", ".tif", ".tiff", ".zarr", ".h5"})

AFFINE_CONVENTION = (
    "4x4 row-major homogeneous matrix in ZYX order, mapping a RAW voxel index "
    "[z, y, x, 1] to a voxel index in the full deskewed volume. Multiply componentwise "
    "by output_voxel_size_um for microns. This records the transform that was APPLIED - "
    "the saved pixels are already deskewed, so it relates them back to the raw "
    "acquisition rather than being something to apply again."
)

ORIGIN_REFERENCE = (
    "Voxel (0, 0, 0) of the full deskewed volume of this acquisition, in deskewed voxels."
)


def _jsonable(value: Any) -> Any:
    """
    Coerce a pydantic `.dict()` tree into something `json.dumps` accepts.

    numpy scalars are the main hazard: the repo pins `numpy<2`, where `np.float32` and
    `np.int64` both raise from the stdlib encoder rather than degrading.
    """
    # Checked before `str`: the StrEnums here are `str` subclasses and would otherwise
    # never reach this branch. Everything round-trips by NAME - `DeskewDirection` is an
    # IntEnum whose value is a meaningless ordinal (Y == 2).
    if isinstance(value, Enum):
        return value.name
    if value is None or isinstance(value, (bool, str)):
        return value
    if isinstance(value, range):
        # The config schema spells a range as [start, stop] - see the YAML examples.
        return [int(value.start), int(value.stop)]
    if isinstance(value, PurePath):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, (int, float)):
        return value
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return str(value)


def build_config(lattice: "LatticeData") -> dict:
    """
    The run configuration, in the same schema as `--json-config` / `--yaml-config`.

    `input_image`, `workflow` and `deconvolution.psf` hold loaded objects that cannot be
    serialised back to the config that produced them, so the source path is recorded
    instead - or `null` where the input was supplied in memory.
    """
    config = lattice.dict(exclude={"input_image", "input_image_path", "derived", "workflow_path"})

    config["input_image"] = str(lattice.input_image_path) if lattice.input_image_path else None
    if lattice.workflow is not None:
        config["workflow"] = str(lattice.workflow_path) if lattice.workflow_path else None

    decon = config.get("deconvolution")
    if isinstance(decon, dict):
        psf_paths = decon.pop("psf_paths", None)
        # `[]` rather than `None` matches the declared `List` type. Note that when the
        # PSFs were passed in memory there are no paths to record, and such a config
        # cannot be re-run: a validator requires one PSF per channel.
        decon["psf"] = [str(p) for p in psf_paths] if psf_paths else []

    # A console affordance, not a property of the data
    config.pop("progress_bar", None)

    return _jsonable(config)


def _crops(lattice: "LatticeData") -> bool:
    """
    Whether this output is one ROI of a cropped run.

    `save_mip` is excluded because `LatticeData.save()` projects straight from the raw
    data and ignores cropping, so an attached `crop` describes nothing about the output.
    """
    return bool(lattice.cropping_enabled and lattice.crop is not None and not lattice.save_mip)


def output_origin_zyx(
    lattice: "LatticeData", roi_index: Optional[int] = None
) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """
    Position of this output's voxel (0, 0, 0) in the full deskewed volume, in deskewed
    voxels.

    An uncropped run is the whole volume, so its origin is the origin. A MIP has
    collapsed Z, so its Z origin is `None` rather than a number that would invite misuse.
    """
    if lattice.save_mip:
        return (None, 0.0, 0.0)
    if not _crops(lattice):
        return (0.0, 0.0, 0.0)

    from lls_core.estimate import _roi_to_shape_array
    from lls_core.utils import calculate_crop_bbox

    index = 0 if roi_index is None else int(roi_index)
    roi_shape = _roi_to_shape_array(lattice.crop.roi_list[index])
    z_start, z_end = lattice.crop.z_range

    if not lattice.coverslip_rotation:
        # The shear-only trim aligns all three axes to the ROI, zero-padding where the
        # sub-block falls short, so the origin is just the ROI origin. See the `_trim`
        # closure in `_crop_volume_deskew_shear_only`.
        crop_bounding_box, _ = calculate_crop_bbox(roi_shape, z_start, z_end)
        x0, y0, z0 = np.asarray(crop_bounding_box).min(axis=0)[:3]
        return (float(z0), float(y0), float(x0))

    # The objective branch trims only the skew axis to the ROI, so its origin is not the
    # ROI's - read it off the same helper the crop uses rather than re-deriving it.
    from lls_core.llsz_core import objective_crop_geometry

    return objective_crop_geometry(
        raw_shape_zyx=tuple(int(s) for s in lattice.get_3d_slice().shape[-3:]),
        roi_shape=roi_shape,
        z_start=z_start,
        z_end=z_end,
        angle_in_degrees=lattice.angle,
        voxel_size_x=lattice.dx,
        voxel_size_y=lattice.dy,
        voxel_size_z=lattice.dz,
        skew_dir=lattice.skew_dir,
    ).origin_zyx


def sidecar_path(image_path: Path) -> Path:
    """
    `<output name without its extension>.lattice.json`, beside the output.

    The extension is stripped rather than appended to: `img.ome.tif.lattice.json` would
    still match a `*.tif*` glob, so anything scanning a results directory for images
    would pick the sidecar up and try to read JSON as an image.
    """
    name = image_path.name
    while True:
        stem, dot, suffix = name.rpartition(".")
        if not dot or ("." + suffix.lower()) not in _OUTPUT_SUFFIXES:
            break
        name = stem
    return image_path.with_name(name + SIDECAR_SUFFIX)


def write_sidecar(
    lattice: "LatticeData", image_path: Path, roi_index: Optional[int] = None
) -> Path:
    """Write the metadata sidecar beside `image_path` and return its path."""
    from lls_core import __version__
    from lls_core.estimate import _roi_to_shape_array

    cropped = _crops(lattice)
    index = (0 if roi_index is None else int(roi_index)) if cropped else None

    origin_px = output_origin_zyx(lattice, index)
    voxel = (float(lattice.new_dz), float(lattice.dy), float(lattice.dx))
    origin_um = [None if o is None else o * v for o, v in zip(origin_px, voxel)]

    roi = None
    if cropped:
        corners = _roi_to_shape_array(lattice.crop.roi_list[index])
        ys, xs = corners[:, 0], corners[:, 1]
        roi = {
            "index": index,
            # CropParams normalises every ROI source to deskewed pixels on the way in
            "units": "Pixels",
            "bbox_yx_px": {
                "top": float(ys.min()), "left": float(xs.min()),
                "bottom": float(ys.max()), "right": float(xs.max()),
            },
            "z_range": _jsonable(lattice.crop.z_range),
        }

    affine = lattice.derived.deskew_affine_transform_zyx if lattice.derived else None
    document = {
        "schema_version": SCHEMA_VERSION,
        "generator": {"name": "napari-lattice", "version": __version__},
        "output": {
            "path": image_path.name,
            "roi_index": index,
            "projection": "mip" if lattice.save_mip else None,
            "origin_zyx_px": _jsonable(list(origin_px)),
            "origin_zyx_um": _jsonable(origin_um),
            "origin_reference": ORIGIN_REFERENCE,
        },
        "roi": roi,
        "derived": {
            "output_voxel_size_um": {"z": voxel[0], "y": voxel[1], "x": voxel[2]},
            "full_output_shape_zyx": _jsonable(
                lattice.derived.deskew_vol_shape if lattice.derived else None
            ),
            "raw_to_deskewed_affine_zyx": (
                _jsonable(np.asarray(affine)) if affine is not None else None
            ),
            "affine_convention": AFFINE_CONVENTION,
        },
        "config": build_config(lattice),
    }

    path = sidecar_path(image_path)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(document, handle, indent=2)
    return path
