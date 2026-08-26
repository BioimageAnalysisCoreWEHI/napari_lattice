"""
Tests for the `.lattice.json` metadata sidecar.

The load-bearing test is `test_recorded_origin_matches_pixels`: it deskews a full volume
and an ROI of it independently, finds where the ROI *actually* landed by scanning for the
best-matching window, and asserts the recorded origin agrees. That checks the metadata
against pixel reality rather than a restatement of the same formula, which is the only way
to catch the origin drifting from what the crop does.
"""
import json

import numpy as np
import pytest
import pyclesperanto as cle
from xarray import DataArray

from lls_core import DeskewDirection
from lls_core.metadata import SIDECAR_SUFFIX, build_config, output_origin_zyx, sidecar_path
from lls_core.models.lattice_data import LatticeData
from lls_core.models.output import SaveFileType
from tests.utils import requires_real_gpu


def _image(shape=(1, 1, 30, 40, 35)):
    rng = np.random.default_rng(0)
    return DataArray(rng.integers(0, 500, shape, dtype=np.uint16), dims=("T", "C", "Z", "Y", "X"))


def _lattice(tmp_path, **kwargs):
    params = dict(
        input_image=_image(),
        physical_pixel_sizes=(0.3, 0.15, 0.15),
        save_dir=tmp_path,
        save_name="meta",
        progress_bar=False,
    )
    params.update(kwargs)
    return LatticeData(**params)


def _sidecars(directory):
    return sorted(directory.glob("*" + SIDECAR_SUFFIX))


def _only_sidecar(directory):
    found = _sidecars(directory)
    assert len(found) == 1, f"expected one sidecar, got {[p.name for p in found]}"
    return json.loads(found[0].read_text(encoding="utf-8"))


@pytest.mark.parametrize("save_type", [SaveFileType.tiff, SaveFileType.h5, SaveFileType.omezarr])
def test_sidecar_written_for_each_format(tmp_path, save_type):
    """Every format gets a parseable sidecar that names the file it sits beside."""
    _lattice(tmp_path, save_type=save_type, angle=32.5).save()
    document = _only_sidecar(tmp_path)

    # Must name the file actually written, not a recomputed name
    assert (tmp_path / document["output"]["path"]).exists()

    derived = document["derived"]
    matrix = np.asarray(derived["raw_to_deskewed_affine_zyx"])
    assert matrix.shape == (4, 4)
    assert np.allclose(matrix[3], [0, 0, 0, 1]), "not a homogeneous matrix"
    assert derived["output_voxel_size_um"]["y"] == pytest.approx(0.15)


@pytest.mark.parametrize("filename,expected", [
    ("img_deskewed.ome.tif", "img_deskewed.lattice.json"),
    # A dot in the base name must survive the stripping
    ("sample.v2_deskewed.ome.zarr", "sample.v2_deskewed.lattice.json"),
])
def test_sidecar_name_drops_the_image_extension(tmp_path, filename, expected):
    """
    Appending would leave `img.ome.tif.lattice.json`, which still matches a `*.tif*` glob,
    so anything scanning a results directory for images would try to read it as an image.
    """
    assert sidecar_path(tmp_path / filename).name == expected


def test_origins_for_uncropped_and_mip(tmp_path):
    """The two cases with no ROI geometry to resolve."""
    _lattice(tmp_path, save_type=SaveFileType.tiff).save()
    document = _only_sidecar(tmp_path)
    assert document["output"]["origin_zyx_px"] == [0.0, 0.0, 0.0]
    assert document["roi"] is None

    mip_dir = tmp_path / "mip"
    mip_dir.mkdir()
    _lattice(mip_dir, save_dir=mip_dir, save_type=SaveFileType.tiff, save_mip=True).save()
    output = _only_sidecar(mip_dir)["output"]
    assert output["projection"] == "mip"
    # Z is projected away, so a Z position would be meaningless rather than unknown
    assert output["origin_zyx_px"][0] is None


def test_mip_ignores_an_attached_crop(tmp_path):
    """
    `LatticeData.save()` projects a MIP straight from the raw data and ignores cropping,
    so an attached crop describes nothing about the output. Reporting it would contradict
    the whole-FOV origin in the same document.
    """
    roi = np.array([[5.0, 4.0], [5.0, 20.0], [24.0, 20.0], [24.0, 4.0]])
    _lattice(tmp_path, save_type=SaveFileType.tiff, save_mip=True,
             crop={"roi_list": [roi], "z_range": (2, 12)}).save()

    document = _only_sidecar(tmp_path)
    assert document["roi"] is None
    assert document["output"]["roi_index"] is None
    assert document["output"]["origin_zyx_px"] == [None, 0.0, 0.0]


def test_each_roi_gets_its_own_placement(tmp_path):
    rois = [
        np.array([[5.0, 4.0], [5.0, 20.0], [24.0, 20.0], [24.0, 4.0]]),
        np.array([[40.0, 6.0], [40.0, 22.0], [60.0, 22.0], [60.0, 6.0]]),
    ]
    _lattice(tmp_path, save_type=SaveFileType.tiff,
             crop={"roi_list": rois, "z_range": (2, 12)}).save()

    documents = [json.loads(p.read_text(encoding="utf-8")) for p in _sidecars(tmp_path)]
    by_index = {d["output"]["roi_index"]: d for d in documents}
    assert set(by_index) == {0, 1}
    assert (by_index[0]["output"]["origin_zyx_px"]
            != by_index[1]["output"]["origin_zyx_px"]), "distinct ROIs reported one origin"

    roi = by_index[1]["roi"]
    assert roi["bbox_yx_px"]["top"] == pytest.approx(40.0)
    assert roi["z_range"] == [2, 12]


def test_config_block_round_trips(tmp_path, image_workflow):
    """The `config` block must be usable as `--json-config`."""
    lattice = _lattice(
        tmp_path,
        save_type=SaveFileType.tiff,
        angle=32.0,
        skew="X",
        invert_scan_direction=True,
        crop={"roi_list": [np.array([[5.0, 4.0], [5.0, 20.0], [24.0, 20.0], [24.0, 4.0]])],
              "z_range": (1, 9)},
    )
    lattice.save()
    config = _only_sidecar(tmp_path)["config"]

    # input_image is null here (the image was passed in memory), so supply one to satisfy
    # the required field; everything else must survive the same parsing the CLI does.
    rebuilt = LatticeData.parse_obj({**config, "input_image": _image(), "progress_bar": False})
    assert rebuilt.angle == 32.0
    assert rebuilt.skew == DeskewDirection.X
    assert rebuilt.invert_scan_direction is True
    assert rebuilt.save_type == SaveFileType.tiff
    assert rebuilt.crop.z_range == (1, 9)
    assert list(rebuilt.time_range) == list(lattice.time_range)
    np.testing.assert_allclose(np.asarray(rebuilt.crop.roi_list[0]),
                               np.asarray(lattice.crop.roi_list[0]))

    # A Workflow object cannot be serialised, so the source path is what gets recorded
    from napari_workflows._io_yaml_v1 import save_workflow

    workflow_path = tmp_path / "flow.yml"
    save_workflow(str(workflow_path), image_workflow)
    with_workflow = _lattice(tmp_path, save_type=SaveFileType.tiff, workflow=workflow_path)
    assert build_config(with_workflow)["workflow"] == str(workflow_path)


def test_shear_only_origin_is_the_roi_origin(tmp_path):
    """
    The shear-only trim aligns all three axes to the ROI (zero-padding the leading edge
    when needed), so unlike the objective branch its origin is exactly the ROI origin.
    """
    roi = np.array([[12.0, 8.0], [12.0, 28.0], [34.0, 28.0], [34.0, 8.0]])
    lattice = _lattice(tmp_path, coverslip_rotation=False, save_type=SaveFileType.tiff,
                       crop={"roi_list": [roi], "z_range": (3, 15)})
    assert output_origin_zyx(lattice, roi_index=0) == pytest.approx((3.0, 12.0, 8.0))


@requires_real_gpu
@pytest.mark.gpu_state_risk
@pytest.mark.parametrize("skew", ["Y", "X"])
def test_recorded_origin_matches_pixels(tmp_path, skew):
    """
    The origin in the sidecar must be where the crop's pixels really are.

    Verified by sliding the crop along the full deskewed volume and taking the offset with
    the lowest error, so this fails if the recorded origin drifts from what
    `crop_volume_deskew` does - the trap being that only the skew axis is trimmed to the
    ROI, so the other two axes' origins are not the ones that were asked for.
    """
    angle = 30.0
    skew_dir = DeskewDirection.Y if skew == "Y" else DeskewDirection.X

    # Uniform background with off-centre markers, wide along the shear axis
    if skew == "Y":
        raw = np.full((90, 260, 90), 30.0, np.float32)
        for z, a in [(20, 60), (45, 130), (70, 200)]:
            raw[z - 3:z + 3, a - 6:a + 6, 40:50] = 800.0
    else:
        raw = np.full((90, 90, 260), 30.0, np.float32)
        for z, a in [(20, 60), (45, 130), (70, 200)]:
            raw[z - 3:z + 3, 40:50, a - 6:a + 6] = 800.0

    deskew = cle.deskew_y if skew == "Y" else cle.deskew_x
    full = np.asarray(cle.pull(deskew(
        raw, angle=angle, voxel_size_x=1, voxel_size_y=1, voxel_size_z=1)))

    axis = 1 if skew == "Y" else 2          # sheared (wide) output axis
    other = 2 if skew == "Y" else 1
    size = 50
    a0 = int(np.clip(0.7 * full.shape[axis] - size / 2, 0, full.shape[axis] - size))
    b0 = full.shape[other] // 2 - size // 2

    if skew == "Y":
        roi = np.array([[a0, b0], [a0, b0 + size], [a0 + size, b0 + size], [a0 + size, b0]], float)
    else:
        roi = np.array([[b0, a0], [b0 + size, a0], [b0 + size, a0 + size], [b0, a0 + size]], float)

    _lattice(
        tmp_path,
        input_image=DataArray(raw[np.newaxis, np.newaxis], dims=("T", "C", "Z", "Y", "X")),
        physical_pixel_sizes=(1.0, 1.0, 1.0),
        save_type=SaveFileType.tiff,
        angle=angle,
        skew=skew_dir,
        crop={"roi_list": [roi], "z_range": (0, full.shape[0])},
    ).save()

    document = _only_sidecar(tmp_path)
    recorded = document["output"]["origin_zyx_px"]

    import tifffile
    crop = np.asarray(tifffile.imread(tmp_path / document["output"]["path"])).astype(np.float32)
    crop = crop.reshape(crop.shape[-3:])   # (T=1, C=1, Z, Y, X) -> (Z, Y, X)

    depth = min(crop.shape[0], full.shape[0])
    assert crop[:depth].std() > 1.0, "crop has no structure; the test would be vacuous"

    width = crop.shape[axis]
    fixed = int(round(recorded[other]))

    def window(k):
        if skew == "Y":
            return full[:depth, k:k + width, fixed:fixed + crop.shape[2]]
        return full[:depth, fixed:fixed + crop.shape[1], k:k + width]

    errors = [float(np.mean(np.abs(window(k) - crop[:depth])))
              for k in range(max(0, full.shape[axis] - width + 1))]
    empirical = int(np.argmin(errors))

    assert abs(empirical - recorded[axis]) <= 1, (
        f"sidecar says the crop starts at {recorded[axis]:.2f} on axis {axis}, but its "
        f"pixels match the full volume best at {empirical} (skew={skew})"
    )
