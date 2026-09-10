"""
ROI input: napari shapes CSV, and the pixel/micron unit conversion.

ROI files carry no unit. ImageJ writes pixels; a napari shapes CSV saved from the
plugin's crop layer is microns, because that layer is unscaled while the image layer
carries the pixel size. Everything downstream of `CropParams` assumes pixels, so the
unit is declared on the way in and converted once.
"""
from __future__ import annotations

import numpy as np
import pytest
from xarray import DataArray

from lls_core.cropping import (
    Roi, RoiUnits, read_napari_csv, scale_rois, units_for_path,
)
from lls_core.models.crop import CropParams
from lls_core.models.lattice_data import LatticeData

CSV = """index,shape-type,vertex-index,axis-0,axis-1
0,polygon,0,10.0,20.0
0,polygon,1,10.0,50.0
0,polygon,2,40.0,50.0
0,polygon,3,40.0,20.0
1,polygon,0,60.0,60.0
1,polygon,1,60.0,80.0
1,polygon,2,90.0,80.0
1,polygon,3,90.0,60.0
"""


def _write(tmp_path, text, name="shapes.csv"):
    path = tmp_path / name
    path.write_text(text)
    return path


def test_reads_a_napari_shapes_csv(tmp_path):
    rois = read_napari_csv(_write(tmp_path, CSV))
    assert rois == [
        Roi((10.0, 20.0), (10.0, 50.0), (40.0, 50.0), (40.0, 20.0)),
        Roi((60.0, 60.0), (60.0, 80.0), (90.0, 80.0), (90.0, 60.0)),
    ]


def test_non_rectangular_shape_becomes_its_bounding_box(tmp_path):
    triangle = ("index,shape-type,vertex-index,axis-0,axis-1\n"
                "0,polygon,0,10.0,20.0\n"
                "0,polygon,1,40.0,35.0\n"
                "0,polygon,2,25.0,50.0\n")
    assert read_napari_csv(_write(tmp_path, triangle)) == [
        Roi((10.0, 20.0), (10.0, 50.0), (40.0, 50.0), (40.0, 20.0))
    ]


def test_leading_axes_of_a_3d_shape_are_ignored(tmp_path):
    # The plugin's own shape-to-ROI conversion takes the last two axes; match it.
    with_z = ("index,shape-type,vertex-index,axis-0,axis-1,axis-2\n"
              "0,polygon,0,7.0,10.0,20.0\n"
              "0,polygon,1,7.0,40.0,50.0\n")
    assert read_napari_csv(_write(tmp_path, with_z)) == [
        Roi((10.0, 20.0), (10.0, 50.0), (40.0, 50.0), (40.0, 20.0))
    ]


def test_a_csv_that_is_not_a_shapes_layer_is_rejected(tmp_path):
    with pytest.raises(Exception, match="not a napari shapes CSV"):
        read_napari_csv(_write(tmp_path, "x,y\n1,2\n"))


def _lattice(tmp_path, rois, units, dy=0.5):
    return LatticeData(
        input_image=DataArray(np.zeros((30, 100, 100), dtype=np.uint16), dims=["Z", "Y", "X"]),
        physical_pixel_sizes=(1, dy, dy),
        save_name="t", save_dir=str(tmp_path), save_type="tiff",
        crop=CropParams(roi_list=rois, roi_units=units, z_range=(0, 5)),
    )


ROI = Roi((10.0, 20.0), (10.0, 50.0), (40.0, 50.0), (40.0, 20.0))


def test_pixel_rois_are_left_alone(tmp_path):
    assert _lattice(tmp_path, [ROI], RoiUnits.Pixels).crop.roi_list == [ROI]


def test_micron_rois_are_converted_to_pixels(tmp_path):
    # dy = 0.5 um, so a 10 um coordinate is pixel 20.
    lattice = _lattice(tmp_path, [ROI], RoiUnits.Microns)
    assert lattice.crop.roi_list == scale_rois([ROI], 2.0)
    assert lattice.crop.roi_units == RoiUnits.Pixels


def test_auto_takes_the_unit_from_the_file_type(tmp_path):
    assert units_for_path("rois.zip") == RoiUnits.Pixels
    assert units_for_path("rois.roi") == RoiUnits.Pixels
    assert units_for_path("shapes.CSV") == RoiUnits.Microns

    csv = _write(tmp_path, CSV)
    assert CropParams(roi_list=[csv], z_range=(0, 5)).roi_units == RoiUnits.Microns
    # ...and the lattice then converts them, so a CSV needs no flag to land correctly.
    lattice = _lattice(tmp_path, [csv], RoiUnits.Auto)
    assert lattice.crop.roi_list == scale_rois(read_napari_csv(csv), 2.0)


@pytest.mark.parametrize("given", ["pixels", "PIXEL"])  # canonical, then miscased singular
def test_cli_accepts_either_spelling_of_a_unit(given):
    from lls_core.cmds.__main__ import RoiUnitsChoice

    choice = RoiUnitsChoice([unit.value for unit in RoiUnits], case_sensitive=False)
    assert choice.convert(given, None, None) == RoiUnits.Pixels


def test_cli_still_rejects_an_unknown_unit():
    import click
    from lls_core.cmds.__main__ import RoiUnitsChoice

    choice = RoiUnitsChoice([unit.value for unit in RoiUnits], case_sensitive=False)
    with pytest.raises(click.UsageError):
        choice.convert("furlongs", None, None)


def test_auto_with_no_files_means_pixels(tmp_path):
    # Coordinates passed directly by an API caller are pixels by convention.
    lattice = _lattice(tmp_path, [ROI], RoiUnits.Auto)
    assert lattice.crop.roi_units == RoiUnits.Pixels
    assert lattice.crop.roi_list == [ROI]


def test_auto_refuses_to_guess_across_mixed_file_types(tmp_path):
    imagej = tmp_path / "rois.zip"
    imagej.write_bytes(b"")
    with pytest.raises(ValueError, match="set roi_units explicitly"):
        CropParams(roi_list=[_write(tmp_path, CSV), imagej], z_range=(0, 5))


def test_rois_outside_the_image_are_called_out(tmp_path, caplog):
    """
    The usual cause is the wrong unit. Without this the run dies later inside the
    writer with 'truncate can only be used with imagej or shaped formats'.
    """
    import logging

    far = scale_rois([ROI], 100.0)
    with caplog.at_level(logging.WARNING, logger="lls_core.models.lattice_data"):
        _lattice(tmp_path, far, RoiUnits.Pixels)
    assert any("roi_units" in r.getMessage() for r in caplog.records), caplog.records


def test_conversion_does_not_repeat_when_the_model_is_revalidated(tmp_path):
    # Parallel workers re-validate a copy of the lattice; converting twice would move
    # every ROI by another 1/dy.
    lattice = _lattice(tmp_path, [ROI], RoiUnits.Microns)
    converted = lattice.crop.roi_list
    assert lattice.copy().crop.roi_list == converted
    assert LatticeData.model_validate(dict(lattice)).crop.roi_list == converted
