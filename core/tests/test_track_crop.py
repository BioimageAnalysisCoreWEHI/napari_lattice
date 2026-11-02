"""
Tests for track-driven cropping (crop window follows a TrackMate track over time).

Covers the helpers and model wiring added for per-timepoint ROIs:
`track_to_rois`, `clamp_rois_to_image`, `CropParams.roi_by_time` /
`roi_for_time`, and the LatticeData validators that scale, clamp, warn and
restrict the time range to the track.
"""
from __future__ import annotations

import logging

import numpy as np
import pytest
from pydantic.v1 import ValidationError
from xarray import DataArray

from lls_core.cropping import (
    Roi,
    RoiUnits,
    clamp_rois_to_image,
    scale_rois,
    track_to_rois,
)
from lls_core.models.crop import CropParams
from lls_core.models.lattice_data import LatticeData


# A TrackMate-driven crop carries a zero ROI in `roi_list` as a placeholder;
# the real per-timepoint windows live in `roi_by_time`, keyed by track then time.
_PLACEHOLDER_ROI = Roi((0, 0), (0, 0), (0, 0), (0, 0))


def _track(*rows):
    """Build an (n, 4) [t, x, y, z] microns array from row tuples."""
    return np.asarray(rows, dtype=float)


def _roi_size(roi: Roi) -> tuple[float, float]:
    ys = [y for y, _ in roi]
    xs = [x for _, x in roi]
    return max(ys) - min(ys), max(xs) - min(xs)


def _roi_bounds(roi: Roi) -> tuple[float, float, float, float]:
    ys = [y for y, _ in roi]
    xs = [x for _, x in roi]
    return min(ys), max(ys), min(xs), max(xs)


def _image(time_points=5, z=20, y=80, x=80):
    return DataArray(
        np.zeros((time_points, z, y, x), dtype=np.uint16),
        dims=["T", "Z", "Y", "X"],
    )


def _lattice(tmp_path, *, roi_by_time, time_points=5, z=20, y=80, x=80, dy=0.5, **kwargs):
    """Minimal LatticeData with a T axis so track time-range logic can run."""
    return LatticeData(
        input_image=_image(time_points, z, y, x),
        physical_pixel_sizes=(1.0, dy, dy),
        save_name="t",
        save_dir=str(tmp_path),
        save_type="tiff",
        crop=CropParams(roi_by_time={0: roi_by_time}, roi_list=[_PLACEHOLDER_ROI], roi_subset=[0], z_range=(0, 5)),
        **kwargs,
    )


def _static_lattice(tmp_path, roi, *, time_points=5, z=20, y=80, x=80):
    return LatticeData(
        input_image=_image(time_points, z, y, x),
        physical_pixel_sizes=(1.0, 1.0, 1.0),
        save_name="t",
        save_dir=str(tmp_path),
        save_type="tiff",
        crop=CropParams(roi_list=[roi], roi_units=RoiUnits.Pixels, z_range=(0, 5)),
    )


def _outside_warnings(caplog):
    return [r for r in caplog.records if "extend to" in r.getMessage()]


def test_track_to_rois_centres_a_fixed_window_on_each_point():
    track = _track([0, 10.0, 20.0, 1.0], [1, 12.0, 24.0, 2.0])
    rois = track_to_rois(track, window_size=10.0)

    assert set(rois) == {0, 1}
    # Points are (y, x): x=10, y=20 -> rows 15..25, columns 5..15.
    assert rois[0] == Roi((15.0, 5.0), (15.0, 15.0), (25.0, 15.0), (25.0, 5.0))
    assert rois[1] == Roi((19.0, 7.0), (19.0, 17.0), (29.0, 17.0), (29.0, 7.0))


def test_track_to_rois_ignores_z():
    a = track_to_rois(_track([0, 5.0, 5.0, 0.0]), window_size=4.0)
    b = track_to_rois(_track([0, 5.0, 5.0, 99.0]), window_size=4.0)
    assert a == b


def test_track_to_rois_casts_fractional_time_to_int():
    rois = track_to_rois(_track([2.7, 1.0, 1.0, 0.0]), window_size=2.0)
    assert list(rois) == [2]


def test_track_to_rois_leaves_gaps_unfilled():
    rois = track_to_rois(_track([0, 0.0, 0.0, 0.0], [2, 2.0, 2.0, 0.0]), window_size=2.0)
    assert set(rois) == {0, 2}


def test_track_to_rois_empty_track_yields_empty_dict():
    assert track_to_rois(_track(), window_size=10.0) == {}


def test_track_to_rois_zero_window_is_a_degenerate_point():
    rois = track_to_rois(_track([0, 3.0, 4.0, 0.0]), window_size=0.0)
    assert _roi_size(rois[0]) == (0.0, 0.0)
    assert rois[0].top_left == (4.0, 3.0)


def test_clamp_leaves_an_in_bounds_roi_alone():
    roi = Roi((10.0, 10.0), (10.0, 30.0), (30.0, 30.0), (30.0, 10.0))
    assert clamp_rois_to_image({0: roi}, height=100.0, width=100.0)[0] == roi


def test_clamp_leaves_a_roi_touching_the_edges_alone():
    roi = Roi((0.0, 0.0), (0.0, 100.0), (100.0, 100.0), (100.0, 0.0))
    assert _roi_bounds(clamp_rois_to_image({0: roi}, 100.0, 100.0)[0]) == (0.0, 100.0, 0.0, 100.0)


def test_clamp_slides_a_window_back_from_the_top_left_without_resizing():
    roi = Roi((-5.0, -10.0), (-5.0, 10.0), (15.0, 10.0), (15.0, -10.0))
    clamped = clamp_rois_to_image({0: roi}, height=100.0, width=100.0)[0]

    assert _roi_size(clamped) == _roi_size(roi)
    assert _roi_bounds(clamped) == (0.0, 20.0, 0.0, 20.0)


def test_clamp_slides_a_window_back_from_the_bottom_right_without_resizing():
    roi = Roi((90.0, 90.0), (90.0, 110.0), (110.0, 110.0), (110.0, 90.0))
    clamped = clamp_rois_to_image({0: roi}, height=100.0, width=100.0)[0]

    assert _roi_size(clamped) == _roi_size(roi)
    assert _roi_bounds(clamped) == (80.0, 100.0, 80.0, 100.0)


def test_clamp_preserves_size_when_the_window_is_larger_than_the_image():
    roi = Roi((-10.0, -10.0), (-10.0, 150.0), (150.0, 150.0), (150.0, -10.0))
    clamped = clamp_rois_to_image({0: roi}, height=100.0, width=100.0)[0]

    assert _roi_size(clamped) == _roi_size(roi)
    top, _, left, _ = _roi_bounds(clamped)
    assert (top, left) == (0.0, 0.0)


def test_clamp_oversized_windows_are_identical_at_every_timepoint():
    # A window too big for the image must land in the same place each frame,
    # otherwise frames would differ in size after trimming.
    rois = {
        t: Roi((c - 80, c - 80), (c - 80, c + 80), (c + 80, c + 80), (c + 80, c - 80))
        for t, c in enumerate([10.0, 50.0, 90.0])
    }
    clamped = clamp_rois_to_image(rois, height=100.0, width=100.0)
    assert len({_roi_bounds(r) for r in clamped.values()}) == 1


def test_clamp_applies_independently_per_timepoint():
    rois = {
        0: Roi((-5.0, 40.0), (-5.0, 60.0), (15.0, 60.0), (15.0, 40.0)),   # past top
        1: Roi((40.0, 90.0), (40.0, 110.0), (60.0, 110.0), (60.0, 90.0)),  # past right
        2: Roi((40.0, 40.0), (40.0, 60.0), (60.0, 60.0), (60.0, 40.0)),    # fine
    }
    clamped = clamp_rois_to_image(rois, height=100.0, width=100.0)

    assert _roi_bounds(clamped[0])[0] == 0.0
    assert _roi_bounds(clamped[1])[3] == 100.0
    assert clamped[2] == rois[2]


def test_clamp_is_idempotent_and_does_not_mutate_input():
    rois = {0: Roi((-5.0, 90.0), (-5.0, 110.0), (15.0, 110.0), (15.0, 90.0))}
    before = dict(rois)
    once = clamp_rois_to_image(rois, 100.0, 100.0)

    assert rois == before
    assert clamp_rois_to_image(once, 100.0, 100.0) == once


def test_crop_params_auto_units_are_microns_when_only_a_track_is_given():
    by_time = track_to_rois(_track([0, 1.0, 1.0, 0.0]), 2.0)
    assert CropParams(roi_by_time={0: by_time}, roi_list=[_PLACEHOLDER_ROI], roi_subset=[0]).roi_units == RoiUnits.Microns


def test_crop_params_explicit_units_override_the_track_default():
    by_time = track_to_rois(_track([0, 1.0, 1.0, 0.0]), 2.0)
    crop = CropParams(roi_by_time={0: by_time}, roi_list=[_PLACEHOLDER_ROI], roi_subset=[0], roi_units=RoiUnits.Pixels)
    assert crop.roi_units == RoiUnits.Pixels


def test_crop_params_auto_units_are_pixels_for_plain_coordinates():
    roi = Roi((0.0, 0.0), (0.0, 5.0), (5.0, 5.0), (5.0, 0.0))
    assert CropParams(roi_list=[roi]).roi_units == RoiUnits.Pixels


def test_crop_params_json_round_trip_keeps_int_timepoints():
    # JSON object keys are strings; they must come back as int timepoints.
    by_time = track_to_rois(_track([2, 10.0, 10.0, 0.0], [3, 11.0, 11.0, 0.0]), 10.0)
    crop = CropParams(roi_by_time={0: by_time}, roi_list=[_PLACEHOLDER_ROI], roi_subset=[0], roi_units=RoiUnits.Microns)
    again = CropParams.parse_raw(crop.json())

    assert set(again.roi_by_time[0]) == {2, 3}
    assert again.roi_for_time(3, 0) == crop.roi_for_time(3, 0)


def test_roi_for_time_returns_the_moving_window_when_tracking():
    by_time = track_to_rois(
        _track([0, 10.0, 10.0, 0.0], [1, 20.0, 20.0, 0.0]),
        window_size=10.0,
    )
    crop = CropParams(roi_by_time={0: by_time}, roi_list=[_PLACEHOLDER_ROI], roi_subset=[0])

    assert crop.roi_for_time(0, 0) == by_time[0]
    assert crop.roi_for_time(1, 0) == by_time[1]
    assert crop.roi_for_time(0, 0) != crop.roi_for_time(1, 0)


def test_roi_for_time_falls_back_to_the_static_roi_list():
    static = Roi((0.0, 0.0), (0.0, 5.0), (5.0, 5.0), (5.0, 0.0))
    crop = CropParams(roi_list=[static], roi_units=RoiUnits.Pixels)

    assert crop.roi_by_time is None
    # Compare by value: pydantic may rebuild the tuple during validation.
    assert crop.roi_for_time(0, 0) == static
    assert crop.roi_for_time(99, 0) == static


def test_roi_for_time_rejects_a_timepoint_the_track_does_not_cover():
    by_time = track_to_rois(_track([0, 1.0, 1.0, 0.0], [2, 2.0, 2.0, 0.0]), 2.0)
    crop = CropParams(roi_by_time={0: by_time}, roi_list=[_PLACEHOLDER_ROI], roi_subset=[0])

    with pytest.raises(ValueError, match="no position at timepoint 1.*covers timepoints 0-2"):
        crop.roi_for_time(1, 0)


def test_lattice_converts_track_rois_from_microns_to_pixels(tmp_path):
    # dy = 0.5 um -> factor 2. A 10 um window centred on (10, 20) becomes 20 px.
    by_time = track_to_rois(_track([0, 10.0, 20.0, 0.0]), window_size=10.0)
    lattice = _lattice(tmp_path, roi_by_time=by_time, dy=0.5)

    assert lattice.crop.roi_units == RoiUnits.Pixels
    expected = scale_rois([by_time[0]], 2.0)[0]
    assert lattice.crop.roi_by_time[0][0] == expected


def test_lattice_conversion_does_not_repeat_on_revalidation(tmp_path):
    by_time = track_to_rois(_track([0, 10.0, 20.0, 0.0]), window_size=10.0)
    lattice = _lattice(tmp_path, roi_by_time=by_time, dy=0.5)
    converted = dict(lattice.crop.roi_by_time)

    # parse_obj re-runs every validator; .copy() would not, so it proves nothing.
    assert LatticeData.parse_obj(dict(lattice)).crop.roi_by_time == converted


def test_lattice_clamps_a_track_roi_that_starts_off_the_image(tmp_path):
    by_time = track_to_rois(_track([0, 1.0, 1.0, 0.0]), window_size=20.0)
    lattice = _lattice(tmp_path, roi_by_time=by_time, dy=1.0, y=50, x=50)

    top, bottom, left, right = _roi_bounds(lattice.crop.roi_by_time[0][0])
    assert top >= 0.0
    assert left >= 0.0
    assert bottom - top == pytest.approx(20.0)
    assert right - left == pytest.approx(20.0)


def test_lattice_clamps_every_timepoint_into_the_deskewed_frame(tmp_path):
    by_time = track_to_rois(
        _track([0, 1.0, 1.0, 0.0], [1, 25.0, 25.0, 0.0], [2, 1000.0, 1000.0, 0.0]),
        window_size=20.0,
    )
    lattice = _lattice(tmp_path, roi_by_time=by_time, dy=1.0, y=50, x=50)
    height, width = lattice.derived.deskew_vol_shape[1:]

    for roi in lattice.crop.roi_by_time[0].values():
        top, bottom, left, right = _roi_bounds(roi)
        assert 0.0 <= top and bottom <= height
        assert 0.0 <= left and right <= width
        assert _roi_size(roi) == pytest.approx((20.0, 20.0))


def test_lattice_does_not_warn_after_a_track_roi_is_clamped(tmp_path, caplog):
    # Find the real deskewed frame size first, then start the track on its
    # bottom-right edge (the warning only checks the max edge).
    probe = track_to_rois(_track([0, 10.0, 10.0, 0.0]), window_size=20.0)
    height, width = _lattice(
        tmp_path, roi_by_time=probe, dy=1.0, y=50, x=50
    ).derived.deskew_vol_shape[1:]

    by_time = track_to_rois(
        _track([0, width - 1, height - 1, 0.0], [1, width - 1, height - 1, 0.0]),
        window_size=20.0,
    )
    caplog.clear()
    with caplog.at_level(logging.WARNING):
        _lattice(tmp_path, roi_by_time=by_time, dy=1.0, y=50, x=50)

    assert not _outside_warnings(caplog)


def test_lattice_still_warns_for_a_static_crop_outside_the_image(tmp_path, caplog):
    big = Roi((0.0, 0.0), (0.0, 10_000.0), (10_000.0, 10_000.0), (10_000.0, 0.0))
    with caplog.at_level(logging.WARNING):
        _static_lattice(tmp_path, big)
    assert _outside_warnings(caplog)


def test_lattice_narrows_time_range_to_the_track_coverage(tmp_path):
    by_time = track_to_rois(
        _track([1, 10.0, 10.0, 0.0], [2, 12.0, 12.0, 0.0]),
        window_size=10.0,
    )
    lattice = _lattice(tmp_path, roi_by_time=by_time, time_points=5)

    # The run-wide range is untouched; each track gets its own covered range.
    assert list(lattice.time_range_by_track[0]) == [1, 2]


def test_lattice_keeps_a_user_time_range_that_is_already_inside_the_track(tmp_path):
    by_time = track_to_rois(
        _track(*[[t, 10.0 + t, 10.0 + t, 0.0] for t in range(4)]),
        window_size=10.0,
    )
    lattice = _lattice(
        tmp_path, roi_by_time=by_time, time_points=5, time_range=range(1, 3)
    )
    assert list(lattice.time_range) == [1, 2]


def test_lattice_trims_a_track_that_runs_past_the_last_frame(tmp_path):
    by_time = track_to_rois(
        _track(*[[t, 10.0, 10.0, 0.0] for t in range(20)]),
        window_size=10.0,
    )
    lattice = _lattice(tmp_path, roi_by_time=by_time, time_points=5)
    assert list(lattice.time_range) == [0, 1, 2, 3, 4]


def test_lattice_rejects_a_time_range_that_misses_the_track(tmp_path):
    by_time = track_to_rois(
        _track([0, 10.0, 10.0, 0.0], [1, 11.0, 11.0, 0.0]),
        window_size=10.0,
    )
    with pytest.raises(ValidationError, match="does not overlap the track"):
        _lattice(tmp_path, roi_by_time=by_time, time_points=5, time_range=range(3, 5))


def test_lattice_rejects_a_track_with_gaps_inside_the_requested_range(tmp_path):
    by_time = track_to_rois(
        _track([0, 10.0, 10.0, 0.0], [2, 12.0, 12.0, 0.0]),
        window_size=10.0,
    )
    with pytest.raises(ValidationError, match=r"no position at timepoints \[1\]"):
        _lattice(tmp_path, roi_by_time=by_time, time_points=5)


def test_lattice_allows_a_gappy_track_when_the_time_range_avoids_the_gap(tmp_path):
    by_time = track_to_rois(
        _track([0, 10.0, 10.0, 0.0], [2, 12.0, 12.0, 0.0], [3, 13.0, 13.0, 0.0]),
        window_size=10.0,
    )
    lattice = _lattice(
        tmp_path, roi_by_time=by_time, time_points=5, time_range=range(2, 4)
    )
    assert list(lattice.time_range) == [2, 3]


def test_static_crop_is_unaffected_by_track_time_restriction(tmp_path):
    static = Roi((0.0, 0.0), (0.0, 10.0), (10.0, 10.0), (10.0, 0.0))
    lattice = _static_lattice(tmp_path, static)

    assert list(lattice.time_range) == [0, 1, 2, 3, 4]
    assert lattice.crop.roi_by_time is None
