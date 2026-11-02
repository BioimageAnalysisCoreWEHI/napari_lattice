"""
Tests for :mod:`lls_core.track_processing`.

Every fixture here is an ``(n, 4)`` ``[t, x, y, z]`` microns array, i.e. the
shape ``load_trackmate_tracks()`` and ``_get_selected_track_data()`` hand over.
"""
from __future__ import annotations

import numpy as np
import pytest

from lls_core.track_processing import interpolate_gaps, microns_to_pixels


def test_interpolate_gaps_leaves_a_complete_track_alone():
    track = np.array(
        [
            [0.0, 10.0, 20.0, 5.0],
            [1.0, 11.0, 22.0, 5.5],
            [2.0, 12.0, 24.0, 6.0],
        ]
    )

    filled = interpolate_gaps(track)

    assert filled.shape == track.shape
    np.testing.assert_allclose(filled, track)


def test_interpolate_gaps_fills_a_single_missing_frame_with_the_midpoint():
    track = np.array(
        [
            [0.0, 10.0, 20.0, 4.0],
            [2.0, 20.0, 30.0, 8.0],
        ]
    )

    filled = interpolate_gaps(track)

    assert filled.shape == (3, 4)
    np.testing.assert_allclose(filled[1], [1.0, 15.0, 25.0, 6.0])
    # The real detections must survive untouched.
    np.testing.assert_allclose(filled[0], track[0])
    np.testing.assert_allclose(filled[2], track[1])


def test_interpolate_gaps_fills_a_multi_frame_gap_linearly():
    track = np.array(
        [
            [0.0, 0.0, 100.0, 0.0],
            [4.0, 40.0, 60.0, 2.0],
        ]
    )

    filled = interpolate_gaps(track)

    expected = np.array(
        [
            [0.0, 0.0, 100.0, 0.0],
            [1.0, 10.0, 90.0, 0.5],
            [2.0, 20.0, 80.0, 1.0],
            [3.0, 30.0, 70.0, 1.5],
            [4.0, 40.0, 60.0, 2.0],
        ]
    )
    np.testing.assert_allclose(filled, expected)


def test_interpolate_gaps_fills_several_separate_gaps():
    track = np.array(
        [
            [0.0, 0.0, 0.0, 0.0],
            [2.0, 2.0, 4.0, 1.0],
            [3.0, 3.0, 6.0, 1.5],
            [6.0, 6.0, 12.0, 3.0],
        ]
    )

    filled = interpolate_gaps(track)

    np.testing.assert_allclose(filled[:, 0], np.arange(7))
    np.testing.assert_allclose(filled[:, 1], np.arange(7, dtype=float))
    np.testing.assert_allclose(filled[:, 2], np.arange(7) * 2.0)
    np.testing.assert_allclose(filled[:, 3], np.arange(7) * 0.5)


def test_interpolate_gaps_does_not_extrapolate_past_either_end():
    # The cell only appears at frame 5 and is lost after frame 9.
    track = np.array(
        [
            [5.0, 50.0, 10.0, 1.0],
            [7.0, 70.0, 14.0, 3.0],
            [9.0, 90.0, 18.0, 5.0],
        ]
    )

    filled = interpolate_gaps(track)

    assert filled[0, 0] == 5.0
    assert filled[-1, 0] == 9.0
    np.testing.assert_allclose(filled[:, 0], np.arange(5, 10))


def test_interpolate_gaps_sorts_unsorted_input():
    track = np.array(
        [
            [4.0, 40.0, 8.0, 2.0],
            [0.0, 0.0, 0.0, 0.0],
            [2.0, 20.0, 4.0, 1.0],
        ]
    )

    filled = interpolate_gaps(track)

    np.testing.assert_allclose(filled[:, 0], np.arange(5))
    np.testing.assert_allclose(filled[:, 1], np.arange(5) * 10.0)
    np.testing.assert_allclose(filled[:, 2], np.arange(5) * 2.0)
    np.testing.assert_allclose(filled[:, 3], np.arange(5) * 0.5)


def test_interpolate_gaps_keeps_the_first_of_two_rows_on_the_same_frame():
    track = np.array(
        [
            [0.0, 0.0, 0.0, 0.0],
            [1.0, 11.0, 21.0, 1.0],
            [1.0, 99.0, 99.0, 9.0],  # duplicate frame, should be dropped
            [2.0, 12.0, 22.0, 2.0],
        ]
    )

    filled = interpolate_gaps(track)

    assert filled.shape == (3, 4)
    np.testing.assert_allclose(filled[1], [1.0, 11.0, 21.0, 1.0])


def test_interpolate_gaps_returns_a_single_detection_unchanged():
    track = np.array([[3.0, 30.0, 40.0, 5.0]])

    filled = interpolate_gaps(track)

    np.testing.assert_allclose(filled, track)


def test_interpolate_gaps_rounds_near_integer_frames_without_moving_coords():
    # Frame indices read back out of a CSV can be a hair off an integer.
    track = np.array(
        [
            [0.0000001, 10.0, 20.0, 1.0],
            [1.9999999, 30.0, 40.0, 3.0],
        ]
    )

    filled = interpolate_gaps(track)

    np.testing.assert_allclose(filled[:, 0], [0.0, 1.0, 2.0])
    np.testing.assert_allclose(filled[0, 1:], [10.0, 20.0, 1.0])
    np.testing.assert_allclose(filled[-1, 1:], [30.0, 40.0, 3.0])


def test_interpolate_gaps_rejects_an_empty_track():
    with pytest.raises(ValueError):
        interpolate_gaps(np.empty((0, 4)))


@pytest.mark.parametrize(
    "bad",
    [
        np.array([]),
        np.array([1.0, 2.0, 3.0, 4.0]),  # 1D, not (1, 4)
        np.zeros((3, 3)),
        np.zeros((3, 5)),
        np.zeros((2, 2, 4)),
    ],
)
def test_interpolate_gaps_rejects_the_wrong_shape(bad):
    with pytest.raises(ValueError):
        interpolate_gaps(bad)


def test_microns_to_pixels_divides_each_axis_by_its_own_size():
    track = np.array(
        [
            [0.0, 10.0, 20.0, 6.0],
            [1.0, 5.0, 40.0, 12.0],
        ]
    )

    pixels = microns_to_pixels(track, dx=0.5, dy=0.2, dz=3.0)

    expected = np.array(
        [
            [0.0, 20.0, 100.0, 2.0],
            [1.0, 10.0, 200.0, 4.0],
        ]
    )
    np.testing.assert_allclose(pixels, expected)


def test_microns_to_pixels_leaves_the_time_column_alone():
    track = np.array(
        [
            [7.0, 1.0, 1.0, 1.0],
            [8.0, 1.0, 1.0, 1.0],
        ]
    )

    pixels = microns_to_pixels(track, dx=0.1, dy=0.1, dz=0.1)

    np.testing.assert_allclose(pixels[:, 0], track[:, 0])


def test_microns_to_pixels_returns_unrounded_pixel_positions():
    track = np.array([[0.0, 1.0, 1.0, 1.0]])

    pixels = microns_to_pixels(track, dx=0.3, dy=0.3, dz=0.3)

    assert pixels.dtype.kind == "f"
    np.testing.assert_allclose(pixels[0, 1:], [1 / 0.3] * 3)


def test_microns_to_pixels_with_all_sizes_one_is_a_no_op():
    track = np.array(
        [
            [0.0, 10.0, 20.0, 5.0],
            [1.0, 11.0, 22.0, 5.5],
        ]
    )

    pixels = microns_to_pixels(track, dx=1.0, dy=1.0, dz=1.0)

    np.testing.assert_allclose(pixels, track)


def test_microns_to_pixels_with_one_size_one_leaves_only_that_axis():
    track = np.array([[0.0, 10.0, 20.0, 5.0]])

    pixels = microns_to_pixels(track, dx=1.0, dy=2.0, dz=0.5)

    np.testing.assert_allclose(pixels[0], [0.0, 10.0, 10.0, 10.0])


def test_microns_to_pixels_does_not_mutate_its_input():
    track = np.array(
        [
            [0.0, 10.0, 20.0, 5.0],
            [1.0, 11.0, 22.0, 5.5],
        ]
    )
    original = track.copy()

    microns_to_pixels(track, dx=0.1, dy=0.2, dz=0.3)

    np.testing.assert_allclose(track, original)


def test_microns_to_pixels_accepts_an_integer_track_array():
    track = np.array([[0, 10, 20, 6]])

    pixels = microns_to_pixels(track, dx=2.0, dy=4.0, dz=4.0)

    np.testing.assert_allclose(pixels[0], [0.0, 5.0, 5.0, 1.5])


@pytest.mark.parametrize(
    "dx, dy, dz",
    [
        (0.0, 1.0, 1.0),
        (1.0, 0.0, 1.0),
        (1.0, 1.0, 0.0),
        (-0.1, 1.0, 1.0),
        (1.0, -1.0, 1.0),
        (1.0, 1.0, float("nan")),
    ],
)
def test_microns_to_pixels_rejects_non_positive_pixel_sizes(dx, dy, dz):
    track = np.array([[0.0, 1.0, 1.0, 1.0]])

    with pytest.raises(ValueError):
        microns_to_pixels(track, dx=dx, dy=dy, dz=dz)


@pytest.mark.parametrize(
    "bad",
    [
        np.array([]),
        np.array([1.0, 2.0, 3.0, 4.0]),  # 1D, not (1, 4)
        np.zeros((3, 3)),
        np.zeros((3, 5)),
        np.zeros((2, 2, 4)),
    ],
)
def test_microns_to_pixels_rejects_the_wrong_shape(bad):
    with pytest.raises(ValueError):
        microns_to_pixels(bad, dx=1.0, dy=1.0, dz=1.0)


def test_gappy_selected_track_becomes_a_per_frame_pixel_trajectory():
    """
    The whole pipeline: an (n, 4) [t, x, y, z] microns track with frames 2 and
    3 missing, the way ``_get_selected_track_data()`` would hand it over,
    through interpolation and then into pixel indices.
    """
    dx, dy, dz = 0.145, 0.145, 0.3
    selected_track = np.array(
        [
            [0.0, 14.5, 29.0, 3.0],
            [1.0, 16.0, 30.0, 3.3],
            # frames 2 and 3 dropped by the detector
            [4.0, 22.0, 34.0, 4.5],
            [5.0, 23.5, 35.0, 4.8],
        ]
    )

    filled = interpolate_gaps(selected_track)
    pixels = microns_to_pixels(filled, dx=dx, dy=dy, dz=dz)

    # One row per frame, none missing, nothing invented past the ends.
    np.testing.assert_allclose(pixels[:, 0], np.arange(6))
    assert pixels.shape == (6, 4)

    # Real detections land where the raw microns say they should.
    np.testing.assert_allclose(
        pixels[0, 1:], [14.5 / dx, 29.0 / dy, 3.0 / dz]
    )
    np.testing.assert_allclose(
        pixels[5, 1:], [23.5 / dx, 35.0 / dy, 4.8 / dz]
    )

    # The interpolated frames sit evenly between frames 1 and 4.
    np.testing.assert_allclose(filled[2, 1:], [18.0, 31.0 + 1 / 3, 3.7])
    np.testing.assert_allclose(pixels[2, 1:], filled[2, 1:] / [dx, dy, dz])

    # A crop indexing the stack wants monotonic, in-bounds pixel positions.
    assert np.all(np.diff(pixels[:, 1]) > 0)
    assert np.all(pixels[:, 1:] >= 0)
