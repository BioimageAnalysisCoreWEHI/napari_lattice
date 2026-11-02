"""
Tests for the helpers behind the ROI Tracking tab: reading per-track window
sizes and working out the crop window preview shown on the canvas.
"""
from __future__ import annotations

import numpy as np
import pytest

from lls_core.cropping import (
    Roi,
    parse_track_window_sizes,
    track_window_previews,
)

TRACKS = {
    "0": {"nSpots": 3, "trackData": np.array([[0, 10, 20, 0], [1, 12, 22, 0], [3, 16, 26, 0]], dtype=float)},
    "4": {"nSpots": 1, "trackData": np.array([[2, 50, 60, 0]], dtype=float)},
}


def _size(roi: Roi) -> tuple[float, float]:
    ys = [y for y, _ in roi]
    xs = [x for _, x in roi]
    return max(ys) - min(ys), max(xs) - min(xs)


def _centre(roi: Roi) -> tuple[float, float]:
    ys = [y for y, _ in roi]
    xs = [x for _, x in roi]
    return (max(ys) + min(ys)) / 2, (max(xs) + min(xs)) / 2


@pytest.mark.parametrize("text", [None, "", "   ", ",", " , "])
def test_blank_sizes_give_no_overrides(text):
    assert parse_track_window_sizes(text) == {}


def test_sizes_are_read_as_track_size_pairs():
    assert parse_track_window_sizes("3:20, 7:45.5") == {"3": 20.0, "7": 45.5}


def test_sizes_accept_whitespace_or_commas_between_pairs():
    assert parse_track_window_sizes("1:10 2:20,3:30") == {"1": 10.0, "2": 20.0, "3": 30.0}


def test_a_repeated_track_keeps_the_last_size():
    assert parse_track_window_sizes("1:10, 1:25") == {"1": 25.0}


def test_track_ids_are_normalised_to_plain_integers():
    assert parse_track_window_sizes("03:20") == {"3": 20.0}


@pytest.mark.parametrize("text, message", [
    ("20", "track:size"),
    ("1:2:3", "track:size"),
    ("a:20", "integers"),
    ("1.5:20", "integers"),
    ("1:big", "numbers"),
    ("1:0", "positive"),
    ("1:-5", "positive"),
])
def test_bad_sizes_are_rejected_with_a_reason(text, message):
    with pytest.raises(ValueError, match=message):
        parse_track_window_sizes(text)


def test_preview_is_a_window_of_the_default_size_centred_on_the_track():
    previews = track_window_previews(TRACKS, ["0"], default_size=10.0, time=0)
    assert _size(previews["0"]) == (10.0, 10.0)
    # TrackMate x/y -> canvas (y, x)
    assert _centre(previews["0"]) == (20.0, 10.0)


def test_preview_follows_the_track_through_time():
    centres = [
        _centre(track_window_previews(TRACKS, ["0"], 10.0, time=t)["0"])
        for t in (0, 1, 3)
    ]
    assert centres == [(20.0, 10.0), (22.0, 12.0), (26.0, 16.0)]


def test_preview_without_a_time_uses_the_first_timepoint():
    previews = track_window_previews(TRACKS, ["0"], 10.0)
    assert _centre(previews["0"]) == (20.0, 10.0)


def test_preview_outside_the_track_uses_the_nearest_timepoint():
    before = track_window_previews(TRACKS, ["4"], 10.0, time=0)["4"]
    after = track_window_previews(TRACKS, ["4"], 10.0, time=99)["4"]
    assert _centre(before) == _centre(after) == (60.0, 50.0)


def test_preview_in_a_gap_uses_the_nearest_timepoint():
    # track 0 has no point at t=2, equally far from t=1 and t=3: the earlier wins
    previews = track_window_previews(TRACKS, ["0"], 10.0, time=2)
    assert _centre(previews["0"]) == (22.0, 12.0)


def test_each_track_can_have_its_own_size():
    previews = track_window_previews(TRACKS, ["0", "4"], 10.0, {"4": 30.0}, time=0)
    assert _size(previews["0"]) == (10.0, 10.0)
    assert _size(previews["4"]) == (30.0, 30.0)


def test_unrequested_and_unknown_tracks_are_left_out():
    previews = track_window_previews(TRACKS, ["4", "99"], 10.0, time=0)
    assert list(previews) == ["4"]


def test_a_track_with_no_points_is_left_out():
    tracks = {"0": {"nSpots": 0, "trackData": np.empty((0, 4))}}
    assert track_window_previews(tracks, ["0"], 10.0) == {}


def test_preview_matches_the_window_a_crop_would_use():
    # The overlay must show what the run will actually crop.
    from lls_core.cropping import track_to_rois
    crop = track_to_rois(TRACKS["0"]["trackData"], 10.0)
    for t in (0, 1, 3):
        assert track_window_previews(TRACKS, ["0"], 10.0, time=t)["0"] == crop[t]
