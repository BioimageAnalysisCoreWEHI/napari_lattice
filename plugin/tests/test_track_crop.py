"""
Headless tests for the plugin side of track-driven cropping. The methods are
called on lightweight stand-ins for `self`, so no napari viewer is needed.
"""
from types import MethodType, SimpleNamespace

import numpy as np
import pytest

from lls_core.cropping import RoiUnits
from lls_core.models.crop import CropParams
from lls_core.trackmate_io import load_trackmate_tracks
from napari_lattice.dock_widget import LLSZWidget
from napari_lattice.fields import TrackmateFields


def fake_dock(fixed, tracked):
    container = SimpleNamespace(
        cropping_fields=SimpleNamespace(_make_model=lambda: fixed),
        trackmate_fields=SimpleNamespace(_make_model=lambda: tracked),
    )
    return SimpleNamespace(LlszMenu=SimpleNamespace(WidgetContainer=container))


FIXED = object()
TRACKED = object()


@pytest.mark.parametrize(
    "fixed, tracked, expected",
    [(None, None, None), (FIXED, None, FIXED), (None, TRACKED, TRACKED)],
)
def test_make_crop_model_picks_enabled_tab(fixed, tracked, expected):
    assert LLSZWidget._make_crop_model(fake_dock(fixed, tracked)) is expected


def test_make_crop_model_refuses_both():
    with pytest.raises(ValueError, match="can only use one"):
        LLSZWidget._make_crop_model(fake_dock(FIXED, TRACKED))


TRACK = np.array([[0, 10, 20, 0], [1, 11, 21, 0]], dtype=float)


CSV = (
    "LABEL,ID,TRACK_ID,QUALITY,POSITION_X,POSITION_Y,POSITION_Z,POSITION_T,FRAME,RADIUS\n"
    "Label,Spot ID,Track ID,Quality,X,Y,Z,T,Frame,Radius\n"
    "Label,Spot ID,Track ID,Quality,X,Y,Z,T,Frame,R\n"
    ",,,(quality),(micron),(micron),(micron),(sec),,(micron)\n"
    "ID0,0,0,1.0,10.0,20.0,0.0,0.0,0,7.5\n"
    "ID1,1,0,1.0,11.0,21.0,0.0,1.0,1,7.5\n"
    "ID2,2,1,1.0,50.0,60.0,0.0,0.0,0,7.5\n"
)


@pytest.fixture
def tracks_file(tmp_path):
    path = tmp_path / "spots.csv"
    path.write_text(CSV)
    return path


def fake_trackmate(path=None, enabled=True, selected=("0",), sizes=None):
    fake = SimpleNamespace(
        fields_enabled=SimpleNamespace(value=enabled),
        tracks_path=SimpleNamespace(value=path),
        track_ids=SimpleNamespace(value=list(selected)),
        _tracks=load_trackmate_tracks(path) if path else {},
        _track_window_sizes=dict(sizes or {}),
        _DEFAULT_WINDOW=TrackmateFields._DEFAULT_WINDOW,
    )
    for name in ("_get_selected_track_ids", "_window_sizes", "_get_track_window_sizes"):
        setattr(fake, name, MethodType(getattr(TrackmateFields, name), fake))
    return fake


def test_disabled_returns_none():
    assert TrackmateFields._make_model(fake_trackmate(enabled=False)) is None


def test_disabled_does_not_require_a_track():
    # nothing selected but tab not enabled -> just viewing tracks, no error
    assert TrackmateFields._make_model(fake_trackmate(enabled=False, selected=())) is None


def test_no_track_selected_is_rejected(tracks_file):
    with pytest.raises(ValueError, match="at least one track"):
        TrackmateFields._make_model(fake_trackmate(tracks_file, selected=()))


def test_builds_micron_crop_from_selected_track(tracks_file):
    crop = TrackmateFields._make_model(fake_trackmate(tracks_file, sizes={"0": 4.0}))
    assert isinstance(crop, CropParams)
    assert crop.roi_units == RoiUnits.Microns
    # Only the selected track is cropped along, not every track in the file.
    assert crop.roi_subset == [0]
    assert set(crop.roi_by_time) == {0}
    assert set(crop.roi_by_time[0]) == {0, 1}
    ys = [y for y, _ in crop.roi_by_time[0][0]]
    xs = [x for _, x in crop.roi_by_time[0][0]]
    assert (min(ys), max(ys), min(xs), max(xs)) == (18, 22, 8, 12)


def test_builds_a_crop_for_every_selected_track(tracks_file):
    crop = TrackmateFields._make_model(fake_trackmate(tracks_file, selected=("1", "0")))
    assert crop.roi_subset == [0, 1]
    assert set(crop.roi_by_time) == {0, 1}


def test_each_track_can_have_its_own_window_size(tracks_file):
    fake = fake_trackmate(tracks_file, selected=("0", "1"), sizes={"0": 4.0, "1": 10.0})
    crop = TrackmateFields._make_model(fake)

    def size(roi):
        ys = [y for y, _ in roi]
        return max(ys) - min(ys)

    assert size(crop.roi_by_time[0][0]) == 4.0
    assert size(crop.roi_by_time[1][0]) == 10.0


def test_sizes_for_unselected_tracks_are_ignored(tracks_file):
    crop = TrackmateFields._make_model(
        fake_trackmate(tracks_file, selected=("0",), sizes={"0": 4.0, "1": 10.0})
    )
    assert crop.trackmate_window_size == {"0": 4.0}


def test_a_track_without_a_size_uses_the_default(tracks_file):
    crop = TrackmateFields._make_model(fake_trackmate(tracks_file, selected=("0",)))
    assert crop.window_for_track(0) == 30.0
