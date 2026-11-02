"""
Headless tests for the ROI Tracking tab (TrackmateFields): loading a track
file, the track dropdown, drawing the tracks layer, the crop window preview and
the model. Methods are called on stand-ins for `self`, so no napari viewer is needed.
"""
from contextlib import nullcontext
from types import MethodType, SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from napari_lattice.fields import TrackmateFields

TRACKS = {
    "0": {"nSpots": 2, "trackData": np.array([[0, 10, 20, 1], [1, 11, 21, 1]], dtype=float)},
    "2": {"nSpots": 1, "trackData": np.array([[0, 30, 40, 2]], dtype=float)},
    "10": {"nSpots": 1, "trackData": np.array([[0, 50, 60, 3]], dtype=float)},
}

CSV = (
    "LABEL,ID,TRACK_ID,QUALITY,POSITION_X,POSITION_Y,POSITION_Z,POSITION_T,FRAME,RADIUS\n"
    "Label,Spot ID,Track ID,Quality,X,Y,Z,T,Frame,Radius\n"
    "Label,Spot ID,Track ID,Quality,X,Y,Z,T,Frame,R\n"
    ",,,(quality),(micron),(micron),(micron),(sec),,(micron)\n"
    "ID0,0,0,1.0,10.0,20.0,1.0,0.0,0,7.5\n"
    "ID1,1,5,1.0,30.0,40.0,2.0,0.0,0,7.5\n"
)


class FakeViewer:
    def __init__(self, layers=None):
        self.layers = dict(layers or {})
        self.added = []

    def add_tracks(self, data, properties=None, name=None, tail_width=None):
        self.added.append(dict(kind="tracks", data=data, properties=properties, name=name, tail_width=tail_width))
        self.layers[name] = data

    def add_shapes(self, data, shape_type=None, name=None, **kwargs):
        self.added.append(dict(kind="shapes", data=data, shape_type=shape_type, name=name))
        self.layers[name] = data


def fake_fields(tracks=None, selected=(), layer_name="", path=None, sizes=None, preview=True):
    fake = SimpleNamespace(
        track_ids=SimpleNamespace(value=list(selected), reset_choices=Mock()),
        layer_name=SimpleNamespace(value=layer_name),
        tracks_path=SimpleNamespace(value=path),
        show_preview=SimpleNamespace(value=preview),
        errors=SimpleNamespace(value=""),
        fields_enabled=SimpleNamespace(value=True),
        _track_window_sizes=dict(sizes or {}),
        _DEFAULT_WINDOW=TrackmateFields._DEFAULT_WINDOW,
        _sync_track_windows=Mock(),
        _render_tracks_layer=Mock(),
        _render_preview_layer=Mock(),
        _set_revealed=Mock(),
        _follow_viewer_time=Mock(),
        # magicclass wraps button methods: it blocks the macro while they run, then records if active
        macro=SimpleNamespace(blocked=nullcontext, active=False),
    )
    # The small helpers are real, so the methods under test see their true behaviour.
    for name in ("_get_selected_track_ids", "_window_sizes", "_get_track_window_sizes",
                 "_tracks_layer_name"):
        setattr(fake, name, MethodType(getattr(TrackmateFields, name), fake))
    if tracks is not None:
        fake._tracks = tracks
    return fake


def test_track_options_before_loading_are_empty():
    assert TrackmateFields._get_track_id_options(fake_fields(), None) == []


def test_track_options_are_ids_in_numeric_order():
    options = TrackmateFields._get_track_id_options(fake_fields(TRACKS), None)
    assert options == ["0", "2", "10"]


def test_nothing_selected_gives_no_tracks():
    assert fake_fields(TRACKS)._get_selected_track_ids() == []


def test_no_file_loaded_gives_no_tracks():
    assert fake_fields(selected=["2"])._get_selected_track_ids() == []


def test_selected_tracks_come_back_in_numeric_order():
    fake = fake_fields(TRACKS, selected=["10", "0", "2"])
    assert fake._get_selected_track_ids() == ["0", "2", "10"]


def test_selected_tracks_not_in_the_file_are_dropped():
    fake = fake_fields(TRACKS, selected=["2", "99"])
    assert fake._get_selected_track_ids() == ["2"]


def test_load_without_a_file_selected():
    with pytest.raises(ValueError, match="No file selected"):
        TrackmateFields.load_tracks(fake_fields(path=None))


def test_load_with_a_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError, match="File not found"):
        TrackmateFields.load_tracks(fake_fields(path=tmp_path / "nope.csv"))


def test_load_stores_tracks_resets_dropdown_reveals_tab_and_draws(tmp_path):
    path = tmp_path / "spots.csv"
    path.write_text(CSV)
    fake = fake_fields(selected=["3"], path=path, sizes={"3": 12.0})

    TrackmateFields.load_tracks(fake)

    assert set(fake._tracks) == {"0", "5"}
    assert fake.track_ids.value == []
    assert fake._track_window_sizes == {}
    fake._sync_track_windows.assert_called_once()
    fake.track_ids.reset_choices.assert_called_once()
    fake._set_revealed.assert_called_once_with(True)
    fake._render_tracks_layer.assert_called_once()
    fake._render_preview_layer.assert_called_once()
    fake._follow_viewer_time.assert_called_once()


def test_a_failed_load_leaves_the_tab_hidden(tmp_path):
    fake = fake_fields(path=tmp_path / "nope.csv")
    with pytest.raises(FileNotFoundError):
        TrackmateFields.load_tracks(fake)
    fake._set_revealed.assert_not_called()


def test_window_sizes_are_limited_to_the_given_tracks():
    fake = fake_fields(TRACKS, sizes={"0": 20.0, "10": 5.0})
    assert fake._get_track_window_sizes(["0"]) == {"0": 20.0}


def test_tracks_without_a_window_size_get_the_default():
    fake = fake_fields(TRACKS, sizes={"0": 20.0})
    assert fake._get_track_window_sizes(["0", "2"]) == {"0": 20.0, "2": 30.0}


def test_editing_a_window_size_stores_it_and_redraws():
    fake = fake_fields(TRACKS)
    fake._on_window_changed = MethodType(TrackmateFields._on_window_changed, fake)
    TrackmateFields._on_track_window_changed(fake, "2", 12)
    assert fake._track_window_sizes == {"2": 12.0}
    fake._render_preview_layer.assert_called_once()


@pytest.fixture
def viewer(monkeypatch):
    v = FakeViewer()
    monkeypatch.setattr("napari_lattice.utils.get_viewer", lambda: v)
    return v


def test_render_with_nothing_selected_draws_every_track(viewer, tmp_path):
    fake = fake_fields(TRACKS, path=tmp_path / "t.xml")
    TrackmateFields._render_tracks_layer(fake)
    assert set(viewer.added[0]["data"][:, 0]) == {0, 2, 10}


def test_render_draws_only_the_selected_tracks(viewer, tmp_path):
    fake = fake_fields(TRACKS, selected=["0", "10"], path=tmp_path / "t.xml")
    TrackmateFields._render_tracks_layer(fake)
    assert set(viewer.added[0]["data"][:, 0]) == {0, 10}


def test_render_layer_name_defaults_to_file_stem(viewer, tmp_path):
    TrackmateFields._render_tracks_layer(fake_fields(TRACKS, path=tmp_path / "my_tracks.csv"))
    assert viewer.added[0]["name"] == "my_tracks"
    assert viewer.added[0]["tail_width"] == 2


def test_render_uses_trimmed_custom_layer_name(viewer, tmp_path):
    fake = fake_fields(TRACKS, layer_name="  cell A  ", path=tmp_path / "t.xml")
    TrackmateFields._render_tracks_layer(fake)
    assert viewer.added[0]["name"] == "cell A"


def test_render_replaces_an_existing_layer_with_the_same_name(monkeypatch, tmp_path):
    viewer = FakeViewer(layers={"t": "old layer"})
    monkeypatch.setattr("napari_lattice.utils.get_viewer", lambda: viewer)

    TrackmateFields._render_tracks_layer(fake_fields(TRACKS, path=tmp_path / "t.xml"))

    assert list(viewer.layers) == ["t"]
    assert isinstance(viewer.layers["t"], np.ndarray)  # the new tracks, not "old layer"
    assert len(viewer.added) == 1


def _bounds(rect):
    rect = np.asarray(rect)
    return rect[:, 0].min(), rect[:, 0].max(), rect[:, 1].min(), rect[:, 1].max()


def _preview_fake(tmp_path, **kwargs):
    fake = fake_fields(TRACKS, path=tmp_path / "t.xml", **kwargs)
    fake._current_time = lambda viewer: None
    return fake


def test_preview_draws_one_rectangle_per_selected_track(viewer, tmp_path):
    fake = _preview_fake(tmp_path, selected=["0", "2"], sizes={"0": 10.0, "2": 10.0})
    TrackmateFields._render_preview_layer(fake)

    layer = viewer.added[0]
    assert layer["kind"] == "shapes"
    assert layer["shape_type"] == "rectangle"
    assert layer["name"] == "t crop window"
    assert [_bounds(r) for r in layer["data"]] == [
        (15.0, 25.0, 5.0, 15.0),
        (35.0, 45.0, 25.0, 35.0),
    ]


def test_preview_uses_each_tracks_own_size(viewer, tmp_path):
    fake = _preview_fake(tmp_path, selected=["0", "2"], sizes={"0": 10.0, "2": 20.0})
    TrackmateFields._render_preview_layer(fake)

    sizes = [_bounds(r)[1] - _bounds(r)[0] for r in viewer.added[0]["data"]]
    assert sizes == [10.0, 20.0]


def test_preview_follows_the_current_timepoint(viewer, tmp_path):
    fake = _preview_fake(tmp_path, selected=["0"], sizes={"0": 10.0})
    fake._current_time = lambda viewer: 1
    TrackmateFields._render_preview_layer(fake)

    assert _bounds(viewer.added[0]["data"][0]) == (16.0, 26.0, 6.0, 16.0)


def test_preview_replaces_the_previous_one(viewer, tmp_path):
    fake = _preview_fake(tmp_path, selected=["0"])
    TrackmateFields._render_preview_layer(fake)
    TrackmateFields._render_preview_layer(fake)
    assert list(viewer.layers) == ["t crop window"]


def test_preview_is_removed_when_nothing_is_selected(viewer, tmp_path):
    fake = _preview_fake(tmp_path, selected=["0"])
    TrackmateFields._render_preview_layer(fake)
    fake.track_ids.value = []
    TrackmateFields._render_preview_layer(fake)
    assert viewer.layers == {}


def test_preview_is_removed_when_switched_off(viewer, tmp_path):
    fake = _preview_fake(tmp_path, selected=["0"])
    TrackmateFields._render_preview_layer(fake)
    fake.show_preview.value = False
    TrackmateFields._render_preview_layer(fake)
    assert viewer.layers == {}


def test_current_time_is_none_without_a_time_axis():
    viewer = SimpleNamespace(dims=SimpleNamespace(ndim=3, current_step=(0, 0, 0)))
    assert TrackmateFields._current_time(viewer) is None


def test_current_time_reads_the_first_of_four_axes():
    viewer = SimpleNamespace(dims=SimpleNamespace(ndim=4, current_step=(3, 7, 0, 0)))
    assert TrackmateFields._current_time(viewer) == 3


def test_changing_tracks_redraws_when_tracks_are_loaded():
    fake = fake_fields(TRACKS)
    TrackmateFields._on_track_ids_changed(fake)
    fake._sync_track_windows.assert_called_once()
    fake._render_tracks_layer.assert_called_once()
    fake._render_preview_layer.assert_called_once()


def test_changing_tracks_does_nothing_before_loading():
    fake = fake_fields()
    TrackmateFields._on_track_ids_changed(fake)
    fake._render_tracks_layer.assert_not_called()
    fake._render_preview_layer.assert_not_called()


def test_changing_the_window_redraws_only_the_preview():
    fake = fake_fields(TRACKS)
    TrackmateFields._on_window_changed(fake)
    fake._render_preview_layer.assert_called_once()
    fake._render_tracks_layer.assert_not_called()


def test_changing_the_window_does_nothing_before_loading():
    fake = fake_fields()
    TrackmateFields._on_window_changed(fake)
    fake._render_preview_layer.assert_not_called()
