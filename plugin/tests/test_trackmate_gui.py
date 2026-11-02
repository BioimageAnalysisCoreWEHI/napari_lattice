"""
Tests of the ROI Tracking tab built inside the real dock widget: what is shown
before and after a TrackMate file is loaded, multi-track selection, per-track
window sizes, and the crop window preview overlay.
"""
from __future__ import annotations

import pytest
from magicclass import MagicTemplate
from magicclass._gui._gui_modes import ErrorMode
from magicclass.widgets import Widget

from napari_lattice.dock_widget import LLSZWidget

CSV = (
    "LABEL,ID,TRACK_ID,QUALITY,POSITION_X,POSITION_Y,POSITION_Z,POSITION_T,FRAME,RADIUS\n"
    "Label,Spot ID,Track ID,Quality,X,Y,Z,T,Frame,Radius\n"
    "Label,Spot ID,Track ID,Quality,X,Y,Z,T,Frame,R\n"
    ",,,(quality),(micron),(micron),(micron),(sec),,(micron)\n"
    "ID0,0,0,1.0,10.0,20.0,0.0,0.0,0,7.5\n"
    "ID1,1,0,1.0,11.0,21.0,0.0,1.0,1,7.5\n"
    "ID2,2,3,1.0,50.0,60.0,0.0,0.0,0,7.5\n"
    "ID3,3,3,1.0,51.0,61.0,0.0,1.0,1,7.5\n"
)

HIDDEN_UNTIL_ENABLED = ["track_ids", "select_all", "deselect_all", "show_preview"]
HIDDEN_UNTIL_LOADED = ["fields_enabled", *HIDDEN_UNTIL_ENABLED]


def _raise_errors(cls: MagicTemplate):
    """
    Stop magicclass showing a modal dialog for errors, which would hang the
    test run, and let them propagate to pytest instead.
    """
    def _handler(e: Exception, parent: Widget):
        raise e
    ErrorMode.get_handler = lambda self: _handler
    cls._error_mode = ErrorMode.stderr
    for child in cls.__magicclass_children__:
        _raise_errors(child)


@pytest.fixture
def tab(make_napari_viewer):
    viewer = make_napari_viewer()
    ui = LLSZWidget()
    _raise_errors(ui)
    viewer.window.add_dock_widget(ui)
    return ui.LlszMenu.WidgetContainer.trackmate_fields, viewer


@pytest.fixture
def loaded(tab, tmp_path):
    fields, viewer = tab
    path = tmp_path / "spots.csv"
    path.write_text(CSV)
    fields.tracks_path.value = path
    fields.load_tracks()
    return fields, viewer


def _shown(widget) -> bool:
    # `.visible` also reflects the parent, which is not on screen under test.
    return not widget.native.isHidden()


def _preview(viewer):
    return viewer.layers["spots crop window"]


def test_only_the_file_picker_is_shown_before_loading(tab):
    fields, _ = tab
    for name in HIDDEN_UNTIL_LOADED:
        assert not _shown(fields[name]), name
    assert _shown(fields.tracks_path)


def test_only_the_enabled_box_is_revealed_after_loading(loaded):
    fields, _ = loaded
    assert _shown(fields.fields_enabled)
    for name in HIDDEN_UNTIL_ENABLED:
        assert not _shown(fields[name]), name


def test_track_options_are_revealed_when_enabled(loaded):
    fields, _ = loaded
    fields.fields_enabled.value = True
    for name in HIDDEN_UNTIL_ENABLED:
        assert _shown(fields[name]), name
    assert list(fields.track_ids.choices) == ["0", "3"]


def test_disabling_hides_the_track_options_and_deselects(loaded):
    fields, viewer = loaded
    fields.fields_enabled.value = True
    fields.track_ids.value = ["0", "3"]

    fields.fields_enabled.value = False
    for name in HIDDEN_UNTIL_ENABLED:
        assert not _shown(fields[name]), name
    assert fields.track_ids.value == []
    assert not _shown(fields.track_windows)
    assert "spots crop window" not in viewer.layers

    fields.fields_enabled.value = True
    assert fields.track_ids.value == []


def test_select_all_picks_every_track(loaded):
    fields, _ = loaded
    fields.select_all()
    assert fields._get_selected_track_ids() == ["0", "3"]
    fields.deselect_all()
    assert fields._get_selected_track_ids() == []


def test_preview_has_one_window_per_selected_track(loaded):
    fields, viewer = loaded
    assert "spots crop window" not in viewer.layers

    fields.track_ids.value = ["0", "3"]
    assert len(_preview(viewer).data) == 2

    fields.track_ids.value = ["3"]
    assert len(_preview(viewer).data) == 1


def _window_boxes(fields) -> dict:
    return {box.label: box for box in fields.track_windows}


def test_window_sizes_are_hidden_until_a_track_is_selected(loaded):
    fields, _ = loaded
    assert not _shown(fields.track_windows)
    fields.track_ids.value = ["3"]
    assert _shown(fields.track_windows)


def test_there_is_a_window_size_for_each_selected_track(loaded):
    fields, _ = loaded
    fields.track_ids.value = ["3", "0"]
    assert list(_window_boxes(fields)) == ["Track 0", "Track 3"]
    assert all(box.value == 30.0 for box in fields.track_windows)

    fields.track_ids.value = ["3"]
    assert list(_window_boxes(fields)) == ["Track 3"]


def test_a_reselected_track_keeps_its_window_size(loaded):
    fields, _ = loaded
    fields.track_ids.value = ["0", "3"]
    _window_boxes(fields)["Track 3"].value = 12.0

    fields.track_ids.value = ["0"]
    fields.track_ids.value = ["0", "3"]
    assert _window_boxes(fields)["Track 3"].value == 12.0


def test_loading_a_file_forgets_the_window_sizes(loaded):
    fields, _ = loaded
    fields.track_ids.value = ["0"]
    _window_boxes(fields)["Track 0"].value = 12.0

    fields.load_tracks()
    assert len(fields.track_windows) == 0
    fields.track_ids.value = ["0"]
    assert _window_boxes(fields)["Track 0"].value == 30.0


def test_preview_is_the_window_size_centred_on_the_track(loaded):
    fields, viewer = loaded
    fields.track_ids.value = ["0"]
    _window_boxes(fields)["Track 0"].value = 10.0

    ys, xs = _preview(viewer).data[0][:, 0], _preview(viewer).data[0][:, 1]
    assert (ys.min(), ys.max(), xs.min(), xs.max()) == (15.0, 25.0, 5.0, 15.0)


def test_preview_updates_when_the_window_size_changes(loaded):
    fields, viewer = loaded
    fields.track_ids.value = ["0"]
    _window_boxes(fields)["Track 0"].value = 10.0
    narrow = _preview(viewer).data[0][:, 0].ptp()

    _window_boxes(fields)["Track 0"].value = 40.0
    assert _preview(viewer).data[0][:, 0].ptp() == 40.0
    assert narrow == 10.0


def test_each_track_has_its_own_window_size(loaded):
    fields, viewer = loaded
    fields.track_ids.value = ["0", "3"]
    _window_boxes(fields)["Track 0"].value = 10.0

    sizes = sorted(float(shape[:, 0].ptp()) for shape in _preview(viewer).data)
    assert sizes == [10.0, 30.0]


def test_preview_can_be_turned_off(loaded):
    fields, viewer = loaded
    fields.track_ids.value = ["0"]
    assert "spots crop window" in viewer.layers

    fields.show_preview.value = False
    assert "spots crop window" not in viewer.layers


def test_model_has_a_window_for_each_selected_track(loaded):
    fields, _ = loaded
    fields.fields_enabled.value = True
    fields.track_ids.value = ["0", "3"]
    _window_boxes(fields)["Track 0"].value = 10.0

    crop = fields._make_model()

    assert crop.roi_subset == [0, 3]
    assert crop.window_for_track(0) == 10.0
    assert crop.window_for_track(3) == 30.0
    assert set(crop.roi_by_time) == {0, 3}


def test_enabling_without_a_track_is_reported_as_invalid(loaded):
    fields, _ = loaded
    fields.fields_enabled.value = True
    assert "at least one track" in fields.errors.value

    fields.track_ids.value = ["0"]
    assert fields.errors.value == ""


def test_disabled_tab_is_valid_without_a_track(loaded):
    fields, _ = loaded
    fields.fields_enabled.value = True
    fields.fields_enabled.value = False
    assert fields.errors.value == ""


def test_model_needs_a_track_to_be_selected(loaded):
    fields, _ = loaded
    fields.fields_enabled.value = True
    with pytest.raises(ValueError, match="at least one track"):
        fields._make_model()
