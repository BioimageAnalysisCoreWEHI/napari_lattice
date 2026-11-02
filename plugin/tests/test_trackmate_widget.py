"""
Tests for the standalone load_trackmate_tracks_widget (trackmate_widget.py).
"""
import importlib

import pytest

CSV = (
    "LABEL,ID,TRACK_ID,QUALITY,POSITION_X,POSITION_Y,POSITION_Z,POSITION_T,FRAME,RADIUS\n"
    "ID0,0,0,1.0,10.0,20.0,1.0,0.0,0,7.5\n"
    "ID1,1,0,1.0,11.0,21.0,1.0,1.0,1,7.5\n"
)


class FakeViewer:
    def __init__(self):
        self.added = []

    def add_tracks(self, data, properties=None, name=None, tail_width=None):
        self.added.append(dict(data=data, properties=properties, name=name, tail_width=tail_width))


def test_widget_module_imports():
    # Fails if the module imports via `core.lls_core...` instead of `lls_core...`,
    # which only resolves when the repo root happens to be on sys.path.
    importlib.import_module("napari_lattice.trackmate_widget")


@pytest.fixture
def load_widget_fn(qtbot):
    module = importlib.import_module("napari_lattice.trackmate_widget")
    factory = module.load_trackmate_tracks_widget
    try:
        return factory.keywords["function"]
    except (AttributeError, KeyError):
        return factory()._function


def test_widget_rejects_a_missing_file(load_widget_fn, tmp_path):
    with pytest.raises(ValueError, match="valid TrackMate tracks file"):
        load_widget_fn(viewer=FakeViewer(), tracks_path=tmp_path / "nope.csv", layer_name="")


def test_widget_adds_a_tracks_layer_named_after_the_file(load_widget_fn, tmp_path):
    path = tmp_path / "spots.csv"
    path.write_text(CSV)
    viewer = FakeViewer()

    load_widget_fn(viewer=viewer, tracks_path=path, layer_name="")

    assert len(viewer.added) == 1
    assert viewer.added[0]["name"] == "spots"
    assert viewer.added[0]["data"].shape == (2, 5)


def test_widget_uses_a_custom_layer_name(load_widget_fn, tmp_path):
    path = tmp_path / "spots.csv"
    path.write_text(CSV)
    viewer = FakeViewer()

    load_widget_fn(viewer=viewer, tracks_path=path, layer_name=" mine ")

    assert viewer.added[0]["name"] == "mine"
