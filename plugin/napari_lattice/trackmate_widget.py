"""
trackmate_widget.py

A small napari widget that lets a user pick a TrackMate track export - either
the "tracks-only" XML or the "spots in tracks statistics" CSV - and adds it
to the viewer as an interactive napari ``Tracks`` layer.
"""
from __future__ import annotations

from pathlib import Path

from magicgui import magic_factory

from lls_core.trackmate_io import trackmate_file_to_napari_tracks

try:
    # Only needed for the type hint below; napari is a runtime dependency
    # of the plugin this widget lives in, so this import is safe.
    from napari.viewer import Viewer
except ImportError:  # pragma: no cover - keeps this file importable in tests
    Viewer = "napari.viewer.Viewer"


@magic_factory(
    call_button="Load TrackMate tracks",
    tracks_path={
        "label": "TrackMate tracks (XML or CSV)",
        "filter": "TrackMate tracks (*.xml *.csv)",
        "mode": "r",
    },
    layer_name={"label": "Layer name (optional)"},
)
def load_trackmate_tracks_widget(
    viewer: "Viewer",
    tracks_path: Path = Path.home(),
    layer_name: str = "",
) -> None:
    """
    Load a TrackMate ROI-tracking export (XML "tracks-only" export, or CSV
    "spots in tracks statistics" export) and add it to the viewer as a
    ``Tracks`` layer.

    Parameters
    ----------
    viewer : napari.Viewer
        Injected automatically by napari/magicgui - the currently active
        viewer instance.
    tracks_path : Path
        Path to the XML file produced by TrackMate's "Export tracks to XML
        file" action, or the CSV produced by "Export spots to CSV".
    layer_name : str
        Optional name for the new layer. Defaults to the file's stem.
    """
    if not tracks_path or not Path(tracks_path).is_file():
        raise ValueError(f"Please choose a valid TrackMate tracks file (got: {tracks_path!r}).")

    data, properties = trackmate_file_to_napari_tracks(tracks_path)

    name = layer_name.strip() or Path(tracks_path).stem
    viewer.add_tracks(
        data,
        properties=properties,
        name=name,
        tail_width=2,
    )