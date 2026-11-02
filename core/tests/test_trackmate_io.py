"""
Tests for lls_core.trackmate_io: loading TrackMate XML / CSV track exports
and converting them into napari Tracks layer data.
"""
from __future__ import annotations

import numpy as np
import pytest

from lls_core.trackmate_io import (
    load_trackmate_csv,
    load_trackmate_tracks,
    load_trackmate_xml,
    trackmate_file_to_napari_tracks,
    trackmate_tracks_to_napari,
)

XML = """<?xml version="1.0" encoding="UTF-8"?>
<Tracks nTracks="2" spaceUnits="micron" frameInterval="1.0" timeUnits="sec">
  <particle nSpots="3">
    <detection t="1" x="11.0" y="21.0" z="1.0" />
    <detection t="0" x="10.0" y="20.0" z="1.0" />
    <detection t="2" x="12.0" y="22.0" z="1.0" />
  </particle>
  <particle nSpots="1">
    <detection t="0" x="30.0" y="40.0" z="2.0" />
  </particle>
</Tracks>
"""

# TrackMate repeats the header three times (labels, short labels, units)
# before the data. POSITION_T is in seconds and deliberately differs from FRAME.
CSV_HEADER = (
    "LABEL,ID,TRACK_ID,QUALITY,POSITION_X,POSITION_Y,POSITION_Z,POSITION_T,FRAME,RADIUS\n"
    "Label,Spot ID,Track ID,Quality,X,Y,Z,T,Frame,Radius\n"
    "Label,Spot ID,Track ID,Quality,X,Y,Z,T,Frame,R\n"
    ",,,(quality),(micron),(micron),(micron),(sec),,(micron)\n"
)
CSV = CSV_HEADER + (
    "ID1,1,0,1.0,11.0,21.0,1.0,5.0,1,7.5\n"
    "ID0,0,0,1.0,10.0,20.0,1.0,0.0,0,7.5\n"
    "ID2,2,0,1.0,12.0,22.0,1.0,10.0,2,7.5\n"
    "ID3,3,1,1.0,30.0,40.0,2.0,0.0,0,7.5\n"
)


@pytest.fixture
def xml_file(tmp_path):
    path = tmp_path / "tracks.xml"
    path.write_text(XML)
    return path


@pytest.fixture
def csv_file(tmp_path):
    path = tmp_path / "spots.csv"
    path.write_text(CSV)
    return path


def _write(tmp_path, name, text):
    path = tmp_path / name
    path.write_text(text)
    return path


def test_xml_tracks_are_keyed_by_particle_index(xml_file):
    tracks = load_trackmate_xml(xml_file)
    assert list(tracks) == ["0", "1"]
    assert tracks["0"]["trackData"].shape == (3, 4)
    assert tracks["1"]["trackData"].shape == (1, 4)


def test_xml_track_data_columns_are_t_x_y_z(xml_file):
    row = load_trackmate_xml(xml_file)["1"]["trackData"][0]
    np.testing.assert_array_equal(row, [0.0, 30.0, 40.0, 2.0])


def test_xml_sorts_detections_by_time(xml_file):
    data = load_trackmate_xml(xml_file)["0"]["trackData"]
    np.testing.assert_array_equal(data[:, 0], [0.0, 1.0, 2.0])
    np.testing.assert_array_equal(data[:, 1], [10.0, 11.0, 12.0])


def test_xml_nspots_is_read_from_the_attribute(xml_file):
    tracks = load_trackmate_xml(xml_file)
    assert tracks["0"]["nSpots"] == 3
    assert tracks["1"]["nSpots"] == 1


def test_xml_nspots_falls_back_to_detection_count(tmp_path):
    path = _write(tmp_path, "t.xml", XML.replace('<particle nSpots="3">', "<particle>"))
    assert load_trackmate_xml(path)["0"]["nSpots"] == 3


def test_xml_wrong_ntracks_attribute_is_tolerated(tmp_path):
    path = _write(tmp_path, "t.xml", XML.replace('nTracks="2"', 'nTracks="9"'))
    assert len(load_trackmate_xml(path)) == 2


def test_xml_with_no_particles_returns_empty_dict(tmp_path):
    path = _write(tmp_path, "t.xml", '<Tracks nTracks="0"></Tracks>')
    assert load_trackmate_xml(path) == {}


def test_xml_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError, match="not found"):
        load_trackmate_xml(tmp_path / "nope.xml")


def test_xml_malformed_file(tmp_path):
    path = _write(tmp_path, "t.xml", "<Tracks><particle>")
    with pytest.raises(ValueError, match="Could not parse"):
        load_trackmate_xml(path)


def test_xml_wrong_root_tag(tmp_path):
    path = _write(tmp_path, "t.xml", "<TrackMate></TrackMate>")
    with pytest.raises(ValueError, match="expected root tag <Tracks>"):
        load_trackmate_xml(path)


def test_csv_skips_the_repeated_header_rows(csv_file):
    tracks = load_trackmate_csv(csv_file)
    total = sum(len(t["trackData"]) for t in tracks.values())
    assert total == 4


def test_csv_tracks_are_keyed_by_trackmate_track_id(tmp_path):
    path = _write(tmp_path, "s.csv", CSV.replace("ID3,3,1,", "ID3,3,7,"))
    assert set(load_trackmate_csv(path)) == {"0", "7"}


def test_csv_time_comes_from_frame_not_position_t(csv_file):
    data = load_trackmate_csv(csv_file)["0"]["trackData"]
    # POSITION_T is 0, 5, 10 s; FRAME is 0, 1, 2.
    np.testing.assert_array_equal(data[:, 0], [0.0, 1.0, 2.0])


def test_csv_track_data_columns_are_t_x_y_z(csv_file):
    row = load_trackmate_csv(csv_file)["1"]["trackData"][0]
    np.testing.assert_array_equal(row, [0.0, 30.0, 40.0, 2.0])


def test_csv_sorts_spots_by_frame(csv_file):
    data = load_trackmate_csv(csv_file)["0"]["trackData"]
    np.testing.assert_array_equal(data[:, 1], [10.0, 11.0, 12.0])


def test_csv_nspots_counts_rows(csv_file):
    tracks = load_trackmate_csv(csv_file)
    assert tracks["0"]["nSpots"] == 3
    assert tracks["1"]["nSpots"] == 1


def test_csv_skips_spots_that_are_not_in_a_track(tmp_path):
    path = _write(tmp_path, "s.csv", CSV + "ID9,9,None,1.0,1.0,1.0,0.0,0.0,0,7.5\n")
    tracks = load_trackmate_csv(path)
    assert set(tracks) == {"0", "1"}


def test_csv_accepts_float_formatted_track_ids(tmp_path):
    path = _write(tmp_path, "s.csv", CSV_HEADER + "ID0,0,4.0,1.0,1.0,2.0,0.0,0.0,0,7.5\n")
    assert list(load_trackmate_csv(path)) == ["4"]


def test_csv_missing_required_columns(tmp_path):
    path = _write(tmp_path, "s.csv", "LABEL,ID,TRACK_ID,POSITION_X\nID0,0,0,1.0\n")
    with pytest.raises(ValueError, match="missing columns"):
        load_trackmate_csv(path)


def test_csv_with_only_header_rows(tmp_path):
    path = _write(tmp_path, "s.csv", CSV_HEADER)
    with pytest.raises(ValueError, match="No spot data"):
        load_trackmate_csv(path)


def test_csv_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError, match="not found"):
        load_trackmate_csv(tmp_path / "nope.csv")


@pytest.mark.parametrize("name, text", [("a.xml", XML), ("a.XML", XML), ("a.csv", CSV), ("a.CSV", CSV)])
def test_dispatch_picks_loader_by_suffix(tmp_path, name, text):
    tracks = load_trackmate_tracks(_write(tmp_path, name, text))
    assert set(tracks) == {"0", "1"}


def test_dispatch_rejects_other_suffixes(tmp_path):
    with pytest.raises(ValueError, match="Unsupported TrackMate track file type"):
        load_trackmate_tracks(_write(tmp_path, "a.txt", CSV))


def _tracks(**by_id):
    return {k: {"nSpots": len(v), "trackData": np.asarray(v, dtype=float)} for k, v in by_id.items()}


def test_napari_columns_are_id_t_z_y_x():
    data, _ = trackmate_tracks_to_napari(_tracks(**{"3": [[5, 1.0, 2.0, 7.0]]}))
    # TrackMate [t, x, y, z] -> napari [id, t, z, y, x]
    np.testing.assert_array_equal(data, [[3, 5, 7.0, 2.0, 1.0]])


def test_napari_rows_sorted_by_track_id_then_time():
    tracks = _tracks(**{
        "10": [[1, 0, 0, 0], [0, 0, 0, 0]],
        "2": [[1, 0, 0, 0], [0, 0, 0, 0]],
    })
    data, _ = trackmate_tracks_to_napari(tracks)
    np.testing.assert_array_equal(data[:, 0], [2, 2, 10, 10])   # numeric, not "10" < "2"
    np.testing.assert_array_equal(data[:, 1], [0, 1, 0, 1])


def test_napari_properties_give_nspots_per_vertex():
    tracks = _tracks(**{"0": [[0, 0, 0, 0], [1, 0, 0, 0]], "1": [[0, 0, 0, 0]]})
    data, props = trackmate_tracks_to_napari(tracks)
    assert len(props["nSpots"]) == len(data)
    np.testing.assert_array_equal(props["nSpots"], [2, 2, 1])


def test_napari_rejects_empty_tracks():
    with pytest.raises(ValueError, match="empty"):
        trackmate_tracks_to_napari({})


def test_xml_and_csv_of_the_same_tracks_give_the_same_napari_data(xml_file, csv_file):
    xml_data, xml_props = trackmate_file_to_napari_tracks(xml_file)
    csv_data, csv_props = trackmate_file_to_napari_tracks(csv_file)
    np.testing.assert_array_equal(xml_data, csv_data)
    np.testing.assert_array_equal(xml_props["nSpots"], csv_props["nSpots"])


def test_output_is_accepted_by_a_napari_tracks_layer(csv_file):
    layers = pytest.importorskip("napari.layers")
    data, props = trackmate_file_to_napari_tracks(csv_file)
    layer = layers.Tracks(data, properties=props)
    assert layer.data.shape == data.shape
