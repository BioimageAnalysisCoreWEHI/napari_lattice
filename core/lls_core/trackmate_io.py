"""
trackmate_io.py

Load ROI trajectories exported from TrackMate - either "Export tracks to XML
file" or "Export spots to CSV" and convert them into the data format napari's 
``Tracks`` layer expects.

TrackMate's "tracks-only" XML looks like this::

    <Tracks nTracks="9" spaceUnits="micron" frameInterval="1.0" timeUnits="sec" ...>
      <particle nSpots="41">
        <detection t="0" x="47.84" y="0.0" z="0.0" />
        <detection t="1" x="48.71" y="0.0" z="0.0" />
        ...
      </particle>
      ...
    </Tracks>

TrackMate's CSV looks like this::

    LABEL,ID,TRACK_ID,QUALITY,POSITION_X,POSITION_Y,POSITION_Z,POSITION_T,FRAME,RADIUS,...
    Label,Spot ID,Track ID,Quality,X,Y,Z,T,Frame,Radius,...
    Label,Spot ID,Track ID,Quality,X,Y,Z,T,Frame,R,...
    ,,,(quality),(micron),(micron),(micron),(sec),,(micron),...
    ID43514,43514,0,0.023,47.85,0.0,0.0,0.0,0,7.5,...
    ...

napari's Tracks layer wants an (N, D+2) array where each row is
``[track_id, t, (z), y, x]`` and rows are sorted by increasing track_id then
time (see https://napari.org/stable/howtos/layers/tracks.html). Note the
axis order flip: TrackMate stores spots as (x, y, z) but napari's array
uses (z, y, x), matching image-axis order.
"""
from __future__ import annotations

import csv
import io
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, Tuple, Union

import numpy as np

PathLike = Union[str, Path]


def load_trackmate_xml(xml_path: PathLike) -> Dict[str, dict]:
    """
    Parse a TrackMate "tracks-only" XML export into a plain dict:

        tracks = {'0': {'nSpots': 41, 'trackData': np.ndarray of shape (41, 4)}, ...}

    where each row of ``trackData`` is ``[t, x, y, z]`` (TrackMate's native
    order), matching the structure produced by ``importTrackMateTracks.m``.

    Parameters
    ----------
    xml_path : str or Path
        Path to the XML file produced by TrackMate's
        "Export tracks to XML file" action.

    Returns
    -------
    dict
        Mapping from track index (as a string) to a dict with keys
        ``'nSpots'`` and ``'trackData'``.

    Raises
    ------
    FileNotFoundError
        If ``xml_path`` does not exist.
    ValueError
        If the file cannot be parsed as TrackMate track XML.
    """
    xml_path = Path(xml_path)
    if not xml_path.is_file():
        raise FileNotFoundError(f"TrackMate XML file not found: {xml_path}")

    try:
        tree = ET.parse(xml_path)
    except ET.ParseError as exc:
        raise ValueError(f"Could not parse '{xml_path}' as XML: {exc}") from exc

    root = tree.getroot()
    if root.tag != "Tracks":
        raise ValueError(
            f"'{xml_path}' does not look like a TrackMate tracks export "
            f"(expected root tag <Tracks>, found <{root.tag}>)."
        )

    particles = root.findall("particle")
    n_tracks = int(root.attrib.get("nTracks", len(particles)))
    if n_tracks != len(particles):
        # Don't hard-fail on a mismatched count attribute - just trust the
        # actual number of <particle> elements found.
        n_tracks = len(particles)

    tracks: Dict[str, dict] = {}
    for i, particle in enumerate(particles):
        detections = particle.findall("detection")
        n_spots = int(particle.attrib.get("nSpots", len(detections)))

        track_data = np.empty((len(detections), 4), dtype=float)
        for j, detection in enumerate(detections):
            track_data[j] = (
                float(detection.attrib["t"]),
                float(detection.attrib["x"]),
                float(detection.attrib["y"]),
                float(detection.attrib["z"]),
            )
        # TrackMate does not guarantee detections are written in time order,
        # so sort defensively on the time column.
        track_data = track_data[np.argsort(track_data[:, 0])]

        tracks[str(i)] = {"nSpots": n_spots, "trackData": track_data}

    return tracks


def load_trackmate_csv(csv_path: PathLike) -> Dict[str, dict]:
    """
    Parse a TrackMate "Spots in tracks statistics" CSV export into the same
    plain dict shape as :func:`load_trackmate_xml`:

        tracks = {'0': {'nSpots': 41, 'trackData': np.ndarray of shape (41, 4)}, ...}

    where each row of ``trackData`` is ``[t, x, y, z]``. ``t`` is taken from
    the CSV's ``FRAME`` column (an integer frame index), not ``POSITION_T``
    (physical time in seconds) - the two coincide only when the export's
    frame interval is 1, and downstream processing indexes the image stack
    by frame number.

    Parameters
    ----------
    csv_path : str or Path
        Path to the CSV file produced by TrackMate's 
        "Export spots to CSV" action

    Returns
    -------
    dict
        Mapping from track index (as a string) to a dict with keys
        ``'nSpots'`` and ``'trackData'``.

    Raises
    ------
    FileNotFoundError
        If ``csv_path`` does not exist.
    ValueError
        If the file does not have the expected TrackMate spots columns, or
        contains no parseable spot rows.
    """
    csv_path = Path(csv_path)
    if not csv_path.is_file():
        raise FileNotFoundError(f"TrackMate CSV file not found: {csv_path}")

    required = {"TRACK_ID", "POSITION_X", "POSITION_Y", "POSITION_Z", "FRAME"}
    rows_by_track: Dict[str, list] = {}
    with open(csv_path, newline="") as handle:
        reader = csv.DictReader(handle)
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(
                f"'{csv_path}' does not look like a TrackMate spots CSV export "
                f"(missing columns: {sorted(missing)})"
            )

        for row in reader:
            # TrackMate repeats the header 3 more times (human-readable
            # labels, abbreviated labels, then a units row like
            # "(quality),(micron),...") before the real data starts. Skip
            # any row whose numeric columns don't parse, rather than
            # assuming a fixed number of rows to skip.
            try:
                track_id = str(int(float(row["TRACK_ID"])))
                t = float(row["FRAME"])
                x = float(row["POSITION_X"])
                y = float(row["POSITION_Y"])
                z = float(row["POSITION_Z"])
            except (TypeError, ValueError):
                continue
            rows_by_track.setdefault(track_id, []).append((t, x, y, z))

    if not rows_by_track:
        raise ValueError(f"No spot data found in '{csv_path}'")

    tracks: Dict[str, dict] = {}
    for track_id, rows in rows_by_track.items():
        track_data = np.asarray(rows, dtype=float)
        track_data = track_data[np.argsort(track_data[:, 0])]
        tracks[track_id] = {"nSpots": len(rows), "trackData": track_data}

    return tracks


def load_trackmate_tracks(path: PathLike) -> Dict[str, dict]:
    """Load a TrackMate track export, dispatching on file suffix (.xml or .csv)."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".xml":
        return load_trackmate_xml(path)
    if suffix == ".csv":
        return load_trackmate_csv(path)
    raise ValueError(
        f"Unsupported TrackMate track file type '{suffix}' for '{path}' "
        "(expected .xml or .csv)"
    )


def tracks_exist(path: PathLike) -> bool:
    """check if there is at least one track"""
    tracks = load_trackmate_tracks(path)
    return len(tracks) > 0

def is_trackmate_file_xml(xml_file: ET.ElementTree) -> bool:
    """check if an xml file is a trackmate file"""
    
    root = xml_file.getroot()
    if root.tag == "Tracks":
        return True
    return False

def is_trackmate_file_csv(csv_file: io.IOBase) -> bool:
    """check if a csv file is a trackmate file"""
    required = {"TRACK_ID", "POSITION_X", "POSITION_Y", "POSITION_Z", "FRAME"}

    reader = csv.DictReader(csv_file)
    missing = required - set(reader.fieldnames or [])
    if not missing:
        return True
    return False


def is_trackmate_file(path: PathLike) -> bool:
    """check if file is a trackmate file. if not it will be assumed to be a roi file"""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".xml":
        if not path.is_file():
            return False
        try:
            tree = ET.parse(path)
        except ET.ParseError as exc:
            raise ValueError(f"Could not parse '{path}' as XML: {exc}") from exc
        return is_trackmate_file_xml(tree)
    if suffix == ".csv":
        if not path.is_file():
            return False
        with open(path, newline="") as file:
            return is_trackmate_file_csv(file)
    return False

def zip_contains_trackmate_file(path: PathLike) -> bool:
    """catch trackmate files in zip files"""
    import zipfile
    contains_trackmate = False
    with zipfile.ZipFile(Path(path), "r") as zip_files:
        for file_name in zip_files.namelist():
            path = Path(file_name)
            suffix = path.suffix.lower()
            if suffix == ".xml":
                with zip_files.open(file_name) as file:
                    try:
                        tree = ET.parse(file)
                    except ET.ParseError as exc:
                        raise ValueError(f"Could not parse '{path}' as XML: {exc}") from exc
                    contains_trackmate = is_trackmate_file_xml(tree) if not contains_trackmate else contains_trackmate
            if suffix == ".csv":
                with zip_files.open(file_name) as raw_file:
                    file = io.TextIOWrapper(raw_file,newline="")
                    contains_trackmate = is_trackmate_file_csv(file) if not contains_trackmate else contains_trackmate
    return contains_trackmate

def trackmate_tracks_to_napari(
    tracks: Dict[str, dict],
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """
    Convert the dict produced by :func:`load_trackmate_xml` into the
    ``(data, properties)`` pair expected by ``napari.Viewer.add_tracks``.

    Parameters
    ----------
    tracks : dict
        Output of :func:`load_trackmate_xml`.

    Returns
    -------
    data : np.ndarray, shape (N, 5)
        Columns are ``[track_id, t, z, y, x]``, sorted by increasing
        track_id then time, as required by napari's Tracks layer.
    properties : dict
        A ``{'nSpots': np.ndarray}`` features table (one value per vertex,
        constant within a track) that can be passed as ``features=`` /
        ``properties=`` to ``add_tracks`` so it's available for
        colour-by-feature in the GUI.
    """
    if not tracks:
        raise ValueError("No tracks to convert - the tracks dict is empty.")

    rows = []
    n_spots_col = []
    for track_id_str in sorted(tracks, key=int):
        track_id = int(track_id_str)
        track_data = tracks[track_id_str]["trackData"]  # (n, 4) = t, x, y, z
        n_spots = tracks[track_id_str]["nSpots"]
        for t, x, y, z in track_data:
            # napari column order is track_id, t, z, y, x
            rows.append((track_id, t, z, y, x))
            n_spots_col.append(n_spots)

    data = np.asarray(rows, dtype=float)
    # Sort by track_id then t, per napari's requirement.
    order = np.lexsort((data[:, 1], data[:, 0]))
    data = data[order]
    properties = {"nSpots": np.asarray(n_spots_col, dtype=int)[order]}

    return data, properties


def trackmate_file_to_napari_tracks(
    path: PathLike,
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """Convenience wrapper: parse a TrackMate .xml or .csv straight to napari-ready data."""
    tracks = load_trackmate_tracks(path)
    return trackmate_tracks_to_napari(tracks)
