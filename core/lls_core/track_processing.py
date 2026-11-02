"""
track_processing.py

Pure array transformations applied to a single TrackMate trajectory before it
is used to drive a moving crop box.

The input is always the ``(n, 4)`` ``[t, x, y, z]`` array produced by
:func:`lls_core.trackmate_io.load_trackmate_tracks` (``tracks[id]['trackData']``)
or, in the GUI, by ``TrackmateFields._get_selected_track_data()``. ``t`` is an
integer frame index and ``x``, ``y``, ``z`` are in microns, TrackMate's native
axis order.

Two things stand between that array and a crop box:

* TrackMate drops a spot whenever the detector loses the cell for a frame, so
  a track can have holes in it. A crop that follows the cell needs a position
  for *every* frame, hence :func:`interpolate_gaps`.
* The coordinates are physical (microns) but the crop is indexed in pixels,
  and lattice data has a different spacing along each axis, hence
  :func:`microns_to_pixels`.

Both functions are pure: no file or model access, and the voxel sizes are
passed in by the caller rather than read from a settings object.
"""
from __future__ import annotations

import numpy as np

# Column indices of the (n, 4) track array, in TrackMate's native order.
T, X, Y, Z = 0, 1, 2, 3


def _as_track_array(track_data: np.ndarray, *, allow_empty: bool) -> np.ndarray:
    """
    Validate a track array and return it as a float array.

    Parameters
    ----------
    track_data : np.ndarray
        Candidate ``(n, 4)`` ``[t, x, y, z]`` array.
    allow_empty : bool
        Whether a track with zero rows is acceptable.

    Returns
    -------
    np.ndarray
        ``track_data`` as a ``float`` array. This is a copy unless the input
        was already a float array, so callers that mutate must copy first.

    Raises
    ------
    ValueError
        If the input is not a 2D array with 4 columns, if it is empty and
        ``allow_empty`` is False, or if it holds non-numeric data.
    """
    try:
        array = np.asarray(track_data, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Track data must be numeric: {exc}") from exc

    if array.ndim != 2 or array.shape[1] != 4:
        raise ValueError(
            "Track data must be a 2D array of shape (n, 4) holding "
            f"[t, x, y, z] rows, got shape {array.shape}."
        )
    if not allow_empty and array.shape[0] == 0:
        raise ValueError("Track data is empty - nothing to process.")

    return array


def interpolate_gaps(track_data: np.ndarray) -> np.ndarray:
    """
    Fill missing frames in a single track by linear interpolation.

    TrackMate omits a spot for any frame in which the detector lost the cell,
    which leaves the track with holes. The returned track has one row for
    every integer frame between the first and last real detection, so a
    downstream crop always has a position to follow.

    Gaps are filled in ``x``, ``y`` and ``z`` independently with
    :func:`numpy.interp`. Real detections are carried through untouched, and
    nothing is extrapolated beyond the ends of the track - a cell that
    appears at frame 5 and vanishes at frame 40 still yields frames 5 to 40.

    Parameters
    ----------
    track_data : np.ndarray, shape (n, 4)
        ``[t, x, y, z]`` rows for one track, as returned by
        :func:`lls_core.trackmate_io.load_trackmate_tracks`. Rows need not be
        sorted by time. ``t`` values are frame indices, and are rounded to the
        nearest integer to decide which frames are present. If two rows round
        to the same frame, the first (in time order) is kept.

    Returns
    -------
    np.ndarray, shape (m, 4)
        ``[t, x, y, z]`` rows sorted by increasing integer ``t``, with
        ``m >= n`` and no missing frames in between. A single-detection track
        is returned as-is.

    Raises
    ------
    ValueError
        If ``track_data`` is empty or is not an ``(n, 4)`` array.
    """
    array = _as_track_array(track_data, allow_empty=False)

    if array.shape[0] == 1:
        return array.copy()

    # A stable sort means that, among rows landing on the same frame, the one
    # that came first in the input stays first and so wins the dedup below.
    array = array[np.argsort(array[:, T], kind="stable")]

    # Snap to integers only to work out which frames exist. Interpolating
    # against the snapped frames (rather than the raw times) keeps the real
    # detections' coordinates exactly as measured.
    frames = np.rint(array[:, T])
    _, first_occurrence = np.unique(frames, return_index=True)
    frames = frames[first_occurrence]
    array = array[first_occurrence]

    if frames.shape[0] == 1:
        # Every detection collapsed onto one frame, so there is nothing to
        # interpolate between.
        return array.copy()

    full_frames = np.arange(frames[0], frames[-1] + 1, dtype=float)

    filled = np.empty((full_frames.shape[0], 4), dtype=float)
    filled[:, T] = full_frames
    for axis in (X, Y, Z):
        filled[:, axis] = np.interp(full_frames, frames, array[:, axis])

    return filled


def microns_to_pixels(
    track_data: np.ndarray,
    dx: float,
    dy: float,
    dz: float,
) -> np.ndarray:
    """
    Convert a track's spatial coordinates from microns to pixel indices.

    Each axis is divided by its own voxel size, since lattice data is
    typically sampled much more coarsely along ``z`` than in the imaging
    plane. Passing ``1.0`` for an axis leaves it alone, which is what a
    caller working in pixel units already wants.

    Parameters
    ----------
    track_data : np.ndarray, shape (n, 4)
        ``[t, x, y, z]`` rows for one track, with ``x``, ``y`` and ``z`` in
        microns.
    dx, dy, dz : float
        Voxel size in microns along each axis.

    Returns
    -------
    np.ndarray, shape (n, 4)
        A new array with ``x``, ``y`` and ``z`` in (fractional) pixels and
        ``t`` unchanged. Values are not rounded, so sub-pixel positions
        survive for the caller to use or discard. The input is not modified.

    Raises
    ------
    ValueError
        If any voxel size is not a finite positive number, or if
        ``track_data`` is not an ``(n, 4)`` array.
    """
    for name, size in (("dx", dx), ("dy", dy), ("dz", dz)):
        size = float(size)
        if not np.isfinite(size) or size <= 0:
            raise ValueError(
                f"Pixel size {name} must be a finite positive number of "
                f"microns, got {size}."
            )

    pixels = _as_track_array(track_data, allow_empty=True).copy()
    pixels[:, X] /= float(dx)
    pixels[:, Y] /= float(dy)
    pixels[:, Z] /= float(dz)

    return pixels
