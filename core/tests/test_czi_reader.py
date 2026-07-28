"""
Tests for `lls_core.czi_reader`, the lazy pylibCZIrw fast path shared by the napari
plugin and the CLI.

Parity against real acquisitions lives in `plugin/tests/test_czi_reader.py`, which
uses the five bundled sample CZIs. This file covers what needs a purpose-built file -
stage drift, multiple scenes - plus the properties the performance work depends on and
the CLI call sites.
"""
from __future__ import annotations

import logging

import numpy as np
import pytest
from bioio import BioImage

from lls_core.czi_reader import czi_dask_array, czi_metadata


def test_drift_czi_reads_the_canvas_not_the_subblock(drift_czi, czi_stub_image):
    """
    Regression for "cannot reshape array of size 432345600 into shape (834,300,1734)".

    When each timepoint records a different stage offset, the subblocks are narrower
    than the canvas they sit on. Reading the subblock width gives an array that cannot
    be reshaped into the canvas bioio reports - 20 vs 25 here, 1728 vs 1734 on the file
    that surfaced the bug.
    """
    path, planes, offsets = drift_czi
    stub = czi_stub_image(path)

    meta = czi_metadata(str(path), stub)
    assert meta is not None, "the fast path declined a plain single-scene CZI"
    assert meta["order"] == "TCZYX"
    assert meta["dtype"] == np.dtype("uint16")
    assert meta["shape"] == (3, 1, 4, 12, 25), "X must be the canvas, not the subblock"

    arr = czi_dask_array(str(path), stub, meta)
    assert arr is not None

    expected = np.zeros((3, 1, 4, 12, 25), dtype=np.uint16)
    for (t, z), plane in planes.items():
        expected[t, 0, z][:, offsets[t]:offsets[t] + 20] = plane
    assert np.array_equal(np.asarray(arr.compute()), expected), (
        "drift offsets were not composited onto the canvas"
    )


def test_drift_czi_matches_bioio(drift_czi, czi_stub_image):
    """
    bioio cannot enumerate scenes on a generated CZI, but `.dask_data` works, so pixel
    parity is still available for the single-scene case.
    """
    path, _planes, _offsets = drift_czi

    arr = czi_dask_array(str(path), czi_stub_image(path))
    assert arr is not None

    ref = BioImage(str(path)).dask_data
    assert arr.shape == ref.shape
    assert arr.dtype == ref.dtype
    assert np.array_equal(np.asarray(arr.compute()), np.asarray(ref.compute()))


@pytest.mark.parametrize("scene_index", [0, 1])
def test_multi_scene_czi_reads_the_selected_scene(
    multi_scene_czi, czi_stub_image, scene_index
):
    """
    Each scene must yield its own pixels from its own bounding rectangle.

    Before the runtime probe was removed this declined - the probe compared a plane
    against `image.dask_data`, so multi-scene files paid for bioio's full per-plane
    graph on every open, and declined outright when bioio could not supply one.
    """
    path, planes = multi_scene_czi
    stub = czi_stub_image(path, n_scenes=2, scene_index=scene_index)

    meta = czi_metadata(str(path), stub)
    assert meta is not None
    assert meta["shape"] == (1, 1, 3, 10, 14), "each scene has its own rectangle"
    assert meta["scene"] == scene_index

    arr = czi_dask_array(str(path), stub, meta)
    assert arr is not None, "multi-scene CZIs must take the fast path"

    expected = np.stack([planes[(scene_index, z)] for z in range(3)])[None, None]
    assert np.array_equal(np.asarray(arr.compute()), expected)
