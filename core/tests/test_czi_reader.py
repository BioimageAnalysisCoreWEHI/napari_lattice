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


@pytest.mark.parametrize("bioio_index", [0, 1])
def test_noncontiguous_scene_keys_map_to_the_czi_scene(
    noncontiguous_scene_czi, czi_stub_image, bioio_index
):
    """
    A CZI's own scene keys need not be zero-based or contiguous; here they are 1 and 2.

    bioio maps its 0..N-1 index through the sorted rectangle keys before handing it to
    pylibCZIrw. Passing the BioIO index straight through instead reads the wrong scene:
    index 0 finds no rectangle at all (so the shape becomes the whole canvas and the
    read raises), and index 1 silently returns CZI scene 1's pixels - right shape, right
    dtype, wrong image, which is the worst failure mode for scientific imaging.
    """
    path, planes = noncontiguous_scene_czi
    stub = czi_stub_image(path, n_scenes=2, scene_index=bioio_index)

    meta = czi_metadata(str(path), stub)
    assert meta is not None
    assert meta["shape"] == (1, 1, 3, 10, 14), "each scene has its own rectangle"
    assert meta["scene"] == bioio_index + 1, "BioIO index must map to the CZI key"

    arr = czi_dask_array(str(path), stub, meta)
    assert arr is not None

    expected = np.stack([planes[(bioio_index + 1, z)] for z in range(3)])[None, None]
    assert np.array_equal(np.asarray(arr.compute()), expected)


def test_bioio_scene_index_out_of_range_declines(
    noncontiguous_scene_czi, czi_stub_image
):
    """The never-raise contract: an impossible scene index falls back, it does not throw."""
    path, _planes = noncontiguous_scene_czi
    stub = czi_stub_image(path, n_scenes=2, scene_index=5)

    assert czi_metadata(str(path), stub) is None


def test_two_dimensional_slice_reads_exactly_one_plane(
    drift_czi, czi_stub_image, czi_read_calls
):
    """
    `_Z_CHUNK` is 32, so a Z chunk spans up to 32 planes. That is a dask-graph-size
    knob, not a read-volume one, only because dask pushes single-plane indexing down
    into `from_array`. If a dask change stops doing that, the chunk becomes a 32x read
    amplifier and nothing else in the suite would notice.
    """
    path, _planes, _offsets = drift_czi
    arr = czi_dask_array(str(path), czi_stub_image(path))
    assert arr is not None

    czi_read_calls.clear()          # discard anything the metadata pass did
    plane = np.asarray(arr[1, 0, 2].compute())

    assert plane.shape == (12, 25)
    assert len(czi_read_calls) == 1, czi_read_calls
    assert czi_read_calls[0] == {"T": 1, "C": 0, "Z": 2}


def test_whole_block_reads_one_plane_per_index(drift_czi, czi_stub_image, czi_read_calls):
    """A genuine whole-array request reads every plane exactly once - no more."""
    path, _planes, _offsets = drift_czi
    arr = czi_dask_array(str(path), czi_stub_image(path))
    assert arr is not None

    czi_read_calls.clear()
    arr.compute()

    assert len(czi_read_calls) == 3 * 4          # T x Z, one read each
    assert len({tuple(sorted(c.items())) for c in czi_read_calls}) == 3 * 4


def test_arrays_share_one_thread_pool(drift_czi, multi_scene_czi, czi_stub_image):
    """
    The executor is module-level, so opening many files does not accumulate a pool
    each. A per-array pool would leak up to eight threads per file opened.
    """
    import threading

    from lls_core import czi_reader

    drift_path, _planes, _offsets = drift_czi
    scene_path, _scene_planes = multi_scene_czi

    pool = czi_reader._pool()
    for path, stub in (
        (drift_path, czi_stub_image(drift_path)),
        (scene_path, czi_stub_image(scene_path, n_scenes=2)),
    ):
        arr = czi_dask_array(str(path), stub)
        assert arr is not None
        arr.compute()          # a whole-block read, which is what uses the pool

    assert czi_reader._pool() is pool
    live = [t for t in threading.enumerate() if t.name.startswith("lls-czi")]
    assert len(live) <= 8, [t.name for t in live]


def test_declining_a_czi_says_why(tmp_path, caplog):
    """
    A silent fallback turns a 1.2 s open back into a 195 s one. Turning on ordinary
    logging has to be enough to explain why a file went slow again.
    """
    broken = tmp_path / "broken.czi"
    broken.write_bytes(b"this is not a CZI")

    with caplog.at_level(logging.INFO, logger="lls_core.czi_reader"):
        assert czi_dask_array(str(broken), None) is None

    messages = [r.getMessage() for r in caplog.records]
    assert any("declined" in m for m in messages), messages


def test_declining_a_non_czi_is_silent(caplog):
    """Every TIFF and HDF5 open would otherwise log; only .czi bailouts are news."""
    with caplog.at_level(logging.DEBUG, logger="lls_core.czi_reader"):
        assert czi_dask_array("not_a_czi.tif", None) is None

    assert caplog.records == [], [r.getMessage() for r in caplog.records]


def _array_name(image) -> str:
    """Our arrays are named "lls-czi-<token>"; bioio's are "reshape-..." etc."""
    data = getattr(image, "data", image)
    return str(getattr(data, "name", ""))


def test_image_like_to_image_takes_the_fast_path(rbc_tiny):
    from lls_core.types import image_like_to_image

    got = image_like_to_image(str(rbc_tiny))
    assert _array_name(got).startswith("lls-czi-"), _array_name(got)

    ref = BioImage(str(rbc_tiny)).xarray_dask_data
    assert got.dims == ref.dims
    assert np.array_equal(np.asarray(got), np.asarray(ref))


def test_load_image_lazy_takes_the_fast_path(rbc_tiny):
    """
    This is the call a parallel ROI worker makes after `_dispatch_payload` strips the
    unpicklable image and sends only the path.
    """
    from pathlib import Path

    from lls_core.models.deskew import load_image_lazy

    got = load_image_lazy(Path(str(rbc_tiny)))
    assert _array_name(got).startswith("lls-czi-"), _array_name(got)

    ref = BioImage(str(rbc_tiny)).xarray_dask_data
    assert got.dims == ref.dims
    assert np.array_equal(np.asarray(got), np.asarray(ref))


def test_deskew_params_read_image_takes_the_fast_path(rbc_tiny):
    """The path every `lls-pipeline` run with a CZI input takes."""
    from lls_core.models.deskew import DeskewParams

    params = DeskewParams(input_image=str(rbc_tiny))
    assert _array_name(params.input_image).startswith("lls-czi-"), _array_name(
        params.input_image
    )
