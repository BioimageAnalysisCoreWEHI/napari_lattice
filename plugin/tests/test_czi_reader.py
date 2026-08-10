"""
Tests for the per-plane pylibCZIrw fast path in the napari reader.
"""
from __future__ import annotations

import numpy as np
import pytest
from importlib_resources import as_file
from bioio import BioImage
from lls_core.sample import resources
from lls_core.czi_reader import _Z_CHUNK
from napari_lattice.reader import _czi_fast_dask_data, _czi_fast_metadata, bioio_reader


@pytest.mark.parametrize("name", ["RBC_tiny.czi", "LLS7_t2_ch3.czi"])
def test_czi_fast_dask_data_matches_bioio_and_is_plane_chunked(name):
    with as_file(resources / name) as path:
        image = BioImage(path)
        order = image.dims.order
        ref = image.dask_data

        fast = _czi_fast_dask_data(str(path), image)
        assert fast is not None, "CZI fast path should engage for a plain CZI"

        # Same shape/dtype as bioio.
        assert fast.shape == ref.shape
        assert fast.dtype == ref.dtype

        # A single block spans Y and X; T and C are one index per block; Z is chunked
        # in _Z_CHUNK-plane blocks purely to keep dask's graph small (a 2D slice still
        # reads exactly one plane, because dask pushes the index into `from_array`).
        sizes = dict(zip(order, ref.shape))
        blocks = dict(zip(order, fast.numblocks))
        chunks = dict(zip(order, fast.chunksize))
        assert blocks["Y"] == 1 and blocks["X"] == 1
        for dim in order:
            if dim in ("Y", "X", "Z"):
                continue
            assert blocks[dim] == sizes[dim], f"expected one chunk per {dim} index"
        assert chunks["Z"] == min(_Z_CHUNK, sizes["Z"])

        # Values identical to bioio.
        assert np.array_equal(np.asarray(fast.compute()), np.asarray(ref.compute()))


@pytest.mark.parametrize("name", ["RBC_tiny.czi", "LLS7_t2_ch3.czi"])
def test_czi_fast_path_plane_slice_matches_bioio(name):
    """A single-plane slice - what napari actually requests - must match bioio."""
    with as_file(resources / name) as path:
        image = BioImage(path)
        order = image.dims.order
        ref = image.dask_data

        fast = _czi_fast_dask_data(str(path), image)
        assert fast is not None

        sizes = dict(zip(order, ref.shape))
        # Mid-stack: Z=0 is frequently near-black and would false-pass.
        idx = tuple(slice(None) if d in ("Y", "X") else sizes[d] // 2 for d in order)
        got = np.asarray(fast[idx])
        assert got.shape == tuple(sizes[d] for d in ("Y", "X"))
        assert np.array_equal(got, np.asarray(ref[idx]))


def test_non_czi_falls_back_to_bioio():
    # A .tif path must not engage the CZI fast path (returns None -> caller uses bioio).
    class _Dummy:
        pass
    assert _czi_fast_dask_data("something.tif", _Dummy()) is None


@pytest.mark.parametrize(
    "name",
    ["RBC_tiny.czi", "LLS7_t1_ch1.czi", "LLS7_t1_ch3.czi", "LLS7_t2_ch1.czi", "LLS7_t2_ch3.czi"],
)
def test_czi_fast_metadata_matches_bioio(name):
    """
    The cheap metadata derivation must agree with bioio exactly. This is the check
    that protects the bypass: `bioio_reader` now takes the dimension order, shape and
    channel names from `_czi_fast_metadata` instead of `image.dims`, because the
    latter builds bioio's whole per-plane graph (~188 s on a 300k-plane timelapse).
    If a bioio upgrade changes how it derives any of these, this fails loudly.
    """
    with as_file(resources / name) as path:
        image = BioImage(path)
        meta = _czi_fast_metadata(str(path), image)
        assert meta is not None

        assert meta["order"] == "".join(image.dims.order)
        assert meta["shape"] == tuple(image.dask_data.shape)
        assert meta["dtype"] == image.dask_data.dtype
        assert list(meta["channel_names"]) == list(image.channel_names)


def test_non_czi_has_no_fast_metadata():
    class _Dummy:
        pass
    assert _czi_fast_metadata("something.tif", _Dummy()) is None


@pytest.mark.parametrize("name", ["RBC_tiny.czi", "LLS7_t2_ch3.czi"])
def test_bioio_reader_never_touches_bioio_graph_for_czi(monkeypatch, name):
    """
    Reading a CZI must not touch any BioImage property that routes through
    `xarray_dask_data` - that is the 188 s the bypass exists to avoid. Make every
    such property explode, and the reader must still produce a layer.
    """
    def _boom(self):
        raise AssertionError("bioio_reader built bioio's dask graph for a CZI")

    for prop in ("dims", "dask_data", "xarray_dask_data", "channel_names", "shape", "dtype"):
        monkeypatch.setattr(BioImage, prop, property(_boom), raising=True)

    with as_file(resources / name) as path:
        layers = bioio_reader(str(path))
        assert layers, "reader should still return a layer"
        data, add_kwargs, layer_type = layers[0]
        assert layer_type == "image"
        assert add_kwargs["scale"]
        assert np.asarray(data[(0,) * (data.ndim - 2)]).ndim == 2
