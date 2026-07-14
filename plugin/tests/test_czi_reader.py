"""
Tests for the per-(T,C) bulk-read CZI fast path in the napari reader.
"""
from __future__ import annotations

import numpy as np
import pytest
from importlib_resources import as_file
from bioio import BioImage
from lls_core.sample import resources
from napari_lattice.reader import _czi_fast_dask_data


@pytest.mark.parametrize("name", ["RBC_tiny.czi", "LLS7_t2_ch3.czi"])
def test_czi_fast_dask_data_matches_bioio_and_is_coarsely_chunked(name):
    with as_file(resources / name) as path:
        image = BioImage(path)
        order = image.dims.order
        ref = image.dask_data

        fast = _czi_fast_dask_data(str(path), image)
        assert fast is not None, "CZI fast path should engage for a plain CZI"

        # Same shape/dtype as bioio.
        assert fast.shape == ref.shape
        assert fast.dtype == ref.dtype

        # Chunked per (T, C): exactly one block along each of Z, Y, X.
        blocks = dict(zip(order, fast.numblocks))
        assert blocks["Z"] == 1 and blocks["Y"] == 1 and blocks["X"] == 1
        # ...and split along T and C (if present).
        assert blocks.get("T", 1) == dict(zip(order, ref.shape)).get("T", 1)
        assert blocks.get("C", 1) == dict(zip(order, ref.shape)).get("C", 1)

        # Values identical to bioio.
        assert np.array_equal(np.asarray(fast.compute()), np.asarray(ref.compute()))


def test_non_czi_falls_back_to_bioio():
    # A .tif path must not engage the CZI fast path (returns None -> caller uses bioio).
    class _Dummy:
        pass
    assert _czi_fast_dask_data("something.tif", _Dummy()) is None
