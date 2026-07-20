"""
Output dtype preservation: deskewing returns float32 from the GPU, so saving
must restore the input dtype - 8-bit stays 8-bit, 32-bit labels stay 32-bit.
"""
import numpy as np
import pytest
import tifffile

from lls_core.models.lattice_data import LatticeData
from lls_core.writers import resolve_output_dtype, to_output_dtype


@pytest.mark.parametrize("in_dtype,out_dtype", [
    ("uint8", "uint8"),
    ("uint16", "uint16"),
    ("int32", "int32"),        # labels: was narrowed to uint16, clipping IDs
    ("float32", "float32"),
    ("bool", "uint8"),         # no 1-bit type in OME-TIFF; was a TypeError
    ("int64", "int32"),        # no 64-bit int in OME-TIFF
])
def test_resolve_output_dtype(in_dtype, out_dtype):
    assert resolve_output_dtype(np.dtype(in_dtype)) == np.dtype(out_dtype)


def test_rounds_rather_than_truncates():
    """A bare .astype() truncates, biasing every voxel down ~0.5 counts."""
    arr = np.array([114.83, 82.099, 0.9, -2.7], dtype=np.float32)
    np.testing.assert_array_equal(to_output_dtype(arr, np.dtype("int16")), [115, 82, 1, -3])


@pytest.mark.parametrize("out_dtype", ["uint8", "uint16", "int32", "uint32"])
def test_out_of_range_saturates_never_wraps(out_dtype, recwarn):
    """iinfo(int32).max is not representable in float32, so the cast overflowed."""
    info = np.iinfo(out_dtype)
    out = to_output_dtype(np.array([1e30, -1e30], dtype=np.float32), np.dtype(out_dtype))
    assert out[0] >= info.max * 0.99, f"positive overflow wrapped to {out[0]}"
    assert out[1] <= 0, f"negative overflow wrapped to {out[1]}"
    assert not [w for w in recwarn if "invalid value" in str(w.message)]


def test_nan_is_zeroed_and_warned(caplog):
    """NaN would silently become 0, indistinguishable from background."""
    with caplog.at_level("WARNING"):
        out = to_output_dtype(np.array([np.nan, 5.7], dtype=np.float32), np.dtype("uint16"))
    assert out[0] == 0 and out[1] == 6
    assert "NaN" in caplog.text


def _lattice(tmp_path, raw):
    return LatticeData(
        input_image=raw, save_dir=str(tmp_path), save_name="t", save_type="tiff",
        physical_pixel_sizes=(0.3, 0.1449, 0.1449),
    )


def test_deconvolution_output_stays_float32(tmp_path, monkeypatch):
    """Deconvolved peaks legitimately exceed the input range, so are not cast."""
    lattice = _lattice(tmp_path, np.zeros((5, 10, 10), dtype=np.uint16))
    monkeypatch.setattr(type(lattice), "deconv_enabled", property(lambda _: True))
    out = np.asarray(lattice._restore_input_dtype(np.array([[[70000.5]]], dtype=np.float32)))
    assert out.dtype == np.float32 and out.max() > 65535


def test_uint8_deskew_saves_as_uint8(tmp_path):
    """The reported bug: 8-bit input came out wider after deskewing."""
    raw = (np.random.default_rng(0).random((15, 40, 40)) * 200 + 50).astype(np.uint8)
    _lattice(tmp_path, raw).save()
    written = list(tmp_path.glob("*.tif*"))
    assert len(written) == 1
    data = tifffile.imread(str(written[0]))
    assert data.dtype == np.uint8
    assert data.max() > 100, f"output collapsed, max={data.max()}"
