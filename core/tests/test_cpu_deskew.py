#filename and function name should start with "test_" when using pytest
import numpy as np
import pytest
from lls_core import DeskewDirection, DeskewEngine
from lls_core.models.crop import CropParams
from lls_core.models.lattice_data import LatticeData
from pydantic import ValidationError
from xarray import DataArray
import tempfile
from tests.utils import requires_real_gpu


def _feature_volume() -> np.ndarray:
    raw = np.zeros((10, 12, 8), dtype=np.float32)
    raw[5, 6, 4] = 100.0
    raw[2, 3, 1] = 40.0
    return raw


def _deskew(raw: np.ndarray, engine: str, skew: DeskewDirection, tmpdir: str) -> np.ndarray:
    lattice = LatticeData(
        input_image=DataArray(raw, dims=["Z", "Y", "X"]),
        # Isotropic XY pixel size: CPU and GPU engines agree on the output geometry
        # and (float32-precision) values whenever dx == dy. See cpu_deskew.py's
        # docstring for the anisotropic-XY caveat.
        physical_pixel_sizes=(0.3, 0.145, 0.145),
        angle=30.0,
        skew=skew,
        engine=engine,
        save_name="test",
        save_dir=tmpdir,
    )
    return np.asarray(next(iter(lattice.process().slices)).data)


@requires_real_gpu
@pytest.mark.parametrize("skew", [DeskewDirection.Y, DeskewDirection.X])
def test_cpu_engine_matches_gpu_engine(skew: DeskewDirection):
    raw = _feature_volume()
    with tempfile.TemporaryDirectory() as tmpdir:
        gpu = _deskew(raw, "GPU", skew, tmpdir)
        cpu = _deskew(raw, "CPU", skew, tmpdir)

    assert gpu.shape == cpu.shape
    np.testing.assert_allclose(gpu.astype(np.float64), cpu.astype(np.float64), atol=0.01)
    # The comparison must be non-trivial (both engines actually produced content)
    assert (gpu > 0).any()


def test_engine_accepts_string():
    raw = _feature_volume()
    with tempfile.TemporaryDirectory() as tmpdir:
        lattice = LatticeData(
            input_image=DataArray(raw, dims=["Z", "Y", "X"]),
            physical_pixel_sizes=(0.3, 0.145, 0.145),
            engine="CPU",
            save_name="test",
            save_dir=tmpdir,
        )
    assert lattice.engine == DeskewEngine.CPU


def test_cpu_engine_rejects_crop():
    raw = np.zeros((5, 5, 5), dtype=np.float32)
    with tempfile.TemporaryDirectory() as tmpdir:
        with pytest.raises(ValidationError, match="ROI cropping is not supported"):
            LatticeData(
                input_image=DataArray(raw, dims=["Z", "Y", "X"]),
                physical_pixel_sizes=(1, 1, 1),
                engine=DeskewEngine.CPU,
                crop=CropParams(roi_list=[[[0, 0], [0, 5], [5, 5], [5, 0]]]),
                save_name="test",
                save_dir=tmpdir,
            )


def test_cpu_engine_rejects_shear_only():
    raw = np.zeros((5, 5, 5), dtype=np.float32)
    with tempfile.TemporaryDirectory() as tmpdir:
        with pytest.raises(ValidationError, match="CPU deskew engine currently only supports"):
            LatticeData(
                input_image=DataArray(raw, dims=["Z", "Y", "X"]),
                physical_pixel_sizes=(1, 1, 1),
                engine=DeskewEngine.CPU,
                coverslip_rotation=False,
                save_name="test",
                save_dir=tmpdir,
            )
