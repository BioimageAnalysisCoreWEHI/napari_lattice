from pathlib import Path
from typing import Sequence
from typer.testing import CliRunner
from lls_core.cmds.__main__ import app
import npy2bdv
import pytest
from bioio import BioImage


def _has_real_gpu() -> bool:
    """
    True if pyclesperanto's active backend is a real GPU, as opposed to a CPU-based
    OpenCL implementation (pocl on Linux, oclgrind on Windows) used as a GPU-less
    fallback in CI.

    cle.deskew_y/deskew_x are confirmed broken on pocl: they return a fixed garbage
    value instead of the interpolated pixel, on both pyclesperanto-opencl 0.22.0 and
    0.24.0 (reproduced in an environment matching CI exactly). This is a bug in
    pyclesperanto itself, not in lls_core - our own vendored deskew kernel
    (crop_volume_deskew) produces the correct answer on the same broken machine.
    Tests that use cle.deskew_y/deskew_x as a ground-truth reference can't be
    meaningfully evaluated on such a backend.
    """
    import pyclesperanto as cle
    try:
        return bool(cle.list_available_devices(device_type="gpu"))
    except Exception:
        return False


requires_real_gpu = pytest.mark.skipif(
    not _has_real_gpu(),
    reason=(
        "cle.deskew_y/deskew_x are broken on CPU-only OpenCL backends (pocl/oclgrind); "
        "this test relies on them as a ground-truth reference. See _has_real_gpu in tests/utils.py."
    ),
)


def invoke(args: Sequence[str]):
    CliRunner().invoke(app, args, catch_exceptions=False)

def valid_image_path(path: Path) -> bool:
    if path.suffix in {".hdf5", ".h5"}:
        npy2bdv.npy2bdv.BdvEditor(str(path)).read_view()
        return True
    else:
        BioImage(path).get_image_data()
        return True
