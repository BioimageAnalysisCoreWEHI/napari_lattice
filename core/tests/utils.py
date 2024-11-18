from pathlib import Path
from typing import Sequence
from typer.testing import CliRunner
from lls_core.cmds.__main__ import app
import npy2bdv
from aicsimageio import AICSImage
from pytest.mark import skipif
import os

def invoke(args: Sequence[str]):
    CliRunner().invoke(app, args, catch_exceptions=False)

def valid_image_path(path: Path) -> bool:
    if path.suffix in {".hdf5", ".h5"}:
        npy2bdv.npy2bdv.BdvEditor(str(path)).read_view()
        return True
    else:
        AICSImage(path).get_image_data()
        return True


def skip_on_github_ci():
    return skipif('GITHUB_ACTIONS' in os.environ)
