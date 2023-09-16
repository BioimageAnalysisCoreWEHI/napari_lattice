from typing import Sequence
from typer.testing import CliRunner
from lls_core.cmds.__main__ import app

def invoke(args: Sequence[str]):
    CliRunner().invoke(app, args, catch_exceptions=False)
