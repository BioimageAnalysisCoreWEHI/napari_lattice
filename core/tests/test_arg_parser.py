from typer.main import get_command
from click import Context
from lls_core.cmds.__main__ import app


def test_voxel_parsing():
    # Tests that we can parse voxel lists correctly
    command = get_command(app).commands["process"]
    ctx = Context(command)
    parser = command.make_parser(ctx)
    args, _, _ = parser.parse_args(args=[
            "process",
            "input",
            "--save-name", "output",
            "--save-type", "tiff",
            "--pixel-sizes", "1", "1", "1"
    ])
    assert args["pixel_sizes"] == ("1", "1", "1")
