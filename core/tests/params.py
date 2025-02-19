import pytest
import json
import yaml

from lls_core.models.output import SaveFileType

parameterized = pytest.mark.parametrize("args", [
    {"skew": "X"},
    {"skew": "Y"},
    {"angle": 30},
    {"physical_pixel_sizes": (1, 1, 1)},
    {"save_type": SaveFileType.h5},
    {"save_type": SaveFileType.tiff},
])

# Allows parameterisation over two serialization formats
config_types = pytest.mark.parametrize(["save_func", "cli_param"], [
     (json.dump, "--json-config"),
     (yaml.safe_dump, "--yaml-config")
])
