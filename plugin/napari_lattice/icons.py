import importlib_resources

resources = importlib_resources.files(__name__)

GREEN = resources / "valid.svg"
GREY = resources / "circle-regular.svg"
RED = resources / "invalid.svg"
