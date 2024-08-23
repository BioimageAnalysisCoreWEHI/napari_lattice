from napari.viewer import current_viewer, Viewer
from napari.layers import Layer
from typing_extensions import TypeVar
from typing import Sequence, Type

def get_viewer() -> Viewer:
    """
    Returns the current viewer, throwing an exception if one doesn't exist
    """
    viewer = current_viewer()
    if viewer is None:
        raise Exception("No viewer present!")
    return viewer

LayerType = TypeVar("LayerType", bound=Layer)
def get_layers(type: Type[LayerType]) -> Sequence[LayerType]:
    """
    Returns all layers in the current napari viewer of a given `Layer` subtype.
    For example, if you pass `napari.layers.Image`, it will return a list of
    Image layers
    """
    viewer = current_viewer()
    if viewer is None:
        return []
    return [layer for layer in viewer.layers if isinstance(layer, type)]
