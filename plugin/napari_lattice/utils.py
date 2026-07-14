from napari.viewer import current_viewer, Viewer
from napari.layers import Layer
from typing_extensions import TypeVar
from typing import Any, List, Optional, Sequence, Type

# Sentinel meaning "leave this transform property untouched".
_UNSET: Any = object()


def _transforms_equal(a: Any, b: Any) -> bool:
    """
    True if two scale/affine values are numerically equal (same shape, close
    values). None equals None. Any comparison we can't make cleanly returns
    False, so the caller errs on the side of assigning (never wrongly skips).
    """
    if a is None and b is None:
        return True
    if a is None or b is None:
        return False
    import numpy as np
    try:
        aa = np.asarray(a, dtype=float)
        bb = np.asarray(b, dtype=float)
    except Exception:
        return False
    return aa.shape == bb.shape and bool(np.allclose(aa, bb))


def apply_layer_transform(image: Any, scale: Any = _UNSET, affine: Any = _UNSET) -> List[str]:
    """
    Assign ``scale``/``affine`` to a napari layer ONLY when the value actually
    changes.

    Re-assigning an identical value still makes napari ``refresh()`` (re-slice
    the data). For a large, lazily-loaded image that single refresh can take
    tens of seconds, so skipping no-op assignments is the difference between an
    instant parameter tweak and a multi-second freeze.

    Pass ``_UNSET`` (the default) to leave a property alone. ``affine=None``
    means "reset to identity"; it is skipped only when the layer is already at
    identity. Returns the list of property names that were actually assigned,
    which callers use for logging and tests.
    """
    changed: List[str] = []

    if scale is not _UNSET and scale is not None:
        if not _transforms_equal(getattr(image, "scale", None), scale):
            image.scale = scale
            changed.append("scale")

    if affine is not _UNSET:
        current = getattr(getattr(image, "affine", None), "affine_matrix", None)
        desired = affine
        if desired is None and current is not None:
            import numpy as np
            desired = np.eye(current.shape[0])
        if not _transforms_equal(current, desired):
            image.affine = affine
            changed.append("affine")

    return changed


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
