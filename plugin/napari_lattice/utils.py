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


def _embed_affine(matrix: Any, target_size: int):
    """
    Embed a ``(k+1)x(k+1)`` square homogeneous affine (k spatial dims) into a
    ``target_size x target_size`` identity so it acts on the LAST k axes —
    matching how napari coerces a smaller affine assigned to a higher-dimensional
    layer (verified against napari for 4x4->5x5 and 4x4->6x6: linear block at
    offset ``n-k``, translation in the last column).

    Only a *square* matrix smaller than ``target_size`` is embedded; anything
    else (non-square, non-2-D, or already ``>= target_size``) is returned
    unchanged so the caller's equality check errs toward assigning rather than
    silently mis-embedding a malformed input.
    """
    import numpy as np
    m = np.asarray(matrix, dtype=float)
    if m.ndim != 2 or m.shape[0] != m.shape[1] or m.shape[0] >= target_size:
        return m
    k = m.shape[0] - 1          # spatial dims of the given affine
    n = target_size - 1         # spatial dims of the layer
    offset = n - k
    result = np.eye(target_size)
    result[offset:offset + k, offset:offset + k] = m[:k, :k]     # linear part
    result[offset:offset + k, n] = m[:k, k]                      # translation
    return result


def apply_layer_transform(image: Any, scale: Any = _UNSET, affine: Any = _UNSET) -> List[str]:
    """
    Assign ``scale``/``affine`` to a napari layer ONLY when the value actually
    changes.

    Re-assigning an identical value still makes napari ``refresh()`` (re-slice
    the data). For a large, lazily-loaded image that single refresh can take
    tens of seconds, so skipping no-op assignments is the difference between an
    instant parameter tweak and a multi-second freeze.

    The desired value is first normalized to the layer's dimensionality exactly
    as napari would coerce it (a short scale is front-padded with 1.0; a small
    affine is embedded into a layer-sized identity). Without this, a 3-tuple
    scale / 4x4 affine assigned to a >3-D layer never shape-matches napari's
    padded value, so the no-op skip is silently defeated for the very case it
    exists to optimize.

    Pass ``_UNSET`` (the default) to leave a property alone. ``affine=None``
    means "reset to identity"; it is skipped only when the layer is already at
    identity. Returns the list of property names that were actually assigned,
    which callers use for logging and tests.
    """
    import numpy as np
    changed: List[str] = []

    if scale is not _UNSET and scale is not None:
        current_scale = getattr(image, "scale", None)
        desired = tuple(scale)
        if current_scale is not None and len(current_scale) > len(desired):
            desired = (1.0,) * (len(current_scale) - len(desired)) + desired
        if not _transforms_equal(current_scale, desired):
            image.scale = desired
            changed.append("scale")

    if affine is not _UNSET:
        current = getattr(getattr(image, "affine", None), "affine_matrix", None)
        if affine is None:
            # Reset to identity; compare against an identity of the layer's size.
            desired = np.eye(current.shape[0]) if current is not None else None
            assign_value = affine   # None -> napari coerces to identity
        else:
            assign_value = affine
            try:
                m = np.asarray(affine, dtype=float)
                if current is not None and m.ndim == 2 and m.shape[0] < current.shape[0]:
                    desired = _embed_affine(m, current.shape[0])
                    assign_value = desired   # assign the embedded form napari would store
                else:
                    desired = m
            except Exception:
                desired = affine        # unusual input -> let _transforms_equal decide (assigns)
        if not _transforms_equal(current, desired):
            image.affine = assign_value
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
