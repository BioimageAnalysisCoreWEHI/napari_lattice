"""
Tests for napari_lattice.utils.apply_layer_transform.

The GUI re-applies a layer's scale/affine on many parameter changes. Assigning
an identical value still forces napari to refresh (re-slice the data), which for
a large lazy image costs tens of seconds. apply_layer_transform must therefore
skip assignments whose value is unchanged, and only assign genuine changes.
"""
from __future__ import annotations

import numpy as np

from napari_lattice.utils import apply_layer_transform


class _FakeAffine:
    def __init__(self, matrix):
        self.affine_matrix = np.asarray(matrix, dtype=float)


class _FakeLayer:
    """Minimal stand-in for a napari Image layer that counts real assignments."""

    def __init__(self, scale, affine_matrix=None):
        self._scale = tuple(scale)
        self._affine = None if affine_matrix is None else _FakeAffine(affine_matrix)
        self.scale_assignments = 0
        self.affine_assignments = 0

    @property
    def scale(self):
        return self._scale

    @scale.setter
    def scale(self, value):
        self.scale_assignments += 1
        self._scale = tuple(value)

    @property
    def affine(self):
        return self._affine

    @affine.setter
    def affine(self, value):
        self.affine_assignments += 1
        self._affine = None if value is None else _FakeAffine(
            value.affine_matrix if isinstance(value, _FakeAffine) else value
        )


def test_unchanged_scale_is_not_reassigned():
    layer = _FakeLayer(scale=(0.3, 0.15, 0.15))
    changed = apply_layer_transform(layer, scale=(0.3, 0.15, 0.15))
    assert changed == []
    assert layer.scale_assignments == 0


def test_changed_scale_is_assigned():
    layer = _FakeLayer(scale=(0.3, 0.15, 0.15))
    changed = apply_layer_transform(layer, scale=(0.4, 0.15, 0.15))
    assert changed == ["scale"]
    assert layer.scale_assignments == 1
    assert layer.scale == (0.4, 0.15, 0.15)


def test_unset_leaves_properties_untouched():
    layer = _FakeLayer(scale=(0.3, 0.15, 0.15), affine_matrix=np.eye(4))
    changed = apply_layer_transform(layer)  # nothing passed
    assert changed == []
    assert layer.scale_assignments == 0
    assert layer.affine_assignments == 0


def test_affine_none_on_identity_layer_is_skipped():
    # affine=None means "reset to identity"; a layer already at identity is a no-op.
    layer = _FakeLayer(scale=(0.3, 0.15, 0.15), affine_matrix=np.eye(4))
    changed = apply_layer_transform(layer, affine=None)
    assert changed == []
    assert layer.affine_assignments == 0


def test_affine_none_clears_a_non_identity_affine():
    flip = np.eye(4)
    flip[0, 0] = -1
    flip[0, 3] = 41
    layer = _FakeLayer(scale=(0.3, 0.15, 0.15), affine_matrix=flip)
    changed = apply_layer_transform(layer, affine=None)
    assert changed == ["affine"]
    assert layer.affine_assignments == 1


def test_new_affine_matrix_is_assigned():
    # Pass a plain ndarray (as production does) so this exercises the numeric
    # equality/embedding path, not the "unparseable input -> assign" fallback.
    layer = _FakeLayer(scale=(0.3, 0.15, 0.15), affine_matrix=np.eye(4))
    flip = np.eye(4)
    flip[0, 0] = -1
    changed = apply_layer_transform(layer, affine=flip)
    assert changed == ["affine"]
    assert layer.affine_assignments == 1
    assert layer.affine.affine_matrix[0, 0] == -1


def test_equal_non_identity_affine_is_skipped():
    # The affine analogue of the scale no-op test: re-applying an equal
    # non-identity matrix must NOT re-assign (regression guard for the dedup).
    flip = np.eye(4)
    flip[0, 0] = -1
    flip[0, 3] = 7
    layer = _FakeLayer(scale=(0.3, 0.15, 0.15), affine_matrix=flip)
    changed = apply_layer_transform(layer, affine=flip.copy())
    assert changed == []
    assert layer.affine_assignments == 0


def test_short_scale_and_affine_skipped_on_higher_dim_layer():
    # A >3D layer stores a length-4 scale and 5x5 affine. Re-applying the
    # equivalent short (3-tuple scale / 4x4 affine) values must be recognized as
    # a no-op. Regression for the dedup being defeated by the shape mismatch
    # between napari's padded stored value and the short value passed in.
    layer = _FakeLayer(scale=(1.0, 0.3, 0.15, 0.15), affine_matrix=np.eye(5))
    changed = apply_layer_transform(layer, scale=(0.3, 0.15, 0.15), affine=np.eye(4))
    assert changed == []
    assert layer.scale_assignments == 0
    assert layer.affine_assignments == 0


def test_short_affine_change_on_higher_dim_layer_is_embedded_and_assigned():
    # A genuine 4x4 change on a 4D layer must still assign, stored as the 5x5
    # form napari would coerce it to (linear block at offset 1).
    layer = _FakeLayer(scale=(1.0, 0.3, 0.15, 0.15), affine_matrix=np.eye(5))
    flip = np.eye(4)
    flip[0, 0] = -1
    changed = apply_layer_transform(layer, affine=flip)
    assert changed == ["affine"]
    assert layer.affine_assignments == 1
    stored = layer.affine.affine_matrix
    assert stored.shape == (5, 5)
    assert stored[1, 1] == -1  # flip's linear part embedded at offset 1
