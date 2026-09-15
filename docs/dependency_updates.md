# Dependency Updates (Pydantic & pyclesperanto)

This page summarises the Sprint 1 dependency upgrades: migrating **Pydantic v1 → v2** and replacing **`pyclesperanto_prototype` with `pyclesperanto`**. It explains what changed, why, how the new code works, and how tests were updated.

## Overview

| Area | Before | After |
|------|--------|--------|
| Parameter models / validation | `pydantic.v1` compatibility layer (`pydantic>=1.10,<3`) | Native Pydantic v2 (`pydantic>=2,<3`) |
| GPU image processing | `pyclesperanto_prototype` | `pyclesperanto` |
| Deskew affine helpers | Private APIs inside the prototype package | Vendored into `lls_core` (`affine.py`, OpenCL kernels) |

These two upgrades are largely independent (separate branches), but both touch shared models, CLI entry points, and tests.

---

## Pydantic v2 migration

### Why

napari-lattice historically used Pydantic **v1 APIs** (either Pydantic 1.x, or v2 via `from pydantic.v1 import ...`). That compatibility path is deprecated, and several dependencies already expect Pydantic 2. Moving to native v2 keeps validation current and removes reliance on the v1 shim.

### What changed

**Dependency** (`core/pyproject.toml`):

- From: `pydantic>=1.10.17,<3` (with v1 compatibility mode)
- To: `pydantic>=2,<3`

**Core API renames** used throughout models, CLI, writers, and tests:

| Pydantic v1 | Pydantic v2 |
|-------------|-------------|
| `from pydantic.v1 import ...` | `from pydantic import ...` |
| `class Config:` / `Extra.forbid` | `model_config = ConfigDict(...)` |
| `__fields__` | `model_fields` |
| `.copy()` / `.dict()` / `.validate()` | `.model_copy()` / `.model_dump()` / `.model_validate()` |
| `.construct()` | `.model_construct()` |
| `@validator` / `@root_validator` | `@field_validator` / `@model_validator` |
| `values` dict in validators | `info: ValidationInfo` → `info.data` |
| `LatticeData.parse_obj(...)` | `LatticeData.model_validate(...)` |

**Main files touched**

- `core/lls_core/models/utils.py` — shared `FieldAccessModel` helpers (`get_default`, `to_definition_dict`, `copy_validate`)
- `core/lls_core/models/lattice_data.py`, `deskew.py`, `crop.py`, `deconvolution.py`, `output.py`, `results.py`
- `core/lls_core/cmds/__main__.py` — CLI builds `LatticeData` via `model_validate`
- `plugin/napari_lattice/fields.py` — GUI field defaults / descriptions via v2 field access

### How it works now

1. **Models still centralise parameters.** `LatticeData` and related models define deskew, crop, deconvolution, and output options. Validation still rejects invalid ranges, missing images, bad workflows, etc.
2. **Validators run under v2 rules.** Cross-field logic (e.g. default `time_range` / `channel_range` from the loaded image) uses `@field_validator` with `ValidationInfo`, reading sibling fields from `info.data`.
3. **`validate_default=True`** was added on selected `LatticeData` fields so defaults still go through validators (this replaces v1’s `always=True` behaviour for those fields).
4. **Extra metadata for the CLI** (`cli_hide`, `cli_description`) now lives in `json_schema_extra` instead of v1 `field_info.extra`.

### Related fix (pixel sizes)

A follow-up fix ensured physical pixel sizes are stored as the correct typed model (`DefinedPixelSizes`), so validation actually applies instead of silently accepting the wrong object type.

### Tests updated for Pydantic

Tests that constructed or mocked models were updated to the v2 API, for example:

- `LatticeData.parse_obj(...)` → `LatticeData.model_validate(...)`
- Monkeypatches of `parse_obj` → `model_validate` in CLI / parallel-processing tests

Affected test modules include `test_process.py`, `test_parallel_processing.py`, `test_roi_units.py`, and `test_validation.py`.

---

## pyclesperanto migration

### Why

`pyclesperanto_prototype` is the older, large prototype API. The maintained package is now **`pyclesperanto`**: a thinner rewrite with a different public surface. Several deskew helpers we relied on were **private** in the prototype and **do not exist** in the new library, so they were ported into `lls_core` to keep deskew geometry stable.

### What changed

**Dependency**

- From: `pyclesperanto_prototype>=0.20.0`
- To: `pyclesperanto`

**Import convention** (code, notebooks, workflow YAML examples):

```python
import pyclesperanto as cle
```

Workflow YAML entries that referenced prototype callables were updated, e.g.:

```yaml
- !!python/name:pyclesperanto.median_sphere ''
```

### How it works now

#### 1. Public GPU ops via `pyclesperanto`

Filters, deskew helpers that still exist publicly, and `cle.push` / `cle.pull` / `cle.execute` come from the new package. Call sites (estimate, shear-only path, MIP, plugin fields, tests) import `pyclesperanto as cle`.

Some keyword names differ from the prototype (e.g. deskew `angle` instead of `angle_in_degrees` where the new API requires it). Tests were updated to match.

#### 2. Vendored affine / deskew math in `lls_core`

Because the new library does not expose the old private classes, the following live in-repo:

| Module / asset | Role |
|----------------|------|
| `core/lls_core/affine.py` | `AffineTransform3D` and `determine_translation_and_bounding_box` (numpy / transforms3d only) |
| `core/lls_core/affine_transform_deskew.py` | Orthogonal-interpolation deskew runner via `pyclesperanto.execute` |
| `core/lls_core/kernels/*.cl` | OpenCL kernels for X/Y deskew |

`DeskewDirection` is defined in `lls_core` itself instead of being imported from the prototype.

#### 3. Array type recognition

`core/lls_core/types.py` treats `pyclesperanto.Array` (`CleArray`) as array-like, alongside numpy, dask, xarray, and raw pyopencl arrays. That matters when workflows return GPU arrays from `pyclesperanto` functions.

#### 4. Crop / workflow robustness fixes (test-driven)

While making the suite pass under the new backend:

- **Degenerate crop guard** in `crop_volume_deskew`: expand crops that would be &lt; 2 voxels wide so GPU kernels always see a valid 3D buffer.
- **Workflow DataFrame export** in `results.py`: explode columns independently and reset indexes before concat, avoiding pandas “duplicate labels” errors when combining ROI results.

### Docs and notebooks

User-facing mentions of `pyclesperanto_prototype` were switched to `pyclesperanto` in README acknowledgements, workflow docs, educational notebooks, and example workflow YAMLs. That is a rename/update pass so examples match the installed package; it is separate from the behaviour notes above.

### Tests updated for pyclesperanto

Broad import and API updates across GPU-related tests, including:

- `conftest.py`, `reference_deskew.py`
- `test_deskew.py`, `test_crop_deskew.py`, `test_crop_placement.py`
- `test_deconvolution.py`, `test_mip.py`, `test_shear_only.py`, `test_parallel_processing.py`
- Workflow YAML under `core/tests/workflows/`

Behavioural regressions caught by those tests drove the crop-size and DataFrame export fixes above.

### Known Windows note

On Windows, `test_workflow_cli` can fail with `PermissionError` on temporary config files (`NamedTemporaryFile` still open when re-opened). That is an environment/test-harness issue, not a pyclesperanto regression. Core deskew / crop / shear-only coverage under the new library is expected to pass when OpenCL is available.

---

## How to verify locally

From the repository root, with the `napari-lattice` conda environment activated:

```bash
pip install -e "core[testing]" -e "plugin[testing]"
python -m pytest core/tests/ plugin/tests -q --tb=line -p no:warnings
```

Confirm:

```bash
python -c "import pyclesperanto as cle; print(cle.__version__); from pydantic import VERSION; print(VERSION)"
```

You should see `pyclesperanto` (not `_prototype`) and Pydantic 2.x.

---

## Developer checklist when touching these areas

**Pydantic**

- Prefer v2 APIs only (`model_validate`, `field_validator`, `model_config`).
- When adding cross-field validation, use `ValidationInfo` / `info.data`.
- Keep CLI metadata in `json_schema_extra` if the field should appear in CLI help.

**pyclesperanto**

- Import `pyclesperanto as cle`.
- Do not reintroduce `pyclesperanto_prototype` imports.
- For deskew geometry that is not in the public API, extend the vendored helpers under `lls_core` rather than depending on private upstream modules.
- If a workflow returns GPU arrays, ensure `is_arraylike` / writers handle `CleArray`.

---

## Related branches / commits (Sprint 1)

**Pydantic (`pydantic-upgrades`)**

- Initial model/CLI/GUI migration to Pydantic v2
- Test updates (`parse_obj` → `model_validate`, etc.)
- Pixel-size typing fix for `DefinedPixelSizes`

**pyclesperanto (`pyclesperanto-upgrades`)**

- Code migration away from `pyclesperanto_prototype`
- Vendored affine / deskew kernel support in `lls_core`
- Notebook / docs / workflow YAML renames
- Test fixes (crop guard, CleArray typing, DataFrame explode)
