# CZI Reader Permanence Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn `lls_core/czi_reader.py` from an untested temporary workaround into maintained architecture: cover the bug it was written for, delete the runtime probe that made multi-scene files pay bioio's cost, and pin the properties its performance depends on.

**Architecture:** The module stays one file with two halves of different lifetimes. `czi_metadata` is temporary — bioio-czi#104 will replace it. `CziPlanes` + `czi_dask_array` are permanent — nothing upstream plans lazy array construction. A docstring marks the seam; no version gates, no file split. All pixel reads continue to go through pylibCZIrw, the same library bioio-czi uses by default, which is why output is byte-identical rather than merely equivalent.

**Tech Stack:** Python 3.11, pylibCZIrw, bioio + bioio-czi, dask.array, xarray, pytest.

**Spec:** [docs/superpowers/specs/2026-07-28-czi-reader-permanence-design.md](../specs/2026-07-28-czi-reader-permanence-design.md)

## Global Constraints

- **Branch:** all work lands on `fix_reader`. **Never push.** The user has asked not to commit during exploration — confirm before the first commit step, then commit per task.
- **Commit authorship is the user's alone.** Do not add a `Co-Authored-By` trailer, a "Generated with Claude Code" line, or any other AI attribution to a commit message or PR body. Commits carry the repo's configured identity (`git config user.name` / `user.email`) and nothing else — do not pass `--author` or otherwise override it. The commit messages quoted in each task are already in this form — use them verbatim.
- **Python:** `C:/Users/rajasekhar.p/.conda/envs/napari_lattice/python.exe`. Never use `conda run`.
- **Reads go through pylibCZIrw only.** Do not introduce an `aicspylibczi` read anywhere in `czi_reader.py`: it returns subblocks without applying their logical offset, which is the original bug.
- **Every entry point returns `None` rather than raising**, so callers fall back to bioio. Preserve this in every edit.
- **No committed binary fixtures.** Test CZIs are generated into `tmp_path` by pylibCZIrw's writer.
- **Generated CZIs need a stub image.** pylibCZIrw's writer never emits the `<Scenes>` XML that bioio's `scene_name()` requires, so `BioImage.scenes` raises `UnsupportedMetadataError: Expected 1 scene for index '0' but found 0` on every writer-produced file — with or without `write_metadata`, with or without an explicit `scene=`. `BioImage.dask_data` still works on them. Tests therefore supply a three-attribute stub where `czi_metadata` expects a `BioImage`.
- **Existing parity tests stay as they are.** `plugin/tests/test_czi_reader.py` covers the five bundled sample CZIs against real files; nothing in this plan changes it except where a task says so.

---

## File Structure

| File | Status | Responsibility |
|---|---|---|
| `core/lls_core/czi_reader.py` | modify | The fast path. Loses the runtime probe and a tautological check; gains INFO-level decline logging and a seam docstring. |
| `core/tests/conftest.py` | modify | Generated-CZI fixtures (`drift_czi`, `multi_scene_czi`), the `czi_stub_image` factory, and the `czi_read_calls` counting proxy. |
| `core/tests/test_czi_reader.py` | create | Geometry and behaviour that needs a purpose-built file, plus the CLI call-site tests. |
| `plugin/napari_lattice/reader.py` | modify | Delete dead `_has_real_channel_metadata`; document the `lattice_params_from_napari` coupling. |

`core/tests/test_czi_reader.py` is new rather than an extension of `plugin/tests/test_czi_reader.py` because the module under test lives in `core`, and the fixtures it needs are pytest fixtures in `core/tests/conftest.py`, which the plugin suite cannot import. The plugin file keeps what is genuinely plugin-specific: that `bioio_reader` never touches bioio's graph.

---

### Task 1: Drift fixture and the original-bug regression test

The reader was written to fix `ValueError: cannot reshape array of size 432345600 into shape (834,300,1734)` and there is no test for it. This task builds the smallest file that reproduces the geometry and pins the fix.

**Files:**
- Modify: `core/tests/conftest.py` (append at end of file)
- Create: `core/tests/test_czi_reader.py`

**Interfaces:**
- Consumes: `lls_core.czi_reader.czi_metadata(path, image) -> Optional[dict]` and `czi_dask_array(path, image, meta=None) -> Optional[dask.array.Array]`, both already implemented.
- Produces:
  - fixture `drift_czi -> (pathlib.Path, dict[tuple[int, int], np.ndarray], dict[int, int])` — `(path, planes, offsets)`, where `planes[(t, z)]` is the array as written and `offsets[t]` its x position.
  - fixture `czi_stub_image -> Callable[[path, int, int], object]` — `czi_stub_image(path, n_scenes=1, scene_index=0)` returns a stand-in for `BioImage` exposing `scenes`, `current_scene_index` and `reader.metadata`.

- [ ] **Step 1: Write the failing test**

Create `core/tests/test_czi_reader.py`:

```python
"""
Tests for `lls_core.czi_reader`, the lazy pylibCZIrw fast path shared by the napari
plugin and the CLI.

Parity against real acquisitions lives in `plugin/tests/test_czi_reader.py`, which
uses the five bundled sample CZIs. This file covers what needs a purpose-built file -
stage drift, multiple scenes - plus the properties the performance work depends on and
the CLI call sites.
"""
from __future__ import annotations

import logging

import numpy as np
import pytest
from bioio import BioImage

from lls_core.czi_reader import czi_dask_array, czi_metadata


def test_drift_czi_reads_the_canvas_not_the_subblock(drift_czi, czi_stub_image):
    """
    Regression for "cannot reshape array of size 432345600 into shape (834,300,1734)".

    When each timepoint records a different stage offset, the subblocks are narrower
    than the canvas they sit on. Reading the subblock width gives an array that cannot
    be reshaped into the canvas bioio reports - 20 vs 25 here, 1728 vs 1734 on the file
    that surfaced the bug.
    """
    path, planes, offsets = drift_czi
    stub = czi_stub_image(path)

    meta = czi_metadata(str(path), stub)
    assert meta is not None, "the fast path declined a plain single-scene CZI"
    assert meta["order"] == "TCZYX"
    assert meta["dtype"] == np.dtype("uint16")
    assert meta["shape"] == (3, 1, 4, 12, 25), "X must be the canvas, not the subblock"

    arr = czi_dask_array(str(path), stub, meta)
    assert arr is not None

    expected = np.zeros((3, 1, 4, 12, 25), dtype=np.uint16)
    for (t, z), plane in planes.items():
        expected[t, 0, z][:, offsets[t]:offsets[t] + 20] = plane
    assert np.array_equal(np.asarray(arr.compute()), expected), (
        "drift offsets were not composited onto the canvas"
    )


def test_drift_czi_matches_bioio(drift_czi, czi_stub_image):
    """
    bioio cannot enumerate scenes on a generated CZI, but `.dask_data` works, so pixel
    parity is still available for the single-scene case.
    """
    path, _planes, _offsets = drift_czi

    arr = czi_dask_array(str(path), czi_stub_image(path))
    assert arr is not None

    ref = BioImage(str(path)).dask_data
    assert arr.shape == ref.shape
    assert arr.dtype == ref.dtype
    assert np.array_equal(np.asarray(arr.compute()), np.asarray(ref.compute()))
```

- [ ] **Step 2: Run test to verify it fails**

Run:
```bash
C:/Users/rajasekhar.p/.conda/envs/napari_lattice/python.exe -m pytest core/tests/test_czi_reader.py -v
```
Expected: ERROR (2 errors) — `fixture 'drift_czi' not found`.

- [ ] **Step 3: Add the fixtures**

Append to `core/tests/conftest.py`:

```python
# --- generated CZI fixtures --------------------------------------------------
#
# pylibCZIrw's writer never emits the <Scenes> element that bioio's `scene_name()`
# requires, so `BioImage.scenes` raises UnsupportedMetadataError on every file it
# produces - with or without `write_metadata`, with or without an explicit `scene=`.
# `czi_metadata` only reads three attributes off the BioImage, so the tests stand a
# stub in for those. That substitutes metadata the writer cannot produce; the geometry
# under test still comes from pylibCZIrw.

class _CziStubImage:
    """Stand-in for a `BioImage` over a generated CZI."""

    def __init__(self, metadata, n_scenes: int, scene_index: int):
        self.scenes = tuple(f"Scene:{i}" for i in range(n_scenes))
        self.current_scene_index = scene_index
        self.reader = SimpleNamespace(metadata=metadata)


@pytest.fixture
def czi_stub_image():
    """Factory: `czi_stub_image(path, n_scenes=1, scene_index=0)`."""
    from xml.etree import ElementTree
    from pylibCZIrw import czi as pyczi

    def make(path, n_scenes: int = 1, scene_index: int = 0) -> _CziStubImage:
        with pyczi.open_czi(str(path)) as czi:
            metadata = ElementTree.fromstring(czi.raw_metadata)
        return _CziStubImage(metadata, n_scenes, scene_index)

    return make


@pytest.fixture(scope="session")
def drift_czi(tmp_path_factory):
    """
    A CZI whose subblocks are narrower than the canvas, because each timepoint records
    a different stage offset. This is the file shape that crashed the old reader:
    aicspylibczi reports the 20-wide subblock while pylibCZIrw and bioio report the
    25-wide canvas, and reshaping one into the other raises.

    Yields `(path, planes, offsets)`; `planes[(t, z)]` is the array as written.
    """
    from pylibCZIrw import czi as pyczi

    path = tmp_path_factory.mktemp("czi") / "drift.czi"
    rng = np.random.default_rng(0)
    offsets = {0: 5, 1: 0, 2: 2}   # x position per timepoint; canvas is 5 + 20 wide
    planes = {}
    with pyczi.create_czi(str(path)) as writer:
        for t in range(3):
            for z in range(4):
                plane = rng.integers(1, 500, size=(12, 20), dtype=np.uint16)
                planes[(t, z)] = plane
                writer.write(
                    plane, location=(offsets[t], 0), plane={"T": t, "Z": z, "C": 0}
                )
    return path, planes, offsets
```

Add `SimpleNamespace` to the imports at the top of `core/tests/conftest.py`:

```python
from types import SimpleNamespace
```

- [ ] **Step 4: Run test to verify it passes**

Run:
```bash
C:/Users/rajasekhar.p/.conda/envs/napari_lattice/python.exe -m pytest core/tests/test_czi_reader.py -v
```
Expected: PASS — 2 passed.

- [ ] **Step 5: Commit**

```bash
git add core/tests/conftest.py core/tests/test_czi_reader.py
git commit -m "test: pin the drift-offset CZI bug the fast path was written for"
```

---

### Task 2: Multi-scene support, and removing the runtime probe

`czi_dask_array` currently validates one plane against `image.dask_data` for multi-scene and exotic-dimension files, which forces bioio's whole per-plane graph on every open of exactly the files that most need the fast path. Replace that runtime trust with a fixture, and decline unknown dimensions outright rather than probing them.

**Files:**
- Modify: `core/tests/conftest.py` (append)
- Modify: `core/tests/test_czi_reader.py` (append)
- Modify: `core/lls_core/czi_reader.py:194-199` (extra-dims handling), `:201-210` (returned dict), `:213-276` (`czi_dask_array` body)

**Interfaces:**
- Consumes: `czi_stub_image` from Task 1.
- Produces:
  - fixture `multi_scene_czi -> (pathlib.Path, dict[tuple[int, int], np.ndarray])` — `(path, planes)`, where `planes[(scene, z)]` is the array as written.
  - `czi_metadata`'s returned dict **no longer has an `"extra_dims"` key**. Later tasks must not read it.

- [ ] **Step 1: Write the failing test**

Append to `core/tests/test_czi_reader.py`:

```python
@pytest.mark.parametrize("scene_index", [0, 1])
def test_multi_scene_czi_reads_the_selected_scene(
    multi_scene_czi, czi_stub_image, scene_index
):
    """
    Each scene must yield its own pixels from its own bounding rectangle.

    Before the runtime probe was removed this declined - the probe compared a plane
    against `image.dask_data`, so multi-scene files paid for bioio's full per-plane
    graph on every open, and declined outright when bioio could not supply one.
    """
    path, planes = multi_scene_czi
    stub = czi_stub_image(path, n_scenes=2, scene_index=scene_index)

    meta = czi_metadata(str(path), stub)
    assert meta is not None
    assert meta["shape"] == (1, 1, 3, 10, 14), "each scene has its own rectangle"
    assert meta["scene"] == scene_index

    arr = czi_dask_array(str(path), stub, meta)
    assert arr is not None, "multi-scene CZIs must take the fast path"

    expected = np.stack([planes[(scene_index, z)] for z in range(3)])[None, None]
    assert np.array_equal(np.asarray(arr.compute()), expected)
```

Append the fixture to `core/tests/conftest.py`:

```python
@pytest.fixture(scope="session")
def multi_scene_czi(tmp_path_factory):
    """
    Two scenes written at different x offsets, so reading the wrong scene's rectangle
    gives visibly wrong pixels rather than a shifted copy of the right ones.

    Yields `(path, planes)`; `planes[(scene, z)]` is the array as written.
    """
    from pylibCZIrw import czi as pyczi

    path = tmp_path_factory.mktemp("czi") / "multi_scene.czi"
    rng = np.random.default_rng(1)
    planes = {}
    with pyczi.create_czi(str(path)) as writer:
        for scene in range(2):
            for z in range(3):
                plane = rng.integers(1, 500, size=(10, 14), dtype=np.uint16)
                planes[(scene, z)] = plane
                writer.write(
                    plane,
                    location=(scene * 20, 0),
                    plane={"Z": z, "C": 0},
                    scene=scene,
                )
    return path, planes
```

- [ ] **Step 2: Run test to verify it fails**

Run:
```bash
C:/Users/rajasekhar.p/.conda/envs/napari_lattice/python.exe -m pytest core/tests/test_czi_reader.py -k multi_scene -v
```
Expected: FAIL — both parametrisations fail on `assert arr is not None`, because the probe reaches for `image.dask_data`, which the stub does not have.

- [ ] **Step 3: Move the extra-dimension decline into `czi_metadata`**

In `core/lls_core/czi_reader.py`, replace:

```python
        # CZI dimensions outside TCZYX. pylibCZIrw defaults them to 0, but their
        # presence forces the validation probe in czi_dask_array.
        extra_dims = sorted(d for d in bbox if d not in ("X", "Y", "Z", "T", "C", "M"))
    except Exception:
        logger.debug("CZI metadata unavailable for %s", path, exc_info=True)
        return None
```

with:

```python
    except Exception:
        logger.debug("CZI metadata unavailable for %s", path, exc_info=True)
        return None

    # CZI dimensions outside TCZYX (H, V, B, ...). We have never seen one, so we
    # decline rather than guess how it folds into the TCZYX bioio normalises to.
    extra_dims = sorted(d for d in bbox if d not in ("X", "Y", "Z", "T", "C", "M"))
    if extra_dims:
        return _decline(path, f"unsupported dimensions {extra_dims}")
```

Then drop the now-unused key from the returned dict — delete this line:

```python
        "extra_dims": extra_dims,
```

And update the `czi_metadata` docstring's last line from:

```python
    Returns ``None`` for non-CZIs, mosaics or anything unexpected.
```

to:

```python
    Returns ``None`` for non-CZIs, mosaics, files carrying dimensions outside TCZYX,
    or anything unexpected.
```

- [ ] **Step 4: Delete the probe and the tautological check**

In `czi_dask_array`, delete this line:

```python
    extra_dims = meta["extra_dims"]
```

and replace everything from the shape check to the end of the function:

```python
    if arr.shape != tuple(shape) or arr.dtype != dtype:
        return _decline(path, f"built {arr.shape}/{arr.dtype}, expected {tuple(shape)}/{dtype}")

    # Multi-scene and exotic-dimension files are not covered by the tests, so check one
    # mid-stack plane against bioio before trusting them (Z=0 is often near-black and
    # would false-pass a scene misalignment). This touches image.dask_data, so those
    # files do pay for bioio's graph: correctness over open time.
    if meta["n_scenes"] > 1 or extra_dims:
        try:
            probe = tuple(
                slice(None) if d in ("Y", "X") else (sizes[d] // 2 if d == "Z" else 0)
                for d in order
            )
            if not np.array_equal(
                np.asarray(arr[probe].compute()),
                np.asarray(image.dask_data[probe].compute()),
            ):
                return _decline(path, "probe plane differs from bioio")
        except Exception:
            logger.debug("could not validate CZI array for %s", path, exc_info=True)
            return None

    return arr
```

with:

```python
    return arr
```

The shape check was tautological: `da.from_array` takes its shape from `source.shape` and its dtype from `meta`, both of which are built from the same `meta` dict the check compares against. The probe is replaced by the `multi_scene_czi` and `drift_czi` fixtures.

Update the `czi_dask_array` docstring's last line from:

```python
    Returns ``None`` for non-CZIs, mosaics, or any read error.
```

to:

```python
    Returns ``None`` whenever ``czi_metadata`` declines, or on any construction error.
```

- [ ] **Step 5: Run tests to verify they pass**

Run:
```bash
C:/Users/rajasekhar.p/.conda/envs/napari_lattice/python.exe -m pytest core/tests/test_czi_reader.py plugin/tests/test_czi_reader.py -v
```
Expected: PASS — 4 core (2 from Task 1, 2 parametrisations here) and 13 plugin.

- [ ] **Step 6: Commit**

```bash
git add core/lls_core/czi_reader.py core/tests/conftest.py core/tests/test_czi_reader.py
git commit -m "feat: give multi-scene CZIs the fast path, replacing the runtime probe"
```

---

### Task 3: Pin read count and the shared thread pool

`_Z_CHUNK = 32` means a Z chunk spans 32 planes. That is only safe because dask pushes single-plane indexing down into `from_array`. If that ever stops, the chunk silently becomes a 32× read amplifier and every other test still passes. The module-level executor is likewise invisible until it leaks eight threads per open.

**Files:**
- Modify: `core/tests/conftest.py` (append)
- Modify: `core/tests/test_czi_reader.py` (append)

**Interfaces:**
- Consumes: `drift_czi`, `multi_scene_czi`, `czi_stub_image` from Tasks 1–2; `lls_core.czi_reader._pool() -> ThreadPoolExecutor`.
- Produces: fixture `czi_read_calls -> list` — every plane dict passed to `pylibCZIrw`'s `read()` for the duration of the test, in call order.

Chunk geometry (Z chunk is `min(_Z_CHUNK, sizeZ)`, Y and X single blocks) is already pinned by `plugin/tests/test_czi_reader.py::test_czi_fast_dask_data_matches_bioio_and_is_plane_chunked`; no new test for it.

- [ ] **Step 1: Write the failing test**

Append to `core/tests/test_czi_reader.py`:

```python
def test_two_dimensional_slice_reads_exactly_one_plane(
    drift_czi, czi_stub_image, czi_read_calls
):
    """
    `_Z_CHUNK` is 32, so a Z chunk spans up to 32 planes. That is a dask-graph-size
    knob, not a read-volume one, only because dask pushes single-plane indexing down
    into `from_array`. If a dask change stops doing that, the chunk becomes a 32x read
    amplifier and nothing else in the suite would notice.
    """
    path, _planes, _offsets = drift_czi
    arr = czi_dask_array(str(path), czi_stub_image(path))
    assert arr is not None

    czi_read_calls.clear()          # discard anything the metadata pass did
    plane = np.asarray(arr[1, 0, 2].compute())

    assert plane.shape == (12, 25)
    assert len(czi_read_calls) == 1, czi_read_calls
    assert czi_read_calls[0] == {"T": 1, "C": 0, "Z": 2}


def test_whole_block_reads_one_plane_per_index(drift_czi, czi_stub_image, czi_read_calls):
    """A genuine whole-array request reads every plane exactly once - no more."""
    path, _planes, _offsets = drift_czi
    arr = czi_dask_array(str(path), czi_stub_image(path))
    assert arr is not None

    czi_read_calls.clear()
    arr.compute()

    assert len(czi_read_calls) == 3 * 4          # T x Z, one read each
    assert len({tuple(sorted(c.items())) for c in czi_read_calls}) == 3 * 4


def test_arrays_share_one_thread_pool(drift_czi, multi_scene_czi, czi_stub_image):
    """
    The executor is module-level, so opening many files does not accumulate a pool
    each. A per-array pool would leak up to eight threads per file opened.
    """
    import threading

    from lls_core import czi_reader

    drift_path, _planes, _offsets = drift_czi
    scene_path, _scene_planes = multi_scene_czi

    pool = czi_reader._pool()
    for path, stub in (
        (drift_path, czi_stub_image(drift_path)),
        (scene_path, czi_stub_image(scene_path, n_scenes=2)),
    ):
        arr = czi_dask_array(str(path), stub)
        assert arr is not None
        arr.compute()          # a whole-block read, which is what uses the pool

    assert czi_reader._pool() is pool
    live = [t for t in threading.enumerate() if t.name.startswith("lls-czi")]
    assert len(live) <= 8, [t.name for t in live]
```

- [ ] **Step 2: Run test to verify it fails**

Run:
```bash
C:/Users/rajasekhar.p/.conda/envs/napari_lattice/python.exe -m pytest core/tests/test_czi_reader.py -k "reads_exactly_one_plane or one_plane_per_index or share_one_thread_pool" -v
```
Expected: FAIL — the first two error with `fixture 'czi_read_calls' not found`; `test_arrays_share_one_thread_pool` passes already (it needs no new fixture).

- [ ] **Step 3: Add the counting fixture**

Append to `core/tests/conftest.py`:

```python
@pytest.fixture
def czi_read_calls(monkeypatch):
    """
    Records the plane dict of every `pylibCZIrw` read for the duration of the test.

    Wraps the library's reader rather than patching `lls_core.czi_reader`, so the test
    pins the observable property - one read per plane - and not how the module happens
    to be written.
    """
    from contextlib import contextmanager
    from pylibCZIrw import czi as pyczi

    calls: list = []
    real_open_czi = pyczi.open_czi

    class _CountingReader:
        def __init__(self, inner):
            self._inner = inner

        def read(self, *args, **kwargs):
            calls.append(kwargs.get("plane"))
            return self._inner.read(*args, **kwargs)

        def __getattr__(self, name):
            return getattr(self._inner, name)

    @contextmanager
    def counting_open_czi(*args, **kwargs):
        with real_open_czi(*args, **kwargs) as reader:
            yield _CountingReader(reader)

    monkeypatch.setattr(pyczi, "open_czi", counting_open_czi)
    return calls
```

- [ ] **Step 4: Run tests to verify they pass**

Run:
```bash
C:/Users/rajasekhar.p/.conda/envs/napari_lattice/python.exe -m pytest core/tests/test_czi_reader.py -v
```
Expected: PASS — 7 passed.

- [ ] **Step 5: Commit**

```bash
git add core/tests/conftest.py core/tests/test_czi_reader.py
git commit -m "test: pin one-read-per-plane and the shared CZI thread pool"
```

---

### Task 4: Make declines visible

Every bailout currently logs at `DEBUG`, which is off by default. A silent fallback turns a 1.2 s open back into a 195 s one with no explanation — the single failure mode this module most needs to make visible. Route every `.czi` bailout through `_decline` at `INFO`, and keep non-CZIs silent so opening a TIFF does not log.

**Files:**
- Modify: `core/lls_core/czi_reader.py:59-62` (`_decline`), and the five bailout sites in `czi_metadata`, `czi_dask_array` and `czi_xarray`
- Modify: `core/tests/test_czi_reader.py` (append)

**Interfaces:**
- Consumes: nothing new.
- Produces: `_decline(path, reason, exc_info=False) -> None` — signature gains a third parameter. All logging is on logger `lls_core.czi_reader` at `INFO`, message `"CZI fast path declined for %s: %s"`.

- [ ] **Step 1: Write the failing test**

Append to `core/tests/test_czi_reader.py`:

```python
def test_declining_a_czi_says_why(tmp_path, caplog):
    """
    A silent fallback turns a 1.2 s open back into a 195 s one. Turning on ordinary
    logging has to be enough to explain why a file went slow again.
    """
    broken = tmp_path / "broken.czi"
    broken.write_bytes(b"this is not a CZI")

    with caplog.at_level(logging.INFO, logger="lls_core.czi_reader"):
        assert czi_dask_array(str(broken), None) is None

    messages = [r.getMessage() for r in caplog.records]
    assert any("declined" in m for m in messages), messages


def test_declining_a_non_czi_is_silent(caplog):
    """Every TIFF and HDF5 open would otherwise log; only .czi bailouts are news."""
    with caplog.at_level(logging.DEBUG, logger="lls_core.czi_reader"):
        assert czi_dask_array("not_a_czi.tif", None) is None

    assert caplog.records == [], [r.getMessage() for r in caplog.records]
```

- [ ] **Step 2: Run test to verify it fails**

Run:
```bash
C:/Users/rajasekhar.p/.conda/envs/napari_lattice/python.exe -m pytest core/tests/test_czi_reader.py -k declining -v
```
Expected: `test_declining_a_czi_says_why` FAILS (the bailout logs at DEBUG, so nothing is captured at INFO); `test_declining_a_non_czi_is_silent` passes.

- [ ] **Step 3: Raise `_decline` to INFO and route every bailout through it**

In `core/lls_core/czi_reader.py`, replace:

```python
def _decline(path: Any, reason: str) -> None:
    """Log why the fast path bowed out. Callers return this (None) to fall back."""
    logger.debug("CZI fast path declined for %s: %s", path, reason)
    return None
```

with:

```python
def _decline(path: Any, reason: str, exc_info: bool = False) -> None:
    """
    Log why the fast path bowed out and return None, so the caller falls back to bioio.

    INFO rather than DEBUG: falling back silently turns a 1.2 s open back into a 195 s
    one, so enabling ordinary logging has to be enough to say why. Callers must return
    before reaching this for a non-CZI, or every TIFF opened would log too.
    """
    logger.info("CZI fast path declined for %s: %s", path, reason, exc_info=exc_info)
    return None
```

Then replace each of the four remaining `logger.debug` bailouts.

In `czi_metadata`:

```python
    except Exception:
        logger.debug("bioio-czi internals unavailable; using bioio", exc_info=True)
        return None
```
becomes
```python
    except Exception:
        return _decline(path, "bioio-czi internals unavailable", exc_info=True)
```

```python
    except Exception:
        logger.debug("CZI metadata unavailable for %s", path, exc_info=True)
        return None
```
becomes
```python
    except Exception:
        return _decline(path, "metadata unreadable", exc_info=True)
```

In `czi_dask_array`:

```python
    except Exception:
        logger.debug("dask.base.tokenize unavailable", exc_info=True)
        return None
```
becomes
```python
    except Exception:
        return _decline(path, "dask.base.tokenize unavailable", exc_info=True)
```

```python
    except Exception:
        logger.debug("could not build CZI array for %s", path, exc_info=True)
        return None
```
becomes
```python
    except Exception:
        return _decline(path, "could not build the array", exc_info=True)
```

In `czi_xarray`:

```python
    except Exception:
        logger.debug("could not wrap CZI array as DataArray for %s", path, exc_info=True)
        return None
```
becomes
```python
    except Exception:
        return _decline(path, "could not wrap the array as a DataArray", exc_info=True)
```

- [ ] **Step 4: Run tests to verify they pass**

Run:
```bash
C:/Users/rajasekhar.p/.conda/envs/napari_lattice/python.exe -m pytest core/tests/test_czi_reader.py -v
```
Expected: PASS — 9 passed.

- [ ] **Step 5: Commit**

```bash
git add core/lls_core/czi_reader.py core/tests/test_czi_reader.py
git commit -m "feat: log CZI fast-path declines at INFO so silent fallbacks are visible"
```

---

### Task 5: Remove dead code and document the remaining graph-building call site

Two findings from the audit for expensive `BioImage` accesses. `_has_real_channel_metadata` has no callers after the refactor to `_has_real_channel_names`, and it calls `image.channel_names` — a loaded gun for whoever picks it up next. `lattice_params_from_napari` has a live branch that builds bioio's graph; behaviour is correct and stays, but the coupling that keeps it unreachable for our own layers is undocumented.

**Files:**
- Modify: `plugin/napari_lattice/reader.py:57-59` (delete), `:119-123` (comment)

**Interfaces:**
- Consumes: nothing.
- Produces: nothing. `_has_real_channel_names(names)` — the surviving helper — is unchanged.

- [ ] **Step 1: Confirm the function is genuinely dead**

Run:
```bash
git grep -n "_has_real_channel_metadata"
```
Expected: exactly one hit, the definition at `plugin/napari_lattice/reader.py:57`. If anything else appears, stop — it is not dead, and this task does not apply.

- [ ] **Step 2: Delete it**

In `plugin/napari_lattice/reader.py`, delete:

```python
def _has_real_channel_metadata(image: BioImage) -> bool:
    """`_has_real_channel_names` for a BioImage whose graph is already built."""
    return _has_real_channel_names(image.channel_names)
```

- [ ] **Step 3: Document the `lattice_params_from_napari` coupling**

In `lattice_params_from_napari`, replace:

```python
            if "dimensions" in img.metadata:
                calculated_order = img.metadata["dimensions"]
            else:
                metadata_order = list(img_data_bioio.dims.order)
```

with:

```python
            if "dimensions" in img.metadata:
                calculated_order = img.metadata["dimensions"]
            else:
                # `bioio_reader` always sets "dimensions", so our own layers never
                # reach here. A layer carrying "bioio_image" without it - from another
                # plugin, a hand-built layer, or an older version of this reader - pays
                # for bioio's full per-plane dask graph on the next two lines (~188 s
                # on a 300k-plane CZI). Keep the two metadata keys set together.
                metadata_order = list(img_data_bioio.dims.order)
```

- [ ] **Step 4: Run the plugin suite to verify nothing regressed**

Run:
```bash
C:/Users/rajasekhar.p/.conda/envs/napari_lattice/python.exe -m pytest plugin/tests -v
```
Expected: PASS — 37 passed (the same count as before this plan).

- [ ] **Step 5: Commit**

```bash
git add plugin/napari_lattice/reader.py
git commit -m "refactor: drop dead _has_real_channel_metadata, document graph-building branch"
```

---

### Task 6: Pin the CLI call sites

`lls-pipeline` reaches the fast path through three functions. Existing core tests exercise all three incidentally via the bundled CZI fixtures, so they would catch a *wrong* array — but nothing fails if a call site quietly stops taking the fast path and falls back to bioio's, which is a 190 s regression that looks like success.

**Files:**
- Modify: `core/tests/test_czi_reader.py` (append)

**Interfaces:**
- Consumes: `rbc_tiny` (existing fixture in `core/tests/conftest.py`, a real bundled CZI); `lls_core.types.image_like_to_image`, `lls_core.models.deskew.load_image_lazy`, `lls_core.models.deskew.DeskewParams`.
- Produces: nothing.

Parallel ROI needs no test here: `core/tests/test_parallel_processing.py::test_parallel_save_matches_serial[file_path_lazy_reload]` already runs `input_image=str(rbc_tiny)` with `process_parallel=2`, so `_dispatch_payload` strips the image to `None` and each worker calls `load_image_lazy` — which now goes through `czi_xarray`. It compares parallel output against serial, so a reader that failed to reconstruct in a spawned subprocess would fail it. `test_load_image_lazy_takes_the_fast_path` below adds the one thing it does not assert.

- [ ] **Step 1: Write the failing test**

Append to `core/tests/test_czi_reader.py`:

```python
def _array_name(image) -> str:
    """Our arrays are named "lls-czi-<token>"; bioio's are "reshape-..." etc."""
    data = getattr(image, "data", image)
    return str(getattr(data, "name", ""))


def test_image_like_to_image_takes_the_fast_path(rbc_tiny):
    from lls_core.types import image_like_to_image

    got = image_like_to_image(str(rbc_tiny))
    assert _array_name(got).startswith("lls-czi-"), _array_name(got)

    ref = BioImage(str(rbc_tiny)).xarray_dask_data
    assert got.dims == ref.dims
    assert np.array_equal(np.asarray(got), np.asarray(ref))


def test_load_image_lazy_takes_the_fast_path(rbc_tiny):
    """
    This is the call a parallel ROI worker makes after `_dispatch_payload` strips the
    unpicklable image and sends only the path.
    """
    from pathlib import Path

    from lls_core.models.deskew import load_image_lazy

    got = load_image_lazy(Path(str(rbc_tiny)))
    assert _array_name(got).startswith("lls-czi-"), _array_name(got)

    ref = BioImage(str(rbc_tiny)).xarray_dask_data
    assert got.dims == ref.dims
    assert np.array_equal(np.asarray(got), np.asarray(ref))


def test_deskew_params_read_image_takes_the_fast_path(rbc_tiny):
    """The path every `lls-pipeline` run with a CZI input takes."""
    from lls_core.models.deskew import DeskewParams

    params = DeskewParams(input_image=str(rbc_tiny))
    assert _array_name(params.input_image).startswith("lls-czi-"), _array_name(
        params.input_image
    )
```

- [ ] **Step 2: Run the mutation check first, so the test is proven to bite**

These pin behaviour that already works, so they would pass on the first run whether or not the assertion means anything. Break the fast path first, then confirm all three fail.

Temporarily insert `return None` as the first statement of `czi_metadata` in `core/lls_core/czi_reader.py`:

```python
def czi_metadata(path: str, image: BioImage) -> Optional[dict]:
    return None   # TEMPORARY - mutation check, revert before continuing
```

Run:
```bash
C:/Users/rajasekhar.p/.conda/envs/napari_lattice/python.exe -m pytest core/tests/test_czi_reader.py -k takes_the_fast_path -v
```
Expected: FAIL — all 3, each on `assert _array_name(...).startswith("lls-czi-")`. If any passes, the assertion is not testing what it claims; stop and fix it.

- [ ] **Step 3: Revert the mutation and confirm the tests pass**

Delete the temporary `return None` line, then run:
```bash
C:/Users/rajasekhar.p/.conda/envs/napari_lattice/python.exe -m pytest core/tests/test_czi_reader.py -k takes_the_fast_path -v
```
Expected: PASS — 3 passed.

Confirm nothing of the mutation survived:
```bash
git diff core/lls_core/czi_reader.py
```
Expected: no output (this task changes no source file).

- [ ] **Step 4: Run the full core suite**

Run:
```bash
C:/Users/rajasekhar.p/.conda/envs/napari_lattice/python.exe -m pytest core/tests -q
```
Expected: 272 passed, 2 failed, 2 skipped — the baseline 260 plus the 12 added by Tasks 1–6. The 2 failures are pre-existing and unrelated: `test_workflows.py::test_workflow_cli[dump---json-config]` and `[safe_dump---yaml-config]`, a Windows `NamedTemporaryFile` reopen-by-name limitation, on a TIFF input that never reaches the CZI path. If any *other* test fails, stop and investigate.

- [ ] **Step 5: Commit**

```bash
git add core/tests/test_czi_reader.py
git commit -m "test: pin that the CLI call sites take the CZI fast path"
```

---

### Task 7: Mark the upstream seam

The module docstring says the whole thing is temporary. Half of it is: bioio-czi#104 covers fast metadata access (`dims`/`shape`/`dtype`) and will replace `czi_metadata`. Nothing upstream plans lazy array construction, per-plane reads or chunking, so `CziPlanes` and `czi_dask_array` are maintained code. Say which is which, so the next person deleting this module deletes the right half.

**Files:**
- Modify: `core/lls_core/czi_reader.py:1-20` (module docstring)

**Interfaces:**
- Consumes: nothing.
- Produces: nothing. Documentation only.

- [ ] **Step 1: Replace the closing paragraph of the module docstring**

In `core/lls_core/czi_reader.py`, replace:

```python
This is a temporary workaround for an upstream bioio-czi issue. Once bioio-czi builds
its array lazily, delete this module and revert its call sites in ``lls_core.types``,
``lls_core.models.deskew`` and ``napari_lattice.reader``.
"""
```

with:

```python
The two halves have different lifetimes, so delete them separately:

* ``czi_metadata`` is temporary. bioio-czi#104 (via bioio#197) tracks fast metadata
  access - dims, shape, dtype - upstream. When that ships, measure it, then delete this
  half along with its three ``bioio_czi`` internal imports.
* ``CziPlanes`` and ``czi_dask_array`` are maintained. Nothing upstream plans lazy
  array construction, per-plane reads or chunking, which is where the per-slice and
  per-volume speedups come from. Revisit only if bioio-czi ships its own lazy reader
  and measures faster.

Note what is *not* maintained here: no CZI is parsed. Reads go through pylibCZIrw, so
what this module owns is the dask graph over one, not the reader itself.

Call sites to revert if the whole module ever goes: ``lls_core.types``,
``lls_core.models.deskew`` and ``napari_lattice.reader``.
"""
```

- [ ] **Step 2: Verify the module still imports and the suite is green**

Run:
```bash
C:/Users/rajasekhar.p/.conda/envs/napari_lattice/python.exe -m pytest core/tests/test_czi_reader.py plugin/tests/test_czi_reader.py -q
```
Expected: PASS — 12 core, 13 plugin.

- [ ] **Step 3: Commit**

```bash
git add core/lls_core/czi_reader.py
git commit -m "docs: mark which half of czi_reader upstream will replace"
```

---

## Final verification

- [ ] **Run both suites end to end**

```bash
C:/Users/rajasekhar.p/.conda/envs/napari_lattice/python.exe -m pytest core/tests plugin/tests -q
```
Expected: 309 passed, 2 failed, 2 skipped — core 272 (baseline 260 + the 12 added here) plus plugin 37 (unchanged; Task 5 touches source, not tests). The 2 failures are the pre-existing `test_workflow_cli` pair.

- [ ] **Confirm the diff is the size the spec predicted**

```bash
git diff --stat master
```
Expected: roughly +65 net across `core/lls_core/czi_reader.py`, `core/tests/conftest.py`, `core/tests/test_czi_reader.py` and `plugin/napari_lattice/reader.py`, on top of the reader work already on the branch. A much larger diff means scope crept.

- [ ] **Do not push.** Report the branch state and wait.

## Side actions (not code, do after the plan lands)

- Comment on bioio-czi#104 asking whether `channel_names` is in scope. Its stated scope is `dims`/`shape`/`dtype`, but `channel_names` also routes through `xarray_dask_data`. If it is out of scope, `czi_metadata` survives #104 and cannot be deleted.
- File the read-side issue upstream. Draft exists: eager `from_delayed` per plane, and chunk count driving per-slice cost. Confirmed a genuine gap in bioio#197's tracker, not a duplicate.

## Known gaps this plan does not close

- **The mosaic guard stays unvalidated.** `size(bbox, "M") > 1` likely never fires: pylibCZIrw auto-stitches tiles, which is bioio's documented behaviour on the same path, so reading a mosaic as a stitched image is the correct result. pylibCZIrw's writer cannot produce a file with `M` in `total_bounding_box` — two 12-wide tiles round-trip as one 24-wide image. Settling it needs a real mosaic CZI; none is available. The guard costs nothing and stays.
- **Dimensions outside TCZYX are declined untested.** We have never seen an `H`/`V`/`B` CZI and cannot write a fixture for one, so the decline is deliberate conservatism rather than covered behaviour.
