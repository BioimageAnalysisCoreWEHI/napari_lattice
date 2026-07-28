# CZI reader: scoped permanence

**Date:** 2026-07-28
**Status:** approved, not yet implemented
**Branch:** `fix_reader`

## Context

`lls_core/czi_reader.py` was written to fix a crash — `cannot reshape array of size
432345600 into shape (834,300,1734)` — when opening a CZI whose subblocks are narrower
than `SizeX`. It grew to also make such files usable: on a 70 GB, 301,908-plane
timelapse, file open went 195 s → 1.2 s, a 2D slice 122 s → 24 ms, and a `(T,C)` volume
60 s → 3.1 s. Output is byte-identical to bioio.

It was written as a disposable bridge, on the assumption that an upstream bioio-czi fix
would obsolete it entirely. That assumption is now only half true.

## The upstream seam

bioio#197 → bioio-czi#104 (opened 2026-06-24, open, assigned) covers *"fast-path access
for metadata properties (dims, shape, dtype) without building full dask task graph"*.
The tracking issue is explicitly metadata-only: it does not cover lazy array
construction, per-plane `from_delayed`, chunking, or slicing performance.

Our module divides along exactly that line:

| component | buys | upstream |
|---|---|---|
| `czi_metadata` (~53 lines, 3 bioio-czi internal imports) | open 195 s → 1.2 s | being fixed (#104) |
| `CziPlanes` + `czi_dask_array` (~90 lines) | slice 122 s → 24 ms, volume 60 s → 3.1 s | **no plan anywhere** |

So the read path is permanent and the metadata path is genuinely temporary.

Note what "permanent" means here: all pixel reads go through **pylibCZIrw**, which is
also what bioio-czi uses by default (`use_aicspylibczi=False`). We do not maintain a CZI
reader. We maintain the dask graph construction over one — which is why output is
byte-identical rather than merely equivalent.

## Decisions

1. **Keep the module.** Driver is that the upstream timeline is unreliable and, for the
   read path, non-existent.
2. **If upstream ships a lazy reader, re-evaluate manually** — compare their numbers
   against ours. No version gate, no automatic retirement.
3. **Keep the bioio-czi internal imports** (`bounding_box.size`, `channels.get_channel_names`,
   `pylibczirw_reader.reader.PIXEL_DICT`). They live in the half #104 replaces;
   inlining them is work on code with a defined end date. The existing parity test
   catches a move.
4. **Mark the seam with a comment, do not split files.** Same clean-deletion property,
   no churn.

## Scope

| change | ~lines | rationale |
|---|---:|---|
| `drift` fixture + parity test | +40 | The original bug has no regression test |
| `multi_scene` fixture | +20 | Prerequisite for removing the probe |
| Remove runtime probe and tautological shape check | −25 | Multi-scene stops paying bioio's graph; removes last `image.dask_data` reach |
| Read-count test | +15 | Pins the property `_Z_CHUNK` depends on |
| CLI call-site tests | +10 | Pins that `image_like_to_image` / `load_image_lazy` actually take the fast path |
| Raise decline logging from `debug` to `info` | +5 | As implemented it is `debug`, invisible by default — see Testing |
| Delete dead `_has_real_channel_metadata` | −5 | Unused, and calls the expensive `channel_names` — see Audit |
| Comment the `lattice_params_from_napari` coupling | +3 | Its `else` branch triggers the graph build — see Audit |
| Docstring: mark the seam | +6 | Read path maintained; metadata path pending #104 |

Net ≈ +65 lines, almost entirely tests, plus two deletions.

### Behaviour changes

- **Multi-scene CZIs get the fast path.** Today they build the array then validate one
  plane against `image.dask_data`, forcing bioio's full graph on every open. That trust
  moves from a runtime check to the fixtures.
- **Unknown extra dimensions (`H`, `V`, `B`, …) now decline outright** instead of being
  probed. More conservative: we cannot write a fixture for a case we have never seen, so
  we should not claim to handle it.
- **Mosaics are supported, and the guard is a safety net rather than a limitation.**
  bioio-czi's `_read_delayed` documents that on the pylibCZIrw path "any scenes with
  multiple tiles will be automatically stitched (where tiles overlap, the highest
  M-index wins)". We call the same `pylibCZIrw.read()`, so we inherit that stitching
  exactly as we inherit drift-offset compositing. The `M` check is retained because it
  costs nothing and would decline safely if `M` ever did surface, but it is not a
  statement that mosaics are unhandled. Its ancestor guard was necessary: aicspylibczi
  exposes `is_mosaic()`/`read_mosaic()` separately and does *not* auto-stitch, so
  reading a mosaic through `read_image()` there would have been wrong.

### Out of scope

`CziImage` dataclass refactor; inlining `size`/`PIXEL_DICT`; a degenerate-axes fixture;
splitting the module into two files; version gates; any attempt at mosaics. Each is
either aesthetics or work on the half that is going away.

## Testing

Fixtures are generated into `tmp_path` by pylibCZIrw's writer — no binaries committed.

The plain single-scene case needs no generated fixture — the five bundled sample CZIs
already cover it against real files, and those tests stay as they are.

- **`drift`** — per-timepoint `location` offset, canvas wider than the subblocks.
  Verified to reproduce the original bug in a 14 KB file: pylibCZIrw/bioio report X=25
  while aicspylibczi reports X=20, the same discrepancy as 1734 vs 1728. Doubles as a
  standing guard against anyone reintroducing an aicspylibczi read.
- **`multi_scene`** — the case whose runtime probe is being removed. Two scenes at
  different `location` offsets; verified that our reader returns each scene's own
  pixels once the probe no longer blocks it.

**Generated fixtures need a stub in place of `BioImage`.** pylibCZIrw's writer never
emits the `<Scenes>` element bioio's `scene_name()` requires, so `BioImage.scenes`
raises `UnsupportedMetadataError: Expected 1 scene for index '0' but found 0` on *every*
writer-produced file — with or without `write_metadata`, with or without an explicit
`scene=` on the write. `czi_metadata` reads `scenes`, `current_scene_index` and
`reader.metadata` off the `BioImage`, so the tests supply a three-attribute stub for
those. This is standing in for metadata the writer cannot produce, not faking the thing
under test: the geometry being verified comes from pylibCZIrw, not from the stub.

Ground truth is therefore the planes the fixture wrote, which is independent of both
libraries and pins drift compositing directly — a stronger check than parity. bioio
parity is kept where it is still available: `BioImage.dask_data` works fine on these
files (only `.scenes` raises), so the single-scene `drift` fixture is compared against
it as well. Multi-scene parity is not available, because selecting a scene requires the
`.scenes` that raises.

Three property tests beyond parity:

1. **Read-count** — a 2D slice must issue exactly one `read()`. If a dask change stops
   pushing indexing down into `from_array`, a 32-deep chunk silently becomes a 32×
   read amplifier and nothing else would catch it.
2. **Chunking** — Z chunk equals `min(_Z_CHUNK, sizeZ)`; Y and X are single blocks.
3. **Shared pool** — two arrays share one executor, pinning the thread-leak fix.

Decline paths: a non-CZI returns `None` silently (expected constantly, would be noise);
a `.czi` we cannot handle logs a reason at `info`, raised from the `debug` currently
implemented so that enabling logging is enough to explain "why is this file slow again".
The log is asserted, because a silent fallback is the failure mode this module most
needs to make visible.

Read-count is measured by wrapping the pylibCZIrw reader with a counting proxy in the
test, not by patching module internals, so the test does not encode implementation
detail beyond "one plane read per 2D slice".

### The CLI path

The reader is shared, so `lls-pipeline` uses it via `image_like_to_image`,
`DeskewParams.read_image` and `load_image_lazy`. Existing core tests cover all three
incidentally through the bundled CZI fixtures. One addition:

**Call-site parity** — assert `image_like_to_image`, `DeskewParams.read_image` and
`load_image_lazy` return the fast path (not bioio's array) for a CZI, and that pixels
match bioio. Today it is only inferred that the CLI benefits; nothing fails if a
call site silently stops taking the fast path.

**Parallel ROI needs no new test.** The worker re-open route is already exercised
end-to-end on a CZI: `test_parallel_save_matches_serial[file_path_lazy_reload]` builds
its lattice with `input_image=str(rbc_tiny)`, so `input_image_path` is set,
`_dispatch_payload` strips `input_image` to `None`, and each worker calls
`load_image_lazy` — which now goes through `czi_xarray`. The test compares parallel
output against serial, so a reader that failed to reconstruct in a `spawn`-ed
subprocess would fail it. The call-site test above adds the one thing it does not
assert: that the array the worker gets is ours rather than bioio's fallback.

The residual risk here is a pre-existing branch, not a reader one: if
`input_image_path` were ever `None` for a large CZI, `_materialized_image` calls
`.compute()` on the whole array — an OOM on a 70 GB timelapse, not a slowdown. That is
reader-independent and out of scope, but worth recording, because the fast path makes
file-backed CZIs materially more attractive to run at scale.

## Why bioio metadata at all

The split is not arbitrary: **bypass bioio only where it is expensive.**

| value | source | cost via bioio |
|---|---|---|
| shape / dims / dtype | pylibCZIrw | **188 s** — builds the graph |
| channel names | `bioio_czi.channels.get_channel_names` | free, given the XML |
| `scenes`, `current_scene_index` | bioio | ~0 s — and it is iteration *state*, not file data |
| `physical_pixel_sizes` | bioio | 0.000 s |
| metadata XML | `image.reader.metadata` | 0.131 s |

Everything except shape is taken from bioio because it is already computed, costs
nothing, and is guaranteed to match what bioio would produce. That guarantee is what
makes this a drop-in substitute rather than a second opinion. pylibCZIrw *could* supply
scenes and pixel sizes too, but each value we derive is a semantic we then own — the
channel-name analysis found four edge-case branches, one of which would produce a wrong
*shape*. Multiply that across scenes, units and dtype mapping and it becomes a reader.

`BioImage` cannot be dropped regardless: `bioio_reader` stashes it in
`add_kwargs["metadata"]["bioio_image"]` and `lattice_params_from_napari` reads it, and
it is needed for the fallback and every non-CZI format.

**Contingency:** if bioio-czi moves `get_channel_names`, pylibCZIrw's `metadata` exposes
the same data at `ImageDocument/Metadata/Information/Image/Dimensions/Channels/Channel/@Name`
(verified matching for all five bundled CZIs). Reimplementing means covering its four
branches — per-scene selection, truncation to the data's channel count, `@Id` fallback,
generated-id fallback. A fallback plan, not a plan.

## Audit: other expensive bioio calls

Prompted by finding one, the codebase was swept for accesses that build bioio's graph.

1. **`_has_real_channel_metadata` is dead code** (`plugin/napari_lattice/reader.py`).
   No live callers after the refactor to `_has_real_channel_names`. It calls
   `image.channel_names`, so it is a loaded gun for the next caller. **Delete it.**

2. **`lattice_params_from_napari` reads `img_data_bioio.dims.order/.shape`** when
   `"dimensions"` is absent from the layer's metadata. `bioio_reader` always sets it, so
   our own layers are safe; a layer carrying `bioio_image` without `dimensions` — from
   another plugin, a hand-built layer, or an older version of this reader — pays the
   full graph build. **Add a comment documenting the coupling**; behaviour unchanged.

3. **PSF `BioImage`s** (`core/lls_core/deconvolution.py`) touch `.data` and `.dims.C`.
   PSFs are a few planes, so this is cheap. Noted, not actioned.

Everything else that matched (`img.dims` in `deconvolution.py`/`deskew.py`, `.dtype` in
`estimate.py`/`lattice_data.py`) operates on `DataArray`s, not `BioImage`s.

## Known gaps

- **The mosaic guard is unvalidated, and may be dead code.** A fixture is not possible:
  pylibCZIrw's writer accepts `M` in the plane dict, but the file round-trips as a single
  *stitched* canvas with no `M` in `total_bounding_box` (two 12-wide tiles came back as
  one 24-wide image). The `UnsupportedMetadataError` that file raised is the general
  writer/`<Scenes>` limitation described under Testing, not specific to mosaics, so the
  stub would get past it — but there is still no `M` to guard against.

  Our check is `size(bbox, "M") > 1`, and it likely never fires: pylibCZIrw auto-stitches
  tiles, which is bioio's documented behaviour on the same path. Reading a mosaic as a
  stitched image is therefore the *correct* result, not a missed guard — we get it from
  the same `read()` call bioio does. What remains unverified is only whether `M` can
  surface in `total_bounding_box` for some real mosaic, in which case we would decline
  rather than stitch. **Settling that needs a real mosaic CZI**; none is available here.
  The guard stays because it is free, and this is recorded as an unvalidated branch
  rather than as coverage.

  One incidental positive: on that malformed file the reader caught bioio's exception,
  declined, and logged the reason — the fallback contract working as intended.
- **`czi_metadata` survives #104 — settled, not open.** Measured against the installed
  bioio 3.4.0 / bioio-czi 2.8.0 by making `xarray_dask_data` raise and reading each
  property:

  | | `dims` | `shape` | `dtype` | `channel_names` |
  |---|---|---|---|---|
  | `reader.*` | cheap | cheap | expensive | expensive |
  | `BioImage.*` | expensive | expensive | expensive | expensive |

  Three findings. `channel_names` is not covered at either level, so the metadata half
  cannot be deleted when #104 closes. `dtype` is not covered either, despite being in
  the issue's stated scope. And the part #104 *did* fix does not reach us: bioio-czi
  already overrides `dims` and `shape` on the Reader and they are genuinely cheap there,
  but `BioImage` does not delegate to the reader — it builds `xarray_dask_data` itself.
  That is bioio#197's half and it has not landed.

- **Possible follow-up, not scheduled.** Because `reader.dims`/`reader.shape` are cheap
  and `reader.czi_scene_index` is public, geometry could come from bioio instead of
  being derived here — deleting our scene-mapping code, which is where the scene-index
  defect lived. It trades tested code of ours for a wider dependency on bioio-czi's
  surface and removes neither the `channel_names` nor the `dtype` bypass, so
  `czi_metadata` survives either way.

## Side actions

- Comment on bioio-czi#104 with the measurement table above: `channel_names` and `dtype`
  are uncovered, and the `dims`/`shape` fix does not reach `BioImage` because the facade
  does not delegate to the reader.
- File the read-side issue (draft exists: eager `from_delayed` per plane, and chunk
  count driving per-slice cost). Confirmed a genuine gap in bioio#197's tracker, not a
  duplicate.
