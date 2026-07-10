# Single-file streaming OME-TIFF — design

**Date:** 2026-07-10
**Branch context:** `ome-tiff-writer`
**Status:** Approved design, ready for implementation plan

## Problem

`TiffWriter` currently flushes once per timepoint, opening and closing a fresh
`tifffile.TiffWriter` each time. For an N-timepoint acquisition this produces **N
separate OME-TIFF files** (each multi-channel), named after the first channel of
that timepoint. Users reasonably expect a **single OME-TIFF** containing all
timepoints and channels (`TCZYX`), which is the standard OME-TIFF shape.

## Constraints (decided)

1. **Must stream (low memory).** Target real lattice-lightsheet data with large
   `T` and big volumes that need not fit in RAM. The writer must hold on the
   order of one plane / one timepoint at a time, preserving the existing lazy,
   push-based pipeline (`slices` is a finite, once-only iterator).
2. **Must not alter existing functionality for `h5` (BDV) and `omezarr`.** This
   is a hard requirement. Their code paths must be byte-for-byte equivalent to
   today.
3. **Scope: the normal save path only** (`ImageSlices.save_image`). The workflow
   image-output path (`WorkflowSlices.process`) is out of scope and unchanged.
4. **Legacy uncompressed path unchanged.** `compression=None` continues to write
   per-timepoint ImageJ-TIFFs exactly as today.

## Key technical finding (why the design is shaped this way)

tifffile binds pages into a single OME series **only within one `write()` call.**
Empirically verified in this environment (tifffile 2026.1.14):

| Method | Push-friendly | tifffile reads as | Usable |
|--------|---------------|-------------------|--------|
| Plain pages, N separate `write()` calls, no OME-XML | yes | 1 series `(N,Y,X)` — no T/C/Z meaning | flat stack only |
| Plain pages + OME-XML injected via `tiffcomment` | yes | **N separate series** | broken |
| Generator: one `write(gen, shape=(T,C,Z,Y,X))` | no (pull) | 1 series `TCZYX` | **correct** |

Consequences:
- "Write a plain TIFF, make it OME later" does **not** work — the resulting file
  is misread as one-series-per-page by tifffile/bioio (what napari-lattice reads
  with). Rejected.
- Reusing `bioio-ome-tiff`'s `OmeTiffWriter` is **incompatible with streaming**:
  it calls `tif.write(full_array)` once and therefore requires the whole array in
  memory. Rejected (would violate constraint 1).
- The only correct streaming path is tifffile's **generator form**, which is
  **pull-based**. Bridging the push pipeline to it is the core of the design.

## Architecture

### `Writer.write_all(slices)` — new base method, default = current behavior

Add to the `Writer` base class:

```python
def write_all(self, slices: Iterable[ProcessedSlice[ArrayLike]]) -> None:
    for slice in slices:
        self.write_slice(slice)
    self.close()
```

This body is identical to the loop that lives in `save_image` today.
`BdvWriter` and `OMEZarrWriter` **inherit it unchanged** → constraint 2 is
satisfied by construction (same calls, same order, same single `close()`).

### `save_image` — one line

```python
# before
for slice in roi_results:
    writer.write_slice(slice)
writer.close()

# after
writer.write_all(roi_results)
```

The "pull" happens entirely inside `TiffWriter.write_all`; the rest of the
codebase is unaffected.

### `TiffWriter.write_all` — override

- **Legacy branch:** if `self.compression is None`, delegate to
  `super().write_all(slices)` → existing per-timepoint ImageJ-TIFF, untouched.
- **OME-TIFF branch** (default, compressed):
  1. `it = iter(slices)`; `first = next(it, None)`. If `None` (empty ROI),
     return without writing a file.
  2. From `first.data`: derive `Z, Y, X` and
     `dtype = resolve_output_dtype(first.data.dtype)`.
  3. `T = len(lattice.time_range)`, `C = len(lattice.channel_range)`.
     Channel names derived from `channel_range`.
  4. Filename: **one file per ROI** — `make_filename_suffix(roi_index=...)` only
     (no `_T`, no `_C`), extension `.ome.tif`.
  5. Plane generator over `itertools.chain([first], it)`: for each `(Z,Y,X)`
     slice, cast with `to_output_dtype(data, dtype)` and `yield` each Z-plane in
     order.
  6. Write once:
     ```python
     with tifffile.TiffWriter(path, bigtiff=True) as tw:
         tw.write(plane_gen, shape=(T, C, Z, Y, X), dtype=dtype,
                  compression=self.compression, metadata=ome_metadata)
     ```
  7. `self.written_files.append(path)`.

## Ordering invariant and safety net

Correctness relies on slices arriving **time-major, channel-minor**
(`lattice_data.py` processing loop: `for time: for channel:`), so flattening
each `(Z,Y,X)` slice into Z-planes yields exactly `for t: for c: for z`, which is
the page order tifffile expects for `shape=(T,C,Z,Y,X)` (DimensionOrder
`XYZCT`, Z fastest).

Safety net: if the number of yielded planes ≠ `T·C·Z`, `tw.write` raises. A
partial or misordered run therefore **fails loudly** rather than writing a
silently wrong file.

## Metadata and dtype

- `ome_metadata`: `{"axes": "TCZYX", PhysicalSizeX/Y/Z (+ µm units), Channel: {"Name": [...]}}`
  — same fields as the current per-timepoint OME-TIFF.
- Dtype is fixed from the first slice (same policy as `OMEZarrWriter`); every
  slice is cast to it via `to_output_dtype`.

## Behavior change (user-visible)

Compressed TIFF output changes from **N files (one per timepoint)** to **one
`.ome.tif` per ROI**. This is the intended improvement; note it in the changelog.
No change to `h5`, `omezarr`, or the legacy `compression=None` output.

## Testing

1. **Regression — `h5` and `omezarr` unchanged.** Assert BDV and OME-Zarr outputs
   for a fixed input are identical (file set + contents) before and after the
   change. This directly guards constraint 2.
2. **Single-file OME-TIFF.** `test_save` (tiff case): exactly one `.ome.tif` per
   ROI; read it back and verify `TCZYX` shape, dtype, per-`(t,c)` pixel values in
   the right location, physical pixel sizes, and channel names.
3. **Serial == parallel** still holds for TIFF output.
4. **MIP** (`Z=1`) still writes correctly.
5. **Legacy `compression=None`** still produces per-timepoint ImageJ-TIFFs.
6. **Empty ROI** writes no file and does not error.

## Non-goals

- Workflow image-output path (`WorkflowSlices.process`) — unchanged.
- Legacy ImageJ-TIFF single-file support — out of scope.
- Any change to `BdvWriter` / `OMEZarrWriter` behavior.

## Risks

- tifffile version differences on CI (design verified against 2026.1.14). The
  generator + `metadata` OME-writing API is stable across recent tifffile, but
  the implementation should be exercised on the CI tifffile version.
- Assumes constant `Z` across slices of an ROI (true for deskew output; MIP is
  `Z=1`). A varying `Z` would trip the plane-count safety net, which is the
  desired fail-loud behavior.
