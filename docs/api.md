# Python Usage

## Introduction

The image processing workflow can also be controlled via Python API.

To do so, first define the parameters:

```python
from lls_core import LatticeData

params = LatticeData(
  input_image="/path/to/some/file.tiff",
  save_dir="/path/to/output"
)
```

Then save the result to disk:
```python
params.save()
```

Or work with the images in memory:
```python
for slice in params.process():
    pass
```

Other more advanced options [are listed below](#lls_core.LatticeData).

## Cropping

Cropping functionality can be enabled by setting the `crop` parameter:

```python
from lls_core import LatticeData, CropParams 

params = LatticeData(
  input_image="/path/to/some/file.tiff",
  save_dir="/path/to/output",
  crop=CropParams(
    roi_list=["/path/to/roi.zip"]
  )
)
```

Other more advanced options [are listed below](#lls_core.CropParams).

### Selecting which ROIs to process

A ROI file (or `roi_list`) may contain many regions. By default **all** of them are
processed. To process only some, pass their indices via `roi_subset`:

```python
from lls_core import LatticeData, CropParams

# Process ALL ROIs in the file (default — roi_subset omitted)
params = LatticeData(
  input_image="/path/to/some/file.tiff",
  save_dir="/path/to/output",
  crop=CropParams(roi_list=["/path/to/roi.zip"])
)

# Process ONLY ROIs 0 and 2
params = LatticeData(
  input_image="/path/to/some/file.tiff",
  save_dir="/path/to/output",
  crop=CropParams(roi_list=["/path/to/roi.zip"], roi_subset=[0, 2])
)
```

The rule is: **omit `roi_subset` to process every ROI; pass a list of indices to
restrict processing to that subset.** The same setting controls the serial and
parallel paths alike.

## Parallel ROI Processing

When cropping is enabled, multiple ROIs can be processed in parallel worker
processes that share the GPU. This is controlled by `process_parallel`:

```python
from lls_core import LatticeData, CropParams

params = LatticeData(
  input_image="/path/to/some/file.tiff",
  save_dir="/path/to/output",
  process_parallel=4,  # distribute the selected ROIs across 4 workers
  crop=CropParams(roi_list=["/path/to/roi.zip"])
)
```

`process_parallel` accepts:

- **`1`** (default) — serial processing, one ROI at a time.
- **`N > 1`** — an explicit number of worker processes. The selected ROIs
  (see [ROI selection](#selecting-which-rois-to-process)) are split into roughly
  equal chunks, one per worker. Useful when a single ROI does not saturate the GPU.
- **`0`** — *auto*: a memory-safe worker count is derived from a memory estimate.
  This is disabled for deconvolution and workflow runs (whose memory cannot be
  sized), which fall back to serial.

Notes:

- The selected ROIs are distributed across workers; parallelism never changes
  *which* ROIs are processed, only how they are spread across processes.
- If only a single ROI is selected, processing always runs serially regardless
  of `process_parallel`, since there is nothing to distribute.
- `process_parallel` is ignored when cropping is disabled.

!!! note "Defaults differ by entry point"

    The Python API and napari GUI default to serial (`process_parallel=1` / the
    GUI's "Parallel ROI Processing" checkbox off). The `lls-pipeline` CLI instead
    defaults to `0` (auto) when you don't set a value, so an unconfigured CLI run
    picks a memory-safe worker count automatically.

## Type Checking

Because of Pydantic idiosyncrasies, the `LatticeData` constructor can accept more data types than the type system realises. 
For example, `input_image="/some/path"` like we used above is not considered correct, because ultimately the input image has to become an `xarray` (aka `DataArray`).
You can solve this in three ways.

The first is to use the types precisely as defined. In this case, we might define the parameters "correctly" (if verbosely) like this:

```python
from lls_core import LatticeData
from bioio import BioImage
from pathlib import Path

params = LatticeData(
  input_image=BioImage("/path/to/some/file.tiff").xarray_dask_data,
  save_dir=Path("/path/to/output")
)
```

The second is to use `LatticeData.parse_obj`, which takes a dictionary of options and allows incorrect types:

```python
params = LatticeData.parse_obj({
  "input_image": "/path/to/some/file.tiff",
  "save_dir": "/path/to/output"
})
```

Finally, if you're using MyPy, you can install the [pydantic plugin](https://docs.pydantic.dev/latest/integrations/mypy/), which solves this problem via the `init_typed = False` option.

## API Docs

::: lls_core.LatticeData
    options:
      members:
        - process
        - process_workflow
        - save

::: lls_core.DeconvolutionParams

::: lls_core.CropParams
    options:
      members: false

::: lls_core.models.results.ImageSlices

::: lls_core.models.results.WorkflowSlices

::: lls_core.models.results.ProcessedWorkflowOutput

::: lls_core.models.deskew.DefinedPixelSizes

::: lls_core.models.deskew.DerivedDeskewFields

::: lls_core.models.output.SaveFileType
