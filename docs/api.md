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

## Type Checking

Because of Pydantic idiosyncrasies, the `LatticeData` constructor can accept more data types than the type system realises. 
For example, `input_image="/some/path"` like we used above is not considered correct, because ultimately the input image has to become an `xarray` (aka `DataArray`).
You can solve this in three ways.

The first is to use the types precisely as defined. In this case, we might define the parameters "correctly" (if verbosely) like this:

```python
from lls_core import LatticeData
from aicsimageio import AICSImage
from pathlib import Path

params = LatticeData(
  input_image=AICSImage("/path/to/some/file.tiff").xarray_dask_data(),
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
