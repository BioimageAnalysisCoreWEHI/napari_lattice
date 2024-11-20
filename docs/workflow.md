# Workflows

`lls_core` supports integration with [`napari-workflows`](https://github.com/haesleinhuepf/napari-workflows).
The advantage of this is that you can design a multi-step automated workflow that uses `lls_core` as the pre-processing step.

## Building a Workflow

You can design your workflow via GUI using [`napari-assistant`](https://github.com/haesleinhuepf/napari-assistant), or directly in the YAML format.

When building your workflow with Napari Assistant, you are actually building a *template* that will be applied to future images.
For this reason, you need to rename your input layer to `deskewed_image`, since this is the exact value that the `lls_core` step produces.

If you want to use YAML, you also have to make sure that the first workflow step to run takes `deskewed_image` as an input.
For example:

```yaml
!!python/object:napari_workflows._workflow.Workflow
_tasks:
  median: !!python/tuple
  - !!python/name:pyclesperanto_prototype.median_sphere ''
  - deskewed_image
  - null
  - 2
  - 2
  - 2
```

Workflows are run once for each 3D slice of the image. In other words, the workflow is run separately for each timepoint, for each channel, for each region of interest (if cropping is enabled).
This means that you should design your workflow expecting that `deskewed_image` is an exactly 3D array.

If you want to define your own custom functions, you can do so in a `.py` file in the same directory as the workflow `.yml` file. 
These will be imported before the workflow is executed.

## Running a Workflow

The `--workflow` command-line flag, the `LatticeData(workflow=)` Python parameter, and the Workflow tab of the plugin can be used to specify the path to a workflow `.yml` file .

If you're using the Python interface, you need to use [`LatticeData.process_workflow()`](api.md#lls_core.LatticeData.process_workflow) rather than `.process()`. 

## Outputs

`lls_core` supports workflows that have exactly one "leaf" task. This is defined as a task that is not used by any other tasks. In other works, it's the final task of the workflow.

If you want multiple return values, this task can return a tuple of values. Each of these values must be:

* An array, in which case it is treated as an image slice
* A `dict`, in which case it is treated as a single row of a data frame whose columns are the keys of the `dict`
* A `list`, in which case it is treated as a single row of a data frame whose columns are unnamed

Then, each slice is combined at the end. Image slices are stacked together into their original dimensionality, and data frame rows are stacked into a data frame with one row per channel, timepoint and ROI.
