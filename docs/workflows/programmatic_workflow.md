
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
