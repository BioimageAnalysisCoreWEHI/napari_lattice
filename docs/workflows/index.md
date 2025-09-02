# Workflows

`lls_core` supports integration with [`napari-workflows`](https://github.com/haesleinhuepf/napari-workflows).
The advantage of this is that you can design a multi-step automated workflow that uses `lls_core` as the pre-processing step.



## Building a Workflow

You can design your workflow via GUI using [`napari-assistant`](https://github.com/haesleinhuepf/napari-assistant), or directly in the YAML format.

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

Workflows are run once for each 3D slice of the image. The workflow is run separately for each timepoint, for each channel, for each region of interest (if cropping is enabled).So, you should design your workflow expecting that `deskewed_image` is a 3D array.

If you want to define your own custom functions, you can do so in a `.py` file in the same directory as the workflow `.yml` file. These will be imported before the workflow is executed.

Workflows can be built in many ways: 

<div class="grid cards" markdown>

-   :octicons-play-24:{ .lg .middle } __Interactive_(no code)__

    ---

    We will use napari-assistant to interactively generate workflows in napari. Requires no coding!

    [:octicons-arrow-right-24: Interactive Workflows](interactive_workflow.md)

-   :octicons-tools-24:{ .lg .middle } __Programmatically__

    ---
    Alternatively, we can generate workflows programmatically using jupyter notebooks.
    This is more powerful and you can integrate many popular software tools.
    [:octicons-arrow-right-24: Jupyter notebooks](notebooks/Image_analysis_workflow_simple.ipynb)

</div>