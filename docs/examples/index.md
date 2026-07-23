# Examples

End-to-end walkthroughs that combine the individual features into a complete analysis.

<div class="grid cards" markdown>

-   :octicons-beaker-24:{ .lg .middle } __Neutrophil NETosis (end-to-end)__

    ---

    Crop, deskew and analyse a live-cell LLS7 timeseries of neutrophils — first in the plugin, then headlessly on the command line with parallel ROI processing.

    [:octicons-arrow-right-24: End-to-end example](neutrophil.md)

-   :octicons-server-24:{ .lg .middle } __HPC processing with SLURM__

    ---

    Run the same pipeline on an HPC cluster as a single SLURM job, distributing many ROIs across workers that share one GPU.

    [:octicons-arrow-right-24: SLURM example](hpc_slurm.md)

</div>

!!! note "About the data paths"

    These examples are based on the neutrophil dataset from the napari-lattice manuscript. The raw data is not bundled with the docs, so the file paths below are **illustrative placeholders** — substitute your own paths. The commands and options themselves are accurate.
