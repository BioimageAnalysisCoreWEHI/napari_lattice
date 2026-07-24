# Examples

End-to-end walkthroughs that combine the individual features into a complete analysis.

<div class="grid cards" markdown>

-   :octicons-broadcast-24:{ .lg .middle } __Oblique plane microscopy (OPM)__

    ---

    Deskew non-Zeiss OPM/SOPi datasets by matching the acquisition geometry — coverslip rotation and scan-direction inversion — on two published example datasets.

    [:octicons-arrow-right-24: OPM example](opm.md)

-   :octicons-image-24:{ .lg .middle } __MIPs and defining ROIs__

    ---

    Generate a deskewed MIP straight from raw data with `--save-mip`, draw cropping ROIs on it in Fiji, and feed them back into a crop run.

    [:octicons-arrow-right-24: MIP walkthrough](mips.md)

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
