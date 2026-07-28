# HPC processing with SLURM

The [neutrophil pipeline](neutrophil.md) was designed to run on an HPC cluster, processing
many ROIs in parallel on a single GPU node. This page shows how to run it from the command
line and how to submit it as a SLURM job.

!!! note "Based on the published scripts"

    The scripts and configs here follow the ones deposited with the data on Zenodo
    ([zenodo.org/records/20837879](https://zenodo.org/records/20837879), under
    `code/hpc_scripts/`). Adjust the `#SBATCH` directives, environment name and paths for
    your own cluster.

## The config file

Rather than a long command line, the parameters live in a YAML config that `lls-pipeline`
reads with `--yaml-config`:

```yaml
# nap_lattice_config.yml
input_image: "../../data/6h_timelapse.czi"
save_dir: "results_hpc_crop_workflow/"
save_type: "h5"
channel_range: [0, 1]
workflow: "../workflow/netosis_seg_measure.yml"
crop:
  roi_list: "../../data/6h_timelapse_ROIs.zip"
```

!!! tip "Writing a config"

    Run `lls-pipeline process --show-schema` to print the full set of JSON/YAML options,
    which you can use as a template for your own config file. Both `--yaml-config` and
    `--json-config` are supported.

## Running from the command line

With the config in place, a full run is a single command:

```bash
lls-pipeline process --yaml-config nap_lattice_config.yml
```

### Sizing the job

Before committing a large job, print a VRAM/RAM estimate (without processing) to help pick
resources and a worker count. `estimate` takes the same options as `process`:

```bash
lls-pipeline estimate --yaml-config nap_lattice_config.yml
```

### Processing ROIs in parallel — the modern way

From napari-lattice **v2.0** onward, ROI-level parallelism is built in via
`--process-parallel`, which distributes the ROIs across worker processes that share the one
GPU. If you don't set it, it defaults to `0` (**auto**) — the largest worker count that fits
the GPU and host memory estimate, the number of ROIs, the CPU count and, in a SLURM
allocation, `SLURM_CPUS_PER_TASK`:

```bash
# Auto: let napari-lattice choose a memory-safe worker count
lls-pipeline process --yaml-config nap_lattice_config.yml --process-parallel 0
```

Or set the number of workers explicitly — for example `8`:

```bash
# Explicit: 8 workers sharing the one GPU
lls-pipeline process --yaml-config nap_lattice_config.yml --process-parallel 8
```

!!! warning "Workflow runs need an explicit worker count"

    `--process-parallel 0` (auto) sizes workers from a memory estimate, but auto is disabled
    for **workflow** and **deconvolution** runs (memory can't be sized) and falls back to
    serial. This pipeline uses a workflow, so pass an explicit number (e.g. `8`).

## Resources used in the manuscript

In the manuscript, all **17 ROIs** of the neutrophil timelapse were processed in a **single
SLURM job on one GPU node** of WEHI's Milton HPC cluster, with:

- **GPU:** 1 × NVIDIA A30, **24 GB VRAM** (OpenCL 3.0 / CUDA)
- **RAM:** 300 GB requested (`--mem=300G`)
- **CPUs:** 35 (`--cpus-per-task=35`)
- **Wall time:** 15 h requested (`--time=15:00:00`)

All 17 ROIs shared the single A30 GPU. This ROI-based approach reduced the deskewed data
~2.6-fold versus the full field of view and completed the end-to-end analysis in ~12 h,
whereas full-FOV processing did not finish within the 48 h limit.

!!! note "Peak usage"

    Processing all **16 ROIs in parallel** on the shared A30 peaked at only **~4 GB GPU
    VRAM** and **~75.6 GB host RAM** — comfortably within the node's 24 GB VRAM and 300 GB
    RAM. By contrast, full field-of-view processing peaked at **~287 GB host RAM** (likely
    driven by ilastik operating on the whole volume), which is why the ROI-based approach
    scales so much better. Use `lls-pipeline estimate` on your own data and hardware to get a
    comparable estimate before submitting.

## Submitting as a SLURM job

A minimal batch script requesting one GPU node and running the whole thing with
`--process-parallel`:

```bash
#!/bin/bash
#SBATCH --job-name=netosis
#SBATCH --time=15:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=35
#SBATCH --mem=300G
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:A30:1
#SBATCH --output=logs/net_%A.out
#SBATCH --mail-type=END
#SBATCH --mail-user=you@example.edu

module load miniconda3
conda activate nap_lat_ilastik

# All ROIs, distributed across 8 workers sharing the one A30 GPU
lls-pipeline process --yaml-config nap_lattice_config.yml --process-parallel 8
```

Submit it with `sbatch`:

```bash
sbatch nap_lat_crop_workflow.sh
```

## What changed: `&` backgrounding → `--process-parallel`

The originally published script pre-dates the built-in parallelism. It ran ROI-level
parallelism **by hand**, launching one `lls-pipeline` process per ROI in the background with
`&` and waiting for them all:

```bash
# Older approach (napari-lattice v1.2.1) — manual shell backgrounding
lls-pipeline --yaml-config nap_lattice_config.yml --roi-subset 0 &
lls-pipeline --yaml-config nap_lattice_config.yml --roi-subset 1 &
lls-pipeline --yaml-config nap_lattice_config.yml --roi-subset 2 &
# ... one line per ROI ...
lls-pipeline --yaml-config nap_lattice_config.yml --roi-subset 16 &
wait
```

This works, but you have to enumerate every ROI, and all the processes compete for GPU
memory with no coordination. From **v2.0** the single `--process-parallel N` flag replaces
the whole block — it splits the selected ROIs across `N` coordinated workers on the shared
GPU:

```bash
# Current approach (napari-lattice v2.0+) — one command
lls-pipeline process --yaml-config nap_lattice_config.yml --process-parallel 8
```

Use `--roi-subset` if you still want to split ROIs across *separate* SLURM jobs (for
example a job array), and `--process-parallel` within each job to parallelise that job's
share.
