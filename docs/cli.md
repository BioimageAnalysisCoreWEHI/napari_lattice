# Command Line Interface

::: mkdocs-click
    :module: lls_core.cmds.__main__
    :command: click_app
    :prog_name: lls-pipeline

## Coverslip-frame deskew (`--no-coverslip-rotation`)

By default, `lls-pipeline` performs the standard deskew (`cle.deskew_y` /
`cle.deskew_x`) — `--coverslip-rotation` is **on** by default. This performs a rotation of the volume based on the skew angle, which is accurate output geometry for **Zeiss LLS**
data.

Pass `--no-coverslip-rotation` to skip that rotation and deskew directly into the
shear-only frame that is level for **OPM/SOPi** (oblique-plane and
single-objective planar illumination) acquisitions. In a single interpolation pass
the data are sheared/deskewed so that the specimen plane is level with the coverslip, rather
than tilted at the light-sheet angle. Within the plugin, you could try "Quick Deskew" to visualize what the output would look like.

Key properties of this mode:

- **Both skew directions supported.** Works for `--skew Y` (default) and
  `--skew X` acquisitions.
- **MIP path.** This mode is also compatible with  `--save-mip`.
- **Cropping path.** When `--roi-list` is provided, each ROI is processed with the selected coverslip rotation mode.
- **Default unchanged.** Omitting `--no-coverslip-rotation` (or passing
  `--coverslip-rotation`) uses the stock deskew, leaving all behaviour for Zeiss
  LLS data exactly as before.

## Flipping the scan direction (`--invert-scan-direction`)

Different microscopes scan the sample in different directions. On some oblique-plane
systems the stage or galvo scans in the *opposite* direction to the Zeiss LLS, which
produces a deskewed volume that is mirrored along the scan axis.

Pass `--invert-scan-direction` to reverse the order of the planes along the scan (Z)
axis **before** deskewing, correcting this handedness:

```bash
lls-pipeline --invert-scan-direction /path/to/input.tiff
```

- **Default off.** Omitting the flag preserves the original Zeiss LLS behaviour.
- **Combine with geometry flags.** It works alongside `--skew X`/`--skew Y` and
  `--no-coverslip-rotation`, so you can match most OPM acquisition geometries by
  combining these three options.

## Generating MIPs (`--save-mip`)

At the end of every acquisition, a top-down (coverslip-view) **maximum-intensity
projection (MIP)** is often all that is needed — for example to define cropping ROIs, or
for a quick overview of a very large dataset. `lls-pipeline` can produce this deskewed MIP
directly, **without ever building the full deskewed volume**:

```bash
lls-pipeline --save-mip /path/to/input.tiff --save-dir /path/to/output
```

The MIP is computed straight from the raw data by mapping each projected pixel back to the
source voxels and accumulating the maximum along the axial direction (a CPU/numba gather).
Because the full deskewed volume is never materialised, this path is **memory-light and
fast**, which makes it well suited to terabyte-scale acquisitions.

Key properties:

- **One MIP per timepoint and channel**, written using the chosen `--save-type`
  (`tiff`, `h5` or `omezarr`).
- **Smoothness.** Use `--mip-interpolation nearest` (default — fastest, blocky) or
  `--mip-interpolation linear` to blend adjacent scan planes for a smoother MIP.
- **Geometry aware.** Compatible with `--skew X`/`--skew Y` and the coverslip-rotation
  modes above.
- **Cropping and deconvolution are ignored** when `--save-mip` is set — the MIP is a
  fast whole-frame projection, not a per-ROI product.

!!! info "MIP output is CLI/API only"

    Generating MIP files is available through the command line (`--save-mip`) and the
    [Python API](api.md#maximum-intensity-projections-mips). It is not currently exposed
    as a checkbox in the napari plugin.
