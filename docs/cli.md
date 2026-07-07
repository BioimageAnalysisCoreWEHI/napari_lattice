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
