# Command Line Interface

::: mkdocs-click
    :module: lls_core.cmds.__main__
    :command: click_app
    :prog_name: lls-pipeline

## Coverslip-frame deskew (`--no-coverslip-rotation`)

By default, `lls-pipeline` performs the standard deskew (`cle.deskew_y` /
`cle.deskew_x`) — `--coverslip-rotation` is **on** by default. This rotates the
volume into the coverslip frame and is the correct output geometry for **Zeiss LLS**
data.

Pass `--no-coverslip-rotation` to skip that rotation and deskew directly into the
shear-only frame that is level for **OPM/SOPi** (oblique-plane and
single-objective planar illumination) acquisitions. In a single interpolation pass
the data are sheared so that the specimen plane is level with the coverslip, rather
than tilted at the light-sheet angle as the standard objective-frame deskew would
leave it.

Key properties of this mode:

- **Both skew directions supported.** Works for `--skew Y` (default) and
  `--skew X` acquisitions.
- **MIP path.** When `--save-mip` is also set, the maximum-intensity projection
  is computed in the shear-only coverslip frame, not the stock objective frame.
- **Cropping path.** When `--roi-list` is provided, each ROI is cropped after the
  coverslip-frame deskew, so crop coordinates refer to the coverslip-frame geometry.
- **Default unchanged.** Omitting `--no-coverslip-rotation` (or passing
  `--coverslip-rotation`) uses the stock deskew, leaving all behaviour for Zeiss
  LLS data exactly as before.
