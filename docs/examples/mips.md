# Generating MIPs and defining ROIs

A top-down **maximum-intensity projection (MIP)** is often all you need to plan an
analysis — to get a quick overview of a very large dataset, or to draw the cropping ROIs
that later runs will process. On the Zeiss LLS7 a MIP is written automatically at the end
of each acquisition, but you can also generate a deskewed MIP yourself from **any**
supported dataset with `lls-pipeline --save-mip`.

This walkthrough goes end to end: generate a MIP, draw ROIs on it in Fiji, and feed those
ROIs back into a cropping run.

!!! note "MIP generation is CLI/API only"

    `--save-mip` is available on the [command line](../cli.md#generating-mips-save-mip) and
    through the [Python API](../api.md#maximum-intensity-projections-mips). It is not
    currently a checkbox in the napari plugin.

## 1. Generate a deskewed MIP

```bash
lls-pipeline --save-mip /path/to/input.tiff --save-dir /path/to/output
```

The MIP is computed straight from the raw data — each projected pixel is mapped back to
its source voxels and the maximum is accumulated along the axial direction. The full
deskewed volume is **never materialised**, so this path is memory-light and fast, which
makes it practical on terabyte-scale acquisitions.

What you get and the knobs that matter:

- **One MIP per timepoint and channel**, written using `--save-type` (`tiff`, `h5` or
  `omezarr`).
- **Smoothness** — `--mip-interpolation nearest` (default; fastest, blocky) or
  `--mip-interpolation linear` to blend adjacent scan planes for a smoother MIP.
- **Geometry aware** — works with `--skew X`/`--skew Y` and both coverslip-rotation modes,
  so a MIP from OPM data uses the same `--no-coverslip-rotation` you deskew with (see the
  [OPM example](opm.md)).
- **Cropping and deconvolution are ignored** on the MIP path — it is a fast whole-frame
  projection, not a per-ROI product.

## 2. Draw ROIs on the MIP in Fiji

Open the MIP in Fiji and draw a rectangle around each region you want to process, adding
each to the ROI Manager, then save the ROI Manager as a `.zip`.

!!! warning "ROIs must be rotated 90°"

    napari-lattice interprets ROIs in the space of the **deskewed** image, so ROIs drawn
    on the MIP need to be rotated 90° before use. You can either rotate the MIP image
    first and then draw, or rotate an existing ROI set with the supplied Fiji macro. The
    full procedure, with screenshots and the macro, is in
    [Defining ROIs for cropping](../miscellaneous/rois_cropping.md).

## 3. Feed the ROIs back into a cropping run

With the rotated ROI file in hand, point a normal (non-MIP) `lls-pipeline` run at it. This
run deskews and crops the full volume — only the ROIs you drew are processed:

```yaml
# crop_config.yml
input_image: "/path/to/input.tiff"
save_dir: "/path/to/output/cropped"
save_type: "h5"
crop:
  roi_list: "/path/to/rois.zip"
```

```bash
lls-pipeline process --yaml-config crop_config.yml
```

To process only some ROIs, or to run many in parallel, see
[ROI selection and parallel ROI processing](../api.md#selecting-which-rois-to-process).

## Next

The [Neutrophil NETosis example](neutrophil.md) picks up from here — it assumes a rotated
ROI file already exists and runs a full crop → deskew → segmentation workflow over it.
