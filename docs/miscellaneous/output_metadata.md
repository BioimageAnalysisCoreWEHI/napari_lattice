# Output metadata (`.lattice.json`)

Every image napari-lattice writes gets a sibling JSON sidecar recording **how the data was
produced** and **where it sits**. For `results/6h_deskewed_ROI_3.ome.tif` the sidecar would be
`results/6h_deskewed_ROI_3.lattice.json`.

The image extension is dropped rather than appended to, so the sidecar never matches a
`*.tif*` glob and tools scanning a results folder for images will not trip over it.

## Why

Pixel data carries voxel size and little else. Two things are otherwise lost at write time:

- **Provenance** — the deskew angle, skew direction, coverslip rotation, scan direction and
  the raw→deskewed affine. The transform is computed for every run but was never saved.
- **Placement** — where a cropped ROI sits in the parent volume. Every ROI file starts at
  voxel `(0,0,0)`, so a set of ROIs from one acquisition could not be related to each other
  or to the full field of view.

## Layout

```json
{
  "schema_version": "1.0",
  "generator": { "name": "napari-lattice", "version": "3.1.1" },
  "output": {
    "path": "6h_deskewed_ROI_3.ome.tif",
    "roi_index": 3,
    "projection": null,
    "origin_zyx_px": [64.0, 1180.0, 512.0],
    "origin_zyx_um": [9.6, 176.91, 76.76],
    "origin_reference": "Voxel (0, 0, 0) of the full deskewed volume ..."
  },
  "roi": {
    "index": 3,
    "units": "Pixels",
    "bbox_yx_px": { "top": 1180.0, "left": 512.0, "bottom": 1660.0, "right": 832.0 },
    "z_range": [64, 274]
  },
  "derived": {
    "output_voxel_size_um": { "z": 0.15, "y": 0.1499, "x": 0.1499 },
    "full_output_shape_zyx": [210, 3100, 2048],
    "raw_to_deskewed_affine_zyx": [[...], [...], [...], [0, 0, 0, 1]],
    "affine_convention": "4x4 row-major homogeneous matrix in ZYX order ..."
  },
  "config": { "angle": 30.0, "skew": "Y", "save_type": "tiff", "...": "..." }
}
```

## `config` — the run, in the config-file schema

The `config` block uses the **same schema as `--json-config` / `--yaml-config`**, so it is
directly reusable:

```bash
# Pull the config back out of a result and re-run it
python -c "import json,sys; json.dump(json.load(open(sys.argv[1]))['config'], sys.stdout)" \
    results/6h_deskewed_ROI_3.lattice.json > rerun.json
lls-pipeline process --json-config rerun.json
```

Run `lls-pipeline process --show-schema` to see the same field set documented.

!!! warning "The config records absolute paths"

    `input_image`, `save_dir`, `workflow` and PSF entries are written as absolute paths
    from the machine that produced the data. Check before sharing sidecars outside your
    group if those paths are sensitive.

!!! note "Three fields record a path, not the value"

    `input_image`, `workflow` and `deconvolution.psf` hold loaded objects in memory, which
    cannot be serialised back to the config that produced them. The sidecar records the
    source path instead, and each is empty when that input was supplied in memory (e.g.
    from the napari plugin, or a script passing an array). The rest of the config is still
    accurate, but that field cannot reproduce the run on its own.

    One case is not merely incomplete but un-runnable: a **deconvolution** run whose PSFs
    were passed as arrays records no PSF paths, and re-parsing then fails because a
    validator requires one PSF per channel. Supply PSFs as paths if you want the config
    to be re-runnable.

## `origin_zyx_px` — placing an output back in the parent volume

`origin_zyx_px` is the position of the output's voxel `(0,0,0)` within the **full deskewed
volume**, in deskewed voxels; `origin_zyx_um` is the same in microns. Loading several ROIs
into a shared coordinate system is then just a translation:

```python
import json
from pathlib import Path
import napari, tifffile

viewer = napari.Viewer()
for sidecar in sorted(Path("results").glob("*.lattice.json")):
    meta = json.loads(sidecar.read_text())
    image = tifffile.imread(Path("results") / meta["output"]["path"])
    voxel = meta["derived"]["output_voxel_size_um"]
    viewer.add_image(
        image,
        name=f"ROI {meta['output']['roi_index']}",
        scale=(voxel["z"], voxel["y"], voxel["x"]),
        translate=meta["output"]["origin_zyx_um"],
    )
```

!!! warning "The origin is not the ROI you drew"

    Cropping only trims the **skew axis** to the ROI; the other two axes keep the bounds the
    clipped raw sub-block deskewed into. So the Z origin is the sub-block's minimum deskewed
    Z, not `z_range[0]`, and if an ROI extends past what the raw data can produce the output
    starts where the data actually begins. `origin_zyx_px` always describes where the pixels
    *are*, which is what you want for placement — compare it against `roi.bbox_yx_px` if you
    need to know whether the request was clipped.

    An uncropped run has origin `[0, 0, 0]`. A MIP has a `null` Z origin, since Z is
    projected away.

## `raw_to_deskewed_affine_zyx`

A 4×4 row-major homogeneous matrix in ZYX order, mapping a **raw** voxel index
`[z, y, x, 1]` to a voxel index in the full deskewed volume.

This records the transform that was **applied**. The saved pixels are already deskewed, so
the matrix is for relating them back to the raw acquisition — not something to apply again.
Multiply componentwise by `output_voxel_size_um` for microns.

The matrix means different things in the two geometries, so read it together with
`config.coverslip_rotation`: `true` (Zeiss LLS7 and similar) is the standard deskew with the
coverslip rotation folded in; `false` is the shear-only OPM/SOPi frame.
