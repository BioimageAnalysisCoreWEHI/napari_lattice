# Oblique plane microscopy (OPM)

napari-lattice can be used for processing  **oblique plane microscopy
(OPM)** and **scanned oblique plane illumination (SOPi)** data. The default parameters work for the Zeiss Lattice Lightsheet 7 (LLS7).  The deskew is **parameter-driven** — you set the skew axis, deskew angle, whether to apply the coverslip rotation, and whether to invert the scan direction. Choosing those values to match a
different acquisition geometry enables napari-lattice to process data from different systems.

This example deskews two published OPM datasets. The two parameters that make an
acquisition "OPM"  are usually to do with **`coverslip_rotation`** and **`invert_scan_direction`**, 
and of course different angles and voxel sizes.

!!! note "Data and configs"

    The two datasets below — raw image, deskewed output, and the `lls-pipeline` config
    that produced it — are published with the manuscript on Zenodo as
    `Supplementary_opm_data.zip`. The file paths in the configs are placeholders; point
    them at your own copies.

## The key setting: coverslip rotation

By default napari-lattice applies a **coverslip rotation** — after shearing, it rotates
the deskewed volume by the deskew angle. This is the correct output geometry for the
**Zeiss LLS7**, where it leaves the specimen level with the coverslip.

Many OPM/SOPi systems need the **opposite** — the volume should *not* be rotated, because
the shear alone already brings the specimen plane level with the coverslip. For these you
turn the coverslip rotation **off**:

| Setting | Plugin | CLI | What it does |
|---|---|---|---|
| Coverslip rotation **on** (default) | `Coverslip Rotation` ticked | omit the flag (or `--coverslip-rotation`) | Standard deskew (`cle.deskew_y`/`cle.deskew_x`) then rotate by the deskew angle. Coverslip-level for **Zeiss LLS7**. |
| Coverslip rotation **off** | `Coverslip Rotation` unticked | `--no-coverslip-rotation` | Shear-only deskew, no rotation. Coverslip-level for many **OPM/SOPi** systems. |

!!! important "This is a geometry choice, not a quality setting"

    Neither value is "better" — the right one depends on how your microscope acquires the
    data. If a deskew looks sheared, tilted, or stretched at the light-sheet angle instead
    of lying flat, the coverslip-rotation setting is the first thing to flip. Both OPM
    datasets below use coverslip rotation **off**.

    See [Coverslip-frame deskew (`--no-coverslip-rotation`)](../cli.md#coverslip-frame-deskew-no-coverslip-rotation)
    for the reference details. 
    
    TIP: In the plugin, **Quick Deskew** gives a fast preview so you
    can check the orientation before committing to a full run.

## The second setting: scan-direction (Z) inversion

Different microscopes scan the sample in opposite directions. On some oblique-plane
systems the stage or galvo scans the **opposite** way to the Zeiss LLS, so the deskewed
volume comes out **mirrored along the scan (Z) axis**.

`invert_scan_direction` reverses the order of the planes along the scan Z axis **before**
deskewing, correcting this handedness. Both datasets below need it:

| Setting | Plugin | CLI | What it does |
|---|---|---|---|
| Invert scan direction | `Invert Scan Direction` ticked | `--invert-scan-direction` | Reverse plane order along the scan (Z) axis before deskewing. Default is **off** (Zeiss LLS behaviour). |

If a coverslip-rotation-corrected deskew is still flipped top-to-bottom along the depth
axis, this is the setting to change. See
[Flipping the scan direction (`--invert-scan-direction`)](../cli.md#flipping-the-scan-direction-invert-scan-direction).

## Datasets

| Folder | Sample | Reference |
|---|---|---|
| `brain_organoid` | 4×-expanded brain organoid, direct-view oblique plane microscopy | Lamb *et al.*, *Optica* **12**, 469–472 (2025); raw data [10.6084/m9.figshare.28324301](https://doi.org/10.6084/m9.figshare.28324301) |
| `thy1_eGFP` | Uncleared coronal Thy1-GFP mouse brain section, scanned oblique plane illumination (SOPi) | Kumar & Kozorovitskiy, *Opt. Lett.* **44**, 1706–1709 (2019); raw data [zenodo.org/records/5088089](https://zenodo.org/records/5088089) |

Both are Y-skew, 45° acquisitions with coverslip rotation **off** and scan direction
**inverted**. They differ only in pixel size.

## Option A — in the napari plugin

1. Drag the raw `.tif` into napari and select it in the `Deskew` tab.
2. Enter the acquisition metadata for the dataset:

    | Parameter | `brain_organoid` | `thy1_eGFP` |
    |---|---|---|
    | dx (µm) | 1.04 | 1 |
    | dy (µm) | 1.04 | 1 |
    | dz (µm) | 2.0 | 1 |
    | Deskew angle | 45 | 45 |
    | **Coverslip Rotation** | **off** | **off** |
    | **Invert Scan Direction** | **on** | **on** |

3. Run **Quick Deskew** (or `Preview`) to check the orientation, then process.

    <!-- SCREENSHOT PLACEHOLDER: Deskew tab with an OPM dataset loaded, Coverslip Rotation unticked and Invert Scan Direction ticked -->
    <!-- SCREENSHOT PLACEHOLDER: Quick Deskew preview of the deskewed OPM volume -->

## Option B — on the command line

Put the parameters in a YAML config and call `lls-pipeline`. This is the config shipped
in each dataset folder, with paths generalised to placeholders.

**`brain_organoid_config.yml`**

```yaml
input_image: "/path/to/brain_organoid/EXP_7_5mm_2um_Steps_1_MMStack_Pos0_14.ome.tif"
save_dir: "/path/to/output/brain_organoid"
save_type: "tiff"
save_name: "brain_organoid_deskewed"
skew: "Y"
angle: 45
coverslip_rotation: False       # shear-only OPM geometry
invert_scan_direction: True     # scan runs opposite to the Zeiss LLS
physical_pixel_sizes:
  Z: 2.0
  Y: 1.04
  X: 1.04
```

**`thy1_eGFP.yml`**

```yaml
input_image: "/path/to/thy1_eGFP/thy1gfp_3_8bit_250um_50fps_20ms_5s_10_MMStack_Pos0.ome.tif"
save_dir: "/path/to/output/thy1_eGFP"
save_type: "tiff"
save_name: "thy1_eGFP_deskewed"
skew: "Y"
angle: 45
coverslip_rotation: False       # shear-only OPM geometry
invert_scan_direction: True     # scan runs opposite to the Zeiss LLS
physical_pixel_sizes:
  Z: 1
  Y: 1
  X: 1
```

Run either config with:

```bash
lls-pipeline process --yaml-config brain_organoid_config.yml
```

!!! tip "Matching a different OPM system"

    Most OPM/SOPi acquisitions can be matched by combining three options — `--skew X`/`--skew Y`,
    `--no-coverslip-rotation`, and `--invert-scan-direction`. If your first deskew looks
    wrong, work through them in that order: skew axis, then coverslip rotation, then scan
    direction.
