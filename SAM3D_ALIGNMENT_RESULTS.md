# SAM3D mesh-prior first-frame alignment — YCB 9-sequence results (2026-08-31)

Pipeline: SAM3D inference with **SAM2 masks** (`masks_sam2`) + GT `cam_K`
(both pointmap and layout postprocess — mandatory; without cam_K the layout
optimization collapses to IoU 0.0), raw `output["mesh"][0]` prior →
area-weighted surfels (`sam3d_prior.py`) → rigid Sim(3) refinement with the
2DGS renderer (`run_sam3d_alignment.py`, config: radius_multiplier 0.75,
photo/SSIM off, depth-driven).

Runs: `logs/<seq>_sam3d_align_sam2mask_{rawdepth,filtdepth,filleddepth}_20260831/`
(each has `sam3d_rts_refined.json`, before/after overlays, `viewer_3d.html`).
SAM3D outputs: `sam-3d-objects/output_YCBInEOAT_sam2mask_mesh/`.

## Depth-target comparison (judged against the raw measured cloud)

"3D" = obs→prior nearest-neighbor median (mm) after refinement; IoU = rendered
alpha vs SAM2 mask at 0.5; z = camera-frame backward shift (mm) absorbed by
the pose (systematic SAM3D front-bulge shape bias).

| sequence | init 3D | raw 3D/IoU | filtered 3D/IoU | filled 3D/IoU | **best depth** |
|---|---|---|---|---|---|
| mustard0 | 5.0 | **4.3**/0.77 | 4.7/0.79 | 5.6/0.83 | raw |
| bleach0 | 5.4 | 3.3/0.87 | 3.2/0.87 | **2.7**/0.86 | filled |
| bleach_hard_00_03_chaitanya | 7.5 | 8.0/0.65 | 8.2/0.67 | **7.0**/**0.76** | filled |
| cracker_box_reorient | 5.6 | 4.1/**0.92** | 4.1/0.90 | **3.5**/0.84 | filled |
| cracker_box_yalehand0 | 11.3 | 3.8/0.74 | **3.4**/0.78 | 4.7/0.76 | filtered |
| mustard_easy_00_02 | 5.2 | **2.8**/0.84 | 3.0/0.83 | 5.1/**0.94** | raw |
| sugar_box1 | 8.5 | 2.9/0.89 | 2.8/0.89 | **2.0**/0.89 | filled |
| sugar_box_yalehand0 | 9.4 | **2.7**/0.53 | 3.1/0.54 | 4.0/0.52 | raw |
| tomato_soup_can_yalehand0 | 4.4 | **2.7**/0.66 | 5.7/0.69 | 2.7/0.65 | raw (=filled) |
| **mean** | 6.9 | **3.83**/0.763 | 4.25/0.775 | 4.16/**0.782** | — |

## Decisions

- **Default depth target: raw** (measurement-only, best mean 3D, no
  inpainting bias). Decided 2026-08-31.
- Per-sequence best (table above) is kept for future max-performance runs;
  `filled` tends to win where sensor dropout is small (better IoU, smaller
  z-shift), but injects interpolation bias exactly where dropout is large
  (cracker_box_yalehand0: raw 3.8 vs filled 4.7).
- cracker_box_yalehand0's sparse first-frame cloud is **sensor dropout in the
  raw data** (34.9% valid in the SAM2 mask; glossy face at a specular angle),
  not a BundleTrack artifact — `depth_filtered` actually fills to 68%,
  dataset `depth_filled` to 96%.
- Known open issues for later stages: systematic +z shift (SAM3D shape
  front-bulge, to be handled by Gaussian refinement, not pose), rotation
  ambiguity on near-symmetric objects (photo/appearance term needed —
  color-transfer experiment), IoU drop on bleach_hard/tomato with the
  depth-only loss.
