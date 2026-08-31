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

## Color-transfer experiment (experiments/exp_color_transfer.py, 2026-08-31)

Hypothesis: transferring SAM3D *gaussian* colors (appearance-trained) onto
the mesh surfels rehabilitates an SSIM photo term. Arms over all 9 seqs,
judged against the raw measured cloud: A = mesh colors / photo off
(current default), B = transferred colors / SSIM w0.5, C = transferred /
SSIM w1.0; a 4-seq D control (mesh colors / SSIM 0.5) proved improvements
come from the colors, not the term itself.

YCB (9 seqs):

| arm | mean 3D (mm) | mean IoU | notes |
|---|---|---|---|
| A | 3.84 | 0.763 | |
| B | 3.96 | 0.773 | sugar_yalehand z-shift 10.6→30.6 mm |
| C | 3.81 | 0.774 | bleach_hard 8.0→6.3 mm / IoU 0.65→0.78 (largest win); bleach0 IoU 0.87→0.80 |

HO3D_v3 (13 seqs, SAM2-mask priors via batch_sam3d_mesh_ho3d.py, raw
RGB-encoded depth): SAM3D initial poses are far worse here (up to 46.5 mm,
hand occlusion), and the alignment recovers every sequence to 1.6–5 mm.

| arm | mean 3D (mm) | mean IoU | notes |
|---|---|---|---|
| A | 2.82 | 0.888 | |
| C | 2.66 | 0.890 | AP12 5.0→2.5 mm / IoU 0.84→0.95 (rescues the worst init 46.5 mm); MPM12 3.6→4.5 regression |

Combined (22 seqs): A 3.24 mm / IoU 0.837 vs **C 3.13 mm / IoU 0.843**.

**Decision (2026-08-31): arm C is the default going forward** — transferred
gaussian colors + SSIM-only photo at w=1.0. Pattern: C rescues the
sequences where SAM3D's initial pose is badly wrong (bleach_hard, AP12) at
the cost of small scattered losses on easy ones; across 22 sequences it
leads on both metrics. Invoke via `run_sam3d_alignment.py --gaussian-ply
<seq>_splat_depth.ply` (without the ply the tool falls back to depth-only;
mesh-color+SSIM is rejected — arm D showed it is ineffective).
Residual note: z-shift sign differs by dataset (+ on YCB, − on many HO3D) —
the SAM3D shape bias direction is dataset-dependent; geometry refinement
downstream owns it. Follow-up idea (not implemented): occlusion-aware
photo weighting (largest color delta 0.084 where SAM3D saw least).
