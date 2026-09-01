# Milestone ④ — online Gaussian backend (2026-09-01)

`bundlesdf.py` gains `run_gaussian()`, selected by `cfg_nerf['backend']`
(`nerf` default; the SDF path is untouched). It keeps `run_nerf`'s exact
`p_dict`/keyframe-list contract: consume pending keyframes, publish
`nerf_num_frames` for the sync gate, return `optimized_cvcam_in_obs`.
**v1 pose feedback is a NO-OP** (tracker poses returned verbatim) — pose
optimization is milestone ⑤.

Per cycle: first batch runs the SAM3D prior pipeline online (surfel
sampling + arm-C color transfer + Sim(3) alignment via
`run_sam3d_alignment.align_prior_sim3`, ~30 s once) and
`initialize_from_prior`; later cycles `refresh_view_poses` (the tracker
keeps bundle-adjusting old keyframes) then `update()` with the lifecycle.
Entry point: `run_custom.py --backend gaussian --prior-mesh-npz ...
--prior-pose-json ... --prior-gaussian-ply ...` (global SDF refinement is
skipped for this backend in v1).

## Validation

- **Port equivalence** (offline replay via the integrated runner vs the
  experiments prototype, mustard0 36 KF): UNSEEN survivors identical
  (9,432), rgb 0.0405 vs 0.0406, depth 3.80 vs 4.11 mm — port confirmed.
  Run: `logs/mustard0_lifecycle_mainpath_20260901/`.
- **Online smoke** (mustard0, 737 frames, stride 1): completed end-to-end;
  33 lifecycle events; online init classified 7,766 verified / 188
  contradicted / 12,046 unseen; final map 60.5k splats with 9,406 UNSEEN
  preserved. Map artifacts in
  `outputs/mustard0_gs_online_20260901/gs_online/`.
- **ADD gate** (first-frame-aligned, AUC@0.1 m):

| mustard0 | ADD-S err | ADD err | ADD-S AUC | ADD AUC |
|---|---|---|---|---|
| July SDF backend (with SDF feedback) | 0.460 cm | 1.369 cm | 95.41 | 86.33 |
| GS backend, NO-OP feedback | **0.369 cm** | **0.758 cm** | **96.32** | **92.43** |

The no-op run tracks *better* than the SDF-feedback baseline — beyond the
established same-code noise floor (±0.03 cm). Single run, but it suggests
the SDF feedback was hurting mustard0, and it fixes the milestone-⑤
success bar: **GS pose feedback must beat the no-op 0.758 cm ADD, not the
SDF 1.369 cm.**

## v1 leftovers (deliberate)

- occ_mask is not forwarded to the online lifecycle classification yet
  (None for YCB anyway; needed for HO3D hands).
- `run_ho3d.py` is not wired to the backend switch (YCB/custom entry only).
- Global GS refinement (offline `--mode global_refine` equivalent) deferred.
- Single-sequence smoke; multi-sequence online eval belongs to the
  evaluation milestone.
