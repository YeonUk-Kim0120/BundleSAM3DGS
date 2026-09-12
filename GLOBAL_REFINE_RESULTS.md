# Global refinement + mesh extraction for the GS backend — results log

Scripts: `experiments/run_gs_global_refine.py` (global stage + 3 mesh extractions), `experiments/eval_mesh_cd.py`
(3 Chamfer protocols). Main code untouched. Logs/JSON: `logs/global_stage1_20260911/`. Meshes: `<run_dir>/final/gs_r2/`.

## Protocols (all meshes, ours and BundleSDF's, scored by the same code)
- **P1 original** = BundleSDF benchmark port: first-frame alignment, 30 cm crop, largest component, 99 999 samples,
  5 mm downsample, ICP (2 cm) onto the GT cloud, mutual Chamfer (cm). GT = HO3D `visible_mesh.ply` (seen parts only),
  YCB full model. Verified: BundleSDF AP12 0.631 (recorded 0.634), mustard0 0.533 (recorded 0.531).
- **P2 full model**: same alignment/ICP, GT = complete CAD model. Holes and invented surfaces both cost.
- **P3 regions**: full-model GT samples split into SEEN / UNSEEN (HO3D: visible_mesh vertices; YCB: ray cast from
  keyframe GT poses, self-occlusion only). GT→pred distance per region + share of UNSEEN GT within 5 mm of the prediction.
- Why BundleSDF's holes are free under P1: the GT has no points where the object was never seen, so neither direction
  sees the hole; a filled surface there is *penalised* in pred→GT (nearest visible GT point). Hence P2/P3.

## Stage 1 (2026-09-11): one setting, three extractions, AP12 + mustard0
Setting: online map continued (`gs_online/checkpoint_final.pt` of the v1 runs), v1 pose refinement on, 2000 steps,
SUSPECT included, Gaussians with in-plane radius > 10 mm hidden for extraction (AP12: 2391, mustard0: 428 — online
training let some prior-lineage surfels grow to 1–29 cm; to be fixed in training later), 2DGS median depth for TSDF.
Extractions: TSDF from the 181/37 training views (6DOPE-GS recipe, voxel 2 mm, trunc 2 cm), TSDF from 120 virtual
cameras on a sphere (2.5× object radius), screened Poisson (depth 9) from Gaussian centres + surfel normals.

| seq | mesh | P1 original | P2 full | pred→GT | GT→pred seen | GT→pred unseen | unseen ≤5 mm | vertices |
|---|---|---|---|---|---|---|---|---|
| AP12 | BundleSDF (SAM2) | 0.631 | 0.706 | 0.876 | 0.276 | 1.297 | 34% | 29113 |
| AP12 | TSDF training views | 0.629 | 0.910 | 0.455 | 0.622 | 3.536 | 3% | 31529 |
| AP12 | TSDF virtual views | 0.939 | 0.798 | 1.006 | 0.620 | 0.504 | 59% | 86395 |
| AP12 | **Poisson** | **0.571** | **0.433** | 0.455 | 0.358 | **0.563** | **61%** | 93667 |
| mustard0 | BundleSDF (SAM2) | 0.533 | 0.533 | 0.510 | 0.281 | 1.082 | 34% | 26662 |
| mustard0 | TSDF training views | 0.720 | 0.720 | 0.196 | 0.380 | 2.897 | 0% | 9549 |
| mustard0 | TSDF virtual views | 0.413 | 0.413 | 0.445 | 0.398 | 0.350 | 74% | 21074 |
| mustard0 | **Poisson** | **0.255** | **0.255** | 0.263 | 0.266 | **0.210** | **95%** | 42234 |

Global stage: AP12 181 views, loss 0.062→0.023, pose refinement moved views max 4.5° / 3.7 mm (mean 0.7° / 0.8 mm), 171 s;
mustard0 37 views, 0.062→0.055, max 0.9° / 0.8 mm, 105 s (both incl. extraction).

Reading:
- **Poisson is the best extraction on both sequences under every protocol**, and it is the only one that beats
  BundleSDF on P1 too (AP12 0.571 vs 0.631; mustard0 0.255 vs 0.533). On the full model (P2) the gap is larger
  (0.433 vs 0.706; 0.255 vs 0.533) because the unseen region is filled: GT→pred unseen 0.56 / 0.21 cm vs 1.30 / 1.08,
  and 61 % / 95 % of the never-seen GT surface lies within 5 mm of our mesh vs 34 % for BundleSDF.
- **TSDF from training views** reproduces the BundleSDF-style result (seen surfaces only; AP12 P1 0.629 ≈ BundleSDF)
  and shows no completion (unseen 3.5 / 2.9 cm, ≤ 3 % within 5 mm) — as expected, it cannot see the prior.
- **TSDF from virtual views** does complete the object (unseen 0.50 / 0.35 cm) but is worse than Poisson on the seen
  region and P1 (rendered-depth fusion from many directions is noisier: expected/median depth of semi-opaque surfels,
  silhouette pixels).
- Seen-region accuracy: BundleSDF is still better on the seen part (0.28 vs 0.36 / 0.27 cm); our advantage is completeness.
- Caveats: single sequence each, one setting; SUSPECT included; oversized Gaussians hidden (a training-side issue);
  YCB unseen labels ignore hand/gripper occlusion.

### Stage-1 follow-up checks (2026-09-11)
**Seen-region breakdown** (GT samples in the seen region → nearest predicted surface; pred → nearest GT):

| seq | mesh | seen GT ≤5 mm | GT(seen)→pred median / p90 mm | pred→GT median / p90 mm | pred pts >1 cm from GT | ICP applied (deg / mm) |
|---|---|---|---|---|---|---|
| AP12 | BundleSDF | 87.4% | 2.3 / 5.3 | 4.2 / 28.4 | 24.7% | 6.1 / 2.0 |
| AP12 | TSDF training views | 59.6% | 3.9 / 14.2 | 3.3 / 9.9 | 9.6% | 17.6 / 14.3 |
| AP12 | TSDF virtual views | 38.6% | 5.9 / 10.6 | 8.6 / 17.7 | 44.2% | 10.9 / 6.1 |
| AP12 | Poisson | 78.1% | 2.4 / 8.0 | 3.1 / 10.1 | 10.3% | 12.5 / 9.9 |
| mustard0 | BundleSDF | 88.5% | 2.4 / 5.2 | 3.8 / 10.8 | 11.2% | 4.3 / 14.4 |
| mustard0 | TSDF training views | 75.8% | 1.7 / 10.1 | 1.3 / 4.6 | 0.4% | 8.6 / 5.6 |
| mustard0 | TSDF virtual views | 69.9% | 3.8 / 7.0 | 4.1 / 8.2 | 3.4% | 9.1 / 3.6 |
| mustard0 | Poisson | 88.1% | 2.2 / 5.2 | 2.2 / 5.2 | 0.0% | 10.6 / 4.6 |

- Seen-surface accuracy of Poisson ≈ BundleSDF (median 2.2–2.4 mm both); BundleSDF covers the seen region slightly
  better on AP12 (87 vs 78 % within 5 mm) but carries spurious geometry (24.7 % of its points > 1 cm from GT, p90 28 mm),
  which is what costs it on P1. Our P1/P2 lead = completion of the unseen region + fewer outliers, not better seen-surface fit.
- Why the smooth-looking TSDF meshes score worse: coverage, not smoothness. TSDF from training views leaves the seen
  region 40 % uncovered on AP12 (alpha/mask thresholds at silhouettes, thin handle, 2 cm truncation) and Chamfer
  charges every uncovered GT point its distance to the nearest surface; visual smoothness is not measured.
- ICP before scoring is part of the original protocol; our meshes need 9–18° / 4–14 mm of it (tracker drift of the
  v1 runs: AP12 keyframe rotation error mean 16.9°), BundleSDF's 4–6°.

**Pose refinement in the global stage vs GT** (180 / 37 keyframes, v1 deltas, 2000 steps):
AP12 rotation error 16.87 → 16.89° (39 % keyframes improved, 52 % worsened; moved 0.75° mean, 4.5° max);
mustard0 4.92 → 4.99° (24 % / 54 %; moved 0.28° mean). The global stage does not move poses toward GT — consistent
with ⑤-4 (the map the poses are refined against has absorbed the tracker error).

**Global-stage loss**: identical to the online update — `GaussianRunner.train()` with 0.8·L1 + 0.2·DSSIM + depth Huber
(+ configured distortion/normal terms), one view per step, map lr = `update_lr_scale`, v1 pose deltas (2 cm / 20°
clamps, lr 0.01 decaying ×0.1 over the call); only the step count (2000 in one call) and the absence of new keyframes
differ. The original's global stage instead re-creates the SDF from scratch with finer settings (2000 steps × 2048
rays from all frames, rgb weight 100, finest_res 256) — that "fresh map" variant is stage 2 (a).

## Stage 2 (2026-09-11): init × pose refinement × steps, AP12 + mustard0 (`logs/global_stage2_20260911/summary_table.md`)
24 configurations = init {online-continued, rebuilt from SAM3D prior + all keyframes, rebuilt from RGB-D only} ×
pose refinement {on, off} × steps {2000, 10000}; Poisson input opacity ≥ 0.1 (changed from 0.5 after a check on the
mustard0 global map: P1 0.247 → 0.232, seen-region 0.245 → 0.213 cm; most observed Gaussians sit below opacity 0.5).
Every configuration: 3 meshes × 3 protocols + keyframe poses vs GT. Full table in the log dir; Poisson rows:

| seq | init | pose | steps | P1 | P2 | seen | unseen | unseen ≤5 mm | keyframe rot err before→after | improved/worsened |
|---|---|---|---|---|---|---|---|---|---|---|
| AP12 | online | on | 2000 | 0.524 | 0.431 | 0.270 | 0.534 | 56% | 16.87→16.89° | 41%/51% |
| AP12 | online | off | 2000 | **0.521** | 0.433 | 0.263 | 0.539 | 56% | – | – |
| AP12 | online | on/off | 10000 | 0.523/0.526 | 0.434/0.436 | 0.270/0.266 | 0.539 | 56% | 16.87→16.94° | 39%/54% |
| AP12 | prior | on | 2000 | 0.797 | 0.588 | 0.810 | 0.999 | 47% | 16.87→16.32° | 58%/41% |
| AP12 | prior | on | 10000 | 0.858 | 0.640 | 0.924 | 1.008 | 45% | 16.87→**15.67°** | **63%/36%** |
| AP12 | prior | off | 2000/10000 | 0.804/0.858 | 0.600/0.638 | 0.82/0.93 | 1.0 | 46–47% | – | – |
| AP12 | fresh | on | 2000/10000 | 0.566/0.514 | 0.731/0.678 | 0.349/0.301 | 2.08/2.00 | 16–17% | 16.87→17.65/18.30° | 26%/69% |
| AP12 | fresh | off | 2000/10000 | 0.582/0.527 | 0.719/0.652 | 0.334/0.270 | 1.83/1.67 | 19–21% | – | – |
| mustard0 | online | on/off | 2000 | 0.232 | 0.232 | 0.214 | 0.214 | 94% | 4.92→5.00° | 22%/49% |
| mustard0 | online | on/off | 10000 | 0.231 | 0.231 | 0.210 | 0.214 | 94% | 4.92→5.02° | 32%/62% |
| mustard0 | prior | on/off | 2000 | 0.280/0.276 | same | 0.315/0.296 | 0.204 | 95–96% | 4.92→5.28° | 32%/62% |
| mustard0 | prior | on/off | 10000 | 0.263/0.265 | same | 0.284 | 0.200 | 96% | 4.92→4.97° | 49%/46% |
| mustard0 | fresh | on/off | 2000/10000 | 0.61–0.63 | same | 0.26–0.28 | 2.25–2.43 | 2–6% | 4.92→4.88/4.84° | 46%/51% |
BundleSDF reference: AP12 P1 0.631 / P2 0.706 / unseen 1.297 (34%); mustard0 0.533 / 0.533 / 1.082 (34%).
TSDF (training views) and TSDF (virtual) rows follow the same ordering; TSDF-train stays ≈ BundleSDF on AP12 (0.62–0.63)
and never completes the object.

Reading:
- **Online-continued map is the best input** on both sequences and under all protocols (AP12 P1 0.52, P2 0.43;
  mustard0 0.23). Rebuilding from the prior in one 2000–10000-step call gives a much worse seen surface on AP12
  (0.81–0.93 cm vs 0.27) — the online map has ~90 k cumulative steps behind it; the rebuilt one has not converged.
- **The prior is what completes the object**: without it (fresh) the seen surface is as good (0.27–0.35) but the
  unseen region stays a hole (1.7–2.4 cm, ≤ 21 % within 5 mm; BundleSDF 34 %).
- **Pose refinement in the global stage is neutral for the mesh** (on vs off differs by ≤ 0.005 cm everywhere) and
  does not move poses toward GT on the online map (AP12 16.87 → 16.89–16.94°, mustard0 4.92 → 5.0°). It does move
  them toward GT on the prior-rebuilt map (AP12 16.87 → 15.67°, 63 % of keyframes improved) — the only
  configuration where the refinement works, and it is the map that is least shaped by the tracker's poses. This
  supports the ⑤-4 reading (contaminated map) and the "prior as anchor" idea (§ MILESTONE5_FEEDBACK_RESULTS §9).
- **10000 steps buys nothing** for the online map (differences ≤ 0.005 cm); it slightly helps the rebuilt maps.
- Decision proposal: **online-continued, pose refinement off, 2000 steps, Poisson (opacity ≥ 0.1) as the primary
  mesh, TSDF-from-training-views kept as the like-for-like control.**

### SUSPECT check on the confirmed setting (online-continued, pose off, 2000; Poisson input)
AP12: included P1 0.521 / P2 0.433 / unseen 0.539 vs excluded 0.519 / 0.431 / 0.542 (points 154 164 vs 153 671);
mustard0: 0.232 / 0.232 / 0.214 both (26 541 vs 26 524 points). No measurable effect (≤ 0.003 cm) — SUSPECT Gaussians are
few (< 0.4 %). Decision: keep them (no filter).

## Confirmed setting (2026-09-11)
Input = online map continued; pose refinement off; 2000 steps; primary mesh = screened Poisson (depth 9, density cut 5 %)
from Gaussians with opacity ≥ 0.1 and in-plane radius ≤ 10 mm (CONTRADICTED excluded, SUSPECT/UNSEEN included);
control mesh = TSDF from training views (median depth, voxel 2 mm, trunc 2 cm, same Gaussian hiding); post-processing =
largest component, metric object frame. To be integrated into the main pipeline (see MILESTONE ledger for the diff).

## Integration into the main pipeline (2026-09-11)
Main code: new module `gaussian_global.py` (verified functions + `run_global_refine`, defaults = confirmed setting);
`run_custom.py` (+`run_one_video_global_gaussian`, `--gs_feedback` default on = v1, `--gs_global_steps`, `--mode
global_refine` honours the backend); `run_ho3d.py` (Gaussian backend wiring mirrored from the experiment driver:
`--backend gaussian`, `--gs_*`, `--prior_root` / `--prior_*`, `resolve_prior_paths`, global stage after tracking,
`--mode global_refine`). Untouched: `gaussian_runner.py`, `bundlesdf.py`, tracker, `benchmark_ho3d.py`, configs.
Two integration bugs found by the end-to-end runs and fixed: (1) the BundleSDF entry points set torch's default tensor
type to CUDA, which breaks the runner's CPU RNG — the global stage now switches to the CPU default and restores it;
(2) `run_ho3d.py`'s `code_dir` is shadowed by `data_reader`'s star import, so the runner-config default pointed into
`BundleTrack/scripts` — the default is now computed from the file's own path. Tests: `tests/test_gaussian_global.py`
(6, CPU) + all suites = 79 OK.

End-to-end check through the official entry points (online + global automatic):

| | ADD / ADD-S | `mesh_real_world.obj` P1 | P2 | unseen GT→pred / ≤5 mm | TSDF-train P1 | reference |
|---|---|---|---|---|---|---|
| mustard0 (`run_custom.py`) | 0.659 / 0.287 | 0.233 | 0.233 | 0.214 cm / 95% | 0.735 | v1 0.650; stage 2 0.232 / 0.720 |
| AP12 (`run_ho3d.py`) | 2.455 / 0.861 | 0.521 | 0.432 | 0.502 cm / 60% | 0.652 | v1 2.379; stage 2 0.521 / 0.629 |
Global stage: mustard0 22 s (38 views), AP12 55 s (185 views, 751 k Gaussians, 2316 over-sized hidden).
Original benchmark check: the untouched baseline `benchmark_ho3d.py` (its own container) picks
`outputs/int_ho3d_20260911/AP12/final/gs/mesh_real_world.obj` and reports chamfer 0.524 cm (our P1 port: 0.521;
difference = surface-sampling noise), ADD 2.45 / ADD-S 0.86 — same numbers as our evaluators.
Note: this repo's own `benchmark_ho3d.py` fails on `Utils.trimesh_clean` (`remove_degenerate_faces` removed in
trimesh ≥ 4) in both of our containers — a pre-existing incompatibility, not touched here; `eval_mesh_cd.py` carries the
API-compatible equivalent.

## Full 22-sequence evaluation through the official pipeline (2026-09-12, `logs/fulleval_20260912`, outputs/fulleval_20260912)
Tracking + online GS (v1 feedback on) + global stage (confirmed setting), scored with ADD (online poses), P1/P2/P3 (our port) and the original benchmark_ho3d.py (HO3D; agrees with P1 within sampling noise). BundleSDF references: SAM2-mask runs in BundleSDF_baseline_outputs (MPM10 has no reference mesh). MPM12: our Chamfer failed — Poisson picked a far-away blob made of mask-leak Gaussians (10 % of the map at 8× the object radius); fix = distance filter on the Poisson input, pending approval.

```
seq                          | ADD: SDF on  tracker    OURS |  P1: BSDF    OURS |  P2: BSDF    OURS | unseen GT->pred: BSDF         OURS | seen: BSDF   OURS
ho3d/SM1                     |       0.551    4.361   3.533 |     0.421   0.488 |     0.420   0.454 |          1.113 ( 25%)  0.268 ( 92%) |      0.322  0.318
ho3d/AP10                    |       0.969    4.552   5.549 |     0.494   0.588 |     0.469   0.548 |          0.459 ( 65%)  0.214 ( 93%) |      0.361  0.374
ho3d/AP11                    |       0.946    1.341   2.418 |     0.547   0.508 |     0.579   0.554 |          1.128 ( 37%)  1.664 (  8%) |      0.294  0.386
ho3d/AP12                    |       0.454    0.888   2.300 |     0.631   0.505 |     0.706   0.420 |          1.297 ( 34%)  0.539 ( 56%) |      0.276  0.252
ho3d/AP13                    |       0.641    0.969   1.406 |     0.619   0.512 |     0.584   0.455 |          0.548 ( 60%)  0.420 ( 66%) |      0.245  0.270
ho3d/AP14                    |       0.499    0.790   0.888 |     0.460   0.757 |     0.593   0.781 |          1.504 ( 29%)  1.083 ( 17%) |      0.345  0.338
ho3d/MPM10                   |       0.964    1.092   1.992 |         -   0.331 |         -   0.297 |               -        0.177 ( 96%) |          -  0.183
ho3d/MPM11                   |       0.819    0.830   1.398 |     0.413   0.309 |     0.389   0.267 |          0.098 (100%)  0.128 (100%) |      0.391  0.194
ho3d/MPM12                   |       0.777    0.457   0.772 |     0.431       - |     0.443       - |          0.830 ( 41%)       -       |      0.327      -
ho3d/MPM13                   |       1.373    2.753   3.462 |     0.437   0.702 |     0.410   0.678 |          0.283 (100%)  0.388 ( 70%) |      0.421  0.282
ho3d/MPM14                   |       0.508    0.910   0.800 |     0.423   0.288 |     0.454   0.199 |          1.032 ( 33%)  0.234 (100%) |      0.340  0.140
ho3d/SB11                    |       0.506    2.144   1.387 |     0.435   0.563 |     0.441   0.514 |          0.910 ( 37%)  0.392 ( 72%) |      0.303  0.315
ho3d/SB13                    |       0.454    0.673   0.924 |     0.452   0.559 |     0.447   0.543 |          0.801 ( 45%)  0.813 ( 45%) |      0.315  0.359
ycb/mustard0                 |       1.380    0.748   0.683 |     0.533   0.236 |     0.533   0.236 |          1.082 ( 34%)  0.214 ( 95%) |      0.281  0.215
ycb/bleach0                  |       1.801    1.764   1.338 |     0.819   0.480 |     0.819   0.480 |          1.087 ( 43%)  0.431 ( 64%) |      0.907  0.440
ycb/bleach_hard_00_03_chaitanya |       0.960    1.011   1.027 |     0.732   0.608 |     0.732   0.608 |          2.393 ( 10%)  0.552 ( 50%) |      0.585  0.335
ycb/cracker_box_reorient     |       0.785    0.756   0.823 |     0.770   0.543 |     0.770   0.543 |          1.556 ( 27%)  0.548 ( 47%) |      0.271  0.602
ycb/cracker_box_yalehand0    |       2.849    2.763   2.658 |     0.800   0.722 |     0.800   0.722 |          0.738 ( 45%)  1.629 ( 13%) |      0.876  0.380
ycb/mustard_easy_00_02       |       0.646    0.784   0.627 |     0.756   0.182 |     0.756   0.182 |          2.318 (  7%)  0.153 ( 99%) |      0.372  0.179
ycb/sugar_box1               |       0.976    0.681   0.777 |     0.713   0.286 |     0.713   0.286 |          1.680 ( 28%)  0.350 ( 86%) |      0.986  0.239
ycb/sugar_box_yalehand0      |       1.555    1.757   1.409 |     0.579   0.479 |     0.579   0.479 |          1.115 ( 34%)  0.696 ( 17%) |      0.314  0.290
ycb/tomato_soup_can_yalehand0 |       2.450    1.211   1.562 |     0.804   0.695 |     0.804   0.695 |          0.616 ( 57%)  0.568 ( 50%) |      1.072  0.485
ho3d mean over 12 done: ADD SDF-on 0.724 tracker 1.775 OURS 2.171 | P1 BSDF 0.485 OURS 0.509 | P2 BSDF 0.499 OURS 0.476 | unseen BSDF 0.834 OURS 0.527
ycb mean over 9 done: ADD SDF-on 1.489 tracker 1.275 OURS 1.212 | P1 BSDF 0.723 OURS 0.470 | P2 BSDF 0.723 OURS 0.470 | unseen BSDF 1.398 OURS 0.571
```
Original benchmark_ho3d.py chamfer (cm): chamfer_dist(cm): 0.508 chamfer_dist(cm): 0.560 chamfer_dist(cm): 0.505 chamfer_dist(cm): 0.509 chamfer_dist(cm): 0.586 chamfer_dist(cm): 0.752 chamfer_dist(cm): 0.490 chamfer_dist(cm): 0.331 chamfer_dist(cm): 0.712 chamfer_dist(cm): 0.310 chamfer_dist(cm): 0.288 chamfer_dist(cm): 0.561 

**MPM12 fix (2026-09-12, approved)**: `gaussian_surfels` now also drops Gaussians beyond 1.25 normalized units from the
object centre (`max_radius_norm`; 1.0 = 1.2 × prior radius). MPM12 re-extracted from its global checkpoint (no
retraining): P1 0.262 / P2 0.175 / unseen 0.230 cm (88 %) vs BundleSDF 0.431 / 0.443 / 0.830 (41 %).
Open map defect recorded for later: 10 % of MPM12's map were observed-lineage Gaussians appended ~8 object radii away
(depth that came through the SAM2 mask, opacity ≈ 0.5); the lifecycle only manages prior-lineage states and never
removes observed outliers — a check on observed Gaussians (distance / support) is missing.

**MPM10 reference (2026-09-12)**: BundleSDF's global stage crashes on MPM10 — reproduced with our copy of the
original SDF code (`run_ho3d.py --mode global_refine`, nerf backend, bundlesdf container, on a copy of the baseline's
online output): the process exits (rc 1, no Python traceback, faulthandler silent) inside `NerfRunner.__init__` after
the hash-grid construction, exactly where the baseline's own log stops. The baseline evaluation therefore scored MPM10
with its documented fallback, the last ONLINE SDF mesh (`1616/nerf/mesh_real_world.obj`, chamfer 0.452); the same mesh is
used here as the reference (P1 0.451 / P2 0.428 / unseen 0.386 cm 78 %). Our MPM10: P1 0.331 / P2 0.297 / unseen 0.177 (96 %).

**Complete 22-sequence table** (`logs/fulleval_20260912/summary_table.txt`; original benchmark_ho3d.py agrees with P1):
```
seq                          | ADD: SDF on  tracker    OURS |  P1: BSDF    OURS |  P2: BSDF    OURS | unseen GT->pred: BSDF         OURS | seen: BSDF   OURS
ho3d/SM1                     |       0.551    4.361   3.533 |     0.421   0.488 |     0.420   0.454 |          1.113 ( 25%)  0.268 ( 92%) |      0.322  0.318
ho3d/AP10                    |       0.969    4.552   5.549 |     0.494   0.588 |     0.469   0.548 |          0.459 ( 65%)  0.214 ( 93%) |      0.361  0.374
ho3d/AP11                    |       0.946    1.341   2.418 |     0.547   0.508 |     0.579   0.554 |          1.128 ( 37%)  1.664 (  8%) |      0.294  0.386
ho3d/AP12                    |       0.454    0.888   2.300 |     0.631   0.505 |     0.706   0.420 |          1.297 ( 34%)  0.539 ( 56%) |      0.276  0.252
ho3d/AP13                    |       0.641    0.969   1.406 |     0.619   0.512 |     0.584   0.455 |          0.548 ( 60%)  0.420 ( 66%) |      0.245  0.270
ho3d/AP14                    |       0.499    0.790   0.888 |     0.460   0.757 |     0.593   0.781 |          1.504 ( 29%)  1.083 ( 17%) |      0.345  0.338
ho3d/MPM10                   |       0.964    1.092   1.992 |     0.451   0.331 |     0.428   0.297 |          0.386 ( 78%)  0.177 ( 96%) |      0.381  0.183
ho3d/MPM11                   |       0.819    0.830   1.398 |     0.413   0.309 |     0.389   0.267 |          0.098 (100%)  0.128 (100%) |      0.391  0.194
ho3d/MPM12                   |       0.777    0.457   0.772 |     0.431   0.262 |     0.443   0.175 |          0.830 ( 41%)  0.230 ( 88%) |      0.327  0.149
ho3d/MPM13                   |       1.373    2.753   3.462 |     0.437   0.702 |     0.410   0.678 |          0.283 (100%)  0.388 ( 70%) |      0.421  0.282
ho3d/MPM14                   |       0.508    0.910   0.800 |     0.423   0.288 |     0.454   0.199 |          1.032 ( 33%)  0.234 (100%) |      0.340  0.140
ho3d/SB11                    |       0.506    2.144   1.387 |     0.435   0.563 |     0.441   0.514 |          0.910 ( 37%)  0.392 ( 72%) |      0.303  0.315
ho3d/SB13                    |       0.454    0.673   0.924 |     0.452   0.559 |     0.447   0.543 |          0.801 ( 45%)  0.813 ( 45%) |      0.315  0.359
ycb/mustard0                 |       1.380    0.748   0.683 |     0.533   0.236 |     0.533   0.236 |          1.082 ( 34%)  0.214 ( 95%) |      0.281  0.215
ycb/bleach0                  |       1.801    1.764   1.338 |     0.819   0.480 |     0.819   0.480 |          1.087 ( 43%)  0.431 ( 64%) |      0.907  0.440
ycb/bleach_hard_00_03_chaitanya |       0.960    1.011   1.027 |     0.732   0.608 |     0.732   0.608 |          2.393 ( 10%)  0.552 ( 50%) |      0.585  0.335
ycb/cracker_box_reorient     |       0.785    0.756   0.823 |     0.770   0.543 |     0.770   0.543 |          1.556 ( 27%)  0.548 ( 47%) |      0.271  0.602
ycb/cracker_box_yalehand0    |       2.849    2.763   2.658 |     0.800   0.722 |     0.800   0.722 |          0.738 ( 45%)  1.629 ( 13%) |      0.876  0.380
ycb/mustard_easy_00_02       |       0.646    0.784   0.627 |     0.756   0.182 |     0.756   0.182 |          2.318 (  7%)  0.153 ( 99%) |      0.372  0.179
ycb/sugar_box1               |       0.976    0.681   0.777 |     0.713   0.286 |     0.713   0.286 |          1.680 ( 28%)  0.350 ( 86%) |      0.986  0.239
ycb/sugar_box_yalehand0      |       1.555    1.757   1.409 |     0.579   0.479 |     0.579   0.479 |          1.115 ( 34%)  0.696 ( 17%) |      0.314  0.290
ycb/tomato_soup_can_yalehand0 |       2.450    1.211   1.562 |     0.804   0.695 |     0.804   0.695 |          0.616 ( 57%)  0.568 ( 50%) |      1.072  0.485
ho3d mean over 13 done: ADD SDF-on 0.728 tracker 1.674 OURS 2.064 | P1 BSDF 0.478 OURS 0.490 | P2 BSDF 0.490 OURS 0.453 | unseen BSDF 0.799 OURS 0.504
ycb mean over 9 done: ADD SDF-on 1.489 tracker 1.275 OURS 1.212 | P1 BSDF 0.723 OURS 0.470 | P2 BSDF 0.723 OURS 0.470 | unseen BSDF 1.398 OURS 0.571
```
