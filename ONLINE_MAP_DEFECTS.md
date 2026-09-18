# Online map defects (B-track) — results record

Started 2026-09-12 after the global-refinement stage was completed (`GLOBAL_REFINE_RESULTS.md`). Order agreed with the
user: B6 (low opacity) → B4 (scale blow-up) → B5 (mask-leak observed Gaussians), then D (milestone ⑤ resumption).
Baseline for every comparison = `outputs/fulleval_20260912` (official pipeline, v1 feedback on, `initial_opacity` 0.1,
`prior_lifecycle.prior_opacity` 0.9, referred to as **o01_p09**).

## 0. The three defects (evidence from the baseline checkpoints, `experiments/eval_map_quality.py`)

| defect | what | baseline evidence (mustard0 / AP12 / MPM12) |
|---|---|---|
| B6 | most observed-lineage Gaussians end with opacity < 0.1; rendered median depth is worse than the centres | observed opacity < 0.1: 79 % / 73 % / 76 % (prior lineage 16 / 21 / 27 %); rendered depth residual 2.82 / 4.31 / 3.65 mm vs centre residual 2.05 / 1.73 / 1.84 mm |
| B4 | prior-lineage Gaussians grow to 1–29 cm in-plane radius (hidden at extraction) | radius > 10 mm: 465 / 2383 / 523 |
| B5 | observed Gaussians appended outside the object (mask leak), never judged by the lifecycle because they never project inside an eroded mask | normalized distance > 1.25: 0 / 16 962 / 21 818 (21 of the 22 fulleval sequences have some; AP14 52 874, AP10 13 221, MPM13 12 204) |

## 1. Map-quality evaluator (`experiments/eval_map_quality.py`, experiment code only)

Reads `final/gs/checkpoint_global.pt` and reports: opacity histogram per lineage, radius > 10 mm and normalized
distance > 1.25 counts, lifecycle state counts, rendered 2DGS median depth vs observed depth and observed point →
nearest centre residuals over 15 training views, and a GT comparison of the centres (opacity ≥ 0.1, radius ≤ 10 mm,
distance ≤ 1.25; benchmark alignment = first online pose + ICP 2 cm onto the visible GT): centres → GT model
(accuracy), GT → centres split into seen / unseen regions (completeness, share within 5 mm). JSONs:
`logs/b6_opacity_20260912/map_*.json`.

## 2. B6 stage 1 — opacity-initialisation sweep (config only, 2026-09-12)

Protocol: official pipeline (`run_custom.py` / `run_ho3d.py --gs_runner_config`) with copies of
`config_gs_2dgs_1mm_lifecycle.yml` that only change `initial_opacity` (observed lineage, o) and
`prior_lifecycle.prior_opacity` (p); global stage included; ADD, Chamfer P1/P2/P3 on `mesh_real_world.obj`, and the map
evaluator. Single runs. Configs: o ∈ {0.1, 0.5, 0.9} × p ∈ {0.9, 0.5}. Scripts and logs: `logs/b6_opacity_20260912/`
(`run_one.sh`, `chain_gpu*.sh`, `progress`, `add_*`, `cd_*`, `map_*`); outputs `outputs/b6_opacity_20260912/<cfg>/`.

Columns: ADD (cm); P1/P2 Chamfer (cm); mesh unseen = GT→mesh distance on the unseen region (share within 5 mm); mesh
seen; N = Gaussians after global; obs/prior opac<0.1 (%); big = radius > 10 mm; far = distance > 1.25; render / centre
residual (mm, median); centres→GT (mm, median); GT→centres seen / unseen (mm, mean); map unseen within 5 mm (%).


### mustard0 (ycb)  — 기준 o01_p09 = fulleval_20260912

| config (obs/prior) | ADD cm | P1 | P2 | mesh unseen cm (5mm내%) | mesh seen | N gauss | obs opac<0.1 % | prior opac<0.1 % | big | far | render mm | centre mm | centres→GT mm | GT→centres seen/unseen mm | map unseen 5mm내 % |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| o01_p09 (기준) | 0.683 | 0.236 | 0.236 | 0.214 (95%) | 0.215 | 71234 | 79 | 16 | 465 | 0 | 2.82 | 2.05 | 2.64 | 1.80 / 1.98 | 98 |
| o05_p09 | 1.088 | 0.365 | 0.365 | 0.371 (72%) | 0.262 | 83261 | 58 | 16 | 818 | 0 | 2.63 | 1.53 | 4.03 | 1.92 / 2.76 | 88 |
| o09_p09 | 2.799 | 0.837 | 0.837 | 0.909 (23%) | 0.397 | 118785 | 14 | 15 | 858 | 0 | 3.74 | 2.32 | 12.20 | 2.43 / 4.67 | 63 |
| o01_p05 | 0.651 | 0.237 | 0.237 | 0.214 (95%) | 0.216 | 70784 | 80 | 20 | 459 | 0 | 2.89 | 2.10 | 2.59 | 1.79 / 1.99 | 98 |
| o05_p05 | 1.054 | 0.356 | 0.356 | 0.360 (74%) | 0.266 | 84423 | 59 | 21 | 769 | 0 | 2.79 | 1.61 | 3.86 | 1.93 / 2.62 | 90 |
| o09_p05 | 2.969 | 0.934 | 0.934 | 0.920 (23%) | 0.428 | 131696 | 12 | 18 | 893 | 0 | 3.87 | 2.24 | 14.16 | 2.49 / 4.77 | 62 |

### AP12 (ho3d)  — 기준 o01_p09 = fulleval_20260912

| config (obs/prior) | ADD cm | P1 | P2 | mesh unseen cm (5mm내%) | mesh seen | N gauss | obs opac<0.1 % | prior opac<0.1 % | big | far | render mm | centre mm | centres→GT mm | GT→centres seen/unseen mm | map unseen 5mm내 % |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| o01_p09 (기준) | 2.300 | 0.505 | 0.420 | 0.539 (56%) | 0.252 | 737009 | 73 | 21 | 2383 | 16962 | 4.31 | 1.73 | 4.11 | 2.12 / 4.80 | 67 |
| o01_p09 (기준, 현재 추출코드로 재추출) | 2.300 | 0.468 | 0.712 | 2.921 (7%) | 0.364 | (지도 동일) |
| o05_p09 | 2.766 | 0.550 | 0.766 | 2.486 (11%) | 0.362 | 1129775 | 66 | 17 | 6791 | 16986 | 4.66 | 1.90 | 5.74 | 2.01 / 8.28 | 39 |
| o09_p09 | 6.262 | 1.159 | 1.385 | 2.660 (12%) | 0.406 | 2150344 | 13 | 15 | 5162 | 22719 | 7.30 | 3.26 | 19.60 | 2.42 / 11.36 | 25 |
| o01_p05 | 2.504 | 0.525 | 0.780 | 3.056 (8%) | 0.430 | 740782 | 72 | 23 | 2330 | 18033 | 4.38 | 1.76 | 4.29 | 2.21 / 4.79 | 67 |
| o05_p05 | 2.750 | 0.559 | 0.787 | 2.617 (7%) | 0.364 | 1126728 | 67 | 19 | 7042 | 17937 | 4.68 | 1.86 | 5.91 | 2.08 / 8.12 | 38 |
| o09_p05 | 6.232 | 1.192 | 1.432 | 2.858 (10%) | 0.440 | 2167608 | 13 | 16 | 4808 | 22730 | 6.96 | 2.61 | 19.66 | 2.31 / 11.55 | 24 |

### MPM12 (ho3d)  — 기준 o01_p09 = fulleval_20260912

| config (obs/prior) | ADD cm | P1 | P2 | mesh unseen cm (5mm내%) | mesh seen | N gauss | obs opac<0.1 % | prior opac<0.1 % | big | far | render mm | centre mm | centres→GT mm | GT→centres seen/unseen mm | map unseen 5mm내 % |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| o01_p09 (기준) | 0.772 | 0.262 | 0.175 | 0.230 (88%) | 0.149 | 174531 | 76 | 27 | 523 | 21818 | 3.65 | 1.84 | 1.75 | 1.17 / 2.22 | 90 |
| o05_p09 | 1.084 | 0.294 | 0.224 | 0.267 (93%) | 0.179 | 200360 | 62 | 20 | 632 | 22343 | 3.41 | 1.75 | 2.49 | 1.12 / 1.84 | 100 |
| o09_p09 | 6.918 | 0.668 | 0.620 | 0.379 (71%) | 0.243 | 599207 | 14 | 13 | 1388 | 177779 | 3.31 | 18.68 | 12.15 | 1.89 / 9.35 | 33 |
| o01_p05 | 0.688 | 0.253 | 0.175 | 0.362 (75%) | 0.146 | 168672 | 78 | 38 | 476 | 21173 | 2.93 | 1.57 | 1.76 | 1.22 / 2.65 | 84 |
| o05_p05 | 0.995 | 0.284 | 0.216 | 0.328 (80%) | 0.177 | 192386 | 62 | 33 | 532 | 22291 | 3.58 | 1.77 | 2.20 | 1.12 / 2.06 | 96 |
| o09_p05 | 6.608 | 0.633 | 0.580 | 0.327 (77%) | 0.182 | 583825 | 15 | 17 | 1200 | 146610 | 3.03 | 14.47 | 13.23 | 1.96 / 8.98 | 32 |

### Reading

- **Observed initial opacity is the decisive knob and 0.1 (current) is best on all three sequences.** 0.5 costs
  0.3–0.5 cm ADD (mustard0 0.68 → 1.05–1.09, AP12 2.30 → 2.75–2.77, MPM12 0.77 → 1.00–1.08) and the map accuracy
  (centres→GT mustard0 2.6 → 3.9–4.0 mm, AP12 4.1 → 5.7–5.9 mm; AP12 unseen coverage 67 → 38 %). 0.9 collapses the
  tracking (ADD 2.8–6.9 cm), the Gaussian count doubles or triples (mustard0 71 k → 119–132 k, AP12 737 k → 2.15 M) and
  on MPM12 the far-away population explodes (22 k → 147–178 k).
- **Prior opacity 0.5 vs 0.9 (at observed 0.1) is on par**: mustard0 0.651 vs 0.683, MPM12 0.688 vs 0.772, AP12 2.504 vs
  2.300, all within run-to-run noise; map metrics are nearly identical (AP12 unseen coverage 67 = 67 %).
- The rendered-depth residual does not improve with a higher birth opacity (mustard0 2.82 → 2.63 → 3.74 mm): the
  low-opacity population is produced by the training, not by the initialisation. Even born at 0.5, 58–67 % of the
  observed Gaussians end below 0.1.
- Interpretation of the ADD loss (calibrated, single runs): a faint birth acts as a temporal low-pass filter. New
  Gaussians are appended at the tracker's (possibly wrong) pose; at 0.1 they barely change what the map renders until
  training raises them where they agree with the existing map, so the next pose-feedback cycle still sees the old map.
  At 0.5–0.9 the new layer immediately dominates the rendering, the map absorbs the current pose error at once, and the
  feedback loop drifts (⑤-4 mechanism). The growing Gaussian count is a symptom of that drift (points at drifted poses
  look novel and are appended).
- Conclusion: keep o01_p09. B6 has to be attacked after the fact (remove the faint observed layer) rather than by the
  initial value. Candidate for stage 2: observed-lineage opacity pruning (prior lineage stays under the lifecycle).

### Side finding: Poisson extraction depended on the far-away Gaussians (evaluation confound)

21 of the 22 `fulleval_20260912` meshes were extracted before the `max_radius_norm` distance filter (ba1cd40; only MPM12
was re-extracted with it). Without the filter the far-away Gaussians enlarge the Poisson bounding box: AP12 74 cm
instead of 28 cm, i.e. a depth-9 cell of 1.6 mm instead of 0.6 mm. Re-extracting the *same* AP12 global checkpoint with
the current code (0 steps, `logs/b6_opacity_20260912/reextract_AP12_baseline/`) gives P1 0.468, P2 0.712, unseen 2.921 cm
(7 % within 5 mm), seen 0.364 — versus 0.505 / 0.420 / 0.539 (56 %) / 0.252 in the fulleval table. The coarse octree
bridged the back of the object (better completion under P2/P3, slightly worse P1); the fine octree loses most of the
prior-completed back (most likely the 5 % density cut removes the sparse prior-only region — not yet tested).
Consequences: (1) the sweep's AP12 rows must be compared with the re-extracted baseline row, not the fulleval row;
(2) the fulleval mesh numbers of sequences with far Gaussians are not reproducible with the current code; (3) the
Poisson resolution should be made independent of outliers (explicit scale, or depth relative to object size) and the
density cut re-examined. Proposed, not done: re-extract the 21 baselines with the current code; diagnostic on AP12 with
density quantile 0 and depth 8.

## 3. Layer analysis and outlier visualisation (2026-09-14, `experiments/analyze_map_layers.py`)

Per checkpoint the script writes coloured layers (OBJ with vertex colours + PLY) in the metric map frame plus the GT model
moved into that frame (`gt_aligned.obj`, benchmark alignment) and `preview.png` (three orthographic projections):
`logs/b6_opacity_20260912/viz/<cfg>_<seq>/` for o01_p09 and o05_p09 × {mustard0, AP12, MPM12}. Colours: prior
VERIFIED green, prior UNSEEN dark green, observed normal grey, observed opacity < 0.1 blue, observed supported in ≤ 1
keyframe (birth view included; `residual_count` of the lifecycle) orange, in-plane radius > 10 mm red (discs drawn
with their true ellipse), normalized distance > 1.25 magenta. Baseline (o01_p09) numbers:

| seq | prior V / U | obs normal | obs opac<0.1 | ≤1-view obs | big | far | obs→nearest prior: total / along normal / tangential (mm, median); normal-dominant; inside prior disc footprint | distance to GT, median mm: normal / low / ≤1-view / big / far |
|---|---|---|---|---|---|---|---|---|
| mustard0 | 10 480 / 9 367 | 10 878 | 40 492 | 8 457 | 465 | 0 | 3.35 / 2.22 / 1.92; 53 %; 89 % | 2.1 / 4.1 / 5.5 / 3.9 / – |
| AP12 | 11 969 / 7 297 | 192 352 | 523 556 | 53 934 | 2 383 | 16 962 | 6.92 / 5.13 / 3.17; 64 %; 79 % | 4.1 / 4.8 / 11.4 / 5.9 / 366 |
| MPM12 | 17 820 / 1 781 | 31 487 | 118 158 | 27 891 (21 794 of them far) | 523 | 21 818 | 4.52 / 3.53 / 1.93; 70 %; 84 % | 1.6 / 4.7 / 450 / 5.3 / 464 |

- All far Gaussians are observed lineage. Big ones are mostly prior lineage on mustard0 (456 / 465) but mixed on AP12
  (1 668 / 2 383) and MPM12 (296 / 523): observed Gaussians grow too.
- The observed layer lies mostly *along the prior normal* (a parallel layer 2–5 mm off the prior surface, inside the
  prior disc footprint for 79–89 % of them), not beside it. Only 26 % (mustard0) / 11 % (AP12) of the observed Gaussians
  are within 2 mm of a prior centre, so a 2 mm novelty radius alone would not stop the duplication.
- Rendering the training views without the opacity < 0.1 Gaussians: alpha inside the mask stays 1.00, holes 0 %, image
  L1 change 0.008–0.022, rendered-depth error gets slightly worse (mustard0 2.97 → 3.60, AP12 4.99 → 5.31, MPM12
  3.78 → 4.10 mm). They are not needed for coverage; they are the dimmed, worse-placed duplicates (4.1–4.8 mm from GT
  vs 1.6–4.1 mm for the opaque ones).
- Support-count histogram of the observed lineage (keyframes with support, |cos| ≥ 0.2): mustard0 0: 5 691, 1: 2 766,
  ≥6: 35 475; AP12 0: 31 710, 1: 22 224, ≥6: 626 258; MPM12 0: 19 209, 1: 8 682, ≥6: 121 488. In the previews the
  ≤1-view Gaussians form a rim along the silhouette (mask-edge points that never fall inside another eroded mask).
- Lifecycle log of the baseline runs (`gs_online/lifecycle_log.json`): appended per keyframe, first / middle / last
  third of the sequence: mustard0 1 820 / 1 665 / 1 173, AP12 7 903 / 4 051 / 2 517, MPM12 1 064 / 655 / 305 — it
  declines but never stops. Removed as CONTRADICTED over the run: mustard0 155, AP12 122 299, MPM12 3 487 (free-space
  events 775 / 2.02 M / 111 k; accepted independent conflicts 2 091 / 984 461 / 68 154). The punishment of Gaussians
  in front of the observed surface works where they project inside an eroded mask; the far ones never do.

## 4. Poisson back-face diagnostic (2026-09-14, `experiments/diag_poisson_backface.py`, AP12 baseline checkpoint)

Same surfel set as the current extraction (155 537 points, 28 cm box); Poisson at depth 9 / 8, with and without the 5 %
density cut, with and without the largest-component step; `logs/b6_opacity_20260912/poisson_diag_AP12/`.

| variant | vertices | P1 | P2 | unseen cm (within 5 mm) | seen cm |
|---|---|---|---|---|---|
| depth 9, no density cut, all components | 550 119 | 0.493 | 0.405 | 0.475 (62 %) | 0.246 |
| depth 9, no density cut, largest component | 501 314 | 0.494 | 0.406 | 0.471 (62 %) | 0.247 |
| depth 9, 5 % cut, all components | 522 613 | 0.469 | 0.712 | 2.915 (7 %) | – |
| depth 9, 5 % cut, largest component (**current**) | 465 035 | 0.469 | 0.713 | 2.921 (7 %) | 0.364 |
| depth 8, no cut, largest | 209 607 | 0.504 | 0.414 | 0.476 (61 %) | – |
| depth 8, 5 % cut, largest | 197 997 | 0.469 | 0.422 | 0.678 (42 %) | 0.277 |
| (fulleval table, old 74 cm box ≈ depth-9 cell 1.6 mm) | 139 991 | 0.505 | 0.420 | 0.539 (56 %) | 0.252 |

**Cause: the 5 % density cut.** At fine resolution the prior-completed back is the sparsest region of the Poisson
output, so the cut removes it; the largest-component step is harmless (0.9 % change). Without the cut the completion is
better than the old coarse extraction (P2 0.405 vs 0.420, unseen 62 % vs 56 %) and P1 moves 0.469 → 0.493 because P1
penalises completed surfaces (pred→GT). Proposed fix: `poisson_density_quantile` 0.05 → 0 (one config value in
`gaussian_global.DEFAULT_GLOBAL_CONFIG`), after checking mustard0 and MPM12 with the same diagnostic; then re-extract the
21 fulleval baselines so the 22-sequence table is reproducible. Approved 2026-09-14 (a)(b)(c). (a) mustard0 / MPM12 with the same diagnostic: depth 9 without the cut gives mustard0 P1 0.228 / P2 0.228 / unseen 0.208 (96 %) (current 0.236 / 0.236 / 0.214 (95 %)), MPM12 P1 0.267 / P2 0.181 / unseen 0.200 (90 %) (current 0.262 / 0.175 / 0.230 (88 %)) — no harm. (b) applied: `DEFAULT_GLOBAL_CONFIG['poisson_density_quantile']` 0.05 → 0.0 (`gaussian_global.py`, one value; `tests/test_gaussian_global.py` pass). (c) re-extraction of all 22 fulleval checkpoints (0 training steps); after the user's approval the new extraction became `final/gs/` and the old one was renamed `final/_gs_before_densitycut_fix/` (BundleSDF reference meshes moved along, duplicate checkpoint copies deleted, 7.1 GB); the new CD results are the canonical `logs/fulleval_20260912/cd_*` and `summary_table_final.txt` (old table in `_before_densitycut_fix/`); scripts `logs/fulleval_20260912/reextract/`.

### 3b. View-support replay (2026-09-14, `experiments/analyze_view_support.py`)

The "≤1-view" layer above used the lifecycle's `residual_count`, which only counts support at |cos(normal, ray)| ≥ 0.2,
so silhouette surfaces (grazing by definition) were mis-labelled. The replay recomputes the lifecycle evidence for every
Gaussian in every keyframe (same code path: eroded mask, valid depth, opacity-1 front depth, grazing-widened tolerance)
and counts judged / supported / independent-support (≥ 10° apart, greedy) / free-space / behind-occluded views.
Observed lineage, baseline o01_p09 (count; median distance to GT in mm):

| seq | observed | never judged | judged, never supported: in front (free space) | judged, never supported: hidden behind the map | supported from 1 viewpoint | supported from ≥ 2 viewpoints | median judged views |
|---|---|---|---|---|---|---|---|
| mustard0 | 51 379 | 1 386 (4.3) | 94 (10.3) | 974 (3.2) | 1 706 (3.1) | 47 219 (3.7) | 24 |
| AP12 | 717 719 | 66 (4.2) | 324 (4.5) | 23 908 (330) | 16 752 (7.1) | 676 669 (4.4) | 139 |
| MPM12 | 154 893 | 729 (569) | 40 (24.8) | 18 852 (465) | 3 731 (449) | 131 541 (3.2) | 140 |

- The far (mask-leak) Gaussians are *not* unjudged: they project inside eroded masks in many views but the object is in
  front of them along those rays, so they are "behind_occluded" (no evidence) every time, or supported only in their
  birth view (MPM12). They are never contradicted. Prior UNSEEN back faces fall in the same "hidden" class (AP12 7 366),
  so any rule must be lineage-specific.
- A rule "observed lineage needs support from ≥ 2 independent viewpoints" (= PROVISIONAL mechanism, birth view + one)
  would remove mustard0 4 160 (8 %, at 3–4 mm from GT, i.e. mostly harmless points seen once), AP12 41 050 (5.7 %,
  incl. all 16 962 far), MPM12 23 352 (15 %, incl. all 21 818 far).
- 94 % (mustard0 92 %) of the observed Gaussians are supported from ≥ 2 viewpoints and were judged in a median of
  24–140 keyframes: the map is overwhelmingly re-observations of the same surfaces — the duplication problem, not an
  outlier problem.
- Cross sections (`xsection_layers.png`, 3 mm slabs through the GT centre): mustard0 shows a 2–8 mm band of faint
  observed Gaussians outside the observed surface on one side; AP12 shows bands up to 1–2 cm and the prior offset by
  ~1 cm from the GT on the top (prior placement error).
- Milestone-5 GT-map probe (`logs/gtmap_probe_20260911`): the same keyframes appended with GT poses give AP12
  454 527 Gaussians vs 705 429 with tracker poses (−36 %), SB13 349 479 vs 374 552 (−7 %). Tracker error explains part
  of the growth; the rest is that the 1 mm centre-distance novelty is below the depth noise, so every re-observation
  of a surface looks novel.

## 5. Re-extraction of the 22 fulleval meshes with density cut 0 (2026-09-14, step (c))

From the unchanged `final/gs/checkpoint_global.pt` of each run (0 training steps); now `final/gs/` (old extraction archived as `final/_gs_before_densitycut_fix/`); CD JSONs `logs/fulleval_20260912/cd_*` (old-extraction JSONs regenerated into `_before_densitycut_fix/`); old = the fulleval table (21 meshes extracted before the far filter, i.e. with an outlier-enlarged Poisson box; MPM12 with the filter and the 5 % cut); BundleSDF = SAM2-mask reference meshes. Duplicate checkpoint copies in the new dirs: 7.0 GB (cleanup pending approval).

### ho3d  (cm; unseen = GT→mesh on the never-seen region, (%) = share within 5 mm)

| seq | BundleSDF P1 | BundleSDF P2 | BundleSDF unseen | old P1 | old P2 | old unseen | old seen | **new P1** | **new P2** | **new unseen** | new seen |
|---|---|---|---|---|---|---|---|---|---|---|---|
| AP10 | 0.494 | 0.469 | 0.459 (65%) | 0.588 | 0.548 | 0.214 (93%) | 0.374 | **0.570** | **0.530** | **0.194 (97%)** | 0.372 |
| AP11 | 0.547 | 0.579 | 1.128 (37%) | 0.508 | 0.554 | 1.664 (8%) | 0.386 | **0.586** | **0.514** | **0.366 (80%)** | 0.359 |
| AP12 | 0.631 | 0.706 | 1.297 (34%) | 0.505 | 0.420 | 0.539 (56%) | 0.252 | **0.493** | **0.404** | **0.474 (62%)** | 0.246 |
| AP13 | 0.619 | 0.584 | 0.548 (60%) | 0.512 | 0.455 | 0.420 (66%) | 0.270 | **0.456** | **0.400** | **0.383 (72%)** | 0.230 |
| AP14 | 0.460 | 0.593 | 1.504 (29%) | 0.757 | 0.781 | 1.083 (17%) | 0.338 | **0.413** | **0.348** | **0.635 (30%)** | 0.219 |
| MPM10 | 0.455 | 0.431 | 0.297 (85%) | 0.331 | 0.297 | 0.177 (96%) | 0.183 | **0.345** | **0.313** | **0.176 (97%)** | 0.190 |
| MPM11 | 0.413 | 0.389 | 0.098 (100%) | 0.309 | 0.267 | 0.128 (100%) | 0.194 | **0.316** | **0.275** | **0.113 (100%)** | 0.190 |
| MPM12 | 0.431 | 0.443 | 0.830 (41%) | 0.262 | 0.175 | 0.230 (88%) | 0.149 | **0.271** | **0.181** | **0.168 (93%)** | 0.146 |
| MPM13 | 0.437 | 0.410 | 0.283 (100%) | 0.702 | 0.678 | 0.388 (70%) | 0.282 | **0.676** | **0.653** | **0.414 (64%)** | 0.288 |
| MPM14 | 0.423 | 0.454 | 1.032 (33%) | 0.288 | 0.199 | 0.234 (100%) | 0.140 | **0.275** | **0.174** | **0.206 (100%)** | 0.131 |
| SB11 | 0.435 | 0.441 | 0.910 (37%) | 0.563 | 0.514 | 0.392 (72%) | 0.315 | **0.582** | **0.537** | **0.423 (67%)** | 0.297 |
| SB13 | 0.452 | 0.447 | 0.801 (45%) | 0.559 | 0.543 | 0.813 (45%) | 0.359 | **0.501** | **0.480** | **0.688 (52%)** | 0.269 |
| SM1 | 0.421 | 0.420 | 1.113 (25%) | 0.488 | 0.454 | 0.268 (92%) | 0.318 | **0.483** | **0.448** | **0.246 (91%)** | 0.311 |
| **mean** | 0.478 | 0.490 | 0.792 (53%) | 0.490 | 0.453 | 0.504 (70%) | 0.274 | 0.459 | 0.404 | 0.345 (77%) | 0.250 |
  (n: ref 13, old 13, new 13)

### ycb  (cm; unseen = GT→mesh on the never-seen region, (%) = share within 5 mm)

| seq | BundleSDF P1 | BundleSDF P2 | BundleSDF unseen | old P1 | old P2 | old unseen | old seen | **new P1** | **new P2** | **new unseen** | new seen |
|---|---|---|---|---|---|---|---|---|---|---|---|
| bleach0 | 0.819 | 0.819 | 1.087 (43%) | 0.480 | 0.480 | 0.431 (64%) | 0.440 | **0.483** | **0.483** | **0.442 (63%)** | 0.416 |
| bleach_hard_00_03_chaitanya | 0.732 | 0.732 | 2.393 (10%) | 0.608 | 0.608 | 0.552 (50%) | 0.335 | **0.620** | **0.620** | **0.512 (55%)** | 0.321 |
| cracker_box_reorient | 0.770 | 0.770 | 1.556 (27%) | 0.543 | 0.543 | 0.548 (47%) | 0.602 | **0.529** | **0.529** | **0.552 (48%)** | 0.509 |
| cracker_box_yalehand0 | 0.800 | 0.800 | 0.738 (45%) | 0.722 | 0.722 | 1.629 (13%) | 0.380 | **0.729** | **0.729** | **1.581 (14%)** | 0.368 |
| mustard0 | 0.533 | 0.533 | 1.082 (34%) | 0.236 | 0.236 | 0.214 (95%) | 0.215 | **0.228** | **0.228** | **0.207 (96%)** | 0.197 |
| mustard_easy_00_02 | 0.756 | 0.756 | 2.318 (7%) | 0.182 | 0.182 | 0.153 (99%) | 0.179 | **0.176** | **0.176** | **0.155 (99%)** | 0.149 |
| sugar_box1 | 0.713 | 0.713 | 1.680 (28%) | 0.286 | 0.286 | 0.350 (86%) | 0.239 | **0.286** | **0.286** | **0.311 (89%)** | 0.228 |
| sugar_box_yalehand0 | 0.579 | 0.579 | 1.115 (34%) | 0.479 | 0.479 | 0.696 (17%) | 0.290 | **0.479** | **0.479** | **0.731 (16%)** | 0.266 |
| tomato_soup_can_yalehand0 | 0.804 | 0.804 | 0.616 (57%) | 0.695 | 0.695 | 0.568 (50%) | 0.485 | **0.667** | **0.667** | **0.513 (58%)** | 0.438 |
| **mean** | 0.723 | 0.723 | 1.398 (32%) | 0.470 | 0.470 | 0.571 (58%) | 0.352 | 0.466 | 0.466 | 0.556 (60%) | 0.321 |
  (n: ref 9, old 9, new 9)

## 6. Mechanism facts established on 2026-09-15 (before the next experiment batch)

- After initialisation the map is positionally frozen: the means learning rate in keyframe updates is 1.6e-5
  (normalized) decaying to 1 % within each 500-step call, so a splat can move at most ≈ 0.28 mm per update (init:
  ≈ 22 mm over 4 000 steps at ×1.0). Views are sampled uniformly per step, so a splat seen by k of N views receives
  ≈ 500·k/N gradient steps per update (the newest keyframe: ~3 steps late in AP12). Opacity moves ≈ 0.005 logit and
  log-scale ≈ 0.0005 per received step. New observations therefore enter the map only by appending at the tracker's
  pose; duplicates are resolved by opacity (B6) and scale (B4), and each appended layer explains its keyframe at its
  (possibly wrong) pose — the concrete path by which the map absorbs tracker error (⑤-4).
- Colour dominates the position gradient 35–46× over the depth Huber term on the final maps (mustard0: colour loss
  0.046 → means-grad norm 1.24e-1; depth loss 9.7e-4 → 2.7e-3; AP12 35×), because L1 has a constant-magnitude
  gradient while Huber's is proportional to a 3–5 mm residual.
- `refresh_view_poses` is inert in the current pipeline: keyframes consumed by the backend are `_nerfed` and fixed in
  BA (Bundler.cpp:919); their poses only change through our write-back (bundlesdf.py:683–686).
- The first 5 keyframes' observations are never appended (initialisation trains the prior only; kept by decision).
- Candidate points are built from the un-eroded mask while the lifecycle judges inside a 2-px-eroded mask (no
  deliberate reason; explains the never-judged rim points).
- Prior surfels deeper than 1 cm behind a mapped surface stay UNSEEN (hidden = no evidence) for ever and are
  included in the extraction; hidden prior distance to GT median 3.0 mm (mustard0) / 2.2 (MPM12) = back faces, but
  AP12 6.4 mm (p90 17.7 mm) = interior/offset prior.
- Sibling BundleGS and original BundleSDF train 500 steps per batch including the first; our 4 000 initial steps were
  never validated.

## 7. Experiment batch 2026-09-15 (approved: F, initial-step sweep, update-step 350)

Rationale: the layer stacking above is both the map defect and the ⑤ bottleneck, so the experiment that matters is
whether absorbing re-observations into existing surfels (instead of appending) restores a usable pose gradient.
- **F — re-observation fusion** (experiment copy `experiments/online_variants/`, main code untouched, switched by
  `GS_ONLINE_VARIANTS=fusion`): each candidate point (1 mm voxel) is projected into its keyframe; among the map splats
  (non-CONTRADICTED, opacity ≥ 0.05) projected to the same pixel, the one closest to the observed depth is taken; if
  its ray depth is within 5 mm of the candidate, the candidate is fused — the splat centre moves to the count-weighted
  mean c ← (n·c + p̄)/(n+1) (p̄ = mean of this keyframe's candidates hitting the splat), n ← min(n+1, 20); otherwise the
  candidate goes to the usual 1 mm novelty test. Prior surfels start with n = 2, appended splats with n = 1. Normals,
  colours, scales, opacities untouched. Expected: no new layers for re-observations within 5 mm; a splat seen n times
  moves only 1/(n+1) toward a drifted observation, so the map resists tracker drift and the pose deltas get a
  gradient; Gaussian count drops sharply; B6/B4 shrink at the source.
- **Initial steps** 500 / 1000 / 2000 vs 4000 (`--gs_initial_steps`, fusion off).
- **Update steps** 350 vs 500 (`--gs_update_steps`, init 4000, fusion off).
- Sequences: mustard0, cracker_box_yalehand0, AP12, MPM12. Runs per sequence: control (copy, all off) ×1, F ×2,
  init500/1000/2000 ×1 each, upd350 ×1 = 28 runs, two GPU chains. Scoring per run: ADD, P1/P2/P3 (current
  extraction), map evaluator, layer analysis, view-support replay (cross sections). Judge: (1) HO3D ADD vs control
  (AP12 tracker-alone 0.89), (2) Gaussian count / layer thickness / rendered-depth error / faint share, (3) P1/P2/P3
  and completion retained. Scripts and logs: `logs/exp_fusion_20260915/` (`run_one.sh`, `chain_gpu*.sh`,
  `progress`); outputs `outputs/exp_fusion_20260915/<cfg>/<ds>/<seq>`.
- Wiring check (mustard0, 2026-09-15, both runs through the copies): control ADD 0.639 / P1 0.230 / unseen 96 % /
  72 806 Gaussians (fulleval reference 0.683 / 0.228 / 96 % / 71 234 → the copy reproduces the pipeline within noise);
  fusion with strict same-pixel association ADD 0.732 / P1 0.239 / 49 552 Gaussians, 65 % of candidates fused, mean
  move 0.41 mm. Diagnosis on that map: with same-pixel matching 34–41 % of re-observed pixels find no splat within
  5 mm although one projects to a neighbouring pixel (1 mm splats vs ~1.5 px spacing); a 3×3 neighbourhood makes
  96–97 % fusable (median |Δz| 0.5–0.7 mm). The batch therefore runs fusion with the 3×3 association (closest ray
  depth among the nine pixels); the strict run is kept as cfg `fusion1px`. Keyframe updates always carried exactly one
  frame (no GS lag batching) in these runs and in the fulleval AP12/MPM12 runs.

### Results (2026-09-15, 29 runs incl. the two wiring runs; all single runs except F ×2; logs/exp_fusion_20260915/, summary script in the session scratchpad → this table)


### mustard0 (ycb) — 참조: SDF-on ADD 1.38, 트래커 단독 0.748, fulleval 우리 0.683 / P1 BundleSDF 0.533, fulleval 우리 0.228

| 설정 | ADD | P1 | P2 | 메시 unseen (5mm내) | seen | 가우시안 수 | 관측 opac<0.1 | big | far | 렌더 깊이 mm | 중심→GT mm | 지도 unseen 5mm내 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 대조(복사본, 4000/500) | 0.639 | 0.230 | 0.230 | 0.207 (96 %) | 0.198 | 72806 | 80 % | 501 | 0 | 2.95 | 2.59 | 98 % |
| F 융합 3×3 | 0.728 | 0.244 | 0.244 | 0.199 (97 %) | 0.231 | 26571 | 82 % | 523 | 0 | 2.93 | 2.97 | 99 % |
| F 융합 3×3 (반복) | 0.670 | 0.244 | 0.244 | 0.200 (97 %) | 0.230 | 25915 | 83 % | 482 | 0 | 2.64 | 2.94 | 99 % |
| F 융합 1px(배선확인) | 0.732 | 0.239 | 0.239 | 0.207 (97 %) | 0.207 | 49552 | 81 % | 537 | 0 | 2.67 | 2.93 | 99 % |
| 초기화 500 | 0.603 | 0.231 | 0.231 | 0.209 (96 %) | 0.200 | 71120 | 77 % | 131 | 0 | 2.86 | 2.55 | 99 % |
| 초기화 1000 | 0.664 | 0.232 | 0.232 | 0.214 (95 %) | 0.195 | 72154 | 78 % | 214 | 0 | 2.91 | 2.65 | 99 % |
| 초기화 2000 | 0.666 | 0.230 | 0.230 | 0.209 (95 %) | 0.199 | 71556 | 80 % | 349 | 0 | 2.96 | 2.65 | 99 % |
| 갱신 350 | 0.673 | 0.232 | 0.232 | 0.207 (96 %) | 0.201 | 70443 | 80 % | 428 | 0 | 2.74 | 2.64 | 98 % |

### cracker_box_yalehand0 (ycb) — 참조: SDF-on ADD 2.849, 트래커 단독 2.763, fulleval 우리 2.658 / P1 BundleSDF 0.8, fulleval 우리 0.729

| 설정 | ADD | P1 | P2 | 메시 unseen (5mm내) | seen | 가우시안 수 | 관측 opac<0.1 | big | far | 렌더 깊이 mm | 중심→GT mm | 지도 unseen 5mm내 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 대조(복사본, 4000/500) | 2.654 | 0.699 | 0.699 | 1.542 (16 %) | 0.348 | 125021 | 65 % | 907 | 2762 | 3.07 | 4.80 | 25 % |
| F 융합 3×3 | 2.652 | 0.787 | 0.787 | 1.546 (15 %) | 0.402 | 41330 | 43 % | 962 | 2560 | 3.30 | 5.89 | 21 % |
| F 융합 3×3 (반복) | 2.656 | 0.770 | 0.770 | 1.527 (17 %) | 0.403 | 41064 | 44 % | 957 | 2568 | 3.59 | 5.70 | 22 % |
| 초기화 500 | 2.692 | 0.719 | 0.719 | 1.537 (16 %) | 0.356 | 121226 | 63 % | 401 | 2690 | 2.30 | 4.88 | 21 % |
| 초기화 1000 | 2.680 | 0.723 | 0.723 | 1.555 (15 %) | 0.362 | 125408 | 65 % | 513 | 2746 | 2.74 | 4.88 | 23 % |
| 초기화 2000 | 2.654 | 0.738 | 0.738 | 1.556 (14 %) | 0.364 | 122757 | 65 % | 659 | 2661 | 2.82 | 5.01 | 21 % |
| 갱신 350 | 2.657 | 0.733 | 0.733 | 1.565 (14 %) | 0.358 | 123274 | 63 % | 843 | 2745 | 2.62 | 5.03 | 22 % |

### AP12 (ho3d) — 참조: SDF-on ADD 0.454, 트래커 단독 0.888, fulleval 우리 2.3 / P1 BundleSDF 0.631, fulleval 우리 0.493

| 설정 | ADD | P1 | P2 | 메시 unseen (5mm내) | seen | 가우시안 수 | 관측 opac<0.1 | big | far | 렌더 깊이 mm | 중심→GT mm | 지도 unseen 5mm내 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 대조(복사본, 4000/500) | 2.478 | 0.519 | 0.427 | 0.464 (64 %) | 0.258 | 745040 | 71 % | 2536 | 19922 | 4.20 | 4.28 | 68 % |
| F 융합 3×3 | 2.243 | 0.544 | 0.434 | 0.454 (64 %) | 0.264 | 370441 | 81 % | 2920 | 18818 | 3.99 | 4.05 | 68 % |
| F 융합 3×3 (반복) | 2.526 | 0.543 | 0.432 | 0.457 (62 %) | 0.271 | 380595 | 82 % | 3029 | 18195 | 4.42 | 4.29 | 68 % |
| 초기화 500 | 2.216 | 0.453 | 0.356 | 0.412 (71 %) | 0.224 | 707937 | 69 % | 2083 | 17639 | 3.88 | 3.48 | 73 % |
| 초기화 1000 | 2.322 | 0.483 | 0.387 | 0.446 (66 %) | 0.245 | 724952 | 70 % | 2208 | 17461 | 4.36 | 3.72 | 67 % |
| 초기화 2000 | 2.361 | 0.509 | 0.418 | 0.473 (62 %) | 0.258 | 729565 | 70 % | 2261 | 18274 | 4.44 | 4.13 | 66 % |
| 갱신 350 | 2.434 | 0.509 | 0.418 | 0.462 (63 %) | 0.257 | 706346 | 72 % | 1787 | 18990 | 4.54 | 4.07 | 69 % |

### MPM12 (ho3d) — 참조: SDF-on ADD 0.777, 트래커 단독 0.457, fulleval 우리 0.772 / P1 BundleSDF 0.431, fulleval 우리 0.271

| 설정 | ADD | P1 | P2 | 메시 unseen (5mm내) | seen | 가우시안 수 | 관측 opac<0.1 | big | far | 렌더 깊이 mm | 중심→GT mm | 지도 unseen 5mm내 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 대조(복사본, 4000/500) | 0.785 | 0.268 | 0.180 | 0.176 (100 %) | 0.148 | 174021 | 79 % | 555 | 21454 | 3.18 | 1.84 | 100 % |
| F 융합 3×3 | 0.989 | 0.343 | 0.268 | 0.304 (95 %) | 0.223 | 102590 | 83 % | 698 | 21136 | 3.24 | 2.85 | 90 % |
| F 융합 3×3 (반복) | 0.892 | 0.284 | 0.191 | 0.160 (99 %) | 0.155 | 98211 | 82 % | 541 | 21219 | 2.96 | 2.33 | 94 % |
| 초기화 500 | 0.684 | 0.263 | 0.181 | 0.262 (85 %) | 0.126 | 170331 | 76 % | 338 | 21828 | 2.72 | 1.74 | 89 % |
| 초기화 1000 | 0.713 | 0.259 | 0.165 | 0.129 (100 %) | 0.136 | 171410 | 78 % | 435 | 21751 | 3.62 | 1.67 | 100 % |
| 초기화 2000 | 0.668 | 0.261 | 0.169 | 0.183 (92 %) | 0.133 | 165987 | 77 % | 394 | 20383 | 2.75 | 1.76 | 90 % |
| 갱신 350 | 0.681 | 0.267 | 0.181 | 0.188 (90 %) | 0.138 | 162775 | 74 % | 312 | 21139 | 2.60 | 1.84 | 88 % |

### 4시퀀스 평균 (설정별, 존재하는 시퀀스만)

| 설정 | n | ADD | P1 | P2 | 메시 unseen 5mm내 | 가우시안 수 | 관측 opac<0.1 | 렌더 깊이 mm | 중심→GT mm |
|---|---|---|---|---|---|---|---|---|---|
| 대조(복사본, 4000/500) | 4 | 1.639 | 0.429 | 0.384 | 69 % | 279222 | 74 % | 3.35 | 3.38 |
| F 융합 3×3 | 4 | 1.653 | 0.480 | 0.433 | 68 % | 135233 | 72 % | 3.37 | 3.94 |
| F 융합 3×3 (반복) | 4 | 1.686 | 0.460 | 0.409 | 69 % | 136446 | 73 % | 3.40 | 3.81 |
| F 융합 1px(배선확인) | 1 | 0.732 | 0.239 | 0.239 | 97 % | 49552 | 81 % | 2.67 | 2.93 |
| 초기화 500 | 4 | 1.548 | 0.417 | 0.372 | 67 % | 267654 | 71 % | 2.94 | 3.16 |
| 초기화 1000 | 4 | 1.595 | 0.424 | 0.377 | 69 % | 273481 | 73 % | 3.41 | 3.23 |
| 초기화 2000 | 4 | 1.587 | 0.435 | 0.389 | 66 % | 272466 | 73 % | 3.24 | 3.39 |
| 갱신 350 | 4 | 1.611 | 0.435 | 0.391 | 66 % | 265710 | 72 % | 3.12 | 3.40 |

Fusion statistics (fused share of candidates over the run / mean centre move): mustard0 95 % / 0.32–0.34 mm,
cracker 92 % / 0.46 mm, AP12 79–81 % / 0.28 mm, MPM12 93 % / 0.13 mm. The doubled mustard0 run (`fusion_doublerun`,
two waiters wrote one directory) is excluded; `fusion1px` is the strict same-pixel wiring run.

### Reading

- **F (re-observation fusion) is not adopted.** It removes most duplicates (Gaussians −50 % AP12, −41 % MPM12,
  −64 % mustard0, −67 % cracker; the cross sections show the observed layer almost gone on mustard0) but ADD does not
  improve anywhere (mustard0 0.73/0.67 vs 0.64, cracker 2.65 vs 2.65, AP12 2.24/2.53 vs 2.48, MPM12 0.99/0.89 vs
  0.79) and the geometry is slightly worse everywhere (P1 +0.01–0.09; centres→GT mustard0 2.59 → 2.9, cracker 4.8 → 5.8,
  MPM12 1.84 → 2.3–2.9 mm). The faint share stays 72–83 %: what remains still gets dimmed, so the faint population is
  not only a duplication artefact. Fusing along the ray (3×3 pixels, 5 mm band) pulls surfels toward noisy / grazing
  observations; the prior gets displaced rather than refined. Consequence for ⑤: a largely de-duplicated map did not
  give the pose feedback a better signal, so "layer stacking blocks the pose gradient" is not confirmed by this test —
  though on AP12 the 2 cm drift exceeds the band and half of the map is still appended layers, so the test is only
  complete on the well-tracked sequences.
- **Initial steps 500 is the actionable result.** Best or tied on all four sequences: ADD mustard0 0.603 (control 0.639),
  AP12 2.216 (2.478), MPM12 0.684 (0.785), cracker 2.692 (2.654, noise); P1 AP12 0.453 vs 0.519, others equal; big
  (radius > 10 mm) Gaussians drop 40–75 % (mustard0 501 → 131, cracker 907 → 401, AP12 2 536 → 2 083, MPM12 555 → 338)
  and the rendered-depth error improves (mean 2.94 vs 3.35 mm). The 4 000-step initialisation at lr ×1.0 is where the
  prior scales blow up (B4) and where the first five keyframes' pose deltas drift; 500 steps matches the sibling
  BundleGS and the original BundleSDF. The sweep is monotone on AP12 (500 < 1000 < 2000 < 4000 in ADD and P1). Proposed
  main-code change (config only): `--gs_initial_steps` default 4000 → 500 in run_custom.py / run_ho3d.py, then a
  22-sequence re-run to confirm. Pending approval.
- **Update steps 350** is neutral (ADD 1.611 vs 1.639, P1 0.435 vs 0.429 over the four sequences); it saves 30 % of the
  per-keyframe GS time. Keep 500 unless speed matters.
- Run-to-run noise on this batch (control vs the fulleval run of the same code): mustard0 0.639 vs 0.683, AP12 2.478
  vs 2.300, MPM12 0.785 vs 0.772, cracker 2.654 vs 2.658 — HO3D differences below ~0.2 cm are noise.
- Outputs `outputs/exp_fusion_20260915` = 81 GB (dumps); cleanup pending approval.
- 2026-09-15 (user): initial steps 500 **adopted** — `--gs_initial_steps` default 4000 → 500 in `run_custom.py` and
  `run_ho3d.py` (uncommitted, one line each). Batch dumps of `outputs/exp_fusion_20260915` deleted the same way as the
  B6 sweep (23 028 entries, 81 → 14 GB; list in `logs/exp_fusion_20260915/cleanup_deleted_list.txt`). Follow-up run
  requested: 500/350 (initial/update steps) on the four sequences, to compare with 500/500 and 4000/500 above.

## 8. Per-cycle analysis of the v1 feedback (2026-09-15, `experiments/analyze_feedback_cycles.py`, logs/exp_fusion_20260915/feedback_cycles/)

Joined per keyframe cycle: the applied pose delta (feedback_log), mask area and RANSAC inliers of the keyframe, and the
per-frame ADD of the run vs the tracker-alone run (archived 9/3 `fbabl_*_off`); harm = error growth until the next
keyframe (on) − (off).

| run | ADD on / off | cycles | Σ harm cm (thirds) | delta mm median / p90 / max | ρ(delta, harm) | ρ(mask ratio, harm) | ρ(inliers, harm) | cycles mask<0.7 / inliers<100 |
|---|---|---|---|---|---|---|---|---|
| AP12 init500/upd350 | 2.39 / 0.89 | 169 | 1.95 (0.54 / −0.01 / 1.42) | 1.2 / 2.0 / 3.4 | 0.01 | −0.17 | 0.00 | 0 / 0 |
| AP12 fulleval (4000/500) | 2.30 / 0.89 | 173 | 1.74 (1.76 / −1.40 / 1.38) | 1.4 / 2.7 / 4.6 | 0.08 | −0.25 | 0.12 | 0 / 0 |
| MPM12 init500/upd350 | 0.64 / 0.46 | 219 | 0.00 | 0.7 / 1.5 / 1.8 | 0.05 | 0.07 | −0.01 | 4 / 0 |
| SM1 fulleval | 3.53 / 4.36 | 298 | 2.25 (2.27 / −2.79 / 2.77) | 0.8 / 2.0 / 3.3 | 0.06 | −0.05 | 0.10 | 19 / 0 |
| AP10 fulleval | 5.55 / 4.55 | 151 | 2.24 (2.95 / 0.83 / −1.53) | 1.6 / 3.1 / 4.7 | 0.00 | 0.09 | −0.19 | 3 / 0 |

- The feedback moves keyframes by only 1–3 mm per cycle (never > 5 mm, rotation median 1–2°), and the harm is
  diffuse: ~0.01 cm per cycle on AP12, positive in ~50–55 % of cycles, uncorrelated with the delta size, the mask area or
  the inlier count. A gate on any of these tracker/observation signals would fire on 0–19 cycles per sequence and
  cannot explain the 1.5 cm gap on AP12. SM1 is the exception where the 19 low-mask cycles carry +0.17 cm each.
- Temporal structure (AP12 plot): keyframes come in three dense bursts; the on/off gap opens in the first burst (young
  map, deltas 2–3.4 mm) and again in the last; between bursts no keyframe arrives and the error stays where the last
  burst left it. The harm is a systematic bias of the small per-cycle corrections, not a few bad cycles.
- Consequence for the "freeze the map when the tracker is unstable" idea: on AP12 nothing marks the harmful cycles, so
  a trigger-based freeze has no handle. The structural alternative is to change the cycle order for every keyframe:
  estimate the new keyframe's pose against the existing (frozen) map first, bake it, and only then append its points
  at the corrected pose — the newest observation must never judge itself (today it is appended before the joint
  optimisation, so the map explains it at the tracker's pose and the pose gradient vanishes).
- Keyframe texture / geometry (`experiments/kf_scores_dataset.py`, same definitions as the 9/11 probe scores, computed
  from the dataset frames): per-cycle harm is uncorrelated with SIFT density, curvature or planar fraction on every
  sequence (|ρ| ≤ 0.10); quartile contrasts point in different directions per sequence (low-SIFT quartile: AP12 fulleval
  +0.05 vs −0.005 cm, AP10 −0.03 vs +0.03; high-curvature quartile: AP10 +0.10 vs −0.015, SM1 +0.03 vs 0.00, MPM12 ≈ 0).
  Only the sequence level shows a pattern: the harmful sequences are the texture-poor objects (AP12 0.67, AP10 1.1
  SIFT keypoints per 1 000 mask px) and the neutral / helped ones are textured (MPM12 8.6, SM1 5.2). Caveat: the
  per-cycle harm is a difference of two diverging trajectories, so cycle-level attribution has little power by
  construction; the sequence-level read is the more reliable one.

### Follow-up runs on the 500/350 base (2026-09-15, approved: 500/350, position lr ×3 / ×10 in updates, append-mask erosion 2 px)

Full per-sequence tables (all configurations incl. these) regenerated below; the new rows:

#### mustard0 (ycb) — 참조: SDF-on ADD 1.38, 트래커 단독 0.748, fulleval 우리 0.683 / P1 BundleSDF 0.533, fulleval 우리 0.228

| 설정 | ADD | P1 | P2 | 메시 unseen (5mm내) | seen | 가우시안 수 | 관측 opac<0.1 | big | far | 렌더 깊이 mm | 중심→GT mm | 지도 unseen 5mm내 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 대조(복사본, 4000/500) | 0.639 | 0.230 | 0.230 | 0.207 (96 %) | 0.198 | 72806 | 80 % | 501 | 0 | 2.95 | 2.59 | 98 % |
| 초기화 500 | 0.603 | 0.231 | 0.231 | 0.209 (96 %) | 0.200 | 71120 | 77 % | 131 | 0 | 2.86 | 2.55 | 99 % |
| 500/350 (기준) | 0.617 | 0.235 | 0.235 | 0.208 (95 %) | 0.204 | 68871 | 78 % | 89 | 0 | 2.74 | 2.58 | 99 % |
| 500/350 + 위치 lr ×3 | 0.604 | 0.237 | 0.237 | 0.209 (95 %) | 0.201 | 70348 | 77 % | 92 | 0 | 2.83 | 2.60 | 99 % |
| 500/350 + 위치 lr ×10 | 0.617 | 0.248 | 0.248 | 0.212 (95 %) | 0.212 | 69761 | 74 % | 64 | 0 | 2.74 | 2.83 | 98 % |
| 500/350 + append 침식 2 px | 0.520 | 0.230 | 0.230 | 0.206 (96 %) | 0.206 | 58021 | 74 % | 86 | 0 | 2.65 | 2.47 | 99 % |

#### cracker_box_yalehand0 (ycb) — 참조: SDF-on ADD 2.849, 트래커 단독 2.763, fulleval 우리 2.658 / P1 BundleSDF 0.8, fulleval 우리 0.729

| 설정 | ADD | P1 | P2 | 메시 unseen (5mm내) | seen | 가우시안 수 | 관측 opac<0.1 | big | far | 렌더 깊이 mm | 중심→GT mm | 지도 unseen 5mm내 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 대조(복사본, 4000/500) | 2.654 | 0.699 | 0.699 | 1.542 (16 %) | 0.348 | 125021 | 65 % | 907 | 2762 | 3.07 | 4.80 | 25 % |
| 초기화 500 | 2.692 | 0.719 | 0.719 | 1.537 (16 %) | 0.356 | 121226 | 63 % | 401 | 2690 | 2.30 | 4.88 | 21 % |
| 500/350 (기준) | 2.680 | 0.730 | 0.730 | 1.558 (14 %) | 0.360 | 123566 | 63 % | 306 | 2710 | 2.65 | 4.98 | 23 % |
| 500/350 + 위치 lr ×3 | 2.687 | 0.726 | 0.726 | 1.555 (14 %) | 0.363 | 122644 | 60 % | 271 | 2705 | 2.48 | 5.00 | 22 % |
| 500/350 + 위치 lr ×10 | 2.709 | 0.716 | 0.716 | 1.517 (16 %) | 0.334 | 126758 | 56 % | 254 | 2700 | 2.63 | 4.61 | 27 % |
| 500/350 + append 침식 2 px | 2.709 | 0.737 | 0.737 | 1.538 (15 %) | 0.374 | 109065 | 61 % | 321 | 1362 | 2.50 | 5.06 | 21 % |

#### AP12 (ho3d) — 참조: SDF-on ADD 0.454, 트래커 단독 0.888, fulleval 우리 2.3 / P1 BundleSDF 0.631, fulleval 우리 0.493

| 설정 | ADD | P1 | P2 | 메시 unseen (5mm내) | seen | 가우시안 수 | 관측 opac<0.1 | big | far | 렌더 깊이 mm | 중심→GT mm | 지도 unseen 5mm내 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 대조(복사본, 4000/500) | 2.478 | 0.519 | 0.427 | 0.464 (64 %) | 0.258 | 745040 | 71 % | 2536 | 19922 | 4.20 | 4.28 | 68 % |
| 초기화 500 | 2.216 | 0.453 | 0.356 | 0.412 (71 %) | 0.224 | 707937 | 69 % | 2083 | 17639 | 3.88 | 3.48 | 73 % |
| 500/350 (기준) | 2.390 | 0.507 | 0.409 | 0.421 (70 %) | 0.254 | 710756 | 70 % | 1241 | 16948 | 4.00 | 3.43 | 69 % |
| 500/350 + 위치 lr ×3 | 2.379 | 0.471 | 0.389 | 0.462 (66 %) | 0.231 | 719968 | 67 % | 1302 | 17159 | 3.99 | 3.32 | 71 % |
| 500/350 + 위치 lr ×10 | 2.414 | 0.466 | 0.398 | 0.475 (63 %) | 0.238 | 756084 | 62 % | 976 | 18663 | 3.63 | 4.11 | 72 % |
| 500/350 + append 침식 2 px | 2.230 | 0.472 | 0.374 | 0.450 (68 %) | 0.236 | 649208 | 68 % | 1345 | 6785 | 4.16 | 3.46 | 71 % |

#### MPM12 (ho3d) — 참조: SDF-on ADD 0.777, 트래커 단독 0.457, fulleval 우리 0.772 / P1 BundleSDF 0.431, fulleval 우리 0.271

| 설정 | ADD | P1 | P2 | 메시 unseen (5mm내) | seen | 가우시안 수 | 관측 opac<0.1 | big | far | 렌더 깊이 mm | 중심→GT mm | 지도 unseen 5mm내 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 대조(복사본, 4000/500) | 0.785 | 0.268 | 0.180 | 0.176 (100 %) | 0.148 | 174021 | 79 % | 555 | 21454 | 3.18 | 1.84 | 100 % |
| 초기화 500 | 0.684 | 0.263 | 0.181 | 0.262 (85 %) | 0.126 | 170331 | 76 % | 338 | 21828 | 2.72 | 1.74 | 89 % |
| 500/350 (기준) | 0.639 | 0.259 | 0.173 | 0.176 (92 %) | 0.129 | 163793 | 76 % | 253 | 20597 | 3.11 | 1.69 | 89 % |
| 500/350 + 위치 lr ×3 | 0.602 | 0.257 | 0.174 | 0.226 (87 %) | 0.124 | 160696 | 74 % | 200 | 20937 | 2.38 | 1.69 | 85 % |
| 500/350 + 위치 lr ×10 | 0.631 | 0.248 | 0.164 | 0.222 (88 %) | 0.113 | 165753 | 70 % | 206 | 21873 | 2.83 | 1.77 | 87 % |
| 500/350 + append 침식 2 px | 0.639 | 0.261 | 0.172 | 0.200 (89 %) | 0.135 | 135180 | 73 % | 237 | 7506 | 2.96 | 1.69 | 86 % |

#### 4시퀀스 평균 (설정별, 존재하는 시퀀스만)

| 설정 | n | ADD | P1 | P2 | 메시 unseen 5mm내 | 가우시안 수 | 관측 opac<0.1 | 렌더 깊이 mm | 중심→GT mm |
|---|---|---|---|---|---|---|---|---|---|
| 대조(복사본, 4000/500) | 4 | 1.639 | 0.429 | 0.384 | 69 % | 279222 | 74 % | 3.35 | 3.38 |
| 초기화 500 | 4 | 1.548 | 0.417 | 0.372 | 67 % | 267654 | 71 % | 2.94 | 3.16 |
| 500/350 (기준) | 4 | 1.582 | 0.433 | 0.387 | 68 % | 266746 | 72 % | 3.12 | 3.17 |
| 500/350 + 위치 lr ×3 | 4 | 1.568 | 0.423 | 0.381 | 66 % | 268414 | 69 % | 2.92 | 3.15 |
| 500/350 + 위치 lr ×10 | 4 | 1.593 | 0.419 | 0.381 | 66 % | 279589 | 65 % | 2.96 | 3.33 |
| 500/350 + append 침식 2 px | 4 | 1.524 | 0.425 | 0.378 | 67 % | 237868 | 69 % | 3.06 | 3.17 |

Reading (single runs; HO3D noise ≈ ±0.2 cm):
- **500/350** is within noise of 500/500 (mean ADD 1.582 vs 1.548; MPM12 better 0.639 vs 0.684, AP12 worse 2.390 vs
  2.216, YCB equal) and 30 % cheaper per keyframe; used as the base for the three variants.
- **Position lr ×3 / ×10 in updates**: marginal. ×3 mean ADD 1.568 / P1 0.423 (base 1.582 / 0.433), ×10 1.593 / 0.419.
  Geometry improves slightly where the map was worst (AP12 P1 0.507 → 0.47) and the faint share drops with ×10
  (72 → 65 %), but mustard0 P1 gets worse with ×10 (0.235 → 0.248). Letting positions move does not change ADD;
  ×3 is harmless, ×10 not clearly better.
- **Append-mask erosion 2 px**: best or equal on every sequence in ADD (mustard0 0.520, cracker 2.709, AP12 2.230,
  MPM12 0.639; mean 1.524 vs 1.582), meshes unchanged, and it removes most of the mask-leak junk at the source
  (far Gaussians MPM12 21 k → 7.5 k, AP12 17 k → 6.8 k, cracker 2.7 k → 1.4 k; map −11 %). Candidate to adopt
  (runner point generation: erode the mask by the same 2 px the lifecycle uses); needs approval and the 22-sequence
  confirmation.

### Decisions and completion-metric view (2026-09-15, user)

- **Initial steps: 500 is better than 4 000 unconditionally** (adopted; default changed in run_custom.py / run_ho3d.py).
- **Update steps 500 → 350 is a speed/accuracy trade-off**: 30 % less GS time per keyframe; ADD mean 1.582 vs 1.548
  and AP12 2.390 vs 2.216 (within HO3D noise but consistently a little worse), while the completion metrics are equal
  (P2 0.387 vs 0.372, unseen coverage 68 vs 67 %).
- Completion view of the follow-ups (P2 / P3 are the contribution metrics): all variants are within noise on P2 and P3
  (mean P2 0.372–0.387, unseen-region coverage 66–69 %, seen 0.224–0.238). The erosion run keeps completion (P2 0.378,
  68 %) while improving ADD (1.524) and removing the far Gaussians; lr ×3 / ×10 improve nothing on P2/P3 (66 %).
  The mesh-level noise between identical-code runs is about ±0.03 cm on P1/P2 and ±5 points on the unseen coverage
  (AP12 control 0.519 / 0.427 / 64 % vs fulleval 0.493 / 0.404 / 62 %; MPM12 unseen coverage 85–100 % across runs).
- **SIFT-density measure validated, texture hypothesis withdrawn (2026-09-15).** Per-sequence medians over 12 frames
  order the objects as expected and are consistent within an object: pitcher (AP10–14) 0.6–1.7 keypoints / 1 000 mask
  px, bleach / mustard 4.8–9.8, potted meat / sugar 8.3–12.1, cracker / tomato 13–18; Spearman with the tracker's own
  RANSAC inlier median is 0.61 over the 22 sequences. But the feedback's effect (fulleval ADD on − tracker alone) is
  not explained by texture: ρ = −0.13 over 22 sequences, −0.24 over HO3D. The pitcher sequences are hurt (+0.1 to
  +1.4 cm) yet so are the well-textured potted-meat ones (MPM10 +0.90, MPM13 +0.71, MPM11 +0.57, MPM12 +0.32), while
  SB11 (−0.76), SM1 (−0.83), bleach0 (−0.43), sugar_box_yalehand0 (−0.35) are helped. The earlier 4-sequence reading
  ("texture-poor objects hurt") was a coincidence of the chosen sequences.

### Repeats and 4-px erosion (2026-09-15 night; paired repeats of the 500/350 base, lr ×3 and erosion 2 px on mustard0 / AP12 / MPM12; erosion 4 px on all four)

| config | mustard0 ADD / P1 | cracker ADD / P1 | AP12 ADD / P1 / unseen % | MPM12 ADD / P1 / unseen % | far Gaussians (AP12 / MPM12 / cracker) |
|---|---|---|---|---|---|
| 500/350 base (2 runs) | 0.617, 0.602 / 0.235, 0.236 | 2.680 / 0.730 | 2.390, 2.321 / 0.507, 0.498 / 70, 75 | 0.639, 0.660 / 0.259, 0.264 / 92, 90 | 16.9 k / 21 k / 2.7 k |
| + lr ×3 (2 runs) | 0.604, 0.612 / 0.237, 0.241 | 2.687 / 0.726 | 2.379, 2.302 / 0.471, 0.474 / 66, 68 | 0.602, 0.612 / 0.257, 0.266 / 87, 85 | unchanged |
| + erosion 2 px (2 runs) | 0.520, 0.511 / 0.230, 0.229 | 2.709 / 0.737 | 2.230, 2.370 / 0.472, 0.498 / 68, 69 | 0.639, 0.644 / 0.261, 0.262 / 89, 89 | 6.7 k / 7.4 k / 1.4 k |
| + erosion 4 px (1 run) | 0.465 / 0.236 | 2.683 / 0.738 | 2.462 / 0.514 / 67 | 0.624 / 0.265 / 87 | 4.0 k / 2.9 k / 0.6 k |

Paired-repeat noise of the base: ADD ±0.01 (mustard0, MPM12), ±0.035 (AP12); P1 ±0.005–0.01; unseen coverage ±2–5 points.
- **Erosion 2 px is real and safe**: mustard0 −0.09 cm in both runs (far beyond ±0.01), AP12 / MPM12 / cracker equal
  within noise, P1 / P2 / completion unchanged, far Gaussians −60–70 %. Adoption candidate.
- **Erosion 4 px**: mustard0 better still (0.465) and far Gaussians −80–90 %, but AP12 2.462 / P1 0.514 is at the
  worse edge of the base spread (single run). 2 px is the safe choice; 4 px would need a repeat on AP12.
- **lr ×3**: MPM12 ADD 0.602 / 0.612 vs 0.639 / 0.660 and AP12 P1 0.471 / 0.474 vs 0.507 / 0.498 are consistent in
  both runs, but the unseen coverage is 2–5 points lower in both runs on both HO3D sequences and mustard0 is unchanged.
  Marginal; not adopted.
- 2026-09-16 (user): update steps stay at **500**; 350 is recorded as a speed option (−30 % GS time per keyframe,
  ADD within noise but slightly worse on AP12, completion equal). Batch dumps deleted again (same recipe, list in
  `logs/exp_fusion_20260915/cleanup_deleted_list_2.txt`). Map-loss balance runs deferred until the sequence/config
  set is fixed.
- 2026-09-16 (user): lr ×3 **not adopted, recorded only**. Launched: 22-sequence confirmation of the adopted
  configuration (initial 500 / update 500 / append-mask erosion 2 px, run through the experiment copy, cfg `adopt`),
  then the map-loss balance runs `dw10` / `dw30` (`depth_loss_weight` 10 / 30 via runner configs
  `logs/exp_fusion_20260915/config_dw*.yml`, same base) on mustard0 / cracker / AP12 / MPM12. Both map and pose deltas
  receive the reweighted total loss in this first stage.

## 9. 22-sequence confirmation of the adopted configuration (2026-09-16; initial 500 / update 500 / append-mask erosion 2 px, run through the experiment copy, cfg `adopt`; single runs)


### ho3d — ADD: SDF-on / 트래커 단독 / 기존(4000/500) / **채택(500/500+침식2px)** | P1: BundleSDF / 기존 / **채택** | P2: BundleSDF / 기존 / **채택** | unseen 5mm내 %: BundleSDF / 기존 / **채택** | far 가우시안 채택

| seq | ADD | P1 | P2 | unseen % | far |
|---|---|---|---|---|---|
| AP10 | 0.969 / 4.552 / 5.549 / **5.740** | 0.494 / 0.570 / **0.600** | 0.469 / 0.530 / **0.564** | 65 / 97 / **99** | 5511 |
| AP11 | 0.946 / 1.341 / 2.418 / **2.000** | 0.547 / 0.586 / **0.582** | 0.579 / 0.514 / **0.508** | 37 / 80 / **77** | 38 |
| AP12 | 0.454 / 0.888 / 2.300 / **2.159** | 0.631 / 0.493 / **0.471** | 0.706 / 0.404 / **0.368** | 34 / 62 / **69** | 7542 |
| AP13 | 0.641 / 0.969 / 1.406 / **1.220** | 0.619 / 0.456 / **0.398** | 0.584 / 0.400 / **0.323** | 60 / 72 / **87** | 863 |
| AP14 | 0.499 / 0.790 / 0.888 / **0.838** | 0.460 / 0.413 / **0.401** | 0.593 / 0.348 / **0.340** | 29 / 30 / **32** | 21477 |
| MPM10 | 0.964 / 1.092 / 1.992 / **1.165** | 0.455 / 0.345 / **0.301** | 0.431 / 0.313 / **0.264** | 85 / 97 / **76** | 1125 |
| MPM11 | 0.819 / 0.830 / 1.398 / **1.300** | 0.413 / 0.316 / **0.273** | 0.389 / 0.275 / **0.232** | 100 / 100 / **100** | 58 |
| MPM12 | 0.777 / 0.457 / 0.772 / **0.673** | 0.431 / 0.271 / **0.261** | 0.443 / 0.181 / **0.172** | 41 / 93 / **89** | 6902 |
| MPM13 | 1.373 / 2.753 / 3.462 / **2.732** | 0.437 / 0.676 / **0.507** | 0.410 / 0.653 / **0.474** | 100 / 64 / **78** | 4367 |
| MPM14 | 0.508 / 0.910 / 0.800 / **0.783** | 0.423 / 0.275 / **0.265** | 0.454 / 0.174 / **0.146** | 33 / 100 / **100** | 697 |
| SB11 | 0.506 / 2.144 / 1.387 / **1.220** | 0.435 / 0.582 / **0.445** | 0.441 / 0.537 / **0.394** | 37 / 67 / **77** | 252 |
| SB13 | 0.454 / 0.673 / 0.924 / **0.810** | 0.452 / 0.501 / **0.456** | 0.447 / 0.480 / **0.431** | 45 / 52 / **48** | 71 |
| SM1 | 0.551 / 4.361 / 3.533 / **3.159** | 0.421 / 0.483 / **0.392** | 0.420 / 0.448 / **0.358** | 25 / 91 / **71** | 505 |
| **평균 (n=13)** | 0.728 / 1.674 / 2.064 / **1.831** | 0.478 / 0.459 / **0.412** | 0.490 / 0.404 / **0.352** | 53 / 77 / **77** | |
  ADD 개선 시퀀스 12/13; 트래커 단독보다 나은 시퀀스 4/13; P1 개선 12/13

### ycb — ADD: SDF-on / 트래커 단독 / 기존(4000/500) / **채택(500/500+침식2px)** | P1: BundleSDF / 기존 / **채택** | P2: BundleSDF / 기존 / **채택** | unseen 5mm내 %: BundleSDF / 기존 / **채택** | far 가우시안 채택

| seq | ADD | P1 | P2 | unseen % | far |
|---|---|---|---|---|---|
| bleach0 | 1.801 / 1.764 / 1.338 / **1.543** | 0.819 / 0.483 / **0.462** | 0.819 / 0.483 / **0.462** | 43 / 63 / **62** | 94 |
| bleach_hard_00_03_chaitanya | 0.960 / 1.011 / 1.027 / **0.799** | 0.732 / 0.620 / **0.580** | 0.732 / 0.620 / **0.580** | 10 / 55 / **59** | 0 |
| cracker_box_reorient | 0.785 / 0.756 / 0.823 / **0.839** | 0.770 / 0.529 / **0.553** | 0.770 / 0.529 / **0.553** | 27 / 48 / **37** | 0 |
| cracker_box_yalehand0 | 2.849 / 2.763 / 2.658 / **2.667** | 0.800 / 0.729 / **0.743** | 0.800 / 0.729 / **0.743** | 45 / 14 / **15** | 1512 |
| mustard0 | 1.380 / 0.748 / 0.683 / **0.512** | 0.533 / 0.228 / **0.226** | 0.533 / 0.228 / **0.226** | 34 / 96 / **96** | 0 |
| mustard_easy_00_02 | 0.646 / 0.784 / 0.627 / **0.617** | 0.756 / 0.176 / **0.172** | 0.756 / 0.176 / **0.172** | 7 / 99 / **99** | 0 |
| sugar_box1 | 0.976 / 0.681 / 0.777 / **0.598** | 0.713 / 0.286 / **0.204** | 0.713 / 0.286 / **0.204** | 28 / 89 / **100** | 0 |
| sugar_box_yalehand0 | 1.555 / 1.757 / 1.409 / **1.474** | 0.579 / 0.479 / **0.465** | 0.579 / 0.479 / **0.465** | 34 / 16 / **18** | 0 |
| tomato_soup_can_yalehand0 | 2.450 / 1.211 / 1.562 / **1.598** | 0.804 / 0.667 / **0.718** | 0.804 / 0.667 / **0.718** | 57 / 58 / **41** | 170 |
| **평균 (n=9)** | 1.489 / 1.275 / 1.212 / **1.183** | 0.723 / 0.466 / **0.458** | 0.723 / 0.466 / **0.458** | 32 / 60 / **59** | |
  ADD 개선 시퀀스 4/9; 트래커 단독보다 나은 시퀀스 7/9; P1 개선 6/9

Reading: HO3D ADD 2.064 → 1.831 (12 of 13 sequences better; tracker alone 1.674, SDF-on 0.728), P1 0.459 → 0.412
(12/13; BundleSDF 0.478), P2 0.404 → 0.352, unseen-region coverage unchanged at 77 %. YCB ADD 1.212 → 1.183
(4/9 better, within noise), P1 0.466 → 0.458, coverage 60 → 59 %. The feedback still loses to the tracker alone on
9 of 13 HO3D sequences (AP10 5.74 vs 4.55, AP12 2.16 vs 0.89, AP11, AP13, MPM10, MPM11, MPM12, SB13, MPM13 ≈), wins on
SB11, SM1, AP14 (≈), MPM14. Adoption of the erosion into the main runner is pending approval.

## 10. Map-loss balance, stage 1 (2026-09-16; `depth_loss_weight` 1 (adopted) / 10 / 30 via runner configs, shared by map and pose deltas; base = adopted configuration; single runs)

| config | mustard0 ADD / P1 / P2 / cov | cracker ADD / P1 / cov | AP12 ADD / P1 / P2 / cov | MPM12 ADD / P1 / P2 / cov | mean ADD / P1 / P2 / cov | mean render-depth err / centres→GT (mm) |
|---|---|---|---|---|---|---|
| weight 1 (adopted) | 0.512 / 0.226 / 0.226 / 96 | 2.667 / 0.743 / 15 | 2.159 / 0.471 / 0.368 / 69 | 0.673 / 0.261 / 0.172 / 89 | 1.503 / 0.425 / 0.378 / 67 | 3.14 / 3.16 |
| weight 10 | 0.582 / 0.239 / 0.239 / 96 | 2.708 / 0.733 / 16 | **1.849** / 0.452 / **0.341** / 72 | 0.648 / 0.257 / 0.160 / 93 | **1.447** / 0.420 / **0.368** / 69 | **1.69** / **2.86** |
| weight 30 | 0.596 / 0.235 / 0.235 / 97 | 2.761 / 0.711 / 14 | **1.656** / 0.460 / **0.333** / **77** | 0.929 / 0.315 / 0.239 / 83 | 1.486 / 0.430 / 0.380 / 68 | 1.53 / 2.89 |

Reading: the depth term is the first lever that moves AP12 (ADD 2.16 → 1.85 → 1.66, P2 0.368 → 0.333, centres→GT 3.45 →
2.25 mm) and it halves the rendered-depth error everywhere (3.1 → 1.7 mm, i.e. the B6 symptom). Weight 10 is a
consistent moderate gain on the HO3D pair and neutral on cracker, at a small cost on mustard0 (0.512 → 0.582, beyond
its ±0.01 noise). Weight 30 over-shoots on MPM12 (0.673 → 0.929, P1 0.315) and mustard0. Both map and pose deltas saw
the reweighted loss here, so the map/pose attribution is still open. Proposed next: repeats of weight 10 (AP12, MPM12,
mustard0), then the two-backward split (map weight vs pose weight) to attribute the AP12 gain, then a 22-sequence run
of the winner.
- 2026-09-16 (user): **append-mask erosion 2 px adopted in the main runner** — `gaussian_runner.DEFAULT_CONFIG["append_mask_erode_px"] = 2`,
  applied in `_frames_to_cloud` before back-projection (0 restores the old behaviour); the experiment copy's default
  mirrors it. `tests/test_gaussian_geometry.py` sets the key to 0 for its 5×6 one-pixel synthetic masks (three
  runner configs); the CPU runner tests pass again. Uncommitted together with the initial-step default.
- BundleSDF (original) uses ONE `config.yml` for YCB-Video/YCBInEOAT and HO3D; `run_ho3d.py` only overrides the
  truncation (`trunc_start`/`trunc` 0.01), `down_scale_ratio` 1 and `far`; every loss weight (rgb_weight 10,
  depth_weight 0, sdf_lambda 5, fs_weight 100, trunc_weight 6000, …) is shared. No per-dataset weight tuning.
- Launched (2026-09-16): second `adopt` run, erosion 3 px ×2, and the loss split (map 10 / pose 1, map 1 / pose 10;
  two backward passes routed with `torch.autograd.backward(inputs=…)` in the copy) on mustard0 / cracker / AP12 / MPM12.

## 11. Erosion 3 px, map/pose loss split (2026-09-16; base = adopted config 500/500/erosion 2 px; single runs unless ×2)

| 설정 | mustard0 ADD / P1 / cov | cracker ADD / P1 / cov | AP12 ADD / P1 / P2 / cov | MPM12 ADD / P1 / P2 / cov | 평균 ADD / P1 / P2 / cov | 평균 렌더 깊이 mm |
|---|---|---|---|---|---|---|
| 채택 2px (지도 1 / 포즈 1) ×2 | 0.512, 0.508 / 0.226, 0.224 / 96, 95 | 2.667, 2.688 / 0.743, 0.712 / 15, 16 | 2.159, 2.217 / 0.471, 0.479 / 0.368, 0.376 / 69, 68 | 0.673, 0.649 / 0.261, 0.263 / 0.172, 0.173 / 89, 90 | 1.509 / 0.422 / 0.374 / 67 % | 3.10 |
| 침식 3px ×2 | 0.508, 0.494 / 0.228, 0.236 / 95, 95 | 2.680, 2.689 / 0.745, 0.740 / 15, 15 | 2.320, 2.069 / 0.493, 0.473 / 0.390, 0.365 / 68, 68 | 0.664, 0.672 / 0.265, 0.261 / 0.174, 0.168 / 90, 89 | 1.512 / 0.430 / 0.381 / 67 % | 3.15 |
| 공유 깊이 10 | 0.582 / 0.239 / 96 | 2.708 / 0.733 / 16 | 1.849 / 0.452 / 0.341 / 72 | 0.648 / 0.257 / 0.160 / 93 | 1.447 / 0.420 / 0.368 / 69 % | 1.69 |
| 공유 깊이 30 | 0.596 / 0.235 / 97 | 2.761 / 0.711 / 14 | 1.656 / 0.460 / 0.333 / 77 | 0.929 / 0.315 / 0.239 / 83 | 1.486 / 0.430 / 0.380 / 68 % | 1.53 |
| 지도 10 / 포즈 1 | 0.490 / 0.228 / 95 | 2.668 / 0.745 / 15 | 2.423 / 0.513 / 0.429 / 57 | 0.535 / 0.258 / 0.176 / 85 | 1.529 / 0.436 / 0.395 / 63 % | 2.31 |
| 지도 1 / 포즈 10 | 0.666 / 0.243 / 95 | 2.739 / 0.726 / 15 | 1.879 / 0.481 / 0.365 / 75 | 0.776 / 0.272 / 0.181 / 87 | 1.515 / 0.431 / 0.379 / 68 % | 2.06 |

Reading:
- **Erosion 3 px = 2 px** within noise (mean ADD 1.512 vs 1.509, P1 0.430 vs 0.422, coverage 67 %); far Gaussians a
  little lower (AP12 5.1–5.7 k vs 6.7–7.8 k). 2 px stays.
- **Loss split attributes the depth-weight gain per sequence**: the map-side weight (map 10 / pose 1) helps mustard0
  (0.490 vs 0.51) and MPM12 (0.535 vs 0.65–0.67) but hurts AP12 (2.423 vs 2.16–2.22, coverage 57 %); the pose-side
  weight (map 1 / pose 10) helps AP12 (1.879, ≈ the shared 1.849) but hurts mustard0 (0.666) and MPM12 (0.776).
  cracker is neutral to everything. The shared weight 10 is the compromise with the best mean (1.447 / P1 0.420 /
  P2 0.368 / coverage 69 %) — the pose-side depth term helps where the tracker drifts a lot and is noise where tracking
  is already good. Single runs; the shared-10 candidate needs repeats before any 22-sequence run.

## 12. Prior alignment with 1 / 3 / 5 keyframes (2026-09-16, `experiments/exp_prior_align_multiframe.py`; 400 steps fixed, round-robin frames; reference = SAM3D mesh at the 1-frame result registered onto the GT model by Sim(3) ICP; mean mesh-vertex displacement in mm)

| seq | init | 1 frame (best / step 400) | 3 frames | 5 frames | note |
|---|---|---|---|---|---|
| mustard0 | 23.1 | 9.4 / 9.3 | 10.2 / 10.4 | 11.4 / 10.0 | worse with more frames |
| cracker_box_yalehand0 | 72.8 | 43.4 / 43.7 | 42.3 / 42.3 | 42.8 / 42.0 | prior 32 % too large (scale ratio 1.32), 37 mm offset remains |
| AP12 | 69.5 | 26.6 / 26.2 | 20.3 / 20.7 | 20.8 / 20.8 | better |
| MPM12 | 23.2 | 14.8 / 11.8 | 13.5 / 11.6 | 16.9 / 11.7 | loss-selected "best" (step 27–43) worse than the final pose |
| AP10 | 17.5 | 7.3 / 7.8 | 6.3 / 7.2 | 5.8 / 6.2 | better |
| MPM10 | 14.3 | 5.7 / 5.7 | 6.1 / 6.1 | 6.2 / 6.4 | slightly worse |
| SM1 | 8.6 | 4.2 / 4.1 | 4.9 / 4.9 | 5.7 / 5.7 | worse |
| mean | 32.7 | 15.9 | 14.8 | 15.6 | |

- Convergence: every run reaches its plateau by step 100 (s100 ≈ s400 in all 21 runs); no steady decrease afterwards,
  so 400 steps is ~4× more than needed and multi-frame sharing does not change that.
- Multi-frame alignment is not a general improvement: it helps the two pitcher sequences (AP12 −6 mm, AP10 −1.5 mm) and
  hurts SM1 / mustard0 / MPM10 by 0.5–2 mm; mean 15.9 → 14.8 (3 frames) → 15.6 (5). Reason: the extra frames are
  rendered through the tracker's relative poses (early tracker error enters the alignment) and the round-robin loss
  fights between views on objects the prior does not fit well.
- Side findings: the cracker prior is mis-scaled by ~32 % and the alignment does not correct it (43 mm residual; this
  is the sequence with 15 % completion); on MPM12 the loss-based best-pose selection picks an early pose 3–5 mm worse
  than the final one (the guards/`best` rule deserve a look).

## 13. Why the per-frame ADD rises and falls, and why feedback-on follows tracker-alone (2026-09-16, `experiments/analyze_error_episodes.py`, logs/exp_fusion_20260915/error_episodes/)

Per-frame ADD of the adopted runs vs the archived tracker-alone runs, joined with GT motion (rotation deg/frame, translation mm/frame), visibility (SAM2 mask area / projected GT convex hull) and the tracker's RANSAC inliers vs the previous frame. Rising / falling episodes = smoothed ADD change over 30 frames beyond ±0.3 cm.

| seq | ADD on / off | r(on,off) curves | r increments | gap cm | rho(Δerr~rot) | rho(Δerr~trans) | rho(Δerr~visible) | rho(Δerr~inliers) | rise: rot / vis / inl | fall: rot / vis / inl |
|---|---|---|---|---|---|---|---|---|---|---|
| AP10 | 5.74 / 4.55 | 0.88 | 0.30 | 1.19 | -0.02 | 0.02 | -0.01 | -0.13 | 0.89 / 0.88 / 488 | 0.85 / 0.81 / 498 |
| AP11 | 2.00 / 1.34 | 0.93 | 0.33 | 0.66 | 0.03 | -0.02 | -0.00 | 0.01 | 1.01 / 0.83 / 483 | 1.17 / 0.91 / 402 |
| AP12 | 2.16 / 0.89 | 0.74 | 0.26 | 1.27 | 0.02 | -0.02 | -0.03 | -0.03 | 0.96 / 0.84 / 450 | 1.06 / 0.89 / 541 |
| AP13 | 1.22 / 0.97 | 0.91 | 0.54 | 0.25 | 0.02 | -0.01 | -0.01 | 0.01 | 1.10 / 0.92 / 482 | 1.00 / 0.95 / 476 |
| AP14 | 0.84 / 0.79 | 0.72 | 0.48 | 0.05 | 0.08 | 0.01 | 0.02 | 0.03 | 0.91 / 0.73 / 555 | 1.15 / 0.72 / 649 |
| MPM10 | 1.17 / 1.09 | 0.64 | -0.07 | 0.07 | 0.00 | 0.00 | 0.00 | -0.00 | 1.68 / 0.47 / 419 | 2.04 / 0.64 / 501 |
| MPM11 | 1.30 / 0.83 | 0.64 | 0.39 | 0.47 | 0.04 | -0.01 | -0.03 | 0.02 | 1.64 / 0.87 / 600 | 2.73 / 0.86 / 566 |
| MPM12 | 0.67 / 0.46 | 0.44 | 0.50 | 0.22 | 0.02 | 0.01 | -0.01 | -0.02 | – / – / – | 0.91 / 0.72 / 1199 |
| MPM13 | 2.73 / 2.75 | 0.94 | 0.33 | -0.02 | 0.03 | -0.06 | -0.05 | -0.02 | 1.51 / 0.86 / 601 | 1.63 / 0.93 / 647 |
| MPM14 | 0.78 / 0.91 | 0.84 | 0.41 | -0.13 | 0.03 | 0.08 | -0.01 | -0.02 | 1.11 / 0.51 / 561 | 2.39 / 0.76 / 698 |
| SB11 | 1.22 / 2.14 | 0.76 | 0.33 | -0.92 | 0.15 | -0.02 | -0.02 | 0.05 | 1.41 / 0.82 / 392 | 1.49 / 0.90 / 360 |
| SB13 | 0.81 / 0.67 | 0.98 | 0.70 | 0.14 | 0.12 | 0.00 | -0.04 | -0.05 | 1.54 / 0.85 / 469 | 1.64 / 0.95 / 560 |
| SM1 | 3.16 / 4.36 | 0.71 | 0.79 | -1.20 | -0.02 | -0.04 | 0.01 | -0.03 | 2.99 / 0.68 / 464 | 2.79 / 0.61 / 392 |
| bleach0 | 1.54 / 1.76 | 0.92 | 0.48 | -0.22 | 0.03 | 0.03 | -0.02 | -0.03 | 2.36 / 0.94 / 318 | – / – / – |
| bleach_hard_00_03_chaitanya | 0.80 / 1.01 | 0.85 | 0.93 | -0.21 | 0.01 | -0.02 | -0.01 | -0.02 | 2.81 / 0.83 / 417 | 0.00 / 0.45 / 567 |
| cracker_box_reorient | 0.84 / 0.76 | 0.92 | 0.82 | 0.08 | -0.04 | -0.12 | 0.05 | -0.00 | 2.72 / 0.83 / 538 | 0.21 / 1.00 / 381 |
| cracker_box_yalehand0 | 2.67 / 2.76 | 1.00 | 0.99 | -0.10 | 0.13 | 0.14 | -0.08 | -0.01 | 2.40 / 0.75 / 407 | 1.16 / 0.80 / 460 |
| mustard0 | 0.51 / 0.74 | 0.74 | 0.75 | -0.23 | -0.01 | 0.07 | 0.01 | -0.01 | 2.59 / 0.94 / 537 | 2.20 / 1.04 / 536 |
| mustard_easy_00_02 | 0.62 / 0.78 | 0.99 | 0.80 | -0.17 | 0.03 | -0.02 | -0.01 | -0.03 | 2.51 / 0.85 / 383 | 0.03 / 0.84 / 399 |
| sugar_box1 | 0.60 / 0.68 | 0.98 | 0.81 | -0.08 | 0.04 | -0.03 | -0.03 | -0.02 | 3.15 / 0.75 / 302 | 0.53 / 0.85 / 545 |
| sugar_box_yalehand0 | 1.47 / 1.76 | 0.98 | 0.96 | -0.28 | 0.00 | 0.00 | 0.00 | 0.00 | 3.09 / 0.82 / 456 | 1.02 / 0.82 / 508 |
| tomato_soup_can_yalehand0 | 1.60 / 1.21 | 0.95 | 0.63 | 0.39 | -0.01 | 0.04 | 0.02 | -0.03 | 1.79 / 0.74 / 479 | 1.07 / 0.70 / 454 |

pooled episode means: {"rise": {"rot_deg_per_frame": 1.913019652686053, "trans_mm_per_frame": 4.353754318744448, "visible_ratio": 0.7953639286311099, "inliers_prev": 466.71675330348745}, "fall": {"rot_deg_per_frame": 1.288090317329542, "trans_mm_per_frame": 1.187511412044569, "visible_ratio": 0.8163536228130058, "inliers_prev": 539.9156107545189}, "flat": {"rot_deg_per_frame": 1.035741742479742, "trans_mm_per_frame": 1.2705807630483028, "visible_ratio": 0.8665670559187906, "inliers_prev": 607.2222117042546}}
pooled mean spearman(Δerr, signal): {"rot_deg_per_frame": 0.030107861984306775, "trans_mm_per_frame": 0.001415648256400326, "visible_ratio": -0.01045367965607146, "inliers_prev": -0.014832200256622984}
comovement mean: {"pearson_curves": 0.8393252813369709, "pearson_increments": 0.566118604171847, "level_gap_cm": 0.05526544825436273}

Reading:
- **Co-movement**: the two curves correlate at r = 0.84 on average (HO3D 0.44–0.98, YCB 0.74–1.00) and their frame-to-frame
  increments at r = 0.57. Both runs share the whole frontend — the same frames, the same LoFTR matches to the previous
  frame, the same BA against keyframes — so the shape of the error is the tracker's; the feedback only moves keyframe
  anchors by 1–3 mm per cycle, which shows up as a slowly varying level offset (mean gap +0.06 cm; per sequence −1.2 to
  +1.3 cm), not as a different shape. The curves separate where a burst of keyframe corrections during fast motion adds
  an error that then persists (AP12 frames 1150–1250: on 1.2 → 3.9 cm while alone stays 1.2).
- **Rise / fall causes**: frame-level correlations of the increments with any signal are ~0 (noise), but the episode
  means separate clearly (pooled over 22 sequences): rising episodes have translation 4.4 mm/frame, rotation 1.9°/frame,
  visibility 0.80, inliers 497; flat stretches 1.3 mm, 1.0°, 0.87, 660; falling episodes 1.2 mm, 1.3°, 0.82, 577. Error
  grows while the object moves fast (3.5× the flat-phase translation speed) with fewer matches and partial occlusion, and
  recovers when the motion slows and visibility/matches return — the BA against keyframes pulls the pose back toward the
  anchored keyframes (AP12: every fall coincides with rotation < 0.5°/frame and visibility ≈ 1.0). Per-sequence plots:
  `episodes_<seq>.png` (rises red, falls green, keyframe ticks).

### 13b. Gap analysis (correction of the 'slowly varying offset' reading, 2026-09-16)

| seq | gap mean / std cm | Δgap std vs Δon std vs Δoff std | r(gap, gap at last keyframe) | gap growth in rise / flat / fall (cm) |
|---|---|---|---|---|
| AP10 | +1.19 / 1.33 | 0.112 / 0.094 / 0.094 | 0.88 | +3.16 / -0.32 / -1.95 |
| AP11 | +0.66 / 0.42 | 0.079 / 0.076 / 0.057 | 0.97 | +2.95 / -0.21 / -1.79 |
| AP12 | +1.27 / 0.60 | 0.086 / 0.083 / 0.053 | 0.97 | +4.47 / +1.85 / -4.50 |
| AP13 | +0.25 / 0.28 | 0.052 / 0.057 / 0.050 | 0.95 | +0.69 / +0.33 / -0.97 |
| AP14 | +0.05 / 0.23 | 0.048 / 0.054 / 0.037 | 0.9 | +0.97 / -0.74 / -0.16 |
| MPM10 | +0.07 / 0.54 | 0.244 / 0.167 / 0.166 | 0.94 | +0.35 / +0.64 / +0.22 |
| MPM11 | +0.47 / 0.33 | 0.054 / 0.052 / 0.046 | 0.94 | +1.19 / +0.61 / -0.95 |
| MPM12 | +0.22 / 0.21 | 0.046 / 0.050 / 0.041 | 0.93 | +0.00 / +0.38 / -0.03 |
| MPM13 | -0.02 / 0.56 | 0.085 / 0.068 / 0.078 | 0.97 | +1.89 / +0.76 / -2.86 |
| MPM14 | -0.13 / 0.21 | 0.054 / 0.055 / 0.042 | 0.81 | +0.48 / -0.74 / +0.04 |
| SB11 | -0.92 / 1.47 | 0.151 / 0.151 / 0.100 | 1.0 | -0.23 / -1.39 / -2.83 |
| SB13 | +0.14 / 0.27 | 0.053 / 0.072 / 0.062 | 0.97 | +2.11 / +0.04 / -1.73 |
| SM1 | -1.20 / 0.57 | 0.057 / 0.092 / 0.078 | 0.99 | +1.45 / -1.20 / -2.48 |
| bleach0 | -0.22 / 0.74 | 0.512 / 0.291 / 0.583 | 0.97 | -12.16 / -0.32 / +0.00 |
| bleach_hard_00_03_chaitanya | -0.21 / 0.23 | 0.099 / 0.259 / 0.254 | 0.62 | -0.93 / +0.78 / +0.03 |
| cracker_box_reorient | +0.08 / 0.13 | 0.070 / 0.113 / 0.119 | 0.83 | +0.65 / -0.46 / -0.01 |
| cracker_box_yalehand0 | -0.10 / 0.14 | 0.044 / 0.322 / 0.318 | 0.39 | +0.14 / +0.16 / -0.04 |
| mustard0 | -0.23 / 0.17 | 0.041 / 0.060 / 0.050 | 0.94 | +0.04 / -0.07 / -0.01 |
| mustard_easy_00_02 | -0.17 / 0.16 | 0.055 / 0.084 / 0.090 | 0.78 | -0.92 / +0.76 / -0.03 |
| sugar_box1 | -0.08 / 0.17 | 0.067 / 0.109 / 0.111 | 0.9 | -0.06 / -0.39 / -0.04 |
| sugar_box_yalehand0 | -0.28 / 0.19 | 0.045 / 0.171 / 0.165 | 0.93 | -0.45 / +0.03 / +0.03 |
| tomato_soup_can_yalehand0 | +0.39 / 0.38 | 0.105 / 0.120 / 0.122 | 0.81 | +1.01 / +0.19 / -0.11 |

pooled gap stats: {"mean_cm": 0.05526544825436273, "std_cm": 0.4233214608213516, "increment_std_cm": 0.09811598260015097, "increment_std_on_cm": 0.11821119082171629, "increment_std_off_cm": 0.12343161802190919, "r_gap_vs_gap_at_last_keyframe": 0.8825178400574998, "mean_abs_gap_change_between_keyframes_cm": 0.08081712123612243, "dgap_in_rise_cm_per_frame": -0.0013851847749358265, "dgap_in_fall_cm_per_frame": -0.0061564996726332895, "dgap_in_flat_cm_per_frame": 2.9704640161172033e-05, "gap_growth_in_rise_total_cm": 0.30819036408031275, "gap_growth_in_fall_total_cm": -0.9171880353457411, "gap_growth_in_flat_total_cm": 0.03242431160271413}

- The gap (on − off) is NOT smooth: its std is 0.2–1.5 cm per sequence (pooled 0.42) and its frame-to-frame increment
  std (0.098 cm) is as large as the curves' own (0.118 / 0.123). What is true is that the gap is *anchored to keyframe
  cycles*: the correlation between the gap at a frame and the gap at the most recent keyframe cycle is 0.8–1.0 (pooled
  ≈ 0.9). The feedback's effect on a frame is whatever offset the last write-back put on the keyframes the frame is
  registered against; between cycles the tracker adds its own jitter on top.
- Where the gap grows: on the sequences the feedback hurts, the gap grows almost entirely inside rising episodes
  (fast motion): AP12 +4.5 cm in rises / +1.9 flat / −4.5 in falls; AP10 +3.2 / −0.3 / −2.0; AP11 +3.0 / −0.2 / −1.8;
  SB13 +2.1 / 0.0 / −1.7; MPM13 +1.9 / +0.8 / −2.9. On the sequences it helps the gap shrinks in flat / fall phases
  (SB11 −0.2 / −1.4 / −2.8, SM1 +1.5 / −1.2 / −2.5). Reading: the corrections written back during fast-motion
  keyframe bursts are the harmful ones and are only partly undone when the tracker re-anchors; corrections written
  during slow, well-observed phases are neutral or helpful.
- Actionable consequence: suspend the write-back (or the map append) during fast-motion cycles, judged by the
  tracker's own velocity / inlier drop, and keep it in slow phases. Unlike the earlier 'protective gate' this is not
  meant to avoid all intervention: SB11 / SM1 / bleach0 / sugar_yalehand0 show the slow-phase corrections beat the
  tracker (−0.4 to −1.2 cm), so removing the fast-phase harm could make the feedback a net gain on more sequences.

## 14. Initial alignment error vs run outcome, and why the loss-selected pose misses (2026-09-16, `experiments/summarize_alignment_vs_runs.py`, logs/exp_fusion_20260915/prior_align_multiframe/alignment_vs_runs.md)

| seq | dataset | align init mm | align best mm | rot deg | scale | ADD | P1 | P2 | unseen % | centres→GT mm |
|---|---|---|---|---|---|---|---|---|---|---|
| SB13 | ho3d | 25.0 | 3.4 | 0.9 | 0.99 | 0.810 | 0.456 | 0.431 | 48 | 2.74 |
| SB11 | ho3d | 19.9 | 3.9 | 4.0 | 1.01 | 1.220 | 0.445 | 0.394 | 77 | 3.12 |
| SM1 | ho3d | 8.6 | 4.2 | 2.1 | 1.00 | 3.159 | 0.392 | 0.358 | 71 | 3.37 |
| MPM13 | ho3d | 13.5 | 4.3 | 5.4 | 1.01 | 2.732 | 0.507 | 0.474 | 78 | 4.39 |
| MPM14 | ho3d | 9.2 | 5.2 | 3.5 | 0.99 | 0.783 | 0.265 | 0.146 | 100 | 1.80 |
| MPM10 | ho3d | 14.3 | 5.7 | 7.8 | 1.00 | 1.165 | 0.301 | 0.264 | 76 | 2.93 |
| mustard_easy_00_02 | ycb | 10.8 | 7.2 | 6.1 | 1.01 | 0.617 | 0.172 | 0.172 | 99 | 1.95 |
| AP10 | ho3d | 17.5 | 7.3 | 5.8 | 0.99 | 5.740 | 0.600 | 0.564 | 99 | 4.10 |
| AP14 | ho3d | 11.7 | 7.6 | 1.9 | 0.95 | 0.838 | 0.401 | 0.340 | 32 | 2.52 |
| AP13 | ho3d | 18.9 | 7.7 | 3.4 | 1.03 | 1.220 | 0.398 | 0.323 | 87 | 2.45 |
| MPM11 | ho3d | 14.6 | 8.0 | 10.4 | 1.02 | 1.300 | 0.273 | 0.232 | 100 | 2.44 |
| mustard0 | ycb | 23.1 | 9.4 | 10.0 | 1.03 | 0.512 | 0.226 | 0.226 | 96 | 2.47 |
| bleach0 | ycb | 12.1 | 9.5 | 8.7 | 1.03 | 1.543 | 0.462 | 0.462 | 62 | 4.42 |
| sugar_box1 | ycb | 25.7 | 10.1 | 2.6 | 1.05 | 0.598 | 0.204 | 0.204 | 100 | 2.39 |
| cracker_box_reorient | ycb | 28.1 | 12.6 | 3.8 | 1.00 | 0.839 | 0.553 | 0.553 | 37 | 4.25 |
| AP11 | ho3d | 31.5 | 13.6 | 10.1 | 1.04 | 2.000 | 0.582 | 0.508 | 77 | 2.70 |
| MPM12 | ho3d | 23.2 | 14.8 | 20.4 | 0.87 | 0.673 | 0.261 | 0.172 | 89 | 1.67 |
| tomato_soup_can_yalehand0 | ycb | 10.7 | 14.9 | 17.1 | 1.16 | 1.598 | 0.718 | 0.718 | 41 | 7.30 |
| bleach_hard_00_03_chaitanya | ycb | 22.4 | 26.2 | 5.0 | 1.18 | 0.799 | 0.580 | 0.580 | 59 | 3.95 |
| AP12 | ho3d | 69.5 | 26.6 | 5.7 | 0.99 | 2.159 | 0.471 | 0.368 | 69 | 3.45 |
| sugar_box_yalehand0 | ycb | 36.1 | 37.3 | 6.8 | 1.20 | 1.474 | 0.465 | 0.465 | 18 | 4.71 |
| cracker_box_yalehand0 | ycb | 72.8 | 43.4 | 3.1 | 1.33 | 2.667 | 0.743 | 0.743 | 15 | 5.06 |

Spearman(align best mm, ·) over all (n=22): ADD 0.06, P1 0.40, P2 0.36, unseen cov -0.37, centres→GT 0.33
Spearman(align best mm, ·) over ho3d (n=13): ADD -0.01, P1 -0.08, P2 -0.23, unseen cov 0.22, centres→GT -0.34
Spearman(align best mm, ·) over ycb (n=9): ADD 0.65, P1 0.83, P2 0.83, unseen cov -0.83, centres→GT 0.72

### best-selection traces (per step: total loss parts vs mesh displacement)

- **Alignment error predicts the outcome on YCB, not on HO3D.** Over the 22 sequences the 1-frame alignment error
  (mesh displacement after registration) correlates with ADD 0.06 / P1 0.40 / P2 0.36 / unseen coverage −0.37
  overall; HO3D alone ≈ 0 (ADD −0.01, P1 −0.08: the tracker dominates), YCB alone 0.65 / 0.83 / 0.83 / −0.83. The four
  worst-completion YCB sequences are exactly the mis-scaled priors: cracker_yalehand0 (scale ratio 1.33, 43 mm),
  sugar_yalehand0 (1.20, 37 mm), bleach_hard (1.18, 26 mm), tomato (1.16, 15 mm); the alignment never corrects a 16–33 %
  scale error although `max_scale_delta` 2.7 allows it.
- **Why 'best' misses**: the guards are not the reason (visible / depth-valid ratios stay ≥ 0.96 after step 40). The total
  loss is simply a weak proxy for the pose error along the trajectory (Spearman total ~ displacement: AP12 −0.16,
  MPM12 +0.03, mustard0 +0.19). Two mechanisms in the traces: (i) MPM12 — the weighted depth term rises 0.11 → 0.32
  while the pose error falls 13.5 → 11.8 mm, because the depth loss is a mean over *rendered* pixels: as the prior comes
  to cover the mask (14.1 k → 16.8 k depth pixels) more rim pixels with prior-shape mismatch enter the mean, so a pose
  that renders fewer pixels scores lower; the loss-min (step 43) is such a pose. (ii) AP12 — the loss keeps decreasing
  (0.376 → 0.352) while the error rises from 21.6 mm (step 58) to 26.4 mm: the loss optimum itself is ~5 mm off the
  true pose (prior shape mismatch of the pitcher). Convergence is reached by step 100–150 in all 22 traces.
- Consequences (proposals): normalise the depth loss by the target-mask pixel count (or add a coverage-weighted
  symmetric term) so covering less is not rewarded; take the final (or EMA) pose instead of the loss-min; 150 steps
  instead of 400; and treat the prior scale explicitly (the YCB priors are too large by 16–33 %).

### 13c. Offline check of a motion / inlier gate on the write-back (2026-09-17) — withdraws the gate proposal

Per keyframe cycle of the adopted runs: gap change (on − off) until the next keyframe, split by whether the tracker's own
signals over the 6 frames before the keyframe would flag the cycle (tracker translation > 2.5 mm/frame or inliers < 60 %
of the sequence median). Assumes that skipping a write-back leaves the tracker as in the tracker-alone run there.

| seq | cycles | flagged | gap growth in flagged cycles | in other cycles | total |
|---|---|---|---|---|---|
| AP10 | 165 | 13 % | +1.87 | −1.54 | +0.33 |
| AP11 | 171 | 28 % | +0.74 | +0.08 | +0.82 |
| AP12 | 173 | 24 % | −0.54 | +2.51 | +1.98 |
| MPM10 | 248 | 42 % | −2.14 | +3.29 | +1.15 |
| MPM11 | 237 | 11 % | −0.14 | +1.71 | +1.57 |
| SB11 | 215 | 24 % | −6.13 | +1.76 | −4.37 |
| SB13 | 201 | 23 % | −0.01 | +0.68 | +0.67 |
| SM1 | 282 | 75 % | −1.25 | −0.75 | −2.00 |
| HO3D total | | | **−7.19** | **+5.82** | |

The cycles a tracker-side gate would skip are, on balance, the cycles where the feedback *helps* (−7.2 cm over
HO3D); the harm accumulates in ordinary cycles (+5.8 cm). The earlier episode reading ("harm concentrates in fast
motion") described when the *error* grows in both runs, not where the feedback's *increment* is harmful. A motion /
inlier gate on the write-back is therefore not supported; the harmful component is the diffuse per-cycle bias of §8.

### 13d. Per-write-back analysis: motion, error jumps and the sign of the feedback effect (2026-09-17) — supersedes 13c

Script `experiments/analyze_feedback_by_motion.py`, outputs `logs/exp_fusion_20260915/feedback_by_motion/` (summary.md,
cycles_<seq>.csv/.png, scatter_<ds>.png).  Verified from the run logs: the tracker waits for the backend, so every write-back
(2949 over 22 sequences) lands in the keyframe's own frame (lag 0), and the saved `ob_in_cam` of a keyframe is the
post-write-back pose.  Measures per write-back: cycle effect = change of the on−off ADD gap over exactly the frames the
write-back influences (< 0 helped); immediate effect at the keyframe frame; motion = mean displacement of the model points
between consecutive frames (GT and tracker); rise_on = on-run ADD change over the 5 frames before the write-back.  No thresholds
(helped/hurt groups, deciles, rankings).

- HO3D, 2653 update write-backs: helped 1333 (mean −0.104 cm) vs hurt 1320 (+0.104).  Group medians are identical:
  GT motion 2.18 vs 2.06 mm/frame, rotation 1.65 vs 1.62 °/frame, tracker motion over the 5 frames before 2.08 vs 2.00,
  rise 0.00 vs 0.00 cm, delta 0.97 vs 1.04 mm, inliers 556 vs 525.  Deciles of GT motion, tracker motion, rise and delta:
  mean effect within ±0.04 cm and helped 45–56 % in every decile; Spearman |ρ| ≤ 0.04.  YCB (296): same (|ρ| ≤ 0.13).
- Sharp tracker jumps: after the top-2 % jumps (≥ 0.35 cm over 5 frames) the error change at the write-back frame is
  −0.045 cm (−0.013 over 3 frames) vs +0.005 / +0.078 at other on-run frames and +0.014 / +0.070 in the tracker-alone run:
  the feedback acts in the good direction but weakly (deltas 1–4 mm against jumps of 0.35–3 cm).  After top-10 % jumps the
  error came down at the next frame in 49 % (write-back) vs 51 % (other on-run frames) vs 45 % (tracker alone).  YCB top 2 %
  (n = 18): −0.40 / −0.59 vs +0.36 / +1.15 (other) and +0.26 / +1.01 (alone).
- The per-event label is noisy because the tracker-alone run is an independent trajectory that jumps on its own
  (MPM10 1330–1415: its jumps of up to +2.3 cm flip the sign of neighbouring write-backs: −2.9, −2.1, +1.7, −1.7, −3.8, +2.7).
  The 13c flagged-cycle sums were driven by such stretches and are evidence neither for nor against a gate.
- Sequence-level harm is a slight excess of hurt write-backs (AP10 58 %, AP12 55 %, AP11 54 % vs SB11 44 %, SM1 42 %), each
  ±0.1 cm; nothing observable at the keyframe (motion, jump, delta, inliers) separates them.  A motion / inlier / jump gate
  cannot work; withdrawn definitively.
- Initial 5-view write-back: its delta is the largest of the run and tracks the prior alignment error (ρ 0.71 over 22:
  AP12 26.6 mm → 7.0 mm, AP11 13.6 → 6.0).  On HO3D larger first deltas hurt at once (AP12 +0.53 cm, AP14 +0.33, AP11 +0.12;
  ρ 0.54, n = 13); on YCB the large ones helped (cracker 6.3 mm → −0.34, tomato 7.3 → −0.29).  Observation only.
- Front-loading: the sum of cycle effects over HO3D is −2.4 cm (end-point gap) while the sum of mean gaps is +2.0 cm: the
  on-run gap is positive during the middle of the sequences and closes towards the end (SB11 −4.4 at the end).

### Parked: prior scale handling (2026-09-17, user decision)

The alignment optimiser leaves 16–33 % scale errors on four YCB priors; whether the clamp / regulariser or a scale-insensitive
loss is responsible is unverified.  To be examined later as its own topic (candidate: closed-form scale initialisation from
the observed depth extent before the gradient refinement).  Depth-term normalisation by mask pixels and final-pose selection
(150 steps) are accepted in principle for a later single-variable test.

## 15. Safety-point verification with the MAIN code (c297f6a) and shared depth weight 10 on all 22 sequences (2026-09-17; single runs)

Batch `logs/safety22_20260917/` (run_one.sh: run_custom.py / run_ho3d.py directly, no experiment copy), outputs `outputs/safety22_20260917/{main,main_dw10}/`, 44 runs, no failures. `main` = the committed defaults (init 500, update 500, erosion 2 px, density cut 0, depth weight 1); `main_dw10` = the same with `depth_loss_weight` 10 in the runner config (the only difference).

| set | ADD tracker alone | ADD main | ADD adopt (copy, §9) | ADD dw10 | P1 main / dw10 | P2 main / dw10 | unseen coverage main / dw10 | render depth err mm main / dw10 |
|---|---|---|---|---|---|---|---|---|
| HO3D 13 | 1.674 | 1.867 | 1.831 | 1.911 | 0.405 / 0.436 | 0.343 / 0.371 | 78 % / 81 % | 3.71 / 2.30 |
| HO3D without MPM10 | 1.722 | 1.901 | 1.886 | 1.669 | 0.414 / 0.430 | 0.350 / 0.362 | 77 % / 80 % | 3.72 / 2.18 |
| YCB 9 | 1.274 | 1.199 | 1.183 | 1.299 | 0.453 / 0.469 | 0.453 / 0.469 | 59 % / 60 % | 3.80 / 2.43 |

Per sequence:

| seq | ADD alone | ADD main | ADD adopt | ADD dw10 | P1 main | P1 dw10 | P2 main | P2 dw10 | cov main | cov dw10 |
|---|---|---|---|---|---|---|---|---|---|---|
| AP10 | 4.552 | 5.530 | 5.740 | 2.906 | 0.609 | 0.612 | 0.565 | 0.574 | 99 | 85 |
| AP11 | 1.341 | 2.153 | 2.000 | 2.039 | 0.579 | 0.644 | 0.507 | 0.573 | 73 | 68 |
| AP12 | 0.888 | 2.271 | 2.159 | 1.890 | 0.476 | 0.462 | 0.372 | 0.349 | 69 | 71 |
| AP13 | 0.969 | 1.244 | 1.220 | 1.322 | 0.393 | 0.402 | 0.321 | 0.322 | 81 | 86 |
| AP14 | 0.790 | 0.878 | 0.838 | 0.752 | 0.401 | 0.393 | 0.332 | 0.287 | 33 | 48 |
| MPM10 | 1.092 | 1.462 | 1.165 | 4.816 | 0.297 | 0.511 | 0.259 | 0.480 | 95 | 100 |
| MPM11 | 0.830 | 1.316 | 1.300 | 1.057 | 0.274 | 0.267 | 0.232 | 0.224 | 100 | 100 |
| MPM12 | 0.457 | 0.641 | 0.673 | 0.661 | 0.260 | 0.261 | 0.170 | 0.169 | 87 | 90 |
| MPM13 | 2.753 | 2.723 | 2.732 | 2.877 | 0.411 | 0.384 | 0.377 | 0.348 | 63 | 93 |
| MPM14 | 0.910 | 0.758 | 0.783 | 0.791 | 0.260 | 0.320 | 0.147 | 0.210 | 100 | 100 |
| SB11 | 2.144 | 1.256 | 1.220 | 1.276 | 0.455 | 0.469 | 0.393 | 0.422 | 90 | 69 |
| SB13 | 0.673 | 0.824 | 0.810 | 0.896 | 0.466 | 0.477 | 0.433 | 0.437 | 53 | 56 |
| SM1 | 4.361 | 3.216 | 3.159 | 3.558 | 0.378 | 0.470 | 0.345 | 0.432 | 72 | 88 |
| bleach0 | 1.764 | 1.493 | 1.543 | 1.683 | 0.476 | 0.543 | 0.476 | 0.543 | 63 | 59 |
| bleach_hard_00_03_chaitanya | 1.011 | 0.779 | 0.799 | 0.810 | 0.572 | 0.583 | 0.572 | 0.583 | 57 | 57 |
| cracker_box_reorient | 0.756 | 0.889 | 0.839 | 0.831 | 0.545 | 0.511 | 0.545 | 0.511 | 38 | 40 |
| cracker_box_yalehand0 | 2.763 | 2.691 | 2.667 | 2.723 | 0.716 | 0.734 | 0.716 | 0.734 | 17 | 16 |
| mustard0 | 0.743 | 0.508 | 0.512 | 0.570 | 0.231 | 0.239 | 0.231 | 0.239 | 95 | 96 |
| mustard_easy_00_02 | 0.784 | 0.616 | 0.617 | 0.646 | 0.172 | 0.179 | 0.172 | 0.179 | 99 | 99 |
| sugar_box1 | 0.681 | 0.643 | 0.598 | 0.822 | 0.217 | 0.243 | 0.217 | 0.243 | 97 | 100 |
| sugar_box_yalehand0 | 1.757 | 1.526 | 1.474 | 1.606 | 0.456 | 0.449 | 0.456 | 0.449 | 18 | 17 |
| tomato_soup_can_yalehand0 | 1.211 | 1.651 | 1.598 | 1.996 | 0.692 | 0.737 | 0.692 | 0.737 | 47 | 59 |

- **Main code = copy runner.** Per-sequence ADD differences main − adopt average +0.04 cm (largest MPM10 +0.30, the unstable sequence); P1/P2/coverage identical within Poisson variation. The safety point c297f6a is verified on 22 sequences; these `main` numbers are the reference for later comparisons.
- **Depth weight 10: not adopted.** ADD better on only 5/13 HO3D and 1/9 YCB sequences (median change +0.02 / +0.06 cm); large gains on the pitcher sequences (AP10 5.53 → 2.91, AP12 2.27 → 1.89, MPM11 1.32 → 1.06) are cancelled by a progressive drift on MPM10 (1.46 → 4.82, error growing from frame 200 to 1000 and plateauing at ~8 cm, no crash) and small losses elsewhere. Geometry vs GT is worse (P1 worse on 17/22; HO3D 0.405 → 0.436, YCB 0.453 → 0.469) although the rendered depth fits the observations better (3.7 → 2.3 mm): the map follows the tracker's biased depth more tightly (§8 mechanism), so the feedback anchors to a worse map. The single loss setting stays at depth weight 1.
- Outputs carry the per-frame dumps (128 GB); cleanup needs approval.

> 2026-09-18: the consolidated 22-sequence tables (original BundleSDF with paper and SAM2 masks, tracker alone, GS 9/12, adopt, main, main + depth 10, the 4-sequence B-track sweep, repeat noise, verdicts) are in `RESULTS_22SEQ.md`, generated by `experiments/build_results_tables.py`; the safety22 dumps were cleaned (128 → 14 GB, list in `logs/safety22_20260917/cleanup_deleted_list.txt`).
