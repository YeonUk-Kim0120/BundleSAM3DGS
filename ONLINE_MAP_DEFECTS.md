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
