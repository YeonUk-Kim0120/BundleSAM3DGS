# Milestone ⑤ — GS→tracker pose feedback: attempt-2 results and diagnosis (2026-09-07/08)

Record of what was run, what came out, and what was ruled out, so the same experiments are not repeated.
Code: branch `milestone5-feedback-attempt2` (uncommitted at the time of writing; snapshot `logs/gsfb_batch_20260907/attempt2_worktree.patch`).
Runs: `outputs/gsfb_{v1,v2,v3,noop}_{ycb,ho3d}_<seq>_20260907`; scoring JSONs, batch scripts, diagnostics and per-cycle CSV: `logs/gsfb_batch_20260907/`.
SDF references: MILESTONES.md ⑤ fair ablation (our runs `outputs/fbabl_*`, baseline folder `~/Desktop/BundleSDF_baseline_outputs/full_eval_sam2_*`, read-only).

## 1. What was implemented (port of BundleSDF's PoseArray into the GS backend)

- `gaussian_runner.py`: `se3_exp_batch`, `PoseDeltas` (per-view 6-DoF, tanh clamp 2 cm / 20°, left-multiplied in the normalized object frame, view 0 fixed, re-created at zero per training call, baked into view poses afterwards), joint map+pose optimization in `train()` (pose Adam lr 0.01 → ×0.1 decay, inf-norm grad clip 0.1), `prealign_views` (v2 pose-only stage on new keyframes with the map frozen), `view_poses_metric`, config block `pose_feedback`, per-cycle `feedback_log`.
- `bundlesdf.py` `run_gaussian`: `feedback` = on / noop / off; refined poses written back for all keyframes (tracker sets `_nerfed`, unchanged C++); `poses_before_gs.txt` / `poses_after_gs.txt` per cycle; `gs_online/feedback_log.json`.
- `experiments/run_sdf_feedback_ablation.py --backend gaussian` (+ `--gs_prealign_steps`, `--gs_update_steps`, `--gs_runner_config`, SAM3D prior auto-resolution).
- `tests/test_pose_feedback.py` (CPU + gated GPU), all passing.
- Conditions: **v1** = original order (append → joint 500 steps); **v2** = pose-only pre-alignment 150 (new keyframes, frozen map) → append with refined poses → joint 350; **v3** = v2 + constant pre-alignment lr 0.05 (`config_gs_v3_prealign_lr05.yml`, YAML only); **noop** = tracker poses back verbatim (= milestone ④).
- Runtime: mustard0 6.7 min (33 cycles, 3–5 s/cycle after the 25–30 s first batch); HO3D 31–61 min per sequence. Two CUDA OOMs on HO3D (tracker LoFTR ~10 GB + GS 10–13 GB + desktop on GPU0); mitigated with `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` and by running large sequences on GPU1. Maps under collapse grow 5–10× (ghost appends: AP10 786k Gaussians vs mustard0 71k).

## 2. Results (ADD cm, SAM2 masks everywhere; single runs)

| mean | SDF on | SDF noop | SDF off (tracker alone) | GS v1 | GS v2 |
|---|---|---|---|---|---|
| YCB 9 | 1.489 | 1.309 | 1.275 | **1.214** | 1.259 |
| HO3D 13 | **0.728** | 1.757 | 1.674 | 2.407 | 3.191 |

Per sequence (v1 / v2 / v3; SDF on / off in parentheses):
- YCB: mustard0 0.638 / 0.683 / 0.850 (1.380 / 0.748); bleach0 1.258 / 1.388 (1.801 / 1.764); bleach_hard 1.055 / 1.134 (0.960 / 1.011); cracker_reorient 0.852 / 0.958 (0.785 / 0.756); cracker_yalehand 2.655 / 2.650 (2.849 / 2.763); mustard_easy 0.647 / 0.584 (0.646 / 0.784); sugar_box1 0.771 / 0.780 (0.976 / 0.681); sugar_yalehand 1.492 / 1.293 (1.555 / 1.757); tomato 1.553 / 1.864 / 2.341 (2.450 / 1.211).
- HO3D: SM1 3.433 / 3.324 / 4.142 (0.551 / 4.361); AP10 5.688 / 5.392 / 5.243 (0.969 / 4.552); AP11 2.736 / 4.067 (0.946 / 1.341); AP12 2.379 / 3.689 (0.454 / 0.888); AP13 1.287 / 2.234 (0.641 / 0.969); AP14 0.841 / 1.492 (0.499 / 0.790); MPM10 4.987 / 5.197 (0.964 / 1.092); MPM11 1.426 / 2.349 (0.819 / 0.830); MPM12 0.742 / 0.776 (0.777 / 0.457); MPM13 3.548 / 4.805 (1.373 / 2.753); MPM14 0.805 / 4.301 (0.508 / 0.910); SB11 2.516 / 1.285 (0.506 / 2.144); SB13 0.908 / 2.575 (0.454 / 0.673). GS-noop SM1 4.363 (= SDF noop 4.449: backend itself is fine).

Reading: YCB — GS feedback beats the original feedback and slightly beats the tracker alone (5 better / 3 worse vs off; it does not add the original's symmetric-object bias). HO3D — GS feedback fails to rescue the collapses and harms non-collapsing sequences too (11/13 worse than tracker alone). v2 < v1 on HO3D and on symmetric YCB objects; v3 worst everywhere. Note the HO3D run-to-run noise can reach ~1 cm on hard sequences (sibling project's repeats), so sub-1 cm differences there are not conclusive.

## 3. Diagnosis (all read-only analyses of saved per-cycle poses vs GT; scripts in logs/gsfb_batch_20260907/)

1. **Timeline** (`diag_timeline.py`): both tracker-alone and GS v1 collapse at the same frames (AP10 243–247, SM1 19–37); the original SDF keeps the tracker within ~5° from there on. Collapses are rotations about near-symmetry axes (ADD-S small).
2. **Direction vs GT** (`diag_direction2.py`, `diag_kbins.py`, kbins.csv): newest keyframe per cycle. HO3D: SDF on moves toward GT in ~60% of cycles (−0.3…−0.5°/cycle, flat in K); GS v1 in ~40% (+0.1…+0.4°/cycle) **from the smallest K bin (5–20) on** — not degrading with K; mean move per cycle is ~1.3–1.9° for both methods in every K bin. Slips (newest-kf error > 15°) start at K = 38–64 on 9/13 HO3D sequences, preceded by mostly-away GS corrections; the SDF tracker slips just as much (max 30–45°) but is pulled back. First cycle (K=5, ~800 updates/view, fresh map): GS pushed away on 5/13 (AP12 +5.0°). YCB: GS neutral (toward 0.46–0.48, Δ≈0); SDF on YCB pushes away at K>20 with 3–6°/cycle (its symmetric-object harm).
3. **Sign-flip test** (`diag_reverse.py`): reversing GS corrections gives toward 0.50, alignment 0.00 — GS moves are mostly *sideways* w.r.t. the true error (near-zero information) with a small systematic tilt away; SDF reversed drops to 0.27 (its signal is real). Conclusion: the fix must add information to the pose gradient, not flip it.
4. **Correction budget**: v1 gives the newest keyframe ~500/K Adam updates; v2/v3 gave 75–150 (lr up to 0.05) — larger moves, worse direction. The sibling project's compute-matched two-stage (map 100 → pose-only 250) also lost to joint. Update count is not the primary cause.
5. **Sequence proxies** (`diag_dynamics.py`): degradation does not correlate with per-frame motion (HO3D is slower than YCB except SM1), keyframe count/density, or mask size (|Spearman| ≤ 0.35). Only HO3D-specific trait: keyframe density 10–16 per 100 frames vs 2.5–7 on YCB. Hand-in-object-mask contamination ruled out on SM1 (SAM2 overlap 0.3%). Map size at K=30 does not separate good from bad sequences (MPM10, the worst, is YCB-sized).
6. **Sibling project (~/Desktop/BundleSDF, "BundleGS")** re-scored with our evaluator: HO3D feedback ON AP10 2.0 (OFF 4.06), SM1 4.2, MPM13 2.6–3.0, SB11 1.6–2.6 (repeat spread 1 cm) — also far below SDF (13/13 HO3D losses in their own audit). Their design differences: `overwrite_only` (no `_nerfed`), 100-step pose freeze at each session start, pose delta/smooth regularizers (0.015/0.0075), near-transparent dense prior (opacity offset −10). Their notes independently report "SAM3D-induced reverse direction" of PoseArray, sampling dilution, rematch/gates/latest-only not helping, and 1-cm run-to-run variance on hard HO3D sequences.

## 4. Hypotheses status

Ruled out as primary cause: update starvation with K; motion dynamics; hand pixels in masks; map size; simple sign error. Consequence, not cause: ghost-surface map growth after a slip.
Remaining: (1) the persistent map's object frame is biased (SAM3D prior anchor with 2.7–8 mm alignment error + observations baked in at historical poses; the SDF re-fits a fresh field to all keyframes every cycle and has no prior); (2) the rasterized RGB/DSSIM/depth-Huber loss carries little pose information under hand occlusion (no gradient where the map has no geometry; the SDF's free-space/truncation losses constrain the whole ray).

## 5. Decisions and next experiments

- `_nerfed`-off (overwrite_only) rejected by the user: it would let BA undo the refinement (coarse→fine intent). Base code for follow-ups: **v1**.
- Next (approved in principle, 2026-09-08): (a) offline gradient-informativeness probe on real HO3D keyframes vs GT — per loss term, with/without prior surfels; (b) oracle prior: GT mesh placed with the GT pose of frame 0 (and SAM3D mesh registered to the GT mesh) instead of the SAM3D alignment — tests hypothesis (1); (c) config-only `prior_lifecycle.prior_opacity` 0.9 → 0.1; (d) explicit new-keyframe sampling quota (every 5th step); (e) loss changes only if (a)/(b) show the gradient stays uninformative with a correct frame. Judge by the K 5–40 toward ratio / alignment (less noisy) plus final ADD; hard HO3D sequences need repeats.

## 6. Cause hunt, 2026-09-08 (experiments A–D; scripts `experiments/exp_feedback_gradient_probe.py`, `experiments/make_oracle_prior.py`; logs `logs/gradprobe_20260908*`, `logs/oracle_priors_20260908`, `logs/gsfb_batch_20260908`)

**A. Offline gradient probe** (replay of the v1 runs; newest keyframe at K = 6…40; gradient of each loss term w.r.t. a world-side 6-DoF delta at the tracker pose, compared with the true correction toward GT; `cos_rot` = rotation-part cosine, share of probes with cos > 0 in parentheses; ΔErr = GT rotation error change after a fixed 0.5° descent step):

| seq (tracker err) | RGB L1 | DSSIM | depth Huber | total (online weights) |
|---|---|---|---|---|
| AP12 (7°) | −0.19 (44%) +0.11° | −0.17 (33%) +0.10° | **+0.50 (100%) −0.23°** | −0.16 (44%) +0.10° |
| MPM10 (8°) | +0.07 (67%) | +0.11 (78%) | **+0.32 (78%) −0.14°** | +0.11 (67%) −0.04° |
| AP10 (5°) | +0.20 (56%) | +0.13 (56%) | −0.14 (22%) +0.10° | +0.18 (56%) −0.06° |
| SM1 (43°, collapsed) | −0.32 (0%) | −0.18 (25%) | +0.06 (38%) | −0.21 (25%) |
| mustard0 (5°, YCB) | +0.39 (88%) | **+0.61 (88%)** | +0.08 (50%) | +0.55 (88%) −0.26° |

- The photometric terms are informative on the textured YCB object and misleading on the blue HO3D pitcher (AP12), where the depth term is right in 100% of probes but is outvoted by the larger photometric gradients in the total. On AP10 the depth term itself points the wrong way; on MPM10 depth is the best term. No single term is reliable everywhere.
- Hiding the frozen UNSEEN prior surfels in the render changes nothing (all sequences); hiding all prior-lineage splats destroys the map early on. The single-step gradient bias is therefore not the frozen prior.
- Pose-only descent on the total loss (30 steps, map frozen — the v2 regime) worsens GT rotation by +0.7…+1.0° on AP12/MPM10/AP10 (improves 22–44%) and improves mustard0 (−1.2°, 62%). Depth-only descent (60 steps, lr 0.01 constant) improves AP12 strongly (−2.2°, 78%; K40 17° → −11.6°) but worsens MPM10 (+1.3°) / AP10 (+1.8°) / mustard0 (+2.1°) — step budget confound (4–9° moved for 3–5° errors); a matched small-budget rerun is pending.

**B. Oracle priors** (`logs/oracle_priors_20260908`): GT mesh at the frame-0 GT pose fits the observed frame-0 surface at 1.2–1.4 mm (HO3D) / 3.9 mm (YCB) (nearest-neighbour; the z-buffer residual is misleading for concave objects). Our ② refined SAM3D placement fits the surface equally well (1.5–2.6 mm) yet differs from the GT-registered placement by **5.0° / 20 mm (AP12), 7.9° / 60 mm (MPM10), 6.5° / 64 mm (AP10), 2.5° / 15 mm (SM1)** — one partial view leaves the prior placement that ambiguous. Online oracle runs (gtmesh / sam3dgt) queued after C.

**C. Faint prior (`prior_lifecycle.prior_opacity` 0.1, v1 base)**: mustard0 0.616 (v1 0.638), AP12 2.212 (2.379), **MPM10 1.271 (4.987; off 1.092)**, AP10 5.158 (5.688), SM1 3.252 (3.433). Rescues the sequence with the largest prior misplacement, marginal elsewhere.

**D. New-keyframe sampling quota** (`pose_feedback.new_view_interval`, driver `--gs_new_view_interval`): implemented (tests: forced draws counted, RNG stream unchanged); online runs queued after B.

Reading so far: the GS pose gradient is a mixture of a geometry term that is informative on some sequences and a photometric term that is informative only on textured, evenly lit objects; on HO3D the sum is dominated by the misleading photometric part. The original SDF avoids this with ray-based geometry losses (free-space/truncation, weight 100) that dominate its objective, and per-frame appearance codes that absorb lighting changes. Candidate fix E: pose gradient from the depth term only (map keeps the full loss), predicted to help AP12/MPM10, not AP10, and to cost a little on YCB.

### 6.1 Online results of B/C/D (2026-09-08, v1 base, single runs, ADD cm; `logs/gsfb_batch_20260908/summary.py`)

| seq | SDF on | tracker alone | GS v1 | C: prior opacity 0.1 | B: GT mesh @ GT pose | B: SAM3D mesh ICP→GT | D: new-keyframe quota (every 5th step) |
|---|---|---|---|---|---|---|---|
| mustard0 (YCB) | 1.380 | 0.748 | 0.638 | 0.616 | 0.703 | – | 0.755 |
| AP12 | 0.454 | 0.888 | 2.379 | 2.212 | 2.440 | 2.968 | **1.764** |
| AP10 | 0.969 | 4.552 | 5.688 | 5.158 | 5.225 | 5.893 | **3.298** |
| MPM10 | 0.964 | 1.092 | 4.987 | **1.271** | **1.817** | **1.916** | 4.484 |
| SM1 | 0.551 | 4.361 | 3.433 | 3.252 | 3.729 | 3.650 | 3.771 |

Reading: the bottleneck differs per sequence. MPM10 is a prior-placement problem (a correctly placed prior, or a faint one, removes the harm; the SAM3D shape is fine). AP10 responds to update count (the quota is the first condition to beat the tracker alone there: 4.55 → 3.30). AP12 responds partly to the quota (2.38 → 1.76) but not to the prior; its probe says the photometric pose gradient is wrong there (candidate E). SM1 responds to nothing so far (collapse at K=7). Caveat: single runs; HO3D run-to-run noise can approach 1 cm on hard sequences — the MPM10 and AP10 wins should be repeated.

### 6.2 Repeats and the C+D combination (2026-09-08 night, v1 base, ADD cm)

| seq | tracker alone | GS v1 | C (prior 0.1) | D (quota) | C+D |
|---|---|---|---|---|---|
| mustard0 | 0.748 | 0.638 | 0.616 | 0.755 | 0.613 |
| AP12 | 0.888 | 2.379 | 2.212 | 1.764 | **1.389** |
| AP10 | 4.552 | 5.688 | 5.158 | 3.298 / 4.400 (repeat) | 2.975 |
| MPM10 | 1.092 | 4.987 | 1.271 / 1.685 (repeat) | 4.484 | 1.776 |
| SM1 | 4.361 | 3.433 | 3.252 | 3.771 | 3.231 |

The MPM10 prior effect reproduces (1.27, 1.69 vs 4.99). The AP10 quota effect is real relative to v1 (3.30, 4.40, 2.98 vs 5.69) but its margin over the tracker alone (4.55) is within run-to-run noise. C+D is the best condition on AP12 (1.39) and AP10 (2.98) and costs nothing on YCB; SM1 is unchanged by everything (3.2–3.8). Overall C+D vs v1 on the four HO3D sequences: 2.34 vs 4.12 mean; vs tracker alone 2.72 → the feedback is no longer harmful on average but still not the SDF-level rescue (SDF on mean 0.73).

### 6.3 Gradient probe on all 22 sequences (2026-09-09, `logs/gradprobe_20260909_all`; v1 run dirs replayed; newest keyframe, K = 6…60, pre-append, no prior hiding)
Mean rotation cosine between each term's descent direction and the true correction (share of probes with cos > 0 in parentheses). Photometric gradients are 20–120× larger than the depth gradient w.r.t. the pose, so "total" follows L1/DSSIM.

| seq (object) | slip° | L1 | DSSIM | depth | total |
|---|---|---|---|---|---|
| SM1 (mustard) | 46 | −0.22 (0.09) | −0.16 (0.27) | **+0.20 (0.73)** | −0.16 |
| AP10 (pitcher) | 16 | +0.02 | −0.00 | −0.15 (0.33) | −0.01 |
| AP11 (pitcher) | 11 | −0.10 | −0.15 | **+0.24 (0.58)** | −0.11 |
| AP12 (pitcher) | 8 | −0.12 | −0.18 | **+0.40 (0.92)** | −0.17 |
| AP13 (pitcher) | 5 | −0.01 | +0.29 | +0.29 | +0.20 |
| AP14 (pitcher) | 2 | +0.14 | +0.12 | **+0.31** | +0.16 |
| MPM10 (meat can) | 8 | +0.01 | +0.07 | **+0.27 (0.83)** | +0.06 |
| MPM11 (meat can) | 9 | +0.05 | +0.14 | −0.08 | +0.12 |
| MPM12 (meat can) | 3 | **+0.33** | +0.19 | −0.12 | +0.21 |
| MPM13 (meat can) | 9 | +0.12 | +0.12 | +0.06 | +0.13 |
| MPM14 (meat can) | 6 | −0.04 | +0.15 | **+0.38** | +0.13 |
| SB11 (bleach) | 4 | +0.20 | **+0.44 (0.92)** | −0.04 | +0.29 |
| SB13 (bleach) | 2 | +0.43 | **+0.53 (0.92)** | −0.29 | +0.54 |
| HO3D mean | | +0.06 | +0.12 | +0.11 | +0.11 |
| mustard0 | 5 | +0.15 | **+0.44** | +0.14 | +0.38 |
| bleach0 | 9 | −0.02 | −0.07 | **+0.24** | −0.04 |
| bleach_hard | 9 | −0.03 | −0.19 | **+0.20** | −0.14 |
| cracker_reorient | 5 | −0.18 | −0.28 | −0.52 (0.00) | −0.25 |
| cracker_yalehand | 6 | −0.26 | −0.27 | −0.38 | −0.29 |
| mustard_easy | 6 | −0.40 | −0.13 | **+0.11 (0.86)** | −0.22 |
| sugar_box1 | 7 | +0.36 | **+0.57 (0.90)** | +0.05 | +0.49 |
| sugar_yalehand | 5 | −0.13 | −0.21 | −0.06 | −0.19 |
| tomato | 17 | +0.12 | +0.09 | −0.32 | +0.10 |
| YCB mean | | −0.04 | −0.00 | −0.06 | −0.02 |

Reading: no term is strongly informative anywhere (|cos| ≤ 0.6); which term is *usable* is object-specific — depth on the pitcher (AP11/12/14), the mustard bottle (SM1, mustard_easy) and the bleach0/bleach_hard views; photometric on the labelled bleach cleanser (SB11/13), sugar box, mustard0 and MPM12. On YCB neither term is informative on average, which means the v1 YCB gains are not from consistently correct corrections. Note SB13: total cos +0.54 yet GS v1 0.908 vs tracker alone 0.673 — a correct newest-keyframe gradient is not sufficient (write-back to all keyframes, `_nerfed` locking and post-append absorption also matter). Supports a per-keyframe choice (agreement gate F / complexity weighting G) rather than a global reweighting.

## 7. Stage 1 — pose-gradient policies (2026-09-09, `logs/gsfb_batch_20260909`)
Decision (user, 2026-09-09): fix the pose first, keep the map loss as is. The pose deltas now take their gradient
from a configurable combination of the two loss terms (`pose_feedback.pose_loss`, driver `--gs_pose_loss`);
the Gaussians keep receiving the unchanged total loss. Default `total` reproduces v1 exactly.

| cond | pose gradient | question |
|---|---|---|
| E1 `depth` | depth term only | does removing the misleading photometric part help AP12/MPM10 (probe: depth cos +0.40 / +0.27)? |
| E2 `weighted` | photometric + 25 × depth | does a fixed rebalancing (both terms comparable) reproduce the SDF-like geometry-led behaviour? |
| E3 `normsum` | unit(photometric) + unit(depth) per part | equal say for both terms regardless of magnitude |
| F `gate` | E3 direction only when the two parts agree (cos > 0), else no step | conservative: move only on consensus |

Implementation notes: `_pose_gradient_policy` computes the two per-term gradients with `autograd.grad`
before `loss.backward()` and replaces `pose_deltas.data.grad` (only the sampled view's row is non-zero);
rotation (rows 3:6) and translation (rows 0:3) parts are normalised/gated separately because of their units.
Cost: two extra backward passes per step (cycle time ≈ 1.6× v1 on AP12: 5.7 s vs 3.3 s).
Test note: `test_pose_gradient_policies` initially failed because the 2-view test scene sampled the fixed view 0
only in 5 steps; the test now forces the movable view every step (`new_view_indices=[1]`, interval 1). Code unchanged.

First observation (E1, AP12, initial cycle, 5 views): max_rot 19.3° / max_trans 15.8 mm (v1: 7.6° / 3.9 mm;
GT-mesh oracle prior: 7.0° / 3.7 mm). Adam is scale-invariant, so a small but *consistent* depth gradient
accumulates to the tanh clamp within the 500 initial steps, whereas the photometric-dominated v1 gradient
partly cancels. Whether that large initial move is right will show in the ADD timeline.
Runs: E1/E3 on GPU0, E2/F on GPU1 (after the vector pass), 6 sequences each; results table via `summary.py`.

### 7.1 G pre-check (offline, read-only): do GTR-style complexity scores predict the usable term? (`logs/gradprobe_20260909_all/complexity_vs_probe.py`)
Per probed keyframe (234 probes, 22 sequences): appearance score = SIFT keypoints per 1 000 mask pixels; geometry
score = median angular deviation of depth normals from their 7×7 mean (deg). Target = per-term rotation cosine from §6.3.

| predictor → target | Pearson r | quartile means of the target (Q1 low … Q4 high) |
|---|---|---|
| appearance → photometric cos | +0.08 | 0.00 / +0.03 / +0.17 / +0.05 |
| appearance → depth cos | **−0.16** | **+0.20 / +0.12 / −0.07 / −0.03** |
| geometry → photometric cos | +0.04 | +0.06 / 0.00 / +0.19 / 0.00 |
| geometry → depth cos | +0.10 | −0.04 / +0.02 / +0.15 / +0.09 |

Sequence level (mean appearance score → which term has the higher cos): pitcher AP10–14 (0.7–2.9 kp/1k px) → depth on
4/5; mustard SM1 (5.2) → depth; cans/boxes/bleach (6.8–18.7) → photometric on 8/11. Reading: the usable signal is
"low texture ⇒ trust depth", which is exactly the GTR switch (use geometry when appearance is poor), but the
effect is weak per frame (|r| ≤ 0.16) and clearer per object. The geometry score barely helps. So G should be a
texture-driven weight (photometric vs depth), decided per object (first keyframes) or smoothed over keyframes,
not a per-frame curvature rule. Threshold suggested by the data: ≈ 5 keypoints / 1k mask px.

## 8. G design draft — texture-weighted pose gradient (GTR-inspired; NOT implemented, for review)
Goal: per keyframe (or per object) decide how much the photometric vs the depth term steers the pose delta,
instead of a global rule (E1–E3) or a pure consensus rule (F). Map loss unchanged.

**Scores (computed once per view when it is appended; never re-computed):**
- `app` = SIFT keypoints inside the mask / (mask pixels / 1000). Costs ~10–30 ms per keyframe (cv2, CPU).
- geometry score dropped for now (7.1: no predictive value); keep the hook so it can be added later.

**Weight rule (v1, two parameters):** `w_photo = sigmoid((app − app_mid) / app_width)`, `w_depth = 1 − w_photo`,
`app_mid = 5`, `app_width = 2` (from 7.1: pitcher ≈ 1–3 → w_photo ≈ 0.1; cans/boxes ≈ 7–19 → w_photo ≈ 0.7–1.0).
Pose gradient for the sampled view (rotation and translation parts separately, as in E3/F):
`g = w_photo · unit(g_photo) + w_depth · unit(g_depth)`. With `w_photo = 0.5` this is exactly E3; with 0/1 it is
E1 / photometric-only. Optional object-level smoothing: `app` averaged over the first 5 keyframes and frozen
(`complexity_scope: object | view`), because 7.1 shows the signal is clearer per object than per frame.

**Code plan (minimal; all inside the existing ⑤ block in `gaussian_runner.py`):**
1. `DEFAULT_CONFIG["pose_feedback"]`: `pose_loss` gains the value `"texture"`; new keys `texture_app_mid`,
   `texture_app_width`, `texture_scope` (+ validator lines).
2. `GaussianRunner.__init__`: `self._view_texture = {}` (view index → app score); filled in `update()` /
   `initialize_from_prior()` right after the views are created (one helper `_texture_score(frame)` using
   `cv2.SIFT_create`; import guarded so the CPU tests without cv2 still run).
3. `_pose_gradient_policy`: new branch `mode == "texture"` reusing the normsum code path with the two weights;
   record fields `texture_w_photo` (mean over moved views) in `feedback_log`.
4. Driver: `--gs_pose_loss texture`, `--gs_texture_app_mid`, `--gs_texture_app_width`, `--gs_texture_scope`.
5. Tests: score helper on a synthetic textured/flat image; policy reduces to E3 at w = 0.5 and to E1 at w = 0.
Not touched: `bundlesdf.py` write-back, tracker, map loss, lifecycle, view dataclass.

**Validation before online runs:** replay the policy on the recorded gradient vectors (`sim_gate.py` extended with
the texture weights) → expected to match or beat `normsum` on the 8 vector sequences; then online on the stage-1
six sequences; go/no-go = HO3D mean vs E1/E3/F and no YCB loss.

**Open choices for the user:** (a) per-view vs per-object scope (recommend per-object for v1); (b) whether the
depth weight floor should be > 0 on textured objects (recommend `w_depth ≥ 0.2` so geometry always has a say);
(c) whether to implement after stage-1 results only if E1/E3/F do not already solve HO3D.

### 7.2 F-offline: policies replayed on the recorded gradient vectors (`logs/gradprobe_20260909_all/sim_gate.py`, 92 probes, 8 sequences)
Score = first-order GT-error change for one fixed step (0.5° rotation / 2 mm translation) along the policy's
direction, from the recorded per-term gradients (v1 map, newest keyframe, pre-append). Negative = improves.
Gate policies score 0 on the steps they block (acceptance rate in parentheses). `texture` = G rule (§8) with
`app_mid 5`, `app_width 2`.

| rotation (deg/step) | total (v1) | depth (E1) | weighted25 (E2) | normsum (E3) | gate (F) | texture (G) |
|---|---|---|---|---|---|---|
| AP10 | **−0.10** | +0.06 | −0.00 | −0.01 | −0.04 (0.50) | +0.05 |
| AP12 | +0.07 | **−0.15** | −0.10 | −0.05 | −0.01 (0.67) | **−0.15** |
| MPM10 | −0.08 | −0.12 | **−0.15** | −0.13 | −0.12 (0.83) | −0.08 |
| MPM13 | −0.04 | −0.05 | **−0.19** | −0.07 | −0.02 (0.58) | −0.03 |
| SB11 | **−0.22** | −0.06 | −0.20 | −0.18 | −0.14 (0.92) | −0.21 |
| SM1 | +0.15 | **−0.07** | +0.04 | +0.10 | −0.02 (0.45) | +0.07 |
| mustard0 | −0.19 | −0.08 | −0.16 | −0.15 | −0.17 (0.89) | **−0.19** |
| tomato | +0.03 | −0.02 | +0.01 | +0.04 | **−0.04** (0.58) | +0.03 |
| **all 92** | −0.046 | −0.061 | **−0.094** | −0.055 | −0.065 | −0.062 |

| translation (mm/step) | total | depth | weighted25 | normsum | gate | texture |
|---|---|---|---|---|---|---|
| AP10 | −0.18 | −0.47 | **−0.48** | −0.28 | −0.04 (0.33) | −0.46 |
| AP12 | +0.26 | +0.13 | **+0.05** | +0.39 | +0.17 (0.83) | +0.16 |
| MPM10 | +0.22 | **−0.44** | −0.33 | −0.03 | −0.06 (0.67) | +0.21 |
| MPM13 | +0.24 | +1.02 | +0.89 | +0.81 | **+0.22** (0.25) | +0.36 |
| SB11 | −0.06 | −0.47 | −0.11 | −0.20 | **−0.52** (0.58) | −0.02 |
| SM1 | **−0.81** | +0.54 | −0.25 | −0.38 | +0.09 (0.45) | −0.26 |
| mustard0 | +0.34 | **−0.28** | +0.08 | −0.10 | +0.14 (0.33) | +0.29 |
| tomato | −0.55 | −0.49 | −0.46 | −0.54 | **−0.59** (0.50) | −0.56 |
| **all 92** | −0.071 | −0.057 | −0.079 | −0.034 | −0.083 | −0.040 |

Reading (first-order, one view, v1 map — a ranking hint, not a prediction of ADD):
- No policy is uniformly best. For rotation, E2 (photometric + 25×depth) has the best average and is never the worst;
  E1 fixes AP12/SM1 (where v1 rotates the wrong way) but is wrong on AP10. F never hurts much but also gains little
  (it blocks 30–70 % of steps). G tracks E1 on the pitcher and v1 on textured objects, as designed, but inherits
  E1's AP10 miss.
- Translation is where the depth term is dangerous: MPM13 (+1.0 mm/step) and SM1 (+0.5) with E1. The
  photometric term is the better translation guide on SM1/tomato, the depth term on AP10/MPM10/SB11/mustard0.
- The effects are small (best |Δ| ≈ 0.1–0.2° per 0.5° step ⇒ cos ≈ 0.2–0.4), consistent with §6.3: the pose gradient is a
  weak signal whichever term is used; a policy choice can remove the wrong-direction cases (AP12, SM1 rotation) but
  cannot turn the feedback into an SDF-like corrector by itself.
- Suggests that stage 1 will most likely help on AP12 (E1/E2/G) and MPM10 (E2), be neutral on AP10, and that F is
  the safe but small option. To be checked against the online ADD table in §7.3.

### 7.3 Stage-1 online results (2026-09-09/10, single runs, ADD cm; `logs/gsfb_batch_20260909/summary.py`)

| seq | SDF on | tracker alone | GS v1 | C+D | E1 depth | E2 photo+25×depth | E3 normsum | F gate |
|---|---|---|---|---|---|---|---|---|
| AP12 | 0.454 | 0.888 | 2.379 | 1.389 | 7.125 | 3.335 | 4.441 | **1.882** |
| AP10 | 0.969 | 4.552 | 5.688 | 2.975 | 6.616 | 5.527 | 6.119 | **3.159** |
| MPM10 | 0.964 | 1.092 | 4.987 | 1.776 | 5.636 | 4.503 | 5.362 | **3.523** |
| SM1 | 0.551 | 4.361 | 3.433 | 3.231 | 7.421 | 3.657 | 3.544 | 3.977 |
| HO3D-4 mean | 0.73 | 2.72 | 4.12 | 2.34 | 6.70 | 4.26 | 4.87 | **3.14** |
| mustard0 | 1.380 | 0.748 | 0.638 | 0.613 | 1.614 | 0.850 | 1.357 | 1.392 |
| tomato | 2.450 | 1.211 | 1.553 | – | 5.117 | 2.580 | 1.896 | 1.850 |

Reading:
- **E1 (depth only) fails everywhere** (6/6 worse than v1, HO3D mean 6.70). **E2 and E3 do not beat v1** on HO3D and
  lose the v1 gain on YCB. **F (gate) is the only policy better than v1 on HO3D** (3.14 vs 4.12, 4/4 sequences)
  but is below C+D (2.34) and the tracker alone (2.72), and it also loses on YCB (mustard0 1.39 vs 0.64).
- **Common failure mechanism: the initial cycle.** It trains 4 000 steps on a prior-dominated map with 5 views.
  Adam is scale-invariant, so any policy whose direction is consistent step to step (depth-only, unit-vector sums)
  accumulates to the tanh clamp (E1 19.3°, E3 21.4°, E2 14.3° on AP12; F 9.8°/10.9 mm on mustard0 vs v1 5.9°/3.9 mm)
  and ADD exceeds 2 cm from the very first feedback frame (66 on AP12). The photometric-dominated v1 gradient
  partly cancels between steps and stays at 7.6°. Interpretation: the depth term at that stage aligns the cameras
  to the (mis-placed) SAM3D prior surface, i.e. it copies the prior placement error into the poses.
- F moves the pose in only ~14 % of steps online (offline estimate was 67 %; the offline probe used the newest
  keyframe on the v1 map, online the gate is evaluated on every sampled view with evolving deltas). Its HO3D gain
  is therefore mostly "less feedback", in line with D and C+D — the conditions that beat the tracker alone on AP10
  all reduce how much the poses are touched.
- The offline first-order ranking (§7.2) did not predict the online outcome (E2 best offline, F best online):
  the per-step direction quality matters less than the *accumulation* over 500–4 000 Adam steps.

Follow-ups queued (config-only, `pose_feedback.in_initial=false` via `config_gs_noinit_feedback.yml`; no code change):
E1/E2/v1 without initial-cycle feedback on AP12 (GPU0), F/v1 without it on AP12 and mustard0 plus F repeats on
AP12/AP10 (GPU1). Question: how much of each policy's loss is the initial cycle, and does F reproduce.

### 7.4 Follow-ups (2026-09-10, config-only; `in_initial=false` = no pose feedback in the initial 4 000-step cycle; ADD cm)

| seq | tracker alone | GS v1 | v1 no-init | C+D (run 1 / 2) | C+D no-init (run 1 / 2) | F gate (1 / 2) | F no-init | F+C+D | E1 no-init | E2 no-init |
|---|---|---|---|---|---|---|---|---|---|---|
| AP12 | 0.888 | 2.379 | 1.906 | 1.389 / 1.324 | **1.245 / 1.191** | 1.882 / 1.994 | 1.698 | 1.865 | 5.162 | 2.320 |
| AP10 | 4.552 | 5.688 | 4.694 | **2.975 / 3.824** | 4.828 / 5.271 | 3.159 / 2.801 | – | 3.056 | – | – |
| MPM10 | 1.092 | 4.987 | 2.325 | **1.776 / 1.942** | 4.177 / 1.750 | 3.523 | – | 3.409 | – | – |
| SM1 | 4.361 | 3.433 | 3.978 | 3.231 | 3.473 | 3.977 | – | 4.381 | – | – |
| mustard0 | 0.748 | 0.638 | 0.788 | **0.613** | 0.749 | 1.392 | 1.007 | 1.095 | – | – |
| tomato | 1.211 | 1.553 | 1.550 | – | 1.603 | 1.850 | – | 1.506 | – | – |

Reading:
- **Initial-cycle feedback is a large part of v1's HO3D harm** (v1 → v1 no-init: AP12 2.38 → 1.91, AP10 5.69 → 4.69,
  MPM10 4.99 → 2.33; SM1 unchanged) and part of its YCB gain (mustard0 0.64 → 0.79). This matches the mechanism in
  7.3: with a prior-dominated map the initial cycle copies the prior placement error into the poses (on MPM10 the
  effect is as large as fainting the prior, C).
- **On top of C+D the switch does not add up**: AP12 gains ~0.15 (reproduced), MPM10 is equal (one run lost the
  frame-700 slip), AP10 and mustard0 lose. Not adopted.
- **E1/E2 stay worse than v1 even without the initial cycle** (AP12 5.16 / 2.32): the depth-led pose gradient is
  harmful beyond the initial cycle, not only through it.
- **F does not combine with C+D** (F+C+D ≈ F alone, below C+D everywhere; loses on YCB). F is a "move less" policy
  (14 % of steps), and D's extra new-keyframe updates are mostly blocked by the gate.
- **C+D reproduces** (AP12 1.39/1.32, AP10 2.98/3.82, MPM10 1.78/1.94) and remains the leading condition. MPM10 is
  decided by one slip event near frame 700 (run-to-run spread up to 2.4 cm on any condition), AP10 spread ≈ 1 cm.
- Decision for the next step: the full 22-sequence pass runs with C+D (`gsfb_po01quota5_*_20260909`), stage-1
  pose-gradient policies are parked (E1/E2/E3 rejected; F kept as a reference "safe" policy).

### 7.5 C+D on the wider sequence set (2026-09-10; 17 of 22 done; ADD cm, single runs unless two values are given)
Runs `outputs/gsfb_po01quota5_*_2026090{8,9}`, scores `logs/gsfb_batch_2026090{8,9}/add_eval_gsfb_po01quota5_*.json`,
table `logs/gsfb_batch_20260909/summary.py`. The two batch chains (`chain_gpu0f.sh`, `chain_gpu1e.sh`) stopped after
SB13 / MPM11 on the evening of 2026-09-10 without a START line for the next sequence (no crash recorded); **MPM12, MPM13,
MPM14, SB11 and tomato are still missing** (~3.5 GPU-hours).

| HO3D | SDF on | tracker alone | GS v1 | C+D (run 1 / run 2) |
|---|---|---|---|---|
| SM1 | 0.551 | 4.361 | 3.433 | 3.231 |
| AP10 | 0.969 | 4.552 | 5.688 | 2.975 / 3.824 |
| AP11 | 0.946 | 1.341 | 2.736 | 1.718 |
| AP12 | 0.454 | 0.888 | 2.379 | 1.389 / 1.324 |
| AP13 | 0.641 | 0.969 | 1.287 | 1.104 |
| AP14 | 0.499 | 0.790 | 0.841 | 0.840 |
| MPM10 | 0.964 | 1.092 | 4.987 | 1.776 / 1.942 |
| MPM11 | 0.819 | 0.830 | 1.426 | **2.525** |
| SB13 | 0.454 | 0.673 | 0.908 | 0.934 |
| **mean (9)** | **0.700** | **1.722** | 2.632 | 1.885 (repeats averaged; 1.938 with run 2 only) |

| YCB | SDF on | tracker alone | GS v1 | C+D |
|---|---|---|---|---|
| mustard0 | 1.380 | 0.748 | 0.638 | 0.613 |
| bleach0 | 1.801 | 1.764 | 1.258 | 0.860 |
| bleach_hard_00_03 | 0.960 | 1.011 | 1.055 | 0.893 |
| cracker_box_reorient | 0.785 | 0.756 | 0.852 | 0.846 |
| cracker_box_yalehand0 | 2.849 | 2.763 | 2.655 | 2.703 |
| mustard_easy_00_02 | 0.646 | 0.784 | 0.647 | 0.620 |
| sugar_box1 | 0.976 | 0.681 | 0.771 | 0.742 |
| sugar_box_yalehand0 | 1.555 | 1.757 | 1.492 | 1.570 |
| **mean (8)** | 1.369 | 1.283 | 1.171 | **1.106** |

Reading:
- **YCB**: C+D is the best condition on average and beats the tracker alone on 6/8 sequences (losses: cracker_reorient
  +0.09, sugar_box1 +0.06, both within the YCB noise floor); it never shows the original SDF feedback's symmetric-object
  harm. This part of ⑤ works.
- **HO3D**: C+D beats the tracker alone only on the two collapsing sequences (SM1, AP10) and even there stays collapsed
  (3.2–3.8 cm vs SDF on 0.55–0.97); it is worse than the tracker alone on the other 7/9, so the 4-sequence stage-1 subset
  (the hardest sequences) over-estimated C+D. **MPM11 regresses 0.83 → 2.53**: its `feedback_log` shows 50 cycles with a
  > 5° correction (v1: 2 cycles; AP14, which is neutral, has 0) — the new-keyframe quota (D) raises the correction
  magnitude on a sequence the tracker was handling well. The remaining gap to the SDF feedback (1.9 vs 0.7) is the
  open ⑤ problem; the evidence in §3–§7 points at the pose *signal* (rasterized loss gradient + Adam accumulation on a
  prior-anchored map), not at the amount of feedback.
- Consequence for the design: a "do not touch a healthy tracker" gate is required before any further signal work,
  because every condition that touches all keyframes every cycle harms some non-collapsing HO3D sequence.

### 7.6 Per-keyframe gradient probe on two contrasting objects (2026-09-11, `logs/gradprobe_20260911_allkf`)
v1 replay, every cycle, newest keyframe, pre-append, no prior hiding; SIFT/curvature scores per keyframe from
`kf_scores.py`; plots and full table in `report_AP12_SB13.html`. Rotation cosine vs the true correction (share > 0).

| seq (probes) | tracker rot err (median) | L1 | DSSIM | depth | total (v1) | translation |
|---|---|---|---|---|---|---|
| AP12 pitcher (175, K 6–181) | 16.8° | −0.18 (35%) | −0.21 (35%) | **+0.42 (91%)** | −0.13 (39%) | no term informative (all \|cos\| ≤ 0.05) |
| SB13 bleach (201, K 6–207) | 7.1° | +0.06 (56%) | +0.08 (53%) | −0.08 (48%) | +0.08 (57%) | photometric +0.09 (62%), depth −0.23 |

Reading: at full-sequence density the §6.3 pattern holds exactly — on the low-texture pitcher only the depth term points
toward GT (91% of cycles) while the photometric terms point away and the online total follows them; on the labelled
bleach bottle every term is weak (rotation \|cos\| ≤ 0.1) and depth is harmful for translation. Translation carries no
usable signal on either object. A per-object term choice (G) would therefore fix the sign on AP12 but cannot give
SDF-like strength (best \|cos\| 0.42), consistent with E1's online failure through accumulation (§7.3).

### 7.5 ⑤-4: is the map or the loss the bottleneck? (2026-09-11, `experiments/exp_feedback_gradient_probe_gtmap.py`, `logs/gtmap_probe_20260911/report.html`)
Same keyframes and schedule as online (SAM3D prior, 4 000 + 500 steps/keyframe), map built either from the tracker
poses v1 saw ("tracker map" = the original probe) or from GT poses ("GT map"; poses locked, map only). Every cycle's
newest keyframe probed (AP12 175, SB13 201) at its tracker pose, and at GT + controlled perturbations (1/3/5/10°,
3 random axes each). Map verified against the GT mesh.

| | map | Chamfer G→mesh / mesh→G (median mm) | mesh covered ≤5 mm | depth residual @GT / @tracker pose (mm) |
|---|---|---|---|---|
| AP12 | GT | 1.9 / 2.8 | 0.70 | 4.1 / 6.5 |
| AP12 | tracker | 2.4 / 5.1 | 0.50 | 7.6 / 5.2 |
| SB13 | GT | 2.3 / 1.8 | 0.90 | 5.4 / 4.2 |
| SB13 | tracker | 2.1 / 2.0 | 0.70 | 8.6 / 3.4 |

The GT-posed map is a correct map (≈2 mm to the GT mesh both ways). The tracker-posed map fits the tracker's own
(wrong) poses better than GT — it has absorbed the tracker error.

Direction of each term's pose gradient at the newest keyframe's **tracker pose** (mean cos rotation, share > 0):

| | map | L1 | DSSIM | depth | total (v1) | translation: total |
|---|---|---|---|---|---|---|
| AP12 (err 17°) | tracker | −0.21 (34%) | −0.16 (38%) | +0.40 (91%) | −0.14 (40%) | −0.06 (47%) |
| AP12 | **GT** | **+0.60 (98%)** | +0.47 (91%) | +0.56 (96%) | **+0.64 (99%)** | **+0.65 (96%)** |
| SB13 (err 20°) | tracker | +0.04 (53%) | +0.07 (54%) | −0.07 (49%) | +0.06 (55%) | +0.12 (64%) |
| SB13 | **GT** | **+0.37 (80%)** | +0.26 (71%) | +0.16 (66%) | **+0.33 (78%)** | −0.01 (50%) |

Controlled perturbations of the GT pose (share of probes with the total gradient pointing back, rotation):

| | 1° | 3° | 5° | 10° |
|---|---|---|---|---|
| AP12 GT map | 76% | 94% | 96% | 100% |
| AP12 tracker map | 52% | 63% | 69% | 77% |
| SB13 GT map | 71% | 86% | 92% | 96% |
| SB13 tracker map | 62% | 72% | 77% | 88% |

Reading:
- **The bottleneck is the map, not the loss.** With a map built from correct poses the *same* loss terms point
  from the tracker pose toward GT in 91–99 % of the cycles on AP12 (v1 total: 40 %) and 78 % on SB13 (55 %), and
  the photometric terms — wrong 65 % of the time on the tracker map — become the best terms. Both terms are
  usable; the earlier "depth right / colour wrong" pattern on AP12 was a property of the contaminated map.
- The tracker-posed map is self-consistent with the tracker's error (its depth residual is lower at the tracker
  pose than at GT), so a gradient taken against it cannot see the error — this is why v1's corrections were
  uncorrelated with GT (§3) and why reweighting/gating the terms (stage 1) could not help.
- Basin: on the GT map the direction is right for 1° errors in 71–76 % of probes and for ≥3° in 86–100 %; on the
  tracker map it needs ≥10° to reach 77–88 %. Translation: informative on AP12 (96 %) but not on SB13 (50 %),
  which still points to a residual loss/geometry limitation for translation on that object.
- Consequence for the order of work: keeping the map's reference correct (which poses it is trained with, prior
  placement, when contaminated views enter) comes before any loss redesign. Recorded here; ⑤ is parked (§ MILESTONES).

## 9. Parked ideas for when ⑤ is resumed (2026-09-11, recorded only — not run)
1. **Map absorbs the tracker error (the ⑤-4 finding).** Hypothesis for why the original works: BundleSDF re-creates
   the SDF and the PoseArray from fresh weights every cycle and solves map + poses jointly, so the map is always
   consistent with the current joint solution and never keeps parts written with last cycle's wrong poses. Our GS
   map persists across cycles. First thing to test on return: re-fit / re-create the map from the current poses
   before the pose step, or otherwise stop contaminated views from being baked into a persistent map.
2. **Multi-view steps.** gsplat rasterizes a camera batch (viewmats [C,4,4]); C views per step give C poses a gradient
   every step (original: all keyframes every step via a pooled ray batch). Cost and memory scale ~C×; keep cycle time
   by trading steps for views (e.g. 500×1 → ~60×8). Does not by itself fix item 1. Pixel subsampling in the SDF
   style is not useful for a rasterizer (tile cost is per image); ROI crops / lower resolution are the levers.
3. Translation on SB13 stays uninformative even with the GT map (50 %) — a loss/geometry limitation to look at
   separately from item 1.
4. ⑤-4 caveat: the perturbation rows have no translation component (perturb_mm = 0), so their translation cos is
   meaningless; translation was assessed only at the tracker poses.
