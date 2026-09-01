# Milestone ③ — observation-gated prior lifecycle (2026-09-01)

Prototype lives in `experiments/` (main pipeline untouched):
`prior_lifecycle.py` (evidence/transition math, 17 CPU tests),
`exp_prior_lifecycle.py` (GaussianRunner subclass: prior-initialized map +
lifecycle + freeze + removal), `bias_field.py`/`test_bias_field.py` (③b).
Design ported from the sibling BundleGS implementation (gs_core_utils.py /
gaussian_runner.py, research.md 2026-08-03) with two extensions: a SUSPECT
hysteresis state and a grazing-aware support test.

States: UNSEEN(0)→VERIFIED(1)→SUSPECT(2)→CONTRADICTED(3, terminal).
SAM3D prior starts UNSEEN; RGB-D appends start VERIFIED. Evidence per
keyframe: support / free_space / behind_occluded (protected via an
opacity-independent 2DGS median-depth front render) / behind_miss, gated by
independent-view direction (≥10°); support resets conflicts. Policy:
only VERIFIED trains; CONTRADICTED is made transparent and removed at
update boundaries; densify/prune and opacity reset stay OFF (v1 constraint).
Thresholds: 1 cm tolerance, 2/3 conflict views, grazing cap 3×.

## mustard0 replay, 36 KF, 30k+500 steps, prior = 20k arm-C surfels

| arm | rgb MAE | depth MAE | IoU | back-side treatment |
|---|---|---|---|---|
| prior + no lifecycle | 0.0404 | 2.94 mm | 0.463 | drifts 3.2 mm mean (max 37.6) |
| lifecycle | 0.0460 | 4.59 | 0.336 | frozen (0 displacement) |
| **lifecycle + grazing-aware support ("A")** | **0.0406** | 4.11 | 0.330 | frozen (0) |
| lifecycle + A + ③b bias | 0.0429 | 4.08 | 0.344 | corrected (rejected, below) |

A(스침각 보정): support tolerance scaled by 1/|cos(normal,ray)| capped at
3× — rescues silhouette-rim splats (initial verified 6,611→8,114) and cuts
false contradictions 591→25 while keeping back-side displacement at 0.

## Ground-truth verdict (GT mesh distance of the never-observed region, mm)

| back-side treatment | mean / median |
|---|---|
| **frozen (lifecycle)** | **4.88 / 3.70 — best** |
| trained freely (no lifecycle) | 5.56 / 4.55 |
| ③b bias-corrected | 5.80 / 5.23 — worst |

**Decisions.**
1. **③ v1 = lifecycle + A, ③b excluded.** Freezing is the GT-best treatment
   of unobservable geometry.
2. **③b (bias extrapolation) v1 rejected**: the global anisotropic stage
   misreads the view-direction-dependent SAM3D bias as an object-frame
   scale error and extrapolates it to the back side with the wrong sign.
   The 22% hold-out CV win was visible-hemisphere *interpolation* and did
   not represent true back-side *extrapolation* (CV design caveat).
   The idea is parked for a view-direction-conditioned residual model.
3. **View-metric gap reinterpreted**: rendering VERIFIED-only equals the
   full map (4.10/0.330 vs 4.11/0.330) — frozen UNSEEN contributes nothing
   to view metrics. The no-lifecycle IoU advantage is largely *corruption
   used as silhouette filler* (its back side moved and got GT-worse), i.e.
   view metrics mis-score the corruption as gain. The remaining honest gap
   (depth 4.11 vs 2.94 on VERIFIED training dynamics) stays an open item.

Runs: `logs/mustard0_{priornolc,lifecycle,lifecycle_grazingA,lifecycle_A_bias}_full_20260901/`
(each: result.json, state_colored.ply, viewer_state.html, lifecycle_fields.pt).
Support-residual accumulators (③b input) remain in the fields for future use.
