# Milestone ⑤ change ledger (branch `milestone5-feedback-attempt2`, base = main `6039ee0`, uncommitted)

Every change to the main code since the safety point, by file and function, with the reason. Experiments-only
files (`experiments/`, `tests/`, `logs/`) are listed separately. Defaults keep the ④ behaviour unless a run
passes the new options.

## Main code

### `gaussian_runner.py`
- `DEFAULT_CONFIG["pose_feedback"]` (new block) and `_validate_config` (new checks): enabled, max_trans_m 0.02,
  max_rot_deg 20, lr 0.01, lr_decay 0.1, grad_max_norm 0.1, fix_first_view, in_initial, prealign_steps/prealign_lr (v2),
  new_view_interval (D), pose_loss / pose_depth_scale (stage-1 pose-loss policies). Default `enabled: False`.
- `se3_exp_batch(delta)` (new module function): batched SE(3) exponential, corrected antisymmetric generator.
- `PoseDeltas` (new class): port of BundleSDF `PoseArray` (tanh-clamped 6-DoF per view, view 0 fixed).
- `GaussianRunner.__init__`: `self.feedback_log = []`.
- `GaussianRunner.update()`: views are prepared before the point cloud; optional v2 pre-alignment
  (`prealign_views`) with `dataclasses.replace` of the frame poses; passes `new_view_indices` to `train()`.
  With `pose_feedback.enabled=False` the order of operations is equivalent to ④.
- `GaussianRunner.initialize_from_prior()`: `train(..., optimize_poses=enabled and in_initial)`.
- `GaussianRunner.train()`: new kwargs `optimize_poses`, `new_view_indices`; creates `PoseDeltas` + Adam per call,
  applies `T_i @ c2w` per sampled view, quota sampling (D), pose-gradient policy (stage 1), pose step with
  clip, bakes deltas into `view.c2w_normalized` at the end. With `optimize_poses=False` the loop is unchanged
  except `self._view_sample_counts` bookkeeping.
- New helpers: `_new_pose_deltas`, `_pose_optimizer`, `_pose_step`, `_bake_pose_deltas` (SVD re-orthonormalisation),
  `prealign_views` (v2), `view_poses_metric`.
- Import: `from dataclasses import asdict, dataclass, replace`.

### `bundlesdf.py` (`run_gaussian` only; tracker side untouched)
- `feedback` switch (`on | noop | off`) from `cfg_gs['feedback']` (default `noop` = ④); `pose_feedback` overrides
  merged into the runner config when `on`.
- `prior.skip_alignment` / `prior.skip_color_transfer` (B oracle priors).
- Per cycle: `write_back` = refined poses (`view_poses_metric`) when `on`; `poses_before_gs.txt` / `poses_after_gs.txt`
  at SPDLOG ≥ 2; cycle time + correction magnitudes logged; `gs_online/feedback_log.json` at exit.

### Not changed
`BundleTrack/` (C++), `nerf_runner.py`, `prior_lifecycle.py`, `sam3d_prior.py`, `run_sam3d_alignment.py`,
`run_custom.py`, `run_ho3d.py`, configs in the repo root (`config_gs*.yml`).

## Experiments / tests / docs (not main code)
- `experiments/run_sdf_feedback_ablation.py`: `--backend gaussian` and GS flags (`--gs_*`), prior overrides,
  manifest fields.
- `experiments/exp_feedback_gradient_probe.py` (A), `experiments/make_oracle_prior.py` (B).
- `tests/test_pose_feedback.py` (CPU + gated GPU). 2026-09-09: `test_pose_gradient_policies` forces the movable view
  every step (`new_view_indices=[1]`, `new_view_interval=1`) so 5-step trainings cannot sample only the fixed view 0.
- `MILESTONE5_FEEDBACK_RESULTS.md`, `MILESTONES.md` (⑤ paragraph), this ledger.
- Runtime YAML variants live under `logs/gsfb_batch_2026090{7,8}/` (v3 prealign lr, prior opacity 0.1).

## Audit 2026-09-11 (requested: "what differs from the pre-⑤ safety point")
Safety points: main `6039ee0` (pre-⑤-attempt-2; differs from `aa7a673` only by the empty-mask guard in `bundlesdf.py`,
commit `1886c44`) — the branch is uncommitted on top of `6039ee0`.

**Tracked files changed vs `6039ee0`** (`git diff --stat`): `gaussian_runner.py` +449/−? (net 449 lines),
`bundlesdf.py` +57, `experiments/run_sdf_feedback_ablation.py` +92, `MILESTONES.md` +2. Nothing else
(`BundleTrack/`, `mycuda/`, `nerf_runner.py`, `prior_lifecycle.py`, `sam3d_prior.py`, `run_custom.py`, `run_ho3d.py`,
repo-root configs: unchanged). Untracked additions: `MILESTONE5_CHANGE_LEDGER.md`, `MILESTONE5_FEEDBACK_RESULTS.md`,
`experiments/exp_feedback_gradient_probe.py`, `experiments/exp_feedback_gradient_probe_gtmap.py` (copy, 2026-09-11),
`experiments/make_oracle_prior.py`, `tests/test_pose_feedback.py`. Full main-code patch saved at
`logs/audit_20260911/main_code_diff_vs_6039ee0.patch`.

**`gaussian_runner.py`, verified line by line against the sections above** — the only items not spelled out before:
- `update()`: `_frames_to_cloud` / `_select_novel` moved from before `before = self.num_gaussians` (outside the
  `try`) to inside the `try`, after `new_views = [_prepare_view(...)]` (needed so v2 pre-alignment can change the
  frame poses before the cloud is built). Effect when feedback is off: identical results; only difference is that an
  exception raised by those two calls is now caught by the existing rollback (`snapshot`) instead of propagating
  before any state change.
- `train()`: `self._view_sample_counts` (new attribute, set on every call, not checkpointed); the loop variable
  `_` → `step`; `loss` is built as `photometric + depth_weight * depth_term` (same arithmetic as before).
- Everything else is additive (new config block, validator lines, `se3_exp_batch`, `PoseDeltas`, `feedback_log`,
  new kwargs with defaults, new helper methods) and inactive unless `pose_feedback.enabled` is true.

**`bundlesdf.py` (`run_gaussian` only)**: `feedback` switch; `skip_color_transfer` / `skip_alignment` branches (default
false = old path); `write_back` prepared before the `try` and published in `finally` (default `noop` publishes the same
verbatim copy as before); per-cycle `[GS backend] cycle …` log line; `poses_before_gs.txt` / `poses_after_gs.txt`
written at SPDLOG ≥ 2 (new files in the run directory); `gs_online/feedback_log.json` at exit. Tracker side untouched.

**Numeric equivalence (`logs/audit_20260911/equiv_check.py`)**: the current runner with defaults (feedback off) and the
`6039ee0` runner on the tests' synthetic RGB-D scene (initialize + 2 updates, 2DGS and 3DGS): losses identical to 6
decimals, Gaussian counts identical, stored view poses bit-identical, and parameter differences (≤ 5e-3) are the same
size as run-to-run noise of either module alone (gsplat rasterization is non-deterministic on GPU). Conclusion: with
`pose_feedback.enabled=False` the main code behaves as at the safety point.
