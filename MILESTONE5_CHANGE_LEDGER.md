# Milestone ⑤ change ledger (branch `milestone5-v1`, base = main `6039ee0`; full attempt archived at `4a84dc8`)

Every change to the main code since the safety point, by file and function, with the reason. Experiments-only
files (`experiments/`, `tests/`, `logs/`) are listed separately. Defaults keep the ④ behaviour unless a run
passes the new options.

## Main code (branch `milestone5-v1`, 2026-09-11: reduced to safety point + v1 only)

The attempt-2 working state (v1 + v2 pre-alignment + v3 config + D quota sampler + stage-1 pose-loss policies +
oracle-prior flags) is archived as commit `4a84dc8` on `milestone5-feedback-attempt2`. This branch keeps only v1.

### `gaussian_runner.py` (+247 lines vs `6039ee0`; nothing removed except the two `self.train(steps)` calls below)
- `DEFAULT_CONFIG["pose_feedback"]` (new block): `enabled` (False), `max_trans_m` 0.02, `max_rot_deg` 20,
  `lr` 0.01, `lr_decay` 0.1, `grad_max_norm` 0.1, `fix_first_view` True, `in_initial` True.
- `_validate_config`: finite/non-negative checks for the four magnitudes, `lr_decay` in (0, 1].
- `se3_exp_batch(delta)` (new module function): batched SE(3) exponential of `[trans(3), rotvec(3)]`.
- `PoseDeltas` (new class): port of BundleSDF `PoseArray` — tanh-clamped 6-DoF per view, view 0 fixed.
- `GaussianRunner.__init__`: `self.feedback_log = []`.
- `GaussianRunner.update()`: `self.train(steps)` → `self.train(steps, optimize_poses=pose_feedback.enabled)`.
- `GaussianRunner.initialize_from_prior()`: `self.train(steps)` → `self.train(steps, optimize_poses=enabled and in_initial)`.
- `GaussianRunner.train()`: new kwarg `optimize_poses` (None = config). When true: fresh `PoseDeltas` + Adam
  (lr, exponential decay to `lr_decay` over the call) per call, `c2w = T_i @ c2w` for the sampled view,
  `pose_optimizer.zero_grad`, inf-norm gradient clip `grad_max_norm`, step, and at the end
  `_bake_pose_deltas` (SVD re-orthonormalised poses written into `view.c2w_normalized`, magnitudes logged).
  When false the loop body is unchanged from `6039ee0`.
- New helpers: `_new_pose_deltas`, `_pose_optimizer`, `_pose_step`, `_bake_pose_deltas`, `view_poses_metric()`.

### `bundlesdf.py` (`run_gaussian` only, +39 lines; tracker side untouched)
- `feedback` switch from `cfg_gs['feedback']`: `on` (GS-refined poses written back), `noop` (default; tracker poses
  back verbatim = ④), `off` (nothing published). `on` merges `cfg_gs['pose_feedback']` into the runner config.
- Per cycle: `write_back` prepared before the `try`; refined poses via `view_poses_metric()` when `on`; feedback
  magnitudes and cycle time logged; `poses_before_gs.txt` / `poses_after_gs.txt` at SPDLOG ≥ 2; publish in
  `finally` (the former two-line no-op write-back is the `noop` branch of the same statement).
- At exit: `gs_online/feedback_log.json`.

### Not changed
`BundleTrack/` (C++), `nerf_runner.py`, `prior_lifecycle.py`, `sam3d_prior.py`, `run_sam3d_alignment.py`,
`run_custom.py`, `run_ho3d.py`, configs in the repo root (`config_gs*.yml`).

## Experiments / tests / docs (not main code)
- `experiments/run_sdf_feedback_ablation.py`: `--backend gaussian`, `--gs_runner_config`, `--gs_initial_steps`,
  `--gs_update_steps`, `--gs_prior_root`; `resolve_prior_paths`, `configure_backend`; manifest `backend` / `gs`.
- `experiments/exp_feedback_gradient_probe.py` (A), `experiments/exp_feedback_gradient_probe_gtmap.py` (⑤-4 copy),
  `experiments/make_oracle_prior.py` (B; its oracle priors need the archived `skip_*` flags to run online).
- `tests/test_pose_feedback.py`: CPU (se3, PoseDeltas clamps/fixed first, normalization consistency, round trip,
  bake, config validation) + gated GPU (v1 joint training reduces an injected 2° error, update() logs one record
  per cycle, disabled path logs nothing) — 2DGS and 3DGS.
- `MILESTONE5_FEEDBACK_RESULTS.md`, `OUTPUTS_INDEX.md`, `MILESTONES.md` (⑤ paragraph), this ledger.

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

## Reduction to v1 (2026-09-11)
Removed from the main code relative to `4a84dc8`: v2 pre-alignment (`prealign_views`, `prealign_steps/lr`,
`dataclasses.replace`, the `update()` reordering — original order restored), D quota sampler (`new_view_interval`,
`new_view_indices`, `_view_sample_counts`), stage-1 policies (`pose_loss`, `pose_depth_scale`,
`_pose_gradient_policy`, photometric/depth split, gate counters, extra record fields), oracle-prior branches in
`bundlesdf.py` (`skip_alignment`, `skip_color_transfer`) and the matching driver flags / tests.
Checks on the reduced code: full CPU+GPU suites 73 tests OK; feedback off ≡ `6039ee0` (equiv_check, GPU-noise level);
feedback on ≡ `4a84dc8` with default policy (equiv_check_on); online v1 reproduction (`logs/v1check_20260911`): mustard0 feedback on ADD 0.650 / ADD-S 0.287 (attempt-2 v1 run: 0.638 / 0.28; first-cycle correction 5.99° / 4.04 mm vs 5.92° / 3.93 mm) — reproduced within run-to-run noise.
