"""EXPERIMENT: milestone-③ prototype — prior-initialized map + lifecycle.

Runs the offline replay with the SAM3D surfel prior as the initial map and
the observation-gated lifecycle managing it. The main pipeline is untouched:
`GaussianRunner` is subclassed here, and the lifecycle fields live alongside
the splats inside this experiment.

Arms:
  lifecycle          prior init + states (freeze non-VERIFIED, remove
                     CONTRADICTED)
  prior_nolifecycle  prior init, everything trainable (corruption control)

Example:
  python3 experiments/exp_prior_lifecycle.py \
    --track-dir logs/mustard0_bundlesam3dgs_..._20260811 \
    --mesh-npz .../mustard0_mesh_depth.npz \
    --pose-json .../mustard0_mesh_depth.json \
    --gaussian-ply .../mustard0_splat_depth.ply \
    --config config_gs_2dgs_1mm.yml --arm lifecycle \
    --output-dir logs/mustard0_lifecycle_<date> --device cuda:0
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gaussian_runner import (  # noqa: E402
    GaussianRunner,
    SceneNormalization,
    load_gaussian_config,
)
from run_gaussian_incremental import (  # noqa: E402
    find_latest_keyframe_snapshot,
    load_fixed_poses,
    load_frame,
    validate_replay_paths,
)
from sam3d_prior import (  # noqa: E402
    load_mesh_prior,
    load_sam3d_gaussian_ply,
    load_sam3d_pose,
    quat_wxyz_to_matrix,
    sample_surfels,
    transfer_gaussian_colors,
    transform_surfels_canonical_to_cv_camera,
)
from experiments.prior_lifecycle import (  # noqa: E402
    STATE_CONTRADICTED,
    STATE_UNSEEN,
    STATE_VERIFIED,
    LifecycleFields,
    TransitionThresholds,
    accumulate_support_residuals,
    apply_transitions,
    depth_evidence_masks,
    independent_view_mask,
)
from experiments.bias_field import (  # noqa: E402
    BiasFieldConfig,
    estimate_bias_offsets,
)

SH_C0 = 0.28209479177387814


def erode_mask(mask: torch.Tensor, radius_px: int) -> torch.Tensor:
    """Binary erosion via inverted max-pooling (mask: [H, W] bool)."""

    if radius_px <= 0:
        return mask
    kernel = 2 * radius_px + 1
    inverted = (~mask).float()[None, None]
    dilated = F.max_pool2d(inverted, kernel, stride=1, padding=radius_px)
    return ~(dilated[0, 0] > 0.5)


class LifecycleRunner(GaussianRunner):
    """GaussianRunner + SAM3D prior initialization + observation lifecycle."""

    def __init__(self, *args, lifecycle_enabled: bool = True,
                 thresholds: TransitionThresholds | None = None,
                 bias_enabled: bool = False,
                 bias_config: BiasFieldConfig | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.lifecycle_enabled = bool(lifecycle_enabled)
        self.thresholds = thresholds or TransitionThresholds()
        self.fields: LifecycleFields | None = None
        self.initial_prior_means_metric: np.ndarray | None = None
        self.lifecycle_log: list[dict] = []
        self.bias_enabled = bool(bias_enabled)
        self.bias_config = bias_config or BiasFieldConfig()
        self.applied_offset: torch.Tensor | None = None
        self.bias_log: list[dict] = []

    # ----- prior initialization -------------------------------------------

    def initialize_from_prior(self, surfels_cv, first_c2w_cv, frames,
                              train_steps):
        """Aligned first-camera-frame surfels → object frame → initial map."""

        if self.is_initialized:
            raise RuntimeError("already initialized")
        R0 = torch.from_numpy(first_c2w_cv[:3, :3].astype(np.float32))
        t0 = torch.from_numpy(first_c2w_cv[:3, 3].astype(np.float32))
        means_metric = surfels_cv.means @ R0.T + t0[None, :]
        normals_obj = F.normalize(surfels_cv.normals @ R0.T, dim=-1)
        radii_metric = surfels_cv.radii

        means_norm = torch.from_numpy(
            self.normalization.normalize_points(means_metric.numpy())
        )
        radii_norm = radii_metric * float(self.normalization.scale)
        quats = self._quats_from_normals(normals_obj)
        log_scales = torch.log(torch.stack(
            (radii_norm, radii_norm, radii_norm * 0.1), dim=-1
        ).clamp_min(1e-9))
        opacity_logit = float(torch.logit(torch.tensor(0.9)))
        sh_count = (int(self.config["sh_degree"]) + 1) ** 2
        count = len(means_norm)
        values = {
            "means": means_norm.float(),
            "scales": log_scales.float(),
            "quats": quats.float(),
            "opacities": torch.full((count,), opacity_logit),
            "sh0": ((surfels_cv.colors - 0.5) / SH_C0)[:, None, :].float(),
            "shN": torch.zeros((count, sh_count - 1, 3), dtype=torch.float32),
        }
        self._set_splats(values)
        self.fields = LifecycleFields.create(
            count, lineage_prior=True, device=self.device
        )
        if not self.lifecycle_enabled:
            # corruption-control arm: everything behaves as verified/trainable
            self.fields.state[:] = STATE_VERIFIED
        self.initial_prior_means_metric = means_metric.numpy().copy()
        self.observed_points_metric = means_metric.numpy().astype(np.float32)
        self.observed_colors = surfels_cv.colors.numpy().astype(np.float32)
        self.applied_offset = torch.zeros(count, device=self.device)
        self.views = [self._prepare_view(frame) for frame in frames]
        self.update_index = 0
        # Classify and compact BEFORE creating optimizers so Adam rows align.
        self.classify(frames, event="initialize")
        self.remove_contradicted()
        self.apply_bias_correction(event="initialize")
        self._reset_optimization_state(
            float(self.config["initial_lr_scale"]), initial=True
        )
        return self.train(train_steps)

    @staticmethod
    def _quats_from_normals(normals: torch.Tensor) -> torch.Tensor:
        from sam3d_prior import quats_from_normals

        return quats_from_normals(normals)

    # ----- lifecycle ------------------------------------------------------

    def _splat_normals(self) -> torch.Tensor:
        R = quat_wxyz_to_matrix(self.splats["quats"].detach())
        return R[:, :, 2]

    @torch.no_grad()
    def _geometric_front_depth(self, c2w_norm: torch.Tensor, K: torch.Tensor,
                               width: int, height: int):
        """Opacity-independent front depth: 2DGS median depth at opacity 1."""

        from gsplat import rasterization_2dgs

        renders, alphas, *_ = rasterization_2dgs(
            means=self.splats["means"].detach(),
            quats=self.splats["quats"].detach(),
            scales=torch.exp(self.splats["scales"].detach()),
            opacities=torch.ones(
                self.num_gaussians, device=self.device
            ),
            colors=torch.zeros(
                self.num_gaussians, 1, 3, device=self.device
            ),
            sh_degree=0,
            viewmats=torch.linalg.inv(c2w_norm)[None],
            Ks=K[None],
            width=width,
            height=height,
            render_mode="RGB+ED",
            depth_mode="median",
        )
        depth_norm = renders[0, ..., 3]
        alpha = alphas[0, ..., 0]
        return depth_norm / float(self.normalization.scale), alpha

    @torch.no_grad()
    def classify(self, frames, event: str):
        """Update lifecycle states from a batch of keyframes (metric space)."""

        if self.fields is None or not self.lifecycle_enabled:
            return
        fields = self.fields.validated()
        device = self.device
        thresholds = self.thresholds
        state_entry = fields.state.clone()
        active = state_entry != STATE_CONTRADICTED
        supported = torch.zeros(len(fields), dtype=torch.bool, device=device)

        means_metric = torch.from_numpy(
            self.normalization.metric_points(
                self.splats["means"].detach().cpu().numpy()
            )
        ).to(device)
        normals = self._splat_normals()
        totals = {"projected_valid": 0, "support": 0, "free_space": 0,
                  "behind_occluded": 0, "behind_miss": 0,
                  "accepted_conflict": 0}

        for frame in frames:
            frame = frame.validated()
            c2w = torch.from_numpy(frame.c2w_cv.astype(np.float32)).to(device)
            w2c = torch.linalg.inv(c2w)
            points_cam = means_metric @ w2c[:3, :3].T + w2c[:3, 3][None, :]
            depth_cam = points_cam[:, 2]
            safe = depth_cam.clamp_min(1e-6)
            K = frame.K
            height, width = frame.mask.shape
            u = points_cam[:, 0] / safe * float(K[0, 0]) + float(K[0, 2])
            v = points_cam[:, 1] / safe * float(K[1, 1]) + float(K[1, 2])
            in_image = ((depth_cam > 0.01) & (u >= 0) & (u < width)
                        & (v >= 0) & (v < height))
            ui = u.round().clamp(0, width - 1).long()
            vi = v.round().clamp(0, height - 1).long()

            observed = torch.from_numpy(frame.depth).to(device)
            mask = erode_mask(
                torch.from_numpy(frame.mask).to(device),
                thresholds.mask_erode_px,
            )
            valid_pixels = (mask & torch.isfinite(observed)
                            & (observed > float(self.config["min_depth"]))
                            & (observed < float(self.config["max_depth"])))
            valid = active & in_image & valid_pixels[vi, ui]

            c2w_norm = torch.from_numpy(
                self.normalization.normalize_c2w(frame.c2w_cv).astype(
                    np.float32
                )
            ).to(device)
            K_t = torch.from_numpy(frame.K.astype(np.float32)).to(device)
            front_depth, front_alpha = self._geometric_front_depth(
                c2w_norm, K_t, width, height
            )

            camera_center = c2w[:3, 3]
            view_dir = F.normalize(
                means_metric - camera_center[None, :], dim=-1, eps=1e-8
            )
            view_abs_cos = (normals * view_dir).sum(dim=-1).abs()
            evidence = depth_evidence_masks(
                gaussian_depth=depth_cam,
                observed_depth=observed[vi, ui],
                valid_observation=valid,
                geometric_front_depth=front_depth[vi, ui],
                geometric_alpha=front_alpha[vi, ui],
                depth_tolerance=thresholds.depth_tolerance_m,
                geometric_alpha_threshold=thresholds.geometric_alpha_threshold,
                view_abs_cos=view_abs_cos,
                grazing_tolerance_cap=thresholds.grazing_tolerance_cap,
            )
            supported |= evidence["support"]
            accumulate_support_residuals(
                fields, evidence["support"], depth_cam, observed[vi, ui],
                normals, view_dir,
            )
            candidate = evidence["free_space"] | evidence["behind_miss"]
            accepted = independent_view_mask(
                candidate, fields.conflict_count, fields.last_conflict_view,
                view_dir, thresholds.min_view_angle_deg,
            )
            if bool(accepted.any()):
                fields.conflict_count[accepted] += 1
                fields.last_conflict_view[accepted] = view_dir[accepted]

            totals["projected_valid"] += int(valid.sum())
            for key in ("support", "free_space", "behind_occluded",
                        "behind_miss"):
                totals[key] += int(evidence[key].sum())
            totals["accepted_conflict"] += int(accepted.sum())

        masks = apply_transitions(fields, supported, thresholds)
        newly_contradicted = masks["to_contradict"]
        if bool(newly_contradicted.any()):
            self.splats["opacities"].data[newly_contradicted] = -10.0
        record = {
            "event": event,
            "n_frames": len(frames),
            "new_verified": int(masks["to_verify"].sum()),
            "new_suspect": int(masks["verified_to_suspect"].sum()),
            "new_contradicted": int(newly_contradicted.sum()),
            **totals,
            "state": fields.summary(),
        }
        self.lifecycle_log.append(record)
        print("[lifecycle] " + json.dumps(record, sort_keys=True))

    @torch.no_grad()
    def remove_contradicted(self):
        if self.fields is None or not self.lifecycle_enabled:
            return 0
        drop = self.fields.state == STATE_CONTRADICTED
        n_drop = int(drop.sum())
        if n_drop == 0:
            return 0
        keep = ~drop
        values = {name: tensor.detach()[keep].cpu()
                  for name, tensor in self.splats.items()}
        self._set_splats(values)
        self.fields = self.fields.keep(keep)
        if self.applied_offset is not None:
            self.applied_offset = self.applied_offset[keep]
        return n_drop

    # ----- ③b bias-extrapolation correction ------------------------------

    @torch.no_grad()
    def apply_bias_correction(self, event: str):
        """Move UNSEEN prior splats by the extrapolated residual field.

        Incremental: only the difference between the newly estimated target
        offset and what was already applied is added, so repeated events
        refine rather than double-apply. VERIFIED splats train directly and
        are never moved here.
        """

        if (not self.bias_enabled or not self.lifecycle_enabled
                or self.fields is None):
            return
        lineage = self.fields.lineage
        if not bool(lineage.any()):
            return
        rows = torch.where(lineage)[0]
        means_metric = torch.from_numpy(
            self.normalization.metric_points(
                self.splats["means"].detach()[rows].cpu().numpy()
            )
        )
        normals = self._splat_normals()[rows].cpu()
        offsets, info = estimate_bias_offsets(
            means_metric,
            normals,
            self.fields.residual_sum[rows].cpu(),
            self.fields.residual_count[rows].cpu(),
            self.bias_config,
        )
        unseen_local = (self.fields.state[rows] == STATE_UNSEEN).cpu()
        delta = offsets - self.applied_offset[rows].cpu()
        delta[~unseen_local] = 0.0
        moved = int((delta.abs() > 1e-6).sum())
        if moved:
            displacement_metric = delta[:, None] * normals
            displacement_norm = (
                displacement_metric * float(self.normalization.scale)
            ).to(self.device)
            self.splats["means"].data[rows] += displacement_norm
            self.applied_offset[rows] += delta.to(self.device)
        record = {"event": event, "moved_unseen": moved, **info}
        self.bias_log.append(record)
        print("[bias] " + json.dumps(
            {k: (round(v, 4) if isinstance(v, float) else v)
             for k, v in record.items()}, sort_keys=True))

    # ----- training with freeze ------------------------------------------

    def _freeze_rows(self) -> torch.Tensor | None:
        if self.fields is None or not self.lifecycle_enabled:
            return None
        frozen = self.fields.state != STATE_VERIFIED
        return frozen if bool(frozen.any()) else None

    def train(self, steps, **kwargs):
        """Wrap the parent loop step-by-step to mask frozen-row gradients.

        The parent's loop applies optimizer.step() inside; to freeze rows we
        run it one step at a time and zero the gradients via a hook installed
        on the parameters (called during backward, before step).
        """

        frozen = self._freeze_rows()
        handles = []
        if frozen is not None:
            frozen_dev = frozen.to(self.device)

            def make_hook():
                def hook(grad):
                    grad = grad.clone()
                    grad[frozen_dev] = 0
                    return grad
                return hook

            for parameter in self.splats.values():
                handles.append(parameter.register_hook(make_hook()))
        try:
            return super().train(steps, **kwargs)
        finally:
            for handle in handles:
                handle.remove()

    # ----- lean update (append → classify → remove → train) ---------------

    def update_with_lifecycle(self, frames, train_steps):
        validated = [frame.validated() for frame in frames]
        points, colors, raw, candidates = self._frames_to_cloud(validated)
        novel_points, novel_colors, _ = self._select_novel(points, colors)
        before = self.num_gaussians
        if len(novel_points):
            # Parent convention: _append_splats reads the pre-append history
            # for its kNN scale reference; extend the history afterwards.
            self._append_splats(novel_points, novel_colors)
            self.observed_points_metric = np.concatenate(
                (self.observed_points_metric, novel_points), axis=0
            )
            self.observed_colors = np.concatenate(
                (self.observed_colors, novel_colors), axis=0
            )
            if self.fields is not None:
                appended = LifecycleFields.create(
                    len(novel_points), lineage_prior=False, device=self.device
                )
                self.fields = self.fields.concat(appended)
            if self.applied_offset is not None:
                self.applied_offset = torch.cat((
                    self.applied_offset,
                    torch.zeros(len(novel_points), device=self.device),
                ))
        self.views.extend(self._prepare_view(frame) for frame in validated)
        self.update_index += 1
        self.classify(validated, event=f"update_{self.update_index:03d}")
        removed = self.remove_contradicted()
        self.apply_bias_correction(event=f"update_{self.update_index:03d}")
        self._reset_optimization_state(
            float(self.config["update_lr_scale"]), initial=False
        )
        first_loss, final_loss = self.train(train_steps)
        return {
            "update_index": self.update_index,
            "novel": int(len(novel_points)),
            "removed_contradicted": int(removed),
            "gaussians_before": before,
            "gaussians_after": self.num_gaussians,
            "first_loss": first_loss,
            "final_loss": final_loss,
        }

    # ----- outputs --------------------------------------------------------

    def export_state_ply(self, path: Path):
        colors_by_state = np.array(
            [[128, 128, 128], [40, 200, 60], [240, 200, 40], [220, 40, 40]],
            dtype=np.uint8,
        )  # UNSEEN grey, VERIFIED green, SUSPECT yellow, CONTRADICTED red
        means = self.normalization.metric_points(
            self.splats["means"].detach().cpu().numpy()
        )
        state = (self.fields.state.cpu().numpy()
                 if self.fields is not None
                 else np.ones(len(means), dtype=np.int8))
        rgb = colors_by_state[state.clip(0, 3)]
        header = ("ply\nformat ascii 1.0\n"
                  f"element vertex {len(means)}\n"
                  "property float x\nproperty float y\nproperty float z\n"
                  "property uchar red\nproperty uchar green\n"
                  "property uchar blue\nend_header\n")
        with open(path, "w") as f:
            f.write(header)
            for p, c in zip(means, rgb):
                f.write(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f} "
                        f"{c[0]} {c[1]} {c[2]}\n")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--track-dir", type=Path, required=True)
    parser.add_argument("--mesh-npz", type=Path, required=True)
    parser.add_argument("--pose-json", type=Path, required=True,
                        help="refined RTS json (alignment output) or SAM3D "
                             "pose json")
    parser.add_argument("--gaussian-ply", type=Path, required=True,
                        help="for the arm-C color transfer")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--arm", choices=("lifecycle", "prior_nolifecycle"),
                        required=True)
    parser.add_argument("--bias-correction", action="store_true",
                        help="③b: extrapolated residual-field correction of "
                             "UNSEEN prior splats")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--initial-keyframes", type=int, default=5)
    parser.add_argument("--max-keyframes", type=int, default=36)
    parser.add_argument("--initial-steps", type=int, default=2000)
    parser.add_argument("--update-steps", type=int, default=200)
    parser.add_argument("--surfel-count", type=int, default=20000)
    return parser.parse_args()


def load_refined_pose(path: Path):
    """Accept either the alignment cache or a raw SAM3D pose json."""

    with path.open() as f:
        meta = json.load(f)
    if "refined" in meta:
        from sam3d_prior import Sim3Pose

        rp = meta["refined"]["sam3d_row_pose"]
        quat = torch.tensor(rp["rotation"], dtype=torch.float32)
        return Sim3Pose(
            scale=torch.tensor(rp["scale"], dtype=torch.float32),
            R_row=quat_wxyz_to_matrix(quat[None])[0],
            T=torch.tensor(rp["translation"], dtype=torch.float32),
        )
    return load_sam3d_pose(path)


def main():
    args = parse_args()
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=False)

    config = load_gaussian_config(args.config)
    # Milestone-③ v1 constraint (approved): densify/prune and opacity reset
    # stay OFF under the lifecycle; density coupling is deferred to ③c.
    for block in ("strategy", "update_strategy"):
        config[block] = dict(config.get(block, {}))
        config[block]["refine_start_iter"] = 1_000_000_000
        config[block]["refine_stop_iter"] = 1_000_000_001
    config["strategy"]["reset_every"] = 1_000_000_000
    snapshot = find_latest_keyframe_snapshot(args.track_dir)
    frame_ids, poses = load_fixed_poses(snapshot)
    frame_ids = frame_ids[: args.max_keyframes]
    poses = poses[: args.max_keyframes]
    validate_replay_paths(args.track_dir, frame_ids)
    K = np.loadtxt(args.track_dir / "cam_K.txt").reshape(3, 3).astype(
        np.float32
    )
    frames = [
        load_frame(args.track_dir, frame_id, K, pose)
        for frame_id, pose in zip(frame_ids, poses)
    ]
    initial = frames[: args.initial_keyframes]
    rest = frames[args.initial_keyframes:]

    # Scene normalization from the prior itself (it covers the full object).
    prior = load_mesh_prior(args.mesh_npz)
    pose0 = load_refined_pose(args.pose_json)
    surfels = sample_surfels(prior, args.surfel_count, seed=0,
                             radius_multiplier=0.75, opacity=0.9)
    gaussian = load_sam3d_gaussian_ply(args.gaussian_ply)
    surfels, transfer_info = transfer_gaussian_colors(surfels, gaussian)
    surfels_cv = transform_surfels_canonical_to_cv_camera(surfels, pose0)

    first_c2w = frames[0].c2w_cv.astype(np.float64)
    means_obj = (surfels_cv.means.numpy() @ first_c2w[:3, :3].T
                 + first_c2w[:3, 3][None, :])
    center = means_obj.mean(axis=0)
    radius = float(np.linalg.norm(means_obj - center, axis=1).max())
    normalization = SceneNormalization(
        scale=1.0 / max(radius * 1.2, 1e-6), translation=-center
    )

    runner = LifecycleRunner(
        dict(config), normalization, device=args.device,
        lifecycle_enabled=(args.arm == "lifecycle"),
        bias_enabled=bool(args.bias_correction),
    )
    runner.initialize_from_prior(
        surfels_cv, frames[0].c2w_cv, initial, args.initial_steps
    )
    updates = []
    for frame in rest:
        updates.append(runner.update_with_lifecycle([frame],
                                                    args.update_steps))

    # ----- final metrics --------------------------------------------------
    rows = []
    for index, view in enumerate(runner.views):
        rgb, alpha, depth = runner.render(index, include_depth=True)
        target = view.rgb.numpy()
        mask = view.mask.numpy()
        rgb_mae = float(np.abs(rgb - target)[mask].mean())
        gt = view.depth.numpy()
        valid = (mask & np.isfinite(gt) & (gt > config["min_depth"])
                 & (gt < config["max_depth"]) & (alpha > 0.5))
        depth_mae = (float(np.abs(depth - gt)[valid].mean() * 1000)
                     if valid.any() else float("nan"))
        pred = alpha > 0.5
        iou = float((pred & mask).sum() / max((pred | mask).sum(), 1))
        rows.append((rgb_mae, depth_mae, iou))
    metrics = {
        "rgb_mae": float(np.nanmean([r[0] for r in rows])),
        "depth_mae_mm": float(np.nanmean([r[1] for r in rows])),
        "alpha_iou": float(np.nanmean([r[2] for r in rows])),
    }

    # displacement of surviving prior-lineage splats vs their initial position
    displacement = {}
    if runner.initial_prior_means_metric is not None:
        current = runner.normalization.metric_points(
            runner.splats["means"].detach().cpu().numpy()
        )
        if runner.fields is not None:
            lineage = runner.fields.lineage.cpu().numpy()
            state = runner.fields.state.cpu().numpy()
        else:
            lineage = np.zeros(len(current), dtype=bool)
            lineage[: len(runner.initial_prior_means_metric)] = True
            state = np.full(len(current), STATE_VERIFIED, dtype=np.int8)
        # surviving prior splats are a prefix only if none were removed;
        # match by nearest initial prior point instead (robust to removal).
        from scipy.spatial import cKDTree

        tree = cKDTree(runner.initial_prior_means_metric)
        idx = np.where(lineage)[0]
        if len(idx):
            d0, _ = tree.query(current[idx], k=1, workers=-1)
            for name, mask_states in (("unseen", (STATE_UNSEEN,)),
                                      ("verified", (STATE_VERIFIED,))):
                sel = np.isin(state[idx], mask_states)
                if sel.any():
                    displacement[name + "_mm"] = {
                        "mean": float(d0[sel].mean() * 1000),
                        "p95": float(np.percentile(d0[sel], 95) * 1000),
                    }

    result = {
        "arm": args.arm,
        "config": str(args.config),
        "transfer": transfer_info,
        "metrics": metrics,
        "prior_displacement": displacement,
        "final_state": (runner.fields.summary()
                        if runner.fields is not None else None),
        "updates": updates,
        "lifecycle_log": runner.lifecycle_log,
        "bias_log": runner.bias_log,
    }
    with (out_dir / "result.json").open("w") as f:
        json.dump(result, f, indent=2)
    runner.export_state_ply(out_dir / "state_colored.ply")
    torch.save(
        {"fields": runner.fields.__dict__ if runner.fields else None},
        out_dir / "lifecycle_fields.pt",
    )
    runner.save_checkpoint(out_dir / "checkpoint_final.pt")
    print(json.dumps({"arm": args.arm, **metrics,
                      "displacement": displacement,
                      "final_state": result["final_state"]}, indent=1))


if __name__ == "__main__":
    main()
