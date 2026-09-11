"""EXPERIMENT ⑤-A: how much pose information does the GS map's gradient carry?

Milestone-⑤ attempt-2 found that the online GS pose corrections are almost
uncorrelated with the true (GT) pose error on HO3D (sign-flip test: toward
ratio 0.42 -> 0.50 when reversed, alignment ~0).  This probe measures the
pose gradient itself, offline, on the map the online feedback saw:

1. replay a saved online run (``outputs/gsfb_*``) cycle by cycle — same
   prior/alignment/normalization recipe as ``bundlesdf.run_gaussian``, the
   tracker poses each cycle actually received (``poses_before_gs.txt``),
   append + 500 steps with poses FIXED (no deltas);
2. at selected keyframe counts K, for the keyframes that arrive in that cycle:
   - regime ``pre_append``: map from cycles < c, new keyframe not in it yet;
   - regime ``post_train``: after append + training (the online v1 regime);
   compute the gradient of each loss term (RGB L1, DSSIM, depth Huber, total)
   w.r.t. a 6-DoF world-side delta at the tracker pose, and compare it with
   the true correction toward GT: cosine of the rotation / translation parts,
   and the change of the GT rotation error after a fixed 0.5-degree descent
   step.  Optionally the prior surfels (UNSEEN only, or all prior lineage)
   are hidden (opacity -> 0) for the render, to test the "map frame is
   anchored by the prior" hypothesis;
   - plus a 30-step pose-only Adam descent (map frozen) as a small-scale
     replica of the v2 pre-alignment.

Read-only w.r.t. the run directories.  GT is used only for evaluation.

  python3 experiments/exp_feedback_gradient_probe.py --dataset ho3d \
    --video-dir datasets/HO3D_v3/evaluation/AP12 \
    --run-dir outputs/gsfb_v1_ho3d_AP12_20260907 \
    --output-dir logs/gradprobe_20260908/AP12 --device cuda:1
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "experiments"))
sys.path.append(str(REPO / "BundleTrack/scripts"))

from gaussian_runner import (  # noqa: E402
    GaussianFrame,
    GaussianRunner,
    SceneNormalization,
    load_gaussian_config,
    se3_exp_batch,
)
from prior_lifecycle import STATE_UNSEEN  # noqa: E402
from run_gaussian_incremental import load_frame  # noqa: E402

SAM3D_ROOT = Path("/home/kist/Desktop/sam-3d-objects")


# --------------------------------------------------------------------------- GT
def rot_deg(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.degrees(np.arccos(np.clip((np.trace(a.T @ b) - 1.0) / 2.0, -1.0, 1.0))))


def rotvec_from_matrix(R: np.ndarray) -> np.ndarray:
    angle = math.acos(max(-1.0, min(1.0, (np.trace(R) - 1.0) / 2.0)))
    if angle < 1e-9:
        return np.zeros(3)
    axis = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / (2.0 * math.sin(angle))
    return axis * angle


class GroundTruth:
    """ob_in_cam GT per frame id, plus the evaluator's first-frame gauge."""

    def __init__(self, dataset: str, video_dir: Path, run_dir: Path) -> None:
        self.dataset = dataset
        if dataset == "ho3d":
            from data_reader import Ho3dReader

            reader = Ho3dReader(str(video_dir))
            self.ids = list(reader.id_strs)
            self._gt = [reader.get_gt_pose(i) for i in range(len(self.ids))]
        else:
            rgb = sorted(glob.glob(str(video_dir / "rgb/*.png")))
            self.ids = [os.path.splitext(os.path.basename(f))[0] for f in rgb]
            files = sorted(glob.glob(str(video_dir / "annotated_poses/*.txt")))
            self._gt = [np.loadtxt(f).reshape(4, 4) for f in files]
            self._gt += [None] * (len(self.ids) - len(self._gt))
        # annotated rotations (YCB especially) are not exactly orthonormal; project them so that
        # validate_c2w_cv accepts GT-derived poses (max correction is reported once)
        max_fix = 0.0
        for i, g in enumerate(self._gt):
            if g is None:
                continue
            u, _, vt = np.linalg.svd(g[:3, :3])
            R = u @ vt
            if np.linalg.det(R) < 0:
                u[:, -1] *= -1.0
                R = u @ vt
            max_fix = max(max_fix, float(np.abs(R - g[:3, :3]).max()))
            fixed = g.copy()
            fixed[:3, :3] = R
            self._gt[i] = fixed
        print(f"[probe] GT rotations orthonormalized (max element change {max_fix:.2e})", flush=True)
        self.index = {s: i for i, s in enumerate(self.ids)}
        first = sorted(os.listdir(run_dir / "ob_in_cam"))[0]
        pred0 = np.loadtxt(run_dir / "ob_in_cam" / first).reshape(4, 4)
        gt0 = self._gt[0]
        if gt0 is None:
            raise RuntimeError("frame 0 has no GT pose")
        # evaluator: aligned_pred_i = pred_i @ inv(pred_0) @ gt_0  (ob_in_cam)
        self.align = np.linalg.inv(pred0) @ gt0

    def gt_c2w(self, frame_id: str) -> np.ndarray | None:
        """GT-consistent cam_in_ob (metric OpenCV) in the tracker's object frame."""
        g = self._gt[self.index[frame_id]] if frame_id in self.index else None
        if g is None:
            return None
        return self.align @ np.linalg.inv(g)

    def rot_error(self, c2w_metric: np.ndarray, frame_id: str) -> float:
        g = self._gt[self.index[frame_id]]
        pred_aligned = np.linalg.inv(c2w_metric) @ self.align
        return rot_deg(pred_aligned[:3, :3], g[:3, :3])


# ----------------------------------------------------------------- saved cycles
def load_cycles(run_dir: Path) -> list[dict]:
    cycles = []
    for f in sorted(glob.glob(str(run_dir / "*/poses_before_gs.txt"))):
        d = Path(f).parent
        kf = d / "nerf_frames.txt"
        if not kf.exists():
            continue
        ids = [l.strip() for l in open(kf) if l.strip()]
        before = np.loadtxt(f).reshape(-1, 4, 4)
        after = np.loadtxt(d / "poses_after_gs.txt").reshape(-1, 4, 4)
        n = min(len(ids), len(before), len(after))
        cycles.append({"dir": d.name, "ids": ids[:n], "before": before[:n], "after": after[:n]})
    # cycles are sorted by directory name = frame id string; for YCB the long ids sort correctly too
    return cycles


# ------------------------------------------------------------------ replay init
def init_runner_like_online(cfg_path: Path, device: str, prior_paths: dict, frames: list[GaussianFrame],
                            cam_in_obs0: np.ndarray, initial_steps: int) -> GaussianRunner:
    """Mirror of bundlesdf.run_gaussian's first batch (prior + online Sim(3) + initialize_from_prior)."""
    from run_sam3d_alignment import DEFAULT_CONFIG as ALIGN_DEFAULTS
    from run_sam3d_alignment import align_prior_sim3, prepare_target
    from sam3d_prior import (
        load_mesh_prior,
        load_sam3d_gaussian_ply,
        load_sam3d_pose_or_refined,
        sample_surfels,
        transfer_gaussian_colors,
        transform_surfels_canonical_to_cv_camera,
    )

    dev = torch.device(device)
    if dev.type == "cuda":
        torch.cuda.set_device(dev)
    runner_config = load_gaussian_config(cfg_path)
    runner_config["device"] = device
    prior = load_mesh_prior(prior_paths["mesh_npz"])
    init_pose = load_sam3d_pose_or_refined(prior_paths["pose_json"])
    surfels = sample_surfels(prior, 20000, seed=0, radius_multiplier=0.75)
    surfels, _ = transfer_gaussian_colors(surfels, load_sam3d_gaussian_ply(prior_paths["gaussian_ply"]))
    align_config = dict(ALIGN_DEFAULTS)
    align_config["use_ssim"] = True
    align_config["use_ms_ssim"] = False
    first = frames[0]
    target = prepare_target({"frame_id": first.frame_id, "rgb": first.rgb, "depth": first.depth,
                             "mask": first.mask, "K": first.K}, align_config, dev)
    refined_pose, _, _, status = align_prior_sim3(surfels, init_pose, target, align_config, dev, verbose=False)
    surfels_cv = transform_surfels_canonical_to_cv_camera(surfels, refined_pose)
    first_c2w = np.asarray(cam_in_obs0, dtype=np.float64)
    means_obj = surfels_cv.means.numpy() @ first_c2w[:3, :3].T + first_c2w[:3, 3][None, :]
    center = means_obj.mean(axis=0)
    radius = float(np.linalg.norm(means_obj - center, axis=1).max())
    normalization = SceneNormalization(scale=1.0 / max(radius * 1.2, 1e-6), translation=-center)
    runner = GaussianRunner(runner_config, normalization, device=device)
    runner.initialize_from_prior(surfels_cv, cam_in_obs0, frames, train_steps=initial_steps)
    print(f"[probe] prior alignment {status}; normalization scale {normalization.scale:.3f}; "
          f"gaussians {runner.num_gaussians}", flush=True)
    return runner


# --------------------------------------------------------------- gradient probe
class PriorHider:
    """Temporarily hide prior-lineage splats (opacity logit -> -30) for a render."""

    def __init__(self, runner: GaussianRunner, mode: str) -> None:
        self.runner, self.mode, self.saved = runner, mode, None

    def __enter__(self):
        if self.mode == "none" or self.runner.lifecycle_fields is None:
            return self
        fields = self.runner.lifecycle_fields
        if self.mode == "unseen":
            rows = fields.state == STATE_UNSEEN
        elif self.mode == "all_prior":
            rows = fields.lineage.bool()
        else:
            raise ValueError(self.mode)
        rows = rows.to(self.runner.device)
        opac = self.runner.splats["opacities"]
        self.saved = opac.data.clone()
        with torch.no_grad():
            opac.data[rows] = -30.0
        self.hidden = int(rows.sum())
        return self

    def __exit__(self, *exc):
        if self.saved is not None:
            with torch.no_grad():
                self.runner.splats["opacities"].data.copy_(self.saved)
        return False


def render_losses(runner: GaussianRunner, view, c2w_norm: torch.Tensor) -> dict[str, torch.Tensor]:
    """Same three loss terms as GaussianRunner.train (weights applied in 'total')."""
    device = runner.device
    target = view.rgb.to(device)[None]
    mask = view.mask.to(device)[None]
    K = view.K.to(device)[None]
    sh_degree = min(runner.total_steps // int(runner.config["sh_degree_interval"]), int(runner.config["sh_degree"]))
    depth_weight = float(runner.config["depth_loss_weight"])
    ssim_weight = float(runner.config["ssim_weight"])
    result = runner._rasterize(K=K, c2w=c2w_norm[None], width=view.width, height=view.height,
                               sh_degree=sh_degree, render_mode="RGB+ED" if depth_weight > 0 else "RGB",
                               absgrad=False)
    rendered = result.colors[..., :3]
    mask_channels = mask[..., None].to(rendered.dtype)
    denominator = mask_channels.sum().clamp_min(1.0) * 3.0
    l1 = (torch.abs(rendered - target) * mask_channels).sum() / denominator
    dssim = runner._masked_dssim(rendered, target, mask)
    depth = runner._masked_depth_loss(result, view, mask) if depth_weight > 0 else torch.zeros((), device=device)
    total = (1.0 - ssim_weight) * l1 + ssim_weight * dssim + depth_weight * depth
    return {"l1": l1, "dssim": dssim, "depth": depth, "total": total}


def probe_view(runner: GaussianRunner, view, c2w_tracker: np.ndarray, gt: GroundTruth, frame_id: str,
               hide: str, step_deg: float, step_mm: float, pose_only_steps: int, pose_only_lr: float,
               pose_only_loss: str = "total") -> dict:
    device = runner.device
    norm = runner.normalization
    c2w_norm = torch.from_numpy(norm.normalize_c2w(c2w_tracker)).float().to(device)
    gt_c2w = gt.gt_c2w(frame_id)
    gt_norm = norm.normalize_c2w(gt_c2w)
    # true world-side correction T*: gt = T* @ tracker (normalized frame)
    T_star = gt_norm.astype(np.float64) @ np.linalg.inv(c2w_norm.cpu().numpy().astype(np.float64))
    true_rot = rotvec_from_matrix(T_star[:3, :3])
    true_trans = T_star[:3, 3]
    err0 = gt.rot_error(c2w_tracker, frame_id)
    out = {"frame_id": frame_id, "hide": hide, "err_rot_before": err0,
           "true_rot_deg": float(np.degrees(np.linalg.norm(true_rot))),
           "true_trans_mm": float(np.linalg.norm(true_trans) / norm.scale * 1000.0), "losses": {}}

    def apply_world(delta6: np.ndarray) -> np.ndarray:
        T = se3_exp_batch(torch.from_numpy(delta6[None]).float())[0].numpy().astype(np.float64)
        c2w_new = T @ c2w_norm.cpu().numpy().astype(np.float64)
        return norm.metric_c2w(c2w_new)

    with PriorHider(runner, hide) as hider:
        out["hidden"] = getattr(hider, "hidden", 0)
        delta = torch.zeros(6, device=device, requires_grad=True)
        T = se3_exp_batch(delta[None])[0]
        losses = render_losses(runner, view, T @ c2w_norm)
        for name, loss in losses.items():
            if not loss.requires_grad:      # e.g. depth term with no valid pixel -> constant 0
                out["losses"][name] = None
                continue
            (g,) = torch.autograd.grad(loss, delta, retain_graph=True, allow_unused=True)
            if g is None or not torch.isfinite(g).all():
                out["losses"][name] = None
                continue
            g = g.detach().cpu().numpy().astype(np.float64)
            g_tr, g_rot = g[:3], g[3:]
            d_rot = -g_rot
            d_tr = -g_tr
            cos_rot = float(d_rot @ true_rot / (np.linalg.norm(d_rot) * np.linalg.norm(true_rot) + 1e-12))
            cos_tr = float(d_tr @ true_trans / (np.linalg.norm(d_tr) * np.linalg.norm(true_trans) + 1e-12))
            # fixed-size descent steps: rotation-only step_deg, translation-only step_mm
            rec = {"loss": float(loss), "cos_rot": cos_rot, "cos_trans": cos_tr,
                   "grad_rot_norm": float(np.linalg.norm(g_rot)), "grad_trans_norm": float(np.linalg.norm(g_tr)),
                   "grad": [float(v) for v in g],                      # raw d(loss)/d(delta), [trans(3), rot(3)]
                   "true_delta": [float(v) for v in np.concatenate((true_trans, true_rot))]}
            if np.linalg.norm(d_rot) > 0:
                step = np.zeros(6)
                step[3:] = d_rot / np.linalg.norm(d_rot) * math.radians(step_deg)
                rec["dErr_rot_step"] = gt.rot_error(apply_world(step), frame_id) - err0
            if np.linalg.norm(d_tr) > 0:
                step = np.zeros(6)
                step[:3] = d_tr / np.linalg.norm(d_tr) * (step_mm / 1000.0 * norm.scale)
                c2w_new = apply_world(step)
                t_err0 = np.linalg.norm(c2w_tracker[:3, 3] - gt_c2w[:3, 3]) * 1000.0
                t_err1 = np.linalg.norm(c2w_new[:3, 3] - gt_c2w[:3, 3]) * 1000.0
                rec["dErr_trans_step_mm"] = float(t_err1 - t_err0)
            out["losses"][name] = rec
        # pose-only descent on the total loss (map frozen), like a short v2 pre-alignment
        if pose_only_steps > 0:
            d2 = torch.zeros(6, device=device, requires_grad=True)
            opt = torch.optim.Adam([d2], lr=pose_only_lr, eps=1e-15)
            for _ in range(pose_only_steps):
                opt.zero_grad(set_to_none=True)
                T2 = se3_exp_batch(d2[None])[0]
                total = render_losses(runner, view, T2 @ c2w_norm)[pose_only_loss]
                if not total.requires_grad:
                    break
                total.backward()
                for p in runner.splats.values():
                    p.grad = None
                torch.nn.utils.clip_grad_norm_([d2], max_norm=0.1, norm_type=float("inf"))
                opt.step()
            d2n = d2.detach().cpu().numpy().astype(np.float64)
            # the online PoseDeltas apply tanh clamps; here raw values are small (lr 0.01 x 30)
            c2w_new = apply_world(d2n)
            out["pose_only"] = {"steps": pose_only_steps, "loss": pose_only_loss, "moved_deg": float(np.degrees(np.linalg.norm(d2n[3:]))),
                                "dErr_rot": gt.rot_error(c2w_new, frame_id) - err0,
                                "dErr_trans_mm": float(np.linalg.norm(c2w_new[:3, 3] - gt_c2w[:3, 3]) * 1000.0
                                                       - np.linalg.norm(c2w_tracker[:3, 3] - gt_c2w[:3, 3]) * 1000.0)}
    return out


# ----------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=("ycb", "ho3d"), required=True)
    ap.add_argument("--video-dir", type=Path, required=True)
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--runner-config", type=Path, default=REPO / "config_gs_2dgs_1mm_lifecycle.yml")
    ap.add_argument("--prior-root", type=Path, default=SAM3D_ROOT)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--initial-steps", type=int, default=4000)
    ap.add_argument("--update-steps", type=int, default=500)
    ap.add_argument("--probe-K", type=int, nargs="+", default=[6, 8, 10, 15, 20, 25, 30, 35, 40])
    ap.add_argument("--hide", nargs="+", default=["none", "unseen", "all_prior"])
    ap.add_argument("--step-deg", type=float, default=0.5)
    ap.add_argument("--step-mm", type=float, default=2.0)
    ap.add_argument("--pose-only-steps", type=int, default=30)
    ap.add_argument("--pose-only-lr", type=float, default=0.01)
    ap.add_argument("--pose-only-loss", choices=("total", "l1", "dssim", "depth"), default="total",
                    help="loss driving the pose-only descent (experiment E preview: depth)")
    ap.add_argument("--dry-run", action="store_true", help="only load data and print GT/true-correction stats")
    args = ap.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_dir}")

    seq = args.video_dir.name
    sub = "output_YCBInEOAT_sam2mask_mesh" if args.dataset == "ycb" else "output_HO3D_sam2mask_mesh"
    prior_paths = {"mesh_npz": args.prior_root / sub / f"{seq}_mesh_depth.npz",
                   "pose_json": args.prior_root / sub / f"{seq}_mesh_depth.json",
                   "gaussian_ply": args.prior_root / sub / f"{seq}_splat_depth.ply"}
    K_cam = np.loadtxt(args.run_dir / "cam_K.txt").reshape(3, 3).astype(np.float32)
    gt = GroundTruth(args.dataset, args.video_dir, args.run_dir)
    cycles = load_cycles(args.run_dir)
    if not cycles:
        raise RuntimeError("no saved cycles (poses_before_gs.txt + nerf_frames.txt) in run dir")
    probe_set = set(args.probe_K)
    max_K = max(args.probe_K)
    print(f"[probe] {seq}: {len(cycles)} cycles, probing K in {sorted(probe_set)}", flush=True)

    if args.dry_run:
        for c in cycles[:12]:
            k = len(c["ids"]) - 1
            fid = c["ids"][k]
            g = gt.gt_c2w(fid)
            if g is None:
                print(f"  cycle {c['dir']} K={len(c['ids'])} newest {fid}: no GT"); continue
            e = gt.rot_error(c["before"][k], fid)
            print(f"  cycle {c['dir']} K={len(c['ids'])} newest {fid}: tracker rot err {e:.2f} deg, "
                  f"after-feedback {gt.rot_error(c['after'][k], fid):.2f}")
        return

    args.output_dir.mkdir(parents=True)
    t0 = time.time()
    rows: list[dict] = []
    runner: GaussianRunner | None = None
    consumed = 0
    for ci, cyc in enumerate(cycles):
        K = len(cyc["ids"])
        if K > max_K:
            break
        new_ids = cyc["ids"][consumed:]
        frames = [load_frame(args.run_dir, fid, K_cam, cyc["before"][consumed + j].astype(np.float32))
                  for j, fid in enumerate(new_ids)]
        frames = [GaussianFrame(f"kf_{consumed + j:05d}", f.rgb, f.depth, f.mask, f.K, f.c2w_cv).validated()
                  for j, f in enumerate(frames)]
        if runner is None:
            runner = init_runner_like_online(args.runner_config, args.device, prior_paths, frames,
                                             cyc["before"][0], args.initial_steps)
            consumed = K
            continue
        # tracker poses the online backend saw at this cycle (includes earlier feedback)
        runner.refresh_view_poses({v.frame_id: cyc["before"][i] for i, v in enumerate(runner.views)})
        do_probe = K in probe_set
        pre_views = []
        if do_probe:
            for j, (f, fid) in enumerate(zip(frames, new_ids)):
                if gt.gt_c2w(fid) is None:
                    continue
                view = runner._prepare_view(f)
                pre_views.append((view, fid, cyc["before"][consumed + j]))
                for hide in args.hide:
                    r = probe_view(runner, view, cyc["before"][consumed + j], gt, fid, hide, args.step_deg,
                                   args.step_mm, args.pose_only_steps, args.pose_only_lr, args.pose_only_loss)
                    r.update({"seq": seq, "cycle": cyc["dir"], "K": K, "regime": "pre_append"})
                    rows.append(r)
        stats = runner.update(frames, train_steps=args.update_steps)
        consumed = K
        if do_probe:
            for view, fid, pose in pre_views:
                # the view object stored in the runner has the same pose (no deltas in replay)
                stored = next(v for v in runner.views if v.frame_id == view.frame_id)
                for hide in args.hide:
                    r = probe_view(runner, stored, pose, gt, fid, hide, args.step_deg, args.step_mm,
                                   args.pose_only_steps, args.pose_only_lr, args.pose_only_loss)
                    r.update({"seq": seq, "cycle": cyc["dir"], "K": K, "regime": "post_train"})
                    rows.append(r)
            print(f"[probe] K={K} cycle {cyc['dir']} gaussians {stats.gaussians_after_train} "
                  f"probed {len(pre_views)} view(s) x {len(args.hide)} hide modes  ({time.time() - t0:.0f}s)", flush=True)
            with open(args.output_dir / "rows.json", "w") as fh:
                json.dump(rows, fh, indent=1)
    with open(args.output_dir / "rows.json", "w") as fh:
        json.dump(rows, fh, indent=1)
    summarize(rows, args.output_dir / "summary.txt")


def summarize(rows: list[dict], path: Path) -> None:
    lines = []
    for regime in ("pre_append", "post_train"):
        for hide in sorted({r["hide"] for r in rows}):
            sel = [r for r in rows if r["regime"] == regime and r["hide"] == hide]
            if not sel:
                continue
            lines.append(f"== {regime} / hide={hide}: {len(sel)} probes, tracker err before mean "
                         f"{np.mean([r['err_rot_before'] for r in sel]):.2f} deg (true rot {np.mean([r['true_rot_deg'] for r in sel]):.2f})")
            for loss in ("l1", "dssim", "depth", "total"):
                recs = [r["losses"][loss] for r in sel if r["losses"].get(loss)]
                if not recs:
                    continue
                cr = np.array([x["cos_rot"] for x in recs]); ct = np.array([x["cos_trans"] for x in recs])
                dr = np.array([x.get("dErr_rot_step", np.nan) for x in recs])
                dt = np.array([x.get("dErr_trans_step_mm", np.nan) for x in recs])
                lines.append(f"   {loss:6s} cos_rot {cr.mean():+.2f} (>0: {(cr > 0).mean():.2f})  cos_trans {ct.mean():+.2f} "
                             f"(>0: {(ct > 0).mean():.2f})  dErr/0.5deg-step {np.nanmean(dr):+.3f} deg  "
                             f"dErr/2mm-step {np.nanmean(dt):+.2f} mm")
            po = [r["pose_only"] for r in sel if r.get("pose_only")]
            if po:
                lines.append(f"   pose-only {po[0]['steps']} steps on {po[0].get('loss', 'total')}: moved {np.mean([p['moved_deg'] for p in po]):.2f} deg, "
                             f"dErr_rot {np.mean([p['dErr_rot'] for p in po]):+.2f} deg "
                             f"(improved in {np.mean([p['dErr_rot'] < -0.05 for p in po]):.2f}), "
                             f"dErr_trans {np.mean([p['dErr_trans_mm'] for p in po]):+.2f} mm")
    text = "\n".join(lines)
    print(text)
    path.write_text(text + "\n")


if __name__ == "__main__":
    main()
