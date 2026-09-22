"""EXP_BATCH_20260921 5.1 — offline spring test (H4): does a keyframe's own Gaussian layer hold its pose in place?

Replay (keyframe images of a finished run) with GT poses for every keyframe, except the target keyframe k* whose pose
is perturbed BEFORE its points are appended (so its layer is born at the wrong pose).  After k* + 10 more keyframes one
pose-optimisation cycle is run in a 2 x 2 design:
  structure  current  : Gaussians fixed in the world, per-view pose deltas only (v1)
             anchored : every observed Gaussian carries its birth view b; rendering uses means_eff = R_d[b] means + t_d[b],
                        quat_eff = q_d[b] (x) quat, i.e. the layer moves rigidly with its birth view's correction (prior
                        lineage -> identity).  Bake afterwards.
  map        fixed (poses only) | joint (map + poses, lifecycle: only VERIFIED learn, update lr scale)
Mandatory sanity checks of the anchored implementation (results are not written unless all pass):
  1 zero deltas: anchored render == current render (max abs diff < 1e-5)
  2 only view k's own layer visible: |dL_k/dDelta_k| / |dL_k/dDelta_k with the full map| < 1e-3
  3 bake: render after baking (zero deltas) == render before baking with the deltas applied
usage: exp_spring_anchor.py --dataset ho3d --video-dir datasets/HO3D_v3/evaluation/AP12 --run-dir outputs/fulleval_20260912/ho3d/AP12 \
         --out-dir logs/exp_batch_20260921/spring/AP12 --kstars 10 20 40 --seeds 0 1
"""
from __future__ import annotations

import argparse
import copy
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
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "experiments")); sys.path.append(str(REPO / "BundleTrack/scripts"))
from exp_feedback_gradient_probe_gtmap import SAM3D_ROOT, GroundTruth, rot_deg  # noqa: E402
from gaussian_runner import GaussianFrame, GaussianRunner, SceneNormalization, load_gaussian_config, se3_exp_batch  # noqa: E402
from prior_lifecycle import STATE_CONTRADICTED, STATE_VERIFIED  # noqa: E402
from run_gaussian_incremental import load_frame  # noqa: E402


class BirthRunner(GaussianRunner):
    """GaussianRunner + the birth view index of every Gaussian (0 = prior lineage / identity anchor)."""

    birth: torch.Tensor | None = None
    _next_birth: int = 0

    def initialize_from_prior(self, *a, **k):
        out = super().initialize_from_prior(*a, **k)
        self.birth = torch.zeros(self.num_gaussians, dtype=torch.long)
        return out

    def update(self, frames, train_steps=None):
        self._next_birth = len(self.views)
        return super().update(frames, train_steps=train_steps)

    def _append_splats(self, points_metric, colors):
        n0 = self.num_gaussians
        super()._append_splats(points_metric, colors)
        self.birth = torch.cat((self.birth, torch.full((self.num_gaussians - n0,), self._next_birth, dtype=torch.long)))

    def _remove_contradicted(self):
        if self.birth is not None and self.lifecycle_enabled and self.lifecycle_fields is not None:
            keep = (self.lifecycle_fields.state != STATE_CONTRADICTED).cpu()
            if not bool(keep.all()):
                self.birth = self.birth[keep]
        n = super()._remove_contradicted()
        assert self.birth is None or len(self.birth) == self.num_gaussians
        return n


def build_runner(cfg_path, device, prior_paths, frames, cam_in_obs0, initial_steps):
    """Mirror of exp_feedback_gradient_probe_gtmap.init_runner_like_online with the BirthRunner class."""
    from run_sam3d_alignment import DEFAULT_CONFIG as ALIGN_DEFAULTS
    from run_sam3d_alignment import align_prior_sim3, prepare_target
    from sam3d_prior import (load_mesh_prior, load_sam3d_gaussian_ply, load_sam3d_pose_or_refined, sample_surfels,
                             transfer_gaussian_colors, transform_surfels_canonical_to_cv_camera)
    dev = torch.device(device); torch.cuda.set_device(dev)
    rc = load_gaussian_config(cfg_path); rc["device"] = device
    prior = load_mesh_prior(prior_paths["mesh_npz"]); init_pose = load_sam3d_pose_or_refined(prior_paths["pose_json"])
    surfels = sample_surfels(prior, 20000, seed=0, radius_multiplier=0.75)
    surfels, _ = transfer_gaussian_colors(surfels, load_sam3d_gaussian_ply(prior_paths["gaussian_ply"]))
    ac = dict(ALIGN_DEFAULTS); ac["use_ssim"] = True; ac["use_ms_ssim"] = False
    f0 = frames[0]
    target = prepare_target({"frame_id": f0.frame_id, "rgb": f0.rgb, "depth": f0.depth, "mask": f0.mask, "K": f0.K}, ac, dev)
    refined, _, _, status = align_prior_sim3(surfels, init_pose, target, ac, dev, verbose=False)
    scv = transform_surfels_canonical_to_cv_camera(surfels, refined)
    c0 = np.asarray(cam_in_obs0, dtype=np.float64)
    m = scv.means.numpy() @ c0[:3, :3].T + c0[:3, 3][None]
    centre = m.mean(0); radius = float(np.linalg.norm(m - centre, axis=1).max())
    norm = SceneNormalization(scale=1.0 / max(radius * 1.2, 1e-6), translation=-centre)
    r = BirthRunner(rc, norm, device=device)
    r.initialize_from_prior(scv, cam_in_obs0, frames, train_steps=initial_steps)
    return r


# ------------------------------------------------------------------ anchored rendering
def rotvec_to_quat(rv: torch.Tensor) -> torch.Tensor:
    theta = torch.linalg.norm(rv, dim=-1, keepdim=True).clamp_min(1e-12)
    half = 0.5 * theta
    return torch.cat((torch.cos(half), torch.sin(half) * rv / theta), dim=-1)


def quat_mul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    aw, ax, ay, az = a.unbind(-1); bw, bx, by, bz = b.unbind(-1)
    return torch.stack((aw * bw - ax * bx - ay * by - az * bz, aw * bx + ax * bw + ay * bz - az * by,
                        aw * by - ax * bz + ay * bw + az * bx, aw * bz + ax * by - ay * bx + az * bw), dim=-1)


def effective_splats(runner, deltas, birth_dev, anchored: bool):
    means, quats = runner.splats["means"], runner.splats["quats"]
    if not anchored or deltas is None:
        return means, quats, None
    d = deltas.deltas()
    if deltas.fix_first:
        gate = torch.ones(len(d), 1, device=d.device, dtype=d.dtype); gate[0] = 0.0
        d = d * gate
    T = se3_exp_batch(d)
    R, t = T[birth_dev, :3, :3], T[birth_dev, :3, 3]
    means_eff = torch.einsum("nij,nj->ni", R, means) + t
    quats_eff = quat_mul(rotvec_to_quat(d[birth_dev, 3:]), quats)   # bilinear: identity delta returns quats exactly
    return means_eff, quats_eff, R


def rasterize(runner, means, quats, K, c2w, width, height, opacities=None, R_body=None):
    _, _, rasterization_2dgs, _ = runner._require_gsplat()
    sh_degree = min(runner.total_steps // int(runner.config["sh_degree_interval"]), int(runner.config["sh_degree"]))
    shs = torch.cat((runner.splats["sh0"], runner.splats["shN"]), dim=1)
    if R_body is not None:
        from gsplat import spherical_harmonics
        dirs_world = means - c2w[0, :3, 3][None]
        dirs_body = torch.einsum("nji,nj->ni", R_body, dirs_world)          # R^T d: direction in the layer's own frame
        colors_in = torch.clamp_min(spherical_harmonics(sh_degree, dirs_body, shs) + 0.5, 0.0)[None]; sh_arg = None   # [C=1, N, 3]
    else:
        colors_in = shs; sh_arg = sh_degree
    colors, alpha, normals, surf_normals, distort, _, info = rasterization_2dgs(
        means=means, quats=quats, scales=torch.exp(runner.splats["scales"]),
        opacities=torch.sigmoid(runner.splats["opacities"]) if opacities is None else opacities,
        colors=colors_in, viewmats=torch.linalg.inv(c2w), Ks=K,
        width=width, height=height, sh_degree=sh_arg, packed=bool(runner.config["packed"]), sparse_grad=False,
        render_mode="RGB+ED", absgrad=False, distloss=False, depth_mode="expected")
    from gaussian_runner import RasterResult
    return RasterResult(colors, alpha, normals, surf_normals, distort, info)


def view_loss(runner, view, result):
    dev = runner.device
    target = view.rgb.to(dev)[None]; mask = view.mask.to(dev)[None]
    rendered = result.colors[..., :3]; mc = mask[..., None].to(rendered.dtype)
    l1 = (torch.abs(rendered - target) * mc).sum() / (mc.sum().clamp_min(1.0) * 3.0)
    w = float(runner.config["ssim_weight"])
    loss = (1.0 - w) * l1 + w * runner._masked_dssim(rendered, target, mask)
    dw = float(runner.config["depth_loss_weight"])
    if dw > 0:
        loss = loss + dw * runner._masked_depth_loss(result, view, mask)
    return loss


def render_view(runner, deltas, birth_dev, anchored, vi, opacities=None):
    view = runner.views[vi]; dev = runner.device
    c2w = view.c2w_normalized.to(dev)[None]
    if deltas is not None:
        c2w = deltas.matrices([vi]) @ c2w
    means, quats, R_body = effective_splats(runner, deltas, birth_dev, anchored)
    return view, rasterize(runner, means, quats, view.K.to(dev)[None], c2w, view.width, view.height, opacities, R_body)


def pose_cycle(runner, steps, anchored, joint, seed):
    """One optimisation cycle mirroring GaussianRunner.train (uniform random view per step, fresh zero deltas, Adam lr /
    decay / inf-norm clip from the config).  joint: the map learns too (update lr scale, only VERIFIED rows)."""
    fb = runner.config["pose_feedback"]; dev = runner.device
    birth_dev = runner.birth.to(dev)
    deltas = runner._new_pose_deltas(len(runner.views), fix_first=bool(fb["fix_first_view"]))
    opt, sched = runner._pose_optimizer(deltas, float(fb["lr"]), steps)
    gen = torch.Generator().manual_seed(1000 + seed)
    frozen = None
    if joint:
        runner._reset_optimization_state(float(runner.config["update_lr_scale"]), initial=False)
        frozen = (runner.lifecycle_fields.state != STATE_VERIFIED).to(dev)
        means_sched = torch.optim.lr_scheduler.ExponentialLR(runner.optimizers["means"], gamma=0.01 ** (1.0 / steps))
    for _ in range(steps):
        vi = int(torch.randint(len(runner.views), (1,), generator=gen).item())
        opt.zero_grad(set_to_none=True)
        if joint:
            for o in runner.optimizers.values(): o.zero_grad(set_to_none=True)
        else:
            for p in runner.splats.values(): p.grad = None
        view, res = render_view(runner, deltas, birth_dev, anchored, vi)
        loss = view_loss(runner, view, res)
        loss.backward()
        if joint:
            for p in runner.splats.values():
                if p.grad is not None: p.grad[frozen] = 0
            for o in runner.optimizers.values(): o.step()
            means_sched.step()
        runner._pose_step(deltas, opt, sched)
        runner.total_steps += 1 if joint else 0
    return deltas


@torch.no_grad()
def bake(runner, deltas, anchored):
    """Fold the deltas into the view poses (as _bake_pose_deltas) and, for anchored, into the stored means / quats."""
    birth_dev = runner.birth.to(runner.device)
    if anchored:
        means_eff, quats_eff, _ = effective_splats(runner, deltas, birth_dev, True)
        runner.splats["means"].data.copy_(means_eff); runner.splats["quats"].data.copy_(quats_eff)
    runner._bake_pose_deltas(deltas, runner.views, event="spring_cycle")


def pose_errors(runner, gt, ids):
    rot, trans = [], []
    P = runner.view_poses_metric(); c = np.append(-runner.normalization.translation.astype(np.float64), 1.0)
    for c2w, fid in zip(P, ids):
        g = gt.gt_c2w(fid)
        rot.append(rot_deg(c2w[:3, :3], g[:3, :3]))
        trans.append(float(np.linalg.norm((np.linalg.inv(c2w) @ c - np.linalg.inv(g) @ c)[:3]) * 1000))   # object centre in the camera frame
    return np.array(rot), np.array(trans)


def perturb(c2w_metric, centre_metric, deg, mm, rng):
    """World-side error (same family as the pose deltas): rotation of `deg` about a random axis through the object
    centre plus a translation of `mm` in a random direction, left-multiplied onto the GT c2w."""
    axis = rng.normal(size=3); axis /= np.linalg.norm(axis); ang = math.radians(deg)
    Kx = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    R = np.eye(3) + math.sin(ang) * Kx + (1 - math.cos(ang)) * Kx @ Kx
    tdir = rng.normal(size=3); tdir /= np.linalg.norm(tdir)
    T = np.eye(4); T[:3, :3] = R; T[:3, 3] = centre_metric - R @ centre_metric + tdir * mm / 1000.0
    return T @ c2w_metric


def sanity_checks(runner, kview, log):
    dev = runner.device; birth_dev = runner.birth.to(dev); fb = runner.config["pose_feedback"]
    out = {}
    with torch.no_grad():
        zero = runner._new_pose_deltas(len(runner.views), fix_first=bool(fb["fix_first_view"]))
        _, a = render_view(runner, zero, birth_dev, True, kview); _, c = render_view(runner, None, birth_dev, False, kview)
        out["check1_max_abs_diff"] = float((a.colors - c.colors).abs().max())
    # check 2: gradient wrt Delta_k, own layer only vs full map (at a non-zero delta so the test is not trivial)
    def grad_norm(opacities):
        d = runner._new_pose_deltas(len(runner.views), fix_first=bool(fb["fix_first_view"]))
        with torch.no_grad(): d.data[kview] = torch.tensor([0.05, -0.04, 0.03, 0.06, -0.05, 0.04], device=dev)
        view, res = render_view(runner, d, birth_dev, True, kview, opacities)
        loss = view_loss(runner, view, res); g, = torch.autograd.grad(loss, d.data); return float(torch.linalg.norm(g[kview]))
    full = grad_norm(None)
    op = torch.sigmoid(runner.splats["opacities"]).detach() * (birth_dev == kview).to(torch.float32)
    own = grad_norm(op)
    out.update({"check2_grad_full_map": full, "check2_grad_own_layer_only": own, "check2_ratio": own / max(full, 1e-30),
                "own_layer_gaussians": int((birth_dev == kview).sum())})
    # check 3: bake consistency on a copy
    r2 = copy.deepcopy(runner)
    d = r2._new_pose_deltas(len(r2.views), fix_first=bool(fb["fix_first_view"]))
    with torch.no_grad():
        d.data.normal_(0, 0.05, generator=None)
        _, before = render_view(r2, d, r2.birth.to(dev), True, kview)
        bake(r2, d, True)
        _, after = render_view(r2, None, r2.birth.to(dev), False, kview)
        out["check3_max_abs_diff"] = float((before.colors - after.colors).abs().max()); out["check3_mean_abs_diff"] = float((before.colors - after.colors).abs().mean())
    r3 = copy.deepcopy(runner)   # negative control: bake the view poses only (layers left behind) -> must differ clearly
    with torch.no_grad():
        d3 = r3._new_pose_deltas(len(r3.views), fix_first=bool(fb["fix_first_view"])); d3.data.copy_(d.data)
        bake(r3, d3, False); _, wrong = render_view(r3, None, r3.birth.to(dev), False, kview)
        out["check3_negative_control_mean_abs_diff"] = float((before.colors - wrong.colors).abs().mean())
    # check 3 compares two float32 pipelines (deltas applied on the fly vs folded into float32 means / SVD-projected
    # poses); SH colours after the bake are evaluated in the world frame again (SH coefficients are not rotated; shN ~ 0)
    out["pass"] = bool(out["check1_max_abs_diff"] < 1e-5 and out["check2_ratio"] < 1e-3
                       and out["check3_mean_abs_diff"] < 0.02 * out["check3_negative_control_mean_abs_diff"] and out["check3_max_abs_diff"] < 5e-3)
    log(f"sanity: {out}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=("ycb", "ho3d"), required=True); ap.add_argument("--video-dir", type=Path, required=True)
    ap.add_argument("--run-dir", type=Path, required=True); ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--runner-config", type=Path, default=REPO / "config_gs_2dgs_1mm_lifecycle.yml")
    ap.add_argument("--kstars", type=int, nargs="+", default=[10, 20, 40]); ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--after", type=int, default=10); ap.add_argument("--inject-deg", type=float, default=3.0); ap.add_argument("--inject-mm", type=float, default=5.0)
    ap.add_argument("--cycle-steps", type=int, nargs="+", default=[500, 1500])
    ap.add_argument("--initial-steps", type=int, default=500); ap.add_argument("--update-steps", type=int, default=500)
    ap.add_argument("--device", default="cuda:0"); ap.add_argument("--sanity-only", action="store_true")
    a = ap.parse_args(); a.out_dir.mkdir(parents=True, exist_ok=True)
    logf = open(a.out_dir / "log.txt", "a")
    def log(msg):
        print(msg, flush=True); logf.write(msg + "\n"); logf.flush()
    seq = a.video_dir.name
    sub = "output_YCBInEOAT_sam2mask_mesh" if a.dataset == "ycb" else "output_HO3D_sam2mask_mesh"
    prior_paths = {"mesh_npz": SAM3D_ROOT / sub / f"{seq}_mesh_depth.npz", "pose_json": SAM3D_ROOT / sub / f"{seq}_mesh_depth.json", "gaussian_ply": SAM3D_ROOT / sub / f"{seq}_splat_depth.ply"}
    K_cam = np.loadtxt(a.run_dir / "cam_K.txt").reshape(3, 3).astype(np.float32)
    gt = GroundTruth(a.dataset, a.video_dir, a.run_dir)
    kf_ids = sorted(os.path.splitext(os.path.basename(f))[0] for f in glob.glob(str(a.run_dir / "color" / "*.png")))
    kf_ids = [f for f in kf_ids if gt.gt_c2w(f) is not None]
    log(f"{seq}: {len(kf_ids)} keyframes with GT in {a.run_dir}")
    results = []; sanity = None
    for kstar in a.kstars:
        last = kstar + a.after
        if last >= len(kf_ids):
            log(f"k*={kstar}: needs {last + 1} keyframes, have {len(kf_ids)} -> skipped"); continue
        for seed in a.seeds:
            t0 = time.time(); rng = np.random.RandomState(100 * kstar + seed)
            poses = {f: gt.gt_c2w(f).astype(np.float64) for f in kf_ids[:last + 1]}
            torch.manual_seed(seed)
            frames0 = [load_frame(a.run_dir, f, K_cam, poses[f].astype(np.float32)) for f in kf_ids[:5]]
            runner = build_runner(a.runner_config, a.device, prior_paths, frames0, poses[kf_ids[0]].astype(np.float32), a.initial_steps)
            runner.config["pose_feedback"]["enabled"] = False            # map building: poses locked
            centre_metric = -runner.normalization.translation.astype(np.float64)
            injected = perturb(poses[kf_ids[kstar]], centre_metric, a.inject_deg, a.inject_mm, rng)
            for j in range(5, last + 1):
                pose = injected if j == kstar else poses[kf_ids[j]]      # injected BEFORE the append of k*
                runner.update([load_frame(a.run_dir, kf_ids[j], K_cam, pose.astype(np.float32))], train_steps=a.update_steps)
            ids = kf_ids[:last + 1]
            rot0, tr0 = pose_errors(runner, gt, ids)
            log(f"k*={kstar} seed={seed}: map built ({runner.num_gaussians} gaussians, own layer {(runner.birth == kstar).sum().item()}), injected error {rot0[kstar]:.2f} deg / {tr0[kstar]:.2f} mm, others max {np.delete(rot0, kstar).max():.3f} deg ({time.time() - t0:.0f}s)")
            if sanity is None:
                sanity = sanity_checks(runner, kstar, log)
                json.dump(sanity, open(a.out_dir / "sanity.json", "w"), indent=1)
                if not sanity["pass"]:
                    log("SANITY FAILED -> no results written"); return
                if a.sanity_only:
                    return
            state = copy.deepcopy(runner)
            for steps in a.cycle_steps:
                for anchored in (False, True):
                    for joint in (False, True):
                        r = copy.deepcopy(state); r.config["pose_feedback"]["enabled"] = True
                        d = pose_cycle(r, steps, anchored, joint, seed); bake(r, d, anchored)
                        rot1, tr1 = pose_errors(r, gt, ids)
                        clean = np.ones(len(ids), bool); clean[kstar] = False; clean[0] = False
                        row = {"seq": seq, "kstar": kstar, "seed": seed, "steps": steps, "structure": "anchored" if anchored else "current", "map": "joint" if joint else "fixed",
                               "inj_rot_deg": float(rot0[kstar]), "inj_trans_mm": float(tr0[kstar]), "after_rot_deg": float(rot1[kstar]), "after_trans_mm": float(tr1[kstar]),
                               "rot_recovery": float(1 - rot1[kstar] / rot0[kstar]), "trans_recovery": float(1 - tr1[kstar] / tr0[kstar]),
                               "clean_drift_rot_median": float(np.median(rot1[clean])), "clean_drift_rot_max": float(rot1[clean].max()),
                               "clean_drift_trans_median": float(np.median(tr1[clean])), "clean_drift_trans_max": float(tr1[clean].max())}
                        results.append(row); log(json.dumps(row))
                        json.dump(results, open(a.out_dir / "results.json", "w"), indent=1)
                        del r; torch.cuda.empty_cache()
            del state, runner; torch.cuda.empty_cache()
    log("done")


if __name__ == "__main__":
    main()
