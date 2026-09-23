"""EXP_BATCH_20260922 Q1 — prior ablation under controlled (BundleSDF) poses.  Same replay as
exp_replay_bsdf_poses.py (last keyframes.yml of the SAM2-on baseline run, its color/depth_filtered/mask, 500/500, noop),
with the prior handled in one of three ways:
  none     : no prior; the map starts from the first 5 keyframes' depth points (GaussianRunner.initialize, normalisation
             from run_gaussian_incremental.compute_initial_normalization); lifecycle fields = observed lineage only.
  frozen   : the aligned prior is inserted and stays UNSEEN forever (no learning, no transitions, no removal):
             classify_lifecycle is a no-op after the initial insertion.
  initonly : the prior is treated like observations: all VERIFIED right after insertion, then no transitions / removal.
'full' (current lifecycle) is the 2026-09-21 4.6 run and is not re-run.
usage: exp_prior_ablation_replay.py --dataset ho3d --seq AP12 --bsdf-run <dir> --out-dir <new dir> --prior-mode none [--max-keyframes 10]
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
import numpy as np, torch, yaml
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "experiments"))
import exp_spring_anchor as E  # noqa: E402
from gaussian_global import run_global_refine  # noqa: E402
from gaussian_runner import GaussianRunner, load_gaussian_config  # noqa: E402
from prior_lifecycle import STATE_UNSEEN, STATE_VERIFIED  # noqa: E402
from run_gaussian_incremental import compute_initial_normalization, load_frame  # noqa: E402


class NoTransitionRunner(GaussianRunner):
    """Prior lifecycle without transitions: classify_lifecycle runs once at insertion (mode-specific), then is a no-op."""
    mode = "frozen"
    _inserted = False

    def classify_lifecycle(self, frames, event, occ_masks=None):
        if not self._inserted:
            self._inserted = True
            lf = self.lifecycle_fields
            if lf is not None:
                lf.state[lf.lineage] = STATE_UNSEEN if self.mode == "frozen" else STATE_VERIFIED
            self.lifecycle_log.append({"event": event, "n_frames": len(frames), "mode": self.mode, "note": "no-transition prior"})
            return None
        return None   # later keyframes: no classification, no state change


class FrozenRunner(NoTransitionRunner):
    mode = "frozen"


class InitOnlyRunner(NoTransitionRunner):
    mode = "initonly"


def build_with_prior(cls, cfg_path, device, prior_paths, frames, cam_in_obs0, initial_steps):
    """exp_spring_anchor.build_runner with a chosen runner class (same alignment / normalisation / init)."""
    from run_sam3d_alignment import DEFAULT_CONFIG as ALIGN_DEFAULTS
    from run_sam3d_alignment import align_prior_sim3, prepare_target
    from sam3d_prior import (load_mesh_prior, load_sam3d_gaussian_ply, load_sam3d_pose_or_refined, sample_surfels,
                             transfer_gaussian_colors, transform_surfels_canonical_to_cv_camera)
    from gaussian_runner import SceneNormalization
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
    c0 = np.asarray(cam_in_obs0, dtype=np.float64); m = scv.means.numpy() @ c0[:3, :3].T + c0[:3, 3][None]
    centre = m.mean(0); radius = float(np.linalg.norm(m - centre, axis=1).max())
    r = cls(rc, SceneNormalization(scale=1.0 / max(radius * 1.2, 1e-6), translation=-centre), device=device)
    r.initialize_from_prior(scv, cam_in_obs0, frames, train_steps=initial_steps)
    return r, status


def load_bsdf_keyframes(bsdf_run: Path):
    snaps = sorted(bsdf_run.glob("*/keyframes.yml"), key=lambda p: p.parent.name); snap = snaps[-1]
    kf = yaml.safe_load(open(snap)); ids = [k.replace("keyframe_", "") for k in kf]
    poses = [np.asarray(kf[k]["cam_in_ob"], dtype=np.float64).reshape(4, 4) for k in kf]
    for P in poses:
        u, _, vt = np.linalg.svd(P[:3, :3]); R = u @ vt
        if np.linalg.det(R) < 0: u[:, -1] *= -1; R = u @ vt
        P[:3, :3] = R
    return snap, ids, poses


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--dataset", required=True); ap.add_argument("--seq", required=True)
    ap.add_argument("--bsdf-run", type=Path, required=True); ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--prior-mode", choices=("none", "frozen", "initonly"), required=True)
    ap.add_argument("--runner-config", type=Path, default=REPO / "config_gs_2dgs_1mm_lifecycle.yml"); ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--max-keyframes", type=int, default=0); ap.add_argument("--skip-global", action="store_true")
    a = ap.parse_args(); t0 = time.time()
    (a.out_dir / "gs_online").mkdir(parents=True, exist_ok=False)
    snap, ids, poses = load_bsdf_keyframes(a.bsdf_run)
    if a.max_keyframes: ids, poses = ids[:a.max_keyframes], poses[:a.max_keyframes]
    K = np.loadtxt(a.bsdf_run / "cam_K.txt").reshape(3, 3).astype(np.float32)
    fr = lambda j: load_frame(a.bsdf_run, ids[j], K, poses[j].astype(np.float32))
    first5 = [fr(j) for j in range(5)]; status = None
    if a.prior_mode == "none":
        rc = load_gaussian_config(a.runner_config); rc["device"] = a.device
        torch.cuda.set_device(torch.device(a.device))
        norm, meta = compute_initial_normalization(first5, a.out_dir, rc)
        r = GaussianRunner(rc, norm, device=a.device)
        r.initialize(first5, train_steps=500)
    else:
        sub = "output_YCBInEOAT_sam2mask_mesh" if a.dataset == "ycb" else "output_HO3D_sam2mask_mesh"
        pp = {k: E.SAM3D_ROOT / sub / f"{a.seq}_{v}" for k, v in (("mesh_npz", "mesh_depth.npz"), ("pose_json", "mesh_depth.json"), ("gaussian_ply", "splat_depth.ply"))}
        cls = FrozenRunner if a.prior_mode == "frozen" else InitOnlyRunner
        r, status = build_with_prior(cls, a.runner_config, a.device, pp, first5, poses[0].astype(np.float32), 500)
    r.config["pose_feedback"]["enabled"] = False
    for j in range(5, len(ids)):
        r.update([fr(j)], train_steps=500)
    lf = r.lifecycle_fields; st = lf.state.cpu().numpy(); ln = lf.lineage.cpu().numpy()
    counts = {"prior_lineage": int(ln.sum()), "prior_unseen": int((ln & (st == STATE_UNSEEN)).sum()), "prior_verified": int((ln & (st == STATE_VERIFIED)).sum()),
              "prior_other": int((ln & (st != STATE_UNSEEN) & (st != STATE_VERIFIED)).sum()), "observed": int((~ln).sum()), "gaussians": int(r.num_gaussians)}
    r.save_checkpoint(a.out_dir / "gs_online" / "checkpoint_final.pt"); np.savetxt(a.out_dir / "cam_K.txt", K)
    json.dump({"prior_mode": a.prior_mode, "bsdf_run": str(a.bsdf_run), "keyframes_yml": str(snap), "keyframes": len(ids), "keyframe_ids": ids,
               "poses_sha": __import__("hashlib").sha1(np.asarray(poses).tobytes()).hexdigest(), "alignment": status, "final_state_counts": counts,
               "seconds_online": time.time() - t0}, open(a.out_dir / "replay_manifest.json", "w"), indent=1)
    print("state counts", counts, flush=True)
    del r; torch.cuda.empty_cache()
    if not a.skip_global:
        run_global_refine(a.out_dir, out_dir=a.out_dir / "final" / "gs_online0", device=a.device, config={"steps": 0})
        run_global_refine(a.out_dir, out_dir=a.out_dir / "final" / "gs", device=a.device, config={"steps": 2000})
    print(f"done {a.seq} {a.prior_mode}: {len(ids)} keyframes, {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
