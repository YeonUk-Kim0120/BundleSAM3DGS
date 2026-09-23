"""EXP_BATCH_20260922 Q2 — hybrid (poses = original NOF, reconstruction = GS) under ONLINE conditions.
Pose source: the BundleSDF run's per-frame <frame>/keyframes.yml snapshots.  Keyframe j is appended with the pose it has in
the snapshot of its own frame (= the pose at consumption time); later snapshots refine it.
  consume     : never update anything after the append (worst case).
  reanchor    : before appending j, for every stored keyframe i whose pose changed in snapshot j (rot > 0.01 deg or
                trans > 0.01 mm): T_i = c2w_new @ inv(c2w_old) (metric object frame) is applied to the Gaussians born
                from i (means, quats; normalised frame via T_n = N T N^-1), to the novelty DB points born from i, and
                the view pose becomes c2w_new.  Prior lineage never moves.  Lifecycle fields untouched.
  refreshonly : view poses updated, layers left where they were (= what refresh_view_poses does in the main code).
Sanity (--sanity, AP12): after a full reanchor replay, re-anchoring once more with the last snapshot must give the 4.6
poses (max |dc2w| < 1e-4) and an online0 mesh P1 within noise of 4.6's.  Pose-change statistics (first consumption ->
final) are written to the manifest for every run.
"""
from __future__ import annotations
import argparse, glob, json, os, sys, time
from pathlib import Path
import numpy as np, torch, yaml
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "experiments"))
import exp_spring_anchor as E  # noqa: E402
from gaussian_global import run_global_refine  # noqa: E402
from run_gaussian_incremental import load_frame  # noqa: E402
from sam3d_prior import matrix_to_quat_wxyz  # noqa: E402
from scipy.spatial import cKDTree  # noqa: E402


def orth(P):
    P = np.array(P, dtype=np.float64); u, _, vt = np.linalg.svd(P[:3, :3]); R = u @ vt
    if np.linalg.det(R) < 0: u[:, -1] *= -1; R = u @ vt
    P[:3, :3] = R; return P


def load_snapshots(bsdf_run: Path):
    """frame id -> {kf id: c2w metric} for every per-frame snapshot, in frame order."""
    snaps = sorted(glob.glob(str(bsdf_run / "*/keyframes.yml")), key=lambda p: os.path.basename(os.path.dirname(p)))
    out = []
    for p in snaps:
        kf = yaml.safe_load(open(p)); out.append((os.path.basename(os.path.dirname(p)), {k.replace("keyframe_", ""): orth(np.asarray(v["cam_in_ob"]).reshape(4, 4)) for k, v in kf.items()}))
    return out


def pose_delta(a, b):
    R = a[:3, :3].T @ b[:3, :3]; return float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1)))), float(np.linalg.norm(a[:3, 3] - b[:3, 3]) * 1000)


class HybridRunner(E.BirthRunner):
    """BirthRunner + birth ids for the novelty DB points + rigid re-anchoring of a birth group."""
    obs_birth: np.ndarray | None = None

    def initialize_from_prior(self, *a, **k):
        out = super().initialize_from_prior(*a, **k)
        self.obs_birth = np.zeros(len(self.observed_points_metric), dtype=np.int64)   # prior points -> group 0 (never moved)
        return out

    def update(self, frames, train_steps=None):
        old_pts, old_birth = self.observed_points_metric.copy(), self.obs_birth.copy(); j = len(self.views)
        out = super().update(frames, train_steps=train_steps)
        # voxel_downsample averages within floor(p / voxel) cells, so a cell that existed before keeps its (first) birth id
        v = float(self.config["voxel_size"]); new = self.observed_points_metric
        if len(new) != len(old_pts) or not np.allclose(new, old_pts):
            keys_old = [tuple(k) for k in np.floor(old_pts.astype(np.float64) / v).astype(np.int64)]
            first = {}
            for k, b in zip(keys_old, old_birth):
                first.setdefault(k, int(b))
            keys_new = [tuple(k) for k in np.floor(new.astype(np.float64) / v).astype(np.int64)]
            self.obs_birth = np.array([first.get(k, j) for k in keys_new], dtype=np.int64)
        return out

    @torch.no_grad()
    def reanchor(self, view_index: int, c2w_new_metric: np.ndarray, move_layers: bool = True):
        view = self.views[view_index]; N = self.normalization
        c2w_old = N.metric_c2w(view.c2w_normalized.numpy()).astype(np.float64)
        T = c2w_new_metric @ np.linalg.inv(c2w_old)                                  # metric object frame
        if move_layers:
            s, t = N.scale, N.translation.astype(np.float64)
            R = T[:3, :3]; tt = T[:3, 3]
            # normalised frame: x_n = s (x + t), x' = R x + tt  ->  x_n' = R x_n + s (tt + t - R t)
            t_n = s * (tt + t - R @ t)   # x_n' = R x_n + s (tt + t - R t)   [bug fixed 2026-09-23: first 13 reanchor runs used s (R t + tt - t)]
            sel = (self.birth == view_index).to(self.device)
            if bool(sel.any()):
                m = self.splats["means"].data; q = self.splats["quats"].data
                Rt = torch.from_numpy(R.astype(np.float32)).to(self.device); tn = torch.from_numpy(t_n.astype(np.float32)).to(self.device)
                m[sel] = m[sel] @ Rt.T + tn
                qR = matrix_to_quat_wxyz(Rt[None].cpu()).to(self.device).float().expand(int(sel.sum()), 4)
                q[sel] = E.quat_mul(qR, q[sel])
            psel = self.obs_birth == view_index
            if psel.any():
                p = self.observed_points_metric[psel].astype(np.float64); self.observed_points_metric[psel] = (p @ R.T + tt).astype(np.float32)
        view.c2w_normalized = torch.from_numpy(N.normalize_c2w(c2w_new_metric).astype(np.float32)).contiguous()
        return pose_delta(c2w_old, c2w_new_metric)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--dataset", required=True); ap.add_argument("--seq", required=True)
    ap.add_argument("--bsdf-run", type=Path, required=True); ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--arm", choices=("consume", "reanchor", "refreshonly"), required=True)
    ap.add_argument("--runner-config", type=Path, default=REPO / "config_gs_2dgs_1mm_lifecycle.yml"); ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--sanity", action="store_true", help="after the replay: re-anchor with the last snapshot, compare with the last-snapshot poses, extract online0 only")
    a = ap.parse_args(); t0 = time.time()
    (a.out_dir / "gs_online").mkdir(parents=True, exist_ok=False)
    snaps = load_snapshots(a.bsdf_run); frame_of = {f: i for i, (f, _) in enumerate(snaps)}
    kf_ids = list(snaps[-1][1].keys()); first = {}
    for f, kf in snaps:
        for i in kf:
            if i not in first: first[i] = (f, kf[i])
    assert all(first[i][0] == i for i in kf_ids), "keyframe id != snapshot frame at first appearance"
    consume_pose = {i: first[i][1] for i in kf_ids}; final_pose = snaps[-1][1]
    K = np.loadtxt(a.bsdf_run / "cam_K.txt").reshape(3, 3).astype(np.float32)
    sub = "output_YCBInEOAT_sam2mask_mesh" if a.dataset == "ycb" else "output_HO3D_sam2mask_mesh"
    pp = {k: E.SAM3D_ROOT / sub / f"{a.seq}_{v}" for k, v in (("mesh_npz", "mesh_depth.npz"), ("pose_json", "mesh_depth.json"), ("gaussian_ply", "splat_depth.ply"))}
    # first 5 keyframes: poses as in the snapshot of keyframe 4 (the online backend starts after 5 keyframes)
    snap5 = snaps[frame_of[kf_ids[4]]][1]
    fr = lambda i, P: load_frame(a.bsdf_run, i, K, P.astype(np.float32))
    frames0 = [fr(i, snap5[i]) for i in kf_ids[:5]]
    rc_path = a.runner_config
    # build with the HybridRunner class (same alignment / normalisation as exp_spring_anchor.build_runner)
    E.BirthRunner, saved = HybridRunner, E.BirthRunner
    try:
        r = E.build_runner(rc_path, a.device, pp, frames0, snap5[kf_ids[0]].astype(np.float32), 500)
    finally:
        E.BirthRunner = saved
    assert isinstance(r, HybridRunner)
    r.config["pose_feedback"]["enabled"] = False
    stats = {"reanchor_events": 0, "reanchor_rot_deg": [], "reanchor_trans_mm": []}
    for j in range(5, len(kf_ids)):
        cur = snaps[frame_of[kf_ids[j]]][1]
        if a.arm in ("reanchor", "refreshonly"):
            for vi, view in enumerate(r.views):
                new = cur.get(view.frame_id)
                if new is None: continue
                old = r.normalization.metric_c2w(view.c2w_normalized.numpy()).astype(np.float64)
                dr, dt = pose_delta(old, new)
                if dr > 0.01 or dt > 0.01:
                    r.reanchor(vi, new, move_layers=(a.arm == "reanchor")); stats["reanchor_events"] += 1; stats["reanchor_rot_deg"].append(dr); stats["reanchor_trans_mm"].append(dt)
        r.update([fr(kf_ids[j], cur[kf_ids[j]])], train_steps=500)
    drift = [pose_delta(consume_pose[i], final_pose[i]) for i in kf_ids]
    rot = np.array([d[0] for d in drift]); tr = np.array([d[1] for d in drift])
    man = {"arm": a.arm, "bsdf_run": str(a.bsdf_run), "keyframes": len(kf_ids), "snapshots": len(snaps), "gaussians": r.num_gaussians,
           "pose_change_first_consume_to_final": {"rot_deg_median": float(np.median(rot)), "rot_deg_p90": float(np.percentile(rot, 90)), "rot_deg_max": float(rot.max()),
                                                  "trans_mm_median": float(np.median(tr)), "trans_mm_p90": float(np.percentile(tr, 90)), "trans_mm_max": float(tr.max())},
           "reanchor_events": stats["reanchor_events"], "reanchor_step_rot_deg_median": float(np.median(stats["reanchor_rot_deg"])) if stats["reanchor_rot_deg"] else None,
           "reanchor_step_trans_mm_median": float(np.median(stats["reanchor_trans_mm"])) if stats["reanchor_trans_mm"] else None, "seconds_online": time.time() - t0}
    if a.sanity:
        for vi, view in enumerate(r.views):
            r.reanchor(vi, final_pose[view.frame_id], move_layers=True)
        P = r.view_poses_metric(); diff = max(float(np.abs(P[vi] - final_pose[v.frame_id]).max()) for vi, v in enumerate(r.views))
        man["sanity_max_abs_pose_diff_vs_last_snapshot"] = diff
    r.save_checkpoint(a.out_dir / "gs_online" / "checkpoint_final.pt"); np.savetxt(a.out_dir / "cam_K.txt", K)
    json.dump(man, open(a.out_dir / "replay_manifest.json", "w"), indent=1); print(json.dumps(man), flush=True)
    del r; torch.cuda.empty_cache()
    run_global_refine(a.out_dir, out_dir=a.out_dir / "final" / "gs_online0", device=a.device, config={"steps": 0})
    if not a.sanity:
        run_global_refine(a.out_dir, out_dir=a.out_dir / "final" / "gs", device=a.device, config={"steps": 2000})
    print(f"done {a.seq} {a.arm}: {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
