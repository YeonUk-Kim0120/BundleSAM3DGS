"""EXP_BATCH_20260921 5.1 follow-up (gate failed): does view k*'s own layer act as a spring?  Same map build as
exp_spring_anchor.py (GT poses, error injected into k* before its append).  At the wrong pose of k*, the gradient of the
loss w.r.t. Delta_k* is compared with the true correction (rotation part / translation part cosines, 1 = points at GT):
  current, full map | current, own layer hidden | current, ONLY own layer (the spring itself: should be ~0 / pull nowhere)
  anchored, loss of view k* only | anchored, sum of the losses of all views (k*'s layer seen from the other views)."""
from __future__ import annotations
import argparse, glob, json, os, sys
from pathlib import Path
import numpy as np, torch
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "experiments"))
import exp_spring_anchor as E  # noqa: E402
from run_gaussian_incremental import load_frame  # noqa: E402

def rotvec(R):
    ang = np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1)); 
    if ang < 1e-9: return np.zeros(3)
    return ang / (2 * np.sin(ang)) * np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--dataset", required=True); ap.add_argument("--video-dir", type=Path, required=True); ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--out-json", type=Path, required=True); ap.add_argument("--kstars", type=int, nargs="+", default=[10, 20]); ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1]); ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args(); seq = a.video_dir.name
    sub = "output_YCBInEOAT_sam2mask_mesh" if a.dataset == "ycb" else "output_HO3D_sam2mask_mesh"
    pp = {k: E.SAM3D_ROOT / sub / f"{seq}_{v}" for k, v in (("mesh_npz", "mesh_depth.npz"), ("pose_json", "mesh_depth.json"), ("gaussian_ply", "splat_depth.ply"))}
    K = np.loadtxt(a.run_dir / "cam_K.txt").reshape(3, 3).astype(np.float32); gt = E.GroundTruth(a.dataset, a.video_dir, a.run_dir)
    ids = sorted(os.path.splitext(os.path.basename(f))[0] for f in glob.glob(str(a.run_dir / "color" / "*.png"))); ids = [f for f in ids if gt.gt_c2w(f) is not None]
    rows = []
    for ks in a.kstars:
        last = ks + 10
        if last >= len(ids): continue
        for seed in a.seeds:
            rng = np.random.RandomState(100 * ks + seed); torch.manual_seed(seed)
            poses = {f: gt.gt_c2w(f).astype(np.float64) for f in ids[:last + 1]}
            r = E.build_runner(REPO / "config_gs_2dgs_1mm_lifecycle.yml", a.device, pp, [load_frame(a.run_dir, f, K, poses[f].astype(np.float32)) for f in ids[:5]], poses[ids[0]].astype(np.float32), 500)
            r.config["pose_feedback"]["enabled"] = False
            inj = E.perturb(poses[ids[ks]], -r.normalization.translation.astype(np.float64), 3.0, 5.0, rng)
            for j in range(5, last + 1):
                r.update([load_frame(a.run_dir, ids[j], K, (inj if j == ks else poses[ids[j]]).astype(np.float32))], train_steps=500)
            # true correction in the normalized frame: T c2w_wrong = c2w_gt
            n = r.normalization; Tn = n.normalize_c2w(poses[ids[ks]]) @ np.linalg.inv(n.normalize_c2w(inj))
            true_rot, true_tr = rotvec(Tn[:3, :3]), Tn[:3, 3]
            dev = r.device; b = r.birth.to(dev); op = torch.sigmoid(r.splats["opacities"]).detach(); own = (b == ks).float()
            def grad(anchored, opac, all_views):
                d = r._new_pose_deltas(len(r.views), fix_first=True); loss = 0
                for vi in (range(len(r.views)) if all_views else [ks]):
                    view, res = E.render_view(r, d, b, anchored, vi, opac); loss = loss + E.view_loss(r, view, res)
                g, = torch.autograd.grad(loss, d.data); g = -g[ks].cpu().numpy() * np.array([d.max_trans_norm] * 3 + [d.max_rot_rad] * 3)  # descent direction in delta units
                cos = lambda x, y: float(x @ y / (np.linalg.norm(x) * np.linalg.norm(y) + 1e-30))
                return {"cos_rot": cos(g[3:], true_rot), "cos_trans": cos(g[:3], true_tr), "norm": float(np.linalg.norm(g))}
            row = {"seq": seq, "kstar": ks, "seed": seed, "own_layer": int(own.sum()), "gaussians": int(len(b)),
                   "current_full": grad(False, None, False), "current_own_hidden": grad(False, op * (1 - own), False), "current_own_only": grad(False, op * own, False),
                   "anchored_view_k": grad(True, None, False), "anchored_all_views": grad(True, None, True)}
            rows.append(row); print(json.dumps(row), flush=True); json.dump(rows, open(a.out_json, "w"), indent=1)
            del r; torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
