"""EXP_BATCH_20260922 §4 — 0/0 training steps under BundleSDF poses (initial_opacity 0.5, full lifecycle, otherwise as
exp_replay_bsdf_poses.py): is any training needed in the hybrid?  Reuses exp_spring_anchor.build_runner (online-like
prior alignment) with initial_steps 0 and update train_steps 0."""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
import numpy as np, torch
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "experiments"))
import exp_spring_anchor as E  # noqa: E402
from exp_prior_ablation_replay import load_bsdf_keyframes  # noqa: E402
from gaussian_global import run_global_refine  # noqa: E402
from run_gaussian_incremental import load_frame  # noqa: E402

ap = argparse.ArgumentParser(); ap.add_argument("--dataset", required=True); ap.add_argument("--seq", required=True)
ap.add_argument("--bsdf-run", type=Path, required=True); ap.add_argument("--out-dir", type=Path, required=True)
ap.add_argument("--runner-config", type=Path, default=REPO / "logs/exp_batch_20260922/config_o05.yml"); ap.add_argument("--device", default="cuda:0")
a = ap.parse_args(); t0 = time.time(); (a.out_dir / "gs_online").mkdir(parents=True, exist_ok=False)
snap, ids, poses = load_bsdf_keyframes(a.bsdf_run); K = np.loadtxt(a.bsdf_run / "cam_K.txt").reshape(3, 3).astype(np.float32)
sub = "output_YCBInEOAT_sam2mask_mesh" if a.dataset == "ycb" else "output_HO3D_sam2mask_mesh"
pp = {k: E.SAM3D_ROOT / sub / f"{a.seq}_{v}" for k, v in (("mesh_npz", "mesh_depth.npz"), ("pose_json", "mesh_depth.json"), ("gaussian_ply", "splat_depth.ply"))}
fr = lambda j: load_frame(a.bsdf_run, ids[j], K, poses[j].astype(np.float32))
r = E.build_runner(a.runner_config, a.device, pp, [fr(j) for j in range(5)], poses[0].astype(np.float32), 0)
r.config["pose_feedback"]["enabled"] = False; cyc = []
for j in range(5, len(ids)):
    t = time.time(); r.update([fr(j)], train_steps=0); cyc.append(time.time() - t)
r.save_checkpoint(a.out_dir / "gs_online" / "checkpoint_final.pt"); np.savetxt(a.out_dir / "cam_K.txt", K)
json.dump({"arm": "zero", "runner_config": str(a.runner_config), "initial_steps": 0, "update_steps": 0, "bsdf_run": str(a.bsdf_run), "keyframes_yml": str(snap), "keyframes": len(ids),
           "gaussians": r.num_gaussians, "cycle_seconds_mean": float(np.mean(cyc)), "seconds_online": time.time() - t0}, open(a.out_dir / "replay_manifest.json", "w"), indent=1)
del r; torch.cuda.empty_cache()
run_global_refine(a.out_dir, out_dir=a.out_dir / "final" / "gs_online0", device=a.device, config={"steps": 0})
run_global_refine(a.out_dir, out_dir=a.out_dir / "final" / "gs", device=a.device, config={"steps": 2000})
print(f"done {a.seq} zero: {len(ids)} keyframes, {time.time() - t0:.0f}s")
