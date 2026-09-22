"""EXP_BATCH_20260921 4.6 — hybrid upper bound: our GS map built with the ORIGINAL BundleSDF's final keyframe poses
(last <run>/*/keyframes.yml of the SAM2-mask baseline runs, read-only) and its saved color / depth_filtered / mask.
Same schedule as online (prior aligned online on keyframe 0, initial 500 steps on 5 keyframes, 500 steps per later
keyframe, poses locked), then gaussian_global.run_global_refine twice: steps 0 -> final/gs_online0, 2000 -> final/gs.
usage: exp_replay_bsdf_poses.py --dataset ho3d --seq AP12 --bsdf-run <dir> --out-dir outputs/exp_batch_20260921/bsdfpose/ho3d/AP12
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
import numpy as np, torch
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "experiments"))
import exp_spring_anchor as E  # noqa: E402  (build_runner = online-like prior alignment + initialize_from_prior)
from gaussian_global import run_global_refine  # noqa: E402
from run_gaussian_incremental import load_frame  # noqa: E402

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--dataset", required=True); ap.add_argument("--seq", required=True)
    ap.add_argument("--bsdf-run", type=Path, required=True); ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--runner-config", type=Path, default=REPO / "config_gs_2dgs_1mm_lifecycle.yml"); ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args(); t0 = time.time()
    (a.out_dir / "gs_online").mkdir(parents=True, exist_ok=False)
    import yaml
    snaps = sorted(a.bsdf_run.glob("*/keyframes.yml"), key=lambda p: p.parent.name); snap = snaps[-1]
    kf = yaml.safe_load(open(snap)); ids = [k.replace("keyframe_", "") for k in kf]; poses = [np.asarray(kf[k]["cam_in_ob"], dtype=np.float64).reshape(4, 4) for k in kf]
    for P in poses:  # project rotations onto SO(3) (validate_c2w_cv tolerance)
        u, _, vt = np.linalg.svd(P[:3, :3]); R = u @ vt
        if np.linalg.det(R) < 0: u[:, -1] *= -1; R = u @ vt
        P[:3, :3] = R
    sub = "output_YCBInEOAT_sam2mask_mesh" if a.dataset == "ycb" else "output_HO3D_sam2mask_mesh"
    pp = {k: E.SAM3D_ROOT / sub / f"{a.seq}_{v}" for k, v in (("mesh_npz", "mesh_depth.npz"), ("pose_json", "mesh_depth.json"), ("gaussian_ply", "splat_depth.ply"))}
    K = np.loadtxt(a.bsdf_run / "cam_K.txt").reshape(3, 3).astype(np.float32)
    fr = lambda j: load_frame(a.bsdf_run, ids[j], K, poses[j].astype(np.float32))
    r = E.build_runner(a.runner_config, a.device, pp, [fr(j) for j in range(5)], poses[0].astype(np.float32), 500)
    r.config["pose_feedback"]["enabled"] = False
    for j in range(5, len(ids)):
        r.update([fr(j)], train_steps=500)
    r.save_checkpoint(a.out_dir / "gs_online" / "checkpoint_final.pt")
    np.savetxt(a.out_dir / "cam_K.txt", K)
    json.dump({"bsdf_run": str(a.bsdf_run), "keyframes_yml": str(snap), "keyframes": len(ids), "frames_source": "bsdf run color/depth_filtered/mask",
               "gaussians": r.num_gaussians, "seconds_online": time.time() - t0}, open(a.out_dir / "replay_manifest.json", "w"), indent=1)
    del r; torch.cuda.empty_cache()
    run_global_refine(a.out_dir, out_dir=a.out_dir / "final" / "gs_online0", device=a.device, config={"steps": 0})
    run_global_refine(a.out_dir, out_dir=a.out_dir / "final" / "gs", device=a.device, config={"steps": 2000})
    print(f"done {a.seq}: {len(ids)} keyframes, {time.time() - t0:.0f}s")

if __name__ == "__main__":
    main()
