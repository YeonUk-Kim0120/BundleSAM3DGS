"""Keyframe poses before/after the global stage vs GT (rotation / translation error, share improved).
usage: eval_global_poses.py --dataset ho3d --video-dir ... --run-dir ... --global-dir <run_dir>/final/<name> --out-json ..."""
from __future__ import annotations
import argparse, glob, json, sys
from pathlib import Path
import numpy as np
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "experiments"))
from exp_feedback_gradient_probe import GroundTruth, rot_deg  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=("ycb", "ho3d"), required=True)
    ap.add_argument("--video-dir", type=Path, required=True)
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--global-dir", type=Path, required=True)
    ap.add_argument("--out-json", type=Path, required=True)
    args = ap.parse_args()
    gt = GroundTruth(args.dataset, args.video_dir, args.run_dir)
    ids = open(sorted(glob.glob(str(args.run_dir / "*/nerf_frames.txt")))[-1]).read().split()
    before = np.loadtxt(args.global_dir / "poses_before_global.txt").reshape(-1, 4, 4)
    after = np.loadtxt(args.global_dir / "poses_after_global.txt").reshape(-1, 4, 4)
    rb, ra, tb, ta, mv = [], [], [], [], []
    for fid, b, a in zip(ids, before, after):
        g = gt.gt_c2w(fid)
        if g is None:
            continue
        rb.append(gt.rot_error(b, fid)); ra.append(gt.rot_error(a, fid))
        tb.append(np.linalg.norm(b[:3, 3] - g[:3, 3]) * 1000); ta.append(np.linalg.norm(a[:3, 3] - g[:3, 3]) * 1000)
        mv.append(rot_deg(b[:3, :3], a[:3, :3]))
    rb, ra, tb, ta, mv = map(np.array, (rb, ra, tb, ta, mv))
    out = {"keyframes": int(len(rb)), "rot_err_before_deg": float(rb.mean()), "rot_err_after_deg": float(ra.mean()),
           "rot_err_median_before": float(np.median(rb)), "rot_err_median_after": float(np.median(ra)),
           "trans_err_before_mm": float(tb.mean()), "trans_err_after_mm": float(ta.mean()),
           "improved_frac": float(np.mean(ra < rb - 0.05)), "worsened_frac": float(np.mean(ra > rb + 0.05)),
           "moved_mean_deg": float(mv.mean()), "moved_max_deg": float(mv.max())}
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(args.out_json, "w"), indent=1)
    print(f"{args.global_dir.name}: rot {out['rot_err_before_deg']:.2f}->{out['rot_err_after_deg']:.2f} deg, trans {out['trans_err_before_mm']:.1f}->{out['trans_err_after_mm']:.1f} mm, "
          f"improved {out['improved_frac']:.0%} worsened {out['worsened_frac']:.0%}, moved {out['moved_mean_deg']:.2f}/{out['moved_max_deg']:.2f} deg")


if __name__ == "__main__":
    main()
