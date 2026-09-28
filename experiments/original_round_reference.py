"""EXP_BATCH_20260928 reference (read-only analysis of the ORIGINAL BundleSDF online runs, SAM2 masks): per NeRF round, the keyframe
d (mean GT-mesh point displacement, mm; same measure and GroundTruth first-frame gauge as the batch, gauge taken from that run's own
first ob_in_cam) before the round (poses_before_nerf.txt = tracker's current keyframe poses incl. earlier write-backs) and after it
(poses_after_nerf.txt).  Two original runs: SAM2 masks (full_eval_sam2_<ds>) and the paper masks (full_eval).  Nothing is written under
~/Desktop/BundleSDF_baseline_outputs.  Output: logs/exp_batch_20260928/analysis/original_rounds.json = {"sam2": {seq: rounds}, "paper": {...}}"""
from __future__ import annotations
import glob, json, sys
from pathlib import Path
import numpy as np
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "experiments")); sys.path.append(str(REPO / "BundleTrack/scripts"))
from exp_feedback_gradient_probe_gtmap import GroundTruth, gt_mesh_in_tracker_frame  # noqa: E402
from exp_round_inputs import orth  # noqa: E402
from joint_refit_eval import Evaluator  # noqa: E402
BASE = Path("/home/kist/Desktop/BundleSDF_baseline_outputs")


def main():
    out = {"sam2": {}, "paper": {}}
    for variant, ds, seq in [(v, d, q) for v in ("sam2", "paper") for d, q in (("ho3d", "SM1"), ("ho3d", "AP12"), ("ho3d", "MPM12"), ("ycb", "mustard0"))]:
        run = (BASE / f"full_eval_sam2_{ds}" if variant == "sam2" else BASE / "full_eval") / ds / seq
        video = REPO / ("datasets/HO3D_v3/evaluation" if ds == "ho3d" else "datasets/YCBInEOAT") / seq
        rounds = sorted(glob.glob(str(run / "*/poses_after_nerf.txt")))
        if not rounds: out[variant][seq] = None; continue
        gt = GroundTruth(ds, video, run); mesh = gt_mesh_in_tracker_frame(ds, seq, gt, n_samples=20000)
        res = []
        for f in rounds:
            d = Path(f).parent
            if not (d / "nerf_frames.txt").exists(): continue  # YCB baseline rounds were saved without the keyframe list
            ids = [l.strip() for l in open(d / "nerf_frames.txt") if l.strip()]
            B = np.loadtxt(d / "poses_before_nerf.txt").reshape(-1, 4, 4); A = np.loadtxt(f).reshape(-1, 4, 4); n = min(len(ids), len(B), len(A))
            gts = [gt.gt_c2w(i) for i in ids[:n]]
            ev = Evaluator(dict(kf_ids=ids[:n], gt=[orth(g).tolist() if g is not None else None for g in gts], mesh_points=mesh.tolist()))
            rows = ev.compare(ids[:n], [orth(b) for b in B[:n]], [orth(a) for a in A[:n]])
            if not rows: continue
            d0 = np.array([r["d0"] for r in rows]); d1 = np.array([r["d1"] for r in rows])
            res.append(dict(round_dir=d.name, n=n, n_gt=len(rows), before_med=float(np.median(d0)), after_med=float(np.median(d1)), before_p90=float(np.percentile(d0, 90)),
                            after_p90=float(np.percentile(d1, 90)), imp=float(np.mean([r["improved"] for r in rows])), wor=float(np.mean([r["worsened"] for r in rows])),
                            final_kf_d=[r["d1"] for r in rows] if f == rounds[-1] else None))
        out[variant][seq] = res
        if not res: print(f"{variant} {seq}: no scorable rounds (no nerf_frames.txt)", flush=True); continue
        last = res[-1]; print(f"{variant} {seq}: {len(res)} rounds; last round n={last['n']}: d median {last['before_med']:.2f} -> {last['after_med']:.2f} mm, p90 {last['before_p90']:.2f} -> {last['after_p90']:.2f}; "
                              f"mean per-round change of median {np.mean([r['after_med'] - r['before_med'] for r in res]):+.2f} mm", flush=True)
    (REPO / "logs/exp_batch_20260928/analysis").mkdir(parents=True, exist_ok=True)
    json.dump(out, open(REPO / "logs/exp_batch_20260928/analysis/original_rounds.json", "w"))


if __name__ == "__main__":
    main()
