"""Per-frame ADD error over the sequence, per sequence and averaged (2026-09-16, user request).

For every sequence: the feedback-on run (default: the adopted configuration runs in outputs/exp_fusion_20260915/adopt)
and, if available, the tracker-alone run archived in outputs/_archive_records_20260912/fbabl_*_off_*.  Per-frame ADD as
in experiments/eval_add_ycbineoat.py (first-frame alignment, YCB model vertices; HO3D frames without GT skipped).
Outputs in --out-dir: add_over_frames_<seq>.png, add_over_frames_mean_<dataset>.png (curves resampled on the
normalised frame index 0..1, 100 bins), per_frame_add.json (all curves).
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "experiments")); sys.path.append(str(REPO / "BundleTrack/scripts"))

HO = ["AP10", "AP11", "AP12", "AP13", "AP14", "MPM10", "MPM11", "MPM12", "MPM13", "MPM14", "SB11", "SB13", "SM1"]
YC = ["bleach0", "bleach_hard_00_03_chaitanya", "cracker_box_reorient", "cracker_box_yalehand0", "mustard0",
      "mustard_easy_00_02", "sugar_box1", "sugar_box_yalehand0", "tomato_soup_can_yalehand0"]


def per_frame_add(dataset: str, video_dir: Path, run_dir: Path):
    import trimesh
    from Utils import add_err  # noqa: E402
    pose_files = sorted(glob.glob(str(run_dir / "ob_in_cam" / "*.txt")))
    if dataset == "ho3d":
        from data_reader import Ho3dReader  # noqa: E402
        reader = Ho3dReader(str(video_dir))
        ids, gts = [], []
        for i in range(len(reader.color_files)):
            gt = reader.get_gt_pose(i)
            if gt is None:
                continue
            ids.append(i); gts.append(gt)
        gts = np.array(gts); pts = np.asarray(reader.get_gt_mesh().vertices, dtype=np.float64)
        n_frames = len(reader.color_files)
    else:
        from eval_add_ycbineoat import VIDEO_TO_OBJECT  # noqa: E402
        color_files = sorted(glob.glob(str(video_dir / "rgb" / "*.png")))
        gt_files = sorted(glob.glob(str(video_dir / "annotated_poses" / "*")))
        gts = np.array([np.loadtxt(f).reshape(4, 4) for f in gt_files]); ids = list(range(len(color_files)))
        mesh = trimesh.load(str(REPO / "datasets/YCB_Video_Models/models" / VIDEO_TO_OBJECT[video_dir.name] / "textured_simple.obj"))
        pts = np.asarray(mesh.vertices, dtype=np.float64); n_frames = len(color_files)
    preds = np.array([np.loadtxt(pose_files[i]).reshape(4, 4) for i in ids])
    preds = preds @ np.linalg.inv(preds[0]) @ gts[0]
    add = np.array([add_err(p, g, pts.copy()) for p, g in zip(preds, gts)]) * 100.0
    return np.array(ids), add, n_frames


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-root", type=Path, default=Path("outputs/exp_fusion_20260915/adopt"))
    ap.add_argument("--off-root", type=Path, default=Path("outputs/_archive_records_20260912"))
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--label", default="feedback on (adopted config)")
    args = ap.parse_args()
    out = args.out_dir; out.mkdir(parents=True, exist_ok=True)
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    curves = {}; bins = np.linspace(0, 1, 101); centres = 0.5 * (bins[1:] + bins[:-1])
    for ds, seqs in (("ho3d", HO), ("ycb", YC)):
        for s in seqs:
            vd = Path("datasets/HO3D_v3/evaluation") / s if ds == "ho3d" else Path("datasets/YCBInEOAT") / s
            rd = args.run_root / ds / s
            if not (rd / "ob_in_cam").exists():
                print("missing run", rd); continue
            ids, add_on, n = per_frame_add(ds, vd, rd)
            off = None
            offs = sorted(glob.glob(str(args.off_root / f"fbabl_{ds}_{s}_off_*"))) or (sorted(glob.glob(str(args.off_root / "ablation_fb_off_r1_*"))) if s == "mustard0" else [])  # mustard0 SAM2-mask tracker-alone (ADD 0.743)
            if offs:
                ids_off, add_off, _ = per_frame_add(ds, vd, Path(offs[-1])); off = add_off if len(ids_off) == len(ids) else None
            x = ids / max(n - 1, 1)
            def resample(y):
                return np.array([y[(x >= bins[k]) & (x < bins[k + 1] + (1e-9 if k == 99 else 0))].mean() if ((x >= bins[k]) & (x < bins[k + 1] + (1e-9 if k == 99 else 0))).any() else np.nan for k in range(100)])
            curves[s] = {"dataset": ds, "frames": int(n), "ids": ids.tolist(), "add_on": add_on.tolist(), "add_off": None if off is None else off.tolist(),
                         "on_resampled": resample(add_on).tolist(), "off_resampled": None if off is None else resample(off).tolist()}
            fig, ax = plt.subplots(figsize=(12, 4))
            ax.plot(ids, add_on, lw=1, label=f"{args.label} (mean {add_on.mean():.2f} cm)")
            if off is not None:
                ax.plot(ids, off, lw=1, alpha=0.8, label=f"tracker alone (mean {off.mean():.2f} cm)")
            ax.set_xlabel("frame"); ax.set_ylabel("ADD (cm)"); ax.set_title(f"{ds}/{s}: per-frame ADD"); ax.legend(); ax.grid(alpha=0.3)
            fig.tight_layout(); fig.savefig(out / f"add_over_frames_{s}.png", dpi=90); plt.close(fig)
            print(f"{ds}/{s}: frames {n}, ADD on {add_on.mean():.3f}" + (f", off {off.mean():.3f}" if off is not None else ""))
    json.dump(curves, open(out / "per_frame_add.json", "w"))
    for ds, seqs in (("ho3d", HO), ("ycb", YC), ("all", HO + YC)):
        fig, ax = plt.subplots(figsize=(10, 5))
        on = np.array([curves[s]["on_resampled"] for s in seqs if s in curves]); ax.plot(centres, np.nanmean(on, 0), lw=2, label=f"{args.label}, mean of {len(on)} sequences")
        offc = [curves[s]["off_resampled"] for s in seqs if s in curves and curves[s]["off_resampled"] is not None]
        if offc:
            ax.plot(centres, np.nanmean(np.array(offc), 0), lw=2, alpha=0.8, label=f"tracker alone, mean of {len(offc)} sequences")
        for s in seqs:
            if s in curves: ax.plot(centres, curves[s]["on_resampled"], lw=0.6, alpha=0.35)
        ax.set_xlabel("normalised frame index (0 = first frame, 1 = last)"); ax.set_ylabel("ADD (cm)"); ax.set_title(f"{ds}: ADD over the sequence (thin lines = individual sequences, feedback on)")
        ax.legend(); ax.grid(alpha=0.3); fig.tight_layout(); fig.savefig(out / f"add_over_frames_mean_{ds}.png", dpi=90); plt.close(fig)
    print("wrote", out)


if __name__ == "__main__":
    main()
