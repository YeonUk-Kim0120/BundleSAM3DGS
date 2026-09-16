"""Per-cycle analysis of the v1 pose feedback (2026-09-15, ⑤ discussion): does the harm come from specific cycles?

For one feedback-on run (HO3D) it joins, per keyframe cycle:
  - the pose delta the GS applied (feedback_log.json: newest view and max over views, mm / deg),
  - the observation quality of that keyframe (mask area from the checkpoint views, ratio to the median of the previous
    10 keyframes; tracker RANSAC inliers vs the previous frame and global correspondences from the run log),
  - the per-frame ADD error of this run and of a tracker-alone run (first-frame alignment as in the ADD evaluator),
    and the error change until the next keyframe ("growth"); harm = growth(on) - growth(off).
Outputs: cycles.csv, summary.json (Spearman correlations and quantile contrasts), feedback_cycles.png.
usage: analyze_feedback_cycles.py --video-dir datasets/HO3D_v3/evaluation/AP12 --run-dir <on run> --off-dir <tracker-alone run>
       --run-log <pipeline log of the on run> --out-dir ... [--device cuda:0]
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "experiments")); sys.path.append(str(REPO / "BundleTrack/scripts"))


def per_frame_add(video_dir: Path, run_dir: Path):
    from data_reader import Ho3dReader  # noqa: E402
    from Utils import add_err  # noqa: E402
    reader = Ho3dReader(str(video_dir))
    pose_files = sorted(glob.glob(str(run_dir / "ob_in_cam" / "*.txt")))
    ids, gts = [], []
    for i in range(len(reader.color_files)):
        gt = reader.get_gt_pose(i)
        if gt is None:
            continue
        ids.append(i); gts.append(gt)
    gts = np.array(gts)
    preds = np.array([np.loadtxt(pose_files[i]).reshape(4, 4) for i in ids])
    preds = preds @ np.linalg.inv(preds[0]) @ gts[0]
    pts = np.asarray(reader.get_gt_mesh().vertices, dtype=np.float64)
    add = np.array([add_err(p, g, pts.copy()) for p, g in zip(preds, gts)]) * 100.0
    names = [os.path.splitext(os.path.basename(pose_files[i]))[0] for i in ids]
    return names, add


def parse_run_log(path: Path):
    inl_prev, inl_max, gcorr = {}, {}, {}
    pat = re.compile(r"ransac makes match betwee frame (\d+) (\d+) #inliers=(\d+)")
    pat_g = re.compile(r"OptimizerGPU begin, global_corres#=(\d+)")
    last_frame = None
    for line in open(path, errors="ignore"):
        m = pat.search(line)
        if m:
            a, b, n = m.group(1), m.group(2), int(m.group(3)); last_frame = a
            inl_max[a] = max(inl_max.get(a, 0), n)
            if int(b) == int(a) - 1:
                inl_prev[a] = n
            continue
        m = pat_g.search(line)
        if m and last_frame is not None:
            gcorr[last_frame] = int(m.group(1))
    return inl_prev, inl_max, gcorr


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video-dir", type=Path, required=True)
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--off-dir", type=Path, default=None)
    ap.add_argument("--run-log", type=Path, default=None)
    ap.add_argument("--checkpoint", type=Path, default=None)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    out = args.out_dir; out.mkdir(parents=True, exist_ok=True)

    names, add_on = per_frame_add(args.video_dir, args.run_dir)
    idx_of = {n: i for i, n in enumerate(names)}
    add_off = None
    if args.off_dir is not None:
        names_off, add_off = per_frame_add(args.video_dir, args.off_dir)
        assert names_off == names, "on/off frame sets differ"

    from gaussian_global import load_online_checkpoint  # noqa: E402
    ckpt = args.checkpoint or args.run_dir / "final" / "gs" / "checkpoint_global.pt"
    r = load_online_checkpoint(ckpt, device=args.device)
    kf = []
    for v in r.views:
        m = v.mask.numpy(); d = v.depth.numpy() if v.depth is not None else None
        kf.append({"frame_id": str(v.frame_id), "mask_px": int(m.sum()), "valid_depth_px": int((m & (d > 0.05)).sum()) if d is not None else None})
    fb = json.load(open(args.run_dir / "gs_online" / "feedback_log.json"))["records"]
    inl_prev, inl_max, gcorr = parse_run_log(args.run_log) if args.run_log else ({}, {}, {})

    # the checkpoint views are named kf_00000...; the pipeline log names each GS cycle by the tracker frame it consumed
    # ("[GS backend] cycle 0075: ..."): cycle 0 = the 5-keyframe initial batch, cycle j >= 1 = update j (newest view).
    cycles = re.findall(r"\[GS backend\] cycle (\S+):", open(args.run_log, errors="ignore").read()) if args.run_log else []
    rows = []
    n_init = int(fb[0]["views"]) if fb else 5
    resolved = []
    for k, view in enumerate(kf):
        fid = view["frame_id"]
        if fid.startswith("kf_"):
            j = k - n_init + 1
            fid = cycles[j] if (1 <= j < len(cycles)) else fid
        resolved.append(fid)
    for k, view in enumerate(kf):
        fid = resolved[k]
        key = fid.lstrip("0") or "0"
        rec = fb[k - n_init + 1] if k >= n_init and (k - n_init + 1) < len(fb) else None
        prev = [x["mask_px"] for x in kf[max(0, k - 10):k]]
        ratio = view["mask_px"] / np.median(prev) if prev else 1.0
        fi = idx_of.get(fid)
        nxt = resolved[k + 1] if k + 1 < len(kf) else None
        fi_next = idx_of.get(nxt) if nxt else None
        g_on = float(add_on[fi_next] - add_on[fi]) if (fi is not None and fi_next is not None) else np.nan
        g_off = float(add_off[fi_next] - add_off[fi]) if (add_off is not None and fi is not None and fi_next is not None) else np.nan
        rows.append({"k": k, "frame_id": fid, "mask_px": view["mask_px"], "mask_ratio": float(ratio), "valid_depth_px": view["valid_depth_px"],
                     "inliers_prev": inl_prev.get(fid, inl_prev.get(key)), "inliers_max": inl_max.get(fid, inl_max.get(key)), "global_corres": gcorr.get(fid, gcorr.get(key)),
                     "delta_new_trans_mm": float(rec["per_view_trans_mm"][-1]) if rec else np.nan, "delta_new_rot_deg": float(rec["per_view_rot_deg"][-1]) if rec else np.nan,
                     "delta_max_trans_mm": float(rec["max_trans_mm"]) if rec else np.nan, "delta_max_rot_deg": float(rec["max_rot_deg"]) if rec else np.nan,
                     "add_on": float(add_on[fi]) if fi is not None else np.nan, "add_off": float(add_off[fi]) if (add_off is not None and fi is not None) else np.nan,
                     "growth_on": g_on, "growth_off": g_off, "harm": g_on - g_off if not (np.isnan(g_on) or np.isnan(g_off)) else np.nan})
    import csv
    with open(out / "cycles.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

    from scipy.stats import spearmanr
    def col(name):
        return np.array([row[name] if row[name] is not None else np.nan for row in rows], dtype=np.float64)
    upd = np.array([not np.isnan(row["delta_new_trans_mm"]) for row in rows])
    summary = {"run": str(args.run_dir), "frames": len(names), "keyframes": len(kf), "ADD_on_mean": float(add_on.mean()),
               "ADD_off_mean": float(add_off.mean()) if add_off is not None else None}
    def corr(a, b):
        m = ~np.isnan(a) & ~np.isnan(b)
        return {"rho": float(spearmanr(a[m], b[m]).correlation), "n": int(m.sum())} if m.sum() > 5 else None
    dt, dr, mr, ip, ho, hf = col("delta_new_trans_mm"), col("delta_new_rot_deg"), col("mask_ratio"), col("inliers_prev"), col("harm"), col("growth_on")
    summary["spearman"] = {"delta_trans~harm": corr(dt, ho), "delta_rot~harm": corr(dr, ho), "delta_trans~growth_on": corr(dt, hf),
                           "mask_ratio~delta_trans": corr(mr, dt), "inliers_prev~delta_trans": corr(ip, dt), "mask_ratio~harm": corr(mr, ho), "inliers_prev~harm": corr(ip, ho)}
    def contrast(mask, label):
        a, b = ho[mask & ~np.isnan(ho)], ho[~mask & ~np.isnan(ho)]
        return {label: {"n": int(len(a)), "mean_harm_cm": float(a.mean()) if len(a) else None}, "rest": {"n": int(len(b)), "mean_harm_cm": float(b.mean()) if len(b) else None}}
    valid = upd & ~np.isnan(dt)
    q75 = np.nanpercentile(dt[valid], 75) if valid.any() else np.nan
    summary["contrasts"] = {"delta_trans_top25%": contrast(valid & (dt >= q75), f"delta>={q75:.2f}mm"),
                            "delta_trans>5mm": contrast(valid & (dt > 5), ">5mm"), "delta_rot>2deg": contrast(valid & (dr > 2), ">2deg"),
                            "mask_ratio<0.7": contrast(valid & (mr < 0.7), "mask<0.7"),
                            "inliers_prev<100": contrast(valid & (ip < 100), "inl<100") if np.isfinite(ip).any() else None}
    summary["delta_stats_mm"] = {"median": float(np.nanmedian(dt[valid])), "p90": float(np.nanpercentile(dt[valid], 90)), "max": float(np.nanmax(dt[valid]))} if valid.any() else None
    json.dump(summary, open(out / "summary.json", "w"), indent=1)

    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, axes = plt.subplots(3, 1, figsize=(16, 11), sharex=True)
    x = np.arange(len(names)); kfx = [idx_of.get(row["frame_id"]) for row in rows]
    axes[0].plot(x, add_on, label=f"feedback on (mean {add_on.mean():.2f})", lw=1)
    if add_off is not None:
        axes[0].plot(x, add_off, label=f"tracker alone (mean {add_off.mean():.2f})", lw=1, alpha=0.8)
    for xx in kfx:
        if xx is not None: axes[0].axvline(xx, color="k", alpha=0.08)
    axes[0].set_ylabel("ADD cm"); axes[0].legend(); axes[0].set_title(f"{args.run_dir.name}: per-frame ADD, keyframe cycles marked")
    kx = np.array([xx if xx is not None else np.nan for xx in kfx], dtype=np.float64)
    axes[1].bar(kx, dt, width=4, label="delta of newest keyframe (mm)"); axes[1].plot(kx, col("delta_max_trans_mm"), "r.", ms=3, label="max over views (mm)")
    axes[1].set_ylabel("pose delta mm"); axes[1].legend()
    axes[2].plot(kx, mr, "g.-", ms=3, label="mask area / median of previous 10 kf")
    ax2 = axes[2].twinx(); ax2.plot(kx, ip, "m.", ms=3, label="RANSAC inliers vs previous frame"); ax2.set_ylabel("inliers")
    axes[2].set_ylabel("mask ratio"); axes[2].legend(loc="upper left"); ax2.legend(loc="upper right"); axes[2].set_xlabel("frame")
    fig.tight_layout(); fig.savefig(out / "feedback_cycles.png", dpi=90); plt.close(fig)
    print(json.dumps({k: summary[k] for k in ("ADD_on_mean", "ADD_off_mean", "spearman", "contrasts", "delta_stats_mm")}, indent=1)[:3000])


if __name__ == "__main__":
    main()
