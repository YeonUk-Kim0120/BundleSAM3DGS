"""Why does the per-frame ADD rise and fall, and why does the feedback-on curve follow the tracker-alone curve?
(2026-09-16, user request).  Uses per_frame_add.json from plot_add_over_frames.py plus per-frame signals:
  speed    : GT rotation (deg/frame) and translation (mm/frame) between consecutive frames
  visible  : SAM2 mask area / area of the GT model's projected convex hull (occlusion + truncation proxy, 0..1)
  inliers  : tracker RANSAC inliers vs the previous frame (run log of the feedback-on run)
  keyframe : frames the tracker added as keyframes
Outputs (--out-dir): episodes_<seq>.png (ADD on/off, speed, visibility, inliers; rises red / falls green shaded),
episodes_summary.json and episodes_summary.md (co-movement statistics, correlations of the error increments with the
signals, and what the signals look like inside rising vs falling episodes).
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial import ConvexHull
from scipy.stats import pearsonr, spearmanr

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "experiments")); sys.path.append(str(REPO / "BundleTrack/scripts"))
HO = ["AP10", "AP11", "AP12", "AP13", "AP14", "MPM10", "MPM11", "MPM12", "MPM13", "MPM14", "SB11", "SB13", "SM1"]
YC = ["bleach0", "bleach_hard_00_03_chaitanya", "cracker_box_reorient", "cracker_box_yalehand0", "mustard0",
      "mustard_easy_00_02", "sugar_box1", "sugar_box_yalehand0", "tomato_soup_can_yalehand0"]


def signals(dataset: str, seq: str, ids: np.ndarray, run_log: Path):
    import trimesh
    vd = REPO / ("datasets/HO3D_v3/evaluation" if dataset == "ho3d" else "datasets/YCBInEOAT") / seq
    if dataset == "ho3d":
        from data_reader import Ho3dReader
        reader = Ho3dReader(str(vd)); gt = [reader.get_gt_pose(i) for i in range(len(reader.color_files))]
        verts = np.asarray(reader.get_gt_mesh().vertices, dtype=np.float64); K = np.asarray(reader.K, dtype=np.float64)
        names = [os.path.basename(f).split(".")[0] for f in reader.color_files]
        get_mask = lambda i: reader.get_mask(i) > 0
    else:
        from data_reader import YcbineoatReader
        from eval_add_ycbineoat import VIDEO_TO_OBJECT
        reader = YcbineoatReader(str(vd), mask_dir="masks_sam2")
        gt = [np.loadtxt(f).reshape(4, 4) for f in sorted(glob.glob(str(vd / "annotated_poses" / "*")))]
        verts = np.asarray(trimesh.load(str(REPO / "datasets/YCB_Video_Models/models" / VIDEO_TO_OBJECT[seq] / "textured_simple.obj")).vertices, dtype=np.float64)
        K = np.asarray(reader.K, dtype=np.float64); names = [os.path.basename(f).split(".")[0] for f in reader.color_files]
        get_mask = lambda i: np.asarray(reader.get_mask(i)) > 0
    sub = verts[np.random.default_rng(0).choice(len(verts), min(len(verts), 3000), replace=False)]
    rot, trans, vis, marea = [], [], [], []
    prev = None
    for i in ids:
        g = gt[i]
        if prev is None or prev is None:
            rot.append(0.0); trans.append(0.0)
        else:
            R = g[:3, :3] @ prev[:3, :3].T; rot.append(float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))))); trans.append(float(np.linalg.norm(g[:3, 3] - prev[:3, 3]) * 1000))
        prev = g
        p = sub @ g[:3, :3].T + g[:3, 3]; z = p[:, 2].clip(1e-6, None)
        u = K[0, 0] * p[:, 0] / z + K[0, 2]; v = K[1, 1] * p[:, 1] / z + K[1, 2]
        m = get_mask(i); h, w = m.shape
        try:
            hull_area = ConvexHull(np.c_[u, v]).volume
        except Exception:
            hull_area = np.nan
        marea.append(int(m.sum())); vis.append(float(m.sum() / hull_area) if hull_area and hull_area > 0 else np.nan)
    inl, kfs = {}, set()
    if run_log and run_log.exists():
        txt = open(run_log, errors="ignore").read()
        for a, b, n in re.findall(r"ransac makes match betwee frame (\d+) (\d+) #inliers=(\d+)", txt):
            if a not in inl: inl[a] = int(n)  # first pair reported for a new frame = match with the previous frame
        kfs = set(re.findall(r"Added frame (\S+) as keyframe", txt))
    inliers = np.array([inl.get(names[i], np.nan) for i in ids], dtype=np.float64)
    kf_flags = np.array([names[i] in kfs for i in ids])
    return dict(rot=np.array(rot), trans=np.array(trans), vis=np.array(vis), mask_px=np.array(marea), inliers=inliers, keyframe=kf_flags)


def smooth(y, w=15):
    k = np.ones(w) / w; return np.convolve(np.nan_to_num(y, nan=np.nanmean(y)), k, mode="same")


def episodes(err, win=30, thr=0.3):
    """Rising / falling segments: smoothed error change over `win` frames beyond ±thr cm."""
    s = smooth(err); d = np.zeros_like(s); d[win:] = s[win:] - s[:-win]
    rises = d > thr; falls = d < -thr
    return rises, falls, s


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--curves", type=Path, default=Path("logs/exp_fusion_20260915/add_over_frames/per_frame_add.json"))
    ap.add_argument("--log-root", type=Path, default=Path("logs/exp_fusion_20260915"))
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()
    out = args.out_dir; out.mkdir(parents=True, exist_ok=True)
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    curves = json.load(open(args.curves)); summary = {}; pooled = {"rise": [], "fall": [], "flat": []}; pooled_inc = []
    for s, c in curves.items():
        ds = c["dataset"]; ids = np.array(c["ids"]); on = np.array(c["add_on"]); off = np.array(c["add_off"]) if c["add_off"] else None
        sig = signals(ds, s, ids, args.log_root / f"run_adopt_{ds}_{s}.log")
        rec = {"dataset": ds, "frames": int(c["frames"]), "ADD_on": float(on.mean()), "ADD_off": float(off.mean()) if off is not None else None}
        if off is not None:
            d_on, d_off = np.diff(on), np.diff(off)
            rec["comovement"] = {"pearson_curves": float(pearsonr(on, off)[0]), "pearson_increments": float(pearsonr(d_on, d_off)[0]),
                                 "spearman_curves": float(spearmanr(on, off).correlation), "level_gap_cm": float((on - off).mean()),
                                 "gap_std_cm": float((on - off).std())}
        inc = np.diff(on); n = len(inc)
        X = {"rot_deg_per_frame": sig["rot"][1:], "trans_mm_per_frame": sig["trans"][1:], "visible_ratio": sig["vis"][1:], "inliers_prev": sig["inliers"][1:]}
        rec["spearman_increment_vs"] = {}
        for k, x in X.items():
            m = ~np.isnan(x) & ~np.isnan(inc)
            rec["spearman_increment_vs"][k] = float(spearmanr(x[m], inc[m]).correlation) if m.sum() > 10 else None
        rises, falls, sm = episodes(on)
        if off is not None:
            gap = on - off; dgap = np.diff(gap)
            # jaggedness of the gap and how much of it is anchored to the last keyframe cycle
            kf_idx = np.nonzero(sig["keyframe"])[0]
            last_kf = np.searchsorted(kf_idx, np.arange(len(on)), side="right") - 1
            gap_kf = np.array([gap[kf_idx[j]] if j >= 0 else np.nan for j in last_kf])
            mm = ~np.isnan(gap_kf)
            rec["gap"] = {"mean_cm": float(gap.mean()), "std_cm": float(gap.std()), "increment_std_cm": float(dgap.std()),
                          "increment_std_on_cm": float(np.diff(on).std()), "increment_std_off_cm": float(np.diff(off).std()),
                          "r_gap_vs_gap_at_last_keyframe": float(pearsonr(gap[mm], gap_kf[mm])[0]) if mm.sum() > 10 else None,
                          "mean_abs_gap_change_between_keyframes_cm": float(np.nanmean(np.abs(gap - gap_kf)[mm])) if mm.any() else None,
                          "dgap_in_rise_cm_per_frame": float(dgap[rises[1:]].mean()) if rises[1:].any() else None,
                          "dgap_in_fall_cm_per_frame": float(dgap[falls[1:]].mean()) if falls[1:].any() else None,
                          "dgap_in_flat_cm_per_frame": float(dgap[(~rises & ~falls)[1:]].mean()) if (~rises & ~falls)[1:].any() else None,
                          "gap_growth_in_rise_total_cm": float(dgap[rises[1:]].sum()), "gap_growth_in_fall_total_cm": float(dgap[falls[1:]].sum()),
                          "gap_growth_in_flat_total_cm": float(dgap[(~rises & ~falls)[1:]].sum())}
        def stats(mask):
            mask = mask[1:] if len(mask) == len(on) else mask
            return {k: float(np.nanmean(x[mask])) if mask.any() else None for k, x in X.items()} | {"frames": int(mask.sum()), "keyframe_rate": float(sig["keyframe"][1:][mask].mean()) if mask.any() else None}
        flat = ~rises & ~falls
        rec["episodes"] = {"rise": stats(rises), "fall": stats(falls), "flat": stats(flat), "whole": stats(np.ones(len(on), bool))}
        for key, mask in (("rise", rises), ("fall", falls), ("flat", flat)):
            mm = mask[1:]
            if mm.any(): pooled[key].append({k: float(np.nanmean(x[mm])) for k, x in X.items()})
        pooled_inc.append({k: rec["spearman_increment_vs"][k] for k in X})
        summary[s] = rec
        fig, axes = plt.subplots(4, 1, figsize=(15, 10), sharex=True)
        axes[0].plot(ids, on, lw=1, label=f"feedback on ({on.mean():.2f})")
        if off is not None: axes[0].plot(ids, off, lw=1, alpha=0.8, label=f"tracker alone ({off.mean():.2f})")
        axes[0].plot(ids, sm, "k--", lw=0.8, alpha=0.5, label="smoothed (on)")
        for m, col in ((rises, "red"), (falls, "green")):
            axes[0].fill_between(ids, 0, np.nanmax(on), where=m, color=col, alpha=0.12)
        kfx = ids[sig["keyframe"]]; axes[0].vlines(kfx, 0, np.nanmax(on) * 0.05, color="k", alpha=0.4)
        axes[0].set_ylabel("ADD cm"); axes[0].legend(fontsize=8); axes[0].set_title(f"{ds}/{s}: red = rising episodes, green = falling; ticks = keyframes")
        axes[1].plot(ids, smooth(sig["rot"], 5), label="GT rotation deg/frame"); ax1b = axes[1].twinx(); ax1b.plot(ids, smooth(sig["trans"], 5), color="orange", label="GT translation mm/frame"); axes[1].set_ylabel("deg/frame"); ax1b.set_ylabel("mm/frame"); axes[1].legend(loc="upper left", fontsize=8); ax1b.legend(loc="upper right", fontsize=8)
        axes[2].plot(ids, sig["vis"], label="mask area / projected GT hull (visibility)"); axes[2].set_ylim(0, 1.3); axes[2].set_ylabel("visible"); axes[2].legend(fontsize=8)
        axes[3].plot(ids, sig["inliers"], ".", ms=2, label="RANSAC inliers vs previous frame"); axes[3].set_ylabel("inliers"); axes[3].legend(fontsize=8); axes[3].set_xlabel("frame")
        fig.tight_layout(); fig.savefig(out / f"episodes_{s}.png", dpi=80); plt.close(fig)
        cm = rec.get("comovement", {})
        print(f"{ds}/{s}: r(on,off)={cm.get('pearson_curves', float('nan')):.2f} r(inc)={cm.get('pearson_increments', float('nan')):.2f} | rho(inc~rot)={rec['spearman_increment_vs']['rot_deg_per_frame']} rho(inc~vis)={rec['spearman_increment_vs']['visible_ratio']} rho(inc~inl)={rec['spearman_increment_vs']['inliers_prev']} | rise frames {rec['episodes']['rise']['frames']} fall {rec['episodes']['fall']['frames']}")
    # pooled
    keys = ["rot_deg_per_frame", "trans_mm_per_frame", "visible_ratio", "inliers_prev"]
    pooled_tab = {key: {k: float(np.nanmean([p[k] for p in pooled[key] if p[k] is not None])) for k in keys} for key in ("rise", "fall", "flat")}
    pooled_rho = {k: float(np.nanmean([p[k] for p in pooled_inc if p[k] is not None])) for k in keys}
    gaps = [r["gap"] for r in summary.values() if isinstance(r, dict) and "gap" in r]
    pooled_gap = {k: float(np.nanmean([g[k] for g in gaps if g.get(k) is not None])) for k in gaps[0]} if gaps else {}
    summary["_pooled"] = {"episode_means": pooled_tab, "mean_spearman_increment_vs": pooled_rho, "gap_pooled": pooled_gap,
                          "comovement_mean": {k: float(np.nanmean([r["comovement"][k] for r in summary.values() if isinstance(r, dict) and "comovement" in r])) for k in ("pearson_curves", "pearson_increments", "level_gap_cm")}}
    json.dump(summary, open(out / "episodes_summary.json", "w"), indent=1)
    with open(out / "episodes_summary.md", "w") as f:
        f.write("| seq | ADD on / off | r(on,off) curves | r increments | gap cm | rho(Δerr~rot) | rho(Δerr~trans) | rho(Δerr~visible) | rho(Δerr~inliers) | rise: rot / vis / inl | fall: rot / vis / inl |\n|---|---|---|---|---|---|---|---|---|---|---|\n")
        for s, r in summary.items():
            if s.startswith("_"): continue
            cm = r.get("comovement", {}); sp = r["spearman_increment_vs"]; ri, fa = r["episodes"]["rise"], r["episodes"]["fall"]
            g = lambda v, spec=".2f": "–" if v is None or (isinstance(v, float) and np.isnan(v)) else format(v, spec)
            f.write(f"| {s} | {r['ADD_on']:.2f} / {g(r['ADD_off'])} | {g(cm.get('pearson_curves'))} | {g(cm.get('pearson_increments'))} | {g(cm.get('level_gap_cm'))} | {g(sp['rot_deg_per_frame'])} | {g(sp['trans_mm_per_frame'])} | {g(sp['visible_ratio'])} | {g(sp['inliers_prev'])} | {g(ri['rot_deg_per_frame'])} / {g(ri['visible_ratio'])} / {g(ri['inliers_prev'], '.0f')} | {g(fa['rot_deg_per_frame'])} / {g(fa['visible_ratio'])} / {g(fa['inliers_prev'], '.0f')} |\n")
        f.write("\n| seq | gap mean / std cm | Δgap std vs Δon std vs Δoff std | r(gap, gap at last keyframe) | gap growth in rise / flat / fall (cm) |\n|---|---|---|---|---|\n")
        for s, r in summary.items():
            if s.startswith("_") or "gap" not in r: continue
            g = r["gap"]; f.write(f"| {s} | {g['mean_cm']:+.2f} / {g['std_cm']:.2f} | {g['increment_std_cm']:.3f} / {g['increment_std_on_cm']:.3f} / {g['increment_std_off_cm']:.3f} | {g['r_gap_vs_gap_at_last_keyframe'] if g['r_gap_vs_gap_at_last_keyframe'] is None else round(g['r_gap_vs_gap_at_last_keyframe'],2)} | {g['gap_growth_in_rise_total_cm']:+.2f} / {g['gap_growth_in_flat_total_cm']:+.2f} / {g['gap_growth_in_fall_total_cm']:+.2f} |\n")
        f.write(f"\npooled gap stats: {json.dumps(summary['_pooled']['gap_pooled'])}\n")
        f.write(f"\npooled episode means: {json.dumps(pooled_tab)}\npooled mean spearman(Δerr, signal): {json.dumps(pooled_rho)}\ncomovement mean: {json.dumps(summary['_pooled']['comovement_mean'])}\n")
    print(json.dumps(summary["_pooled"], indent=1))


if __name__ == "__main__":
    main()
