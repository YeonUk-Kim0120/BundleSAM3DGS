"""Per-keyframe appearance / geometry scores computed from the DATASET frames (HO3D), joined with the per-cycle feedback
harm of ``analyze_feedback_cycles.py`` (2026-09-15).  Same definitions as logs/gradprobe_20260911_allkf/kf_scores.py:
sift_density = SIFT keypoints inside the SAM2 mask per 1 000 mask pixels (and _interior: mask eroded 5 px);
curvature_deg = median angle between each surface normal and the mean normal of its 15x15 window (eroded mask);
planar_frac = share of those pixels with angle < 3 deg.
usage: kf_scores_dataset.py --video-dir datasets/HO3D_v3/evaluation/AP12 --cycles <analysis dir>/cycles.csv --out-dir <analysis dir>
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.append(str(REPO / "BundleTrack/scripts"))


def normals_from_points(P, ok):
    gx = np.stack([cv2.Sobel(P[..., i], cv2.CV_32F, 1, 0, ksize=5) for i in range(3)], -1)
    gy = np.stack([cv2.Sobel(P[..., i], cv2.CV_32F, 0, 1, ksize=5) for i in range(3)], -1)
    n = np.cross(gx, gy); nn = np.linalg.norm(n, axis=-1, keepdims=True)
    n = n / np.maximum(nn, 1e-9)
    n[n[..., 2] > 0] *= -1
    return n, (nn[..., 0] > 0) & ok


def score(gray, mask, depth, K, sift):
    m8 = mask.astype(np.uint8) * 255
    er = cv2.erode(m8, np.ones((11, 11), np.uint8)) > 0
    n_all = len(sift.detect(gray, m8)); n_in = len(sift.detect(gray, er.astype(np.uint8) * 255))
    px = int(mask.sum()); px_in = int(er.sum())
    valid = mask & (depth > 0.05)
    d = depth.copy(); d[~valid] = 0
    d_s = cv2.medianBlur((d * 1000).astype(np.uint16), 5).astype(np.float32) / 1000.0
    ok = er & (d_s > 0.05)
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    v, u = np.mgrid[:depth.shape[0], :depth.shape[1]]
    P = np.stack([(u - cx) / fx * d_s, (v - cy) / fy * d_s, d_s], -1).astype(np.float32)
    n, okn = normals_from_points(P, ok)
    okn &= cv2.erode(ok.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    w = 15
    nm = cv2.blur(n * okn[..., None].astype(np.float32), (w, w)); cnt = cv2.blur(okn.astype(np.float32), (w, w))
    nm = nm / np.maximum(cnt[..., None], 1e-6); nm /= np.maximum(np.linalg.norm(nm, axis=-1, keepdims=True), 1e-9)
    ang = np.degrees(np.arccos(np.clip((n * nm).sum(-1), -1, 1)))[okn]
    return {"mask_px": px, "depth_valid_frac": float(valid.sum() / max(px, 1)), "sift_n": n_all, "sift_density": 1000.0 * n_all / max(px, 1),
            "sift_density_interior": 1000.0 * n_in / max(px_in, 1), "curvature_deg": float(np.median(ang)) if len(ang) else None,
            "planar_frac": float((ang < 3).mean()) if len(ang) else None}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video-dir", type=Path, required=True)
    ap.add_argument("--cycles", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()
    from data_reader import Ho3dReader  # noqa: E402
    reader = Ho3dReader(str(args.video_dir))
    ids = {os.path.basename(f).split(".")[0]: i for i, f in enumerate(reader.color_files)}
    rows = list(csv.DictReader(open(args.cycles)))
    sift = cv2.SIFT_create()
    out_rows = []
    for row in rows:
        fid = row["frame_id"]
        if fid not in ids:
            continue
        i = ids[fid]
        img = cv2.imread(reader.color_files[i]); gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        mask = reader.get_mask(i) > 0
        depth = reader.get_depth(i).astype(np.float32)
        s = score(gray, mask, depth, np.asarray(reader.K, dtype=np.float64), sift)
        s.update({"frame_id": fid, "harm": float(row["harm"]) if row["harm"] not in ("", "nan") else np.nan,
                  "delta_new_trans_mm": float(row["delta_new_trans_mm"]) if row["delta_new_trans_mm"] not in ("", "nan") else np.nan,
                  "growth_on": float(row["growth_on"]) if row["growth_on"] not in ("", "nan") else np.nan})
        out_rows.append(s)
    json.dump(out_rows, open(args.out_dir / "kf_scores.json", "w"), indent=1)
    from scipy.stats import spearmanr
    def col(k):
        return np.array([r[k] if r.get(k) is not None else np.nan for r in out_rows], dtype=np.float64)
    harm, sd, sdi, cur, pf, dt = col("harm"), col("sift_density"), col("sift_density_interior"), col("curvature_deg"), col("planar_frac"), col("delta_new_trans_mm")
    def corr(a, b):
        m = ~np.isnan(a) & ~np.isnan(b)
        return {"rho": float(spearmanr(a[m], b[m]).correlation), "n": int(m.sum())} if m.sum() > 5 else None
    def contrast(mask, label):
        a, b = harm[mask & ~np.isnan(harm)], harm[~mask & ~np.isnan(harm)]
        return {label: {"n": int(len(a)), "mean_harm": float(a.mean()) if len(a) else None}, "rest": {"n": int(len(b)), "mean_harm": float(b.mean()) if len(b) else None}}
    ok = ~np.isnan(harm)
    q_sd, q_cur, q_pf = np.nanpercentile(sd[ok], 25), np.nanpercentile(cur[ok], 75), np.nanpercentile(pf[ok], 75)
    summary = {"n": int(ok.sum()), "spearman": {"sift_density~harm": corr(sd, harm), "sift_interior~harm": corr(sdi, harm), "curvature~harm": corr(cur, harm),
                                                 "planar_frac~harm": corr(pf, harm), "sift_density~delta": corr(sd, dt), "curvature~delta": corr(cur, dt)},
               "contrasts": {"sift_bottom25%": contrast(ok & (sd <= q_sd), f"sift<={q_sd:.2f}"), "curvature_top25%": contrast(ok & (cur >= q_cur), f"curv>={q_cur:.1f}deg"),
                             "planar_top25%": contrast(ok & (pf >= q_pf), f"planar>={q_pf:.2f}")},
               "score_stats": {"sift_density_median": float(np.nanmedian(sd)), "curvature_median": float(np.nanmedian(cur)), "planar_median": float(np.nanmedian(pf))}}
    json.dump(summary, open(args.out_dir / "kf_scores_summary.json", "w"), indent=1)
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, (x, lab) in zip(axes, [(sd, "SIFT density (kp / 1000 mask px)"), (cur, "curvature (deg)"), (pf, "planar fraction")]):
        ax.scatter(x[ok], harm[ok], s=10); ax.axhline(0, color="k", lw=0.5); ax.set_xlabel(lab); ax.set_ylabel("harm per cycle (cm)")
    fig.suptitle(f"{args.video_dir.name}: per-cycle harm vs keyframe texture / geometry"); fig.tight_layout(); fig.savefig(args.out_dir / "harm_vs_scores.png", dpi=90)
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
