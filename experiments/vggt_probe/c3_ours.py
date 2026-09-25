"""C3 (brief section 3): consumption-time keyframe poses of OUR system (outputs/fulleval_20260912) vs GT, per sequence.
Separate rows only (keyframe sets differ from BSDF).  Writes logs/exp_vggt_probe1/table_C3_ours.md"""
import sys, glob, os
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import probe_lib as L
from exp_hybrid_realistic_replay import load_snapshots
OURS = Path("/home/kist/Desktop/BundleSAM3DGS/outputs/fulleval_20260912")
rows = []
for ds, seqs in [("ho3d", L.__dict__.get("HO3D", None) or ["AP10","AP11","AP12","AP13","AP14","MPM10","MPM11","MPM12","MPM13","MPM14","SB11","SB13","SM1"]), ("ycb", ["bleach0","bleach_hard_00_03_chaitanya","cracker_box_reorient","cracker_box_yalehand0","mustard0","mustard_easy_00_02","sugar_box1","sugar_box_yalehand0","tomato_soup_can_yalehand0"])]:
    for sq in seqs:
        run = OURS / ds / sq
        if not (run / "ob_in_cam").exists(): rows.append((ds, sq, "ob_in_cam 없음", "", "", "")); continue
        snaps = load_snapshots(run)
        if not snaps: rows.append((ds, sq, "스냅샷 없음", "", "", "")); continue
        video = L.REPO / ("datasets/HO3D_v3/evaluation" if ds == "ho3d" else "datasets/YCBInEOAT") / sq
        gt = L.GroundTruth(ds, video, run); c1 = {}; c2 = dict(snaps[-1][1])
        for f, kf in snaps:
            for k, P in kf.items(): c1.setdefault(k, P)
        e1 = [L.pose_err(c1[k], gt.gt_c2w(k)) for k in c1 if gt.gt_c2w(k) is not None]; e2 = [L.pose_err(c2[k], gt.gt_c2w(k)) for k in c2 if gt.gt_c2w(k) is not None]
        rows.append((ds, sq, len(c1), f"{np.median([e[0] for e in e1]):.2f}/{np.percentile([e[0] for e in e1],90):.2f}", f"{np.median([e[1] for e in e1]):.1f}", f"{np.median([e[0] for e in e2]):.2f} / {np.median([e[1] for e in e2]):.1f}"))
        print(rows[-1], flush=True)
out = ["| ds | seq | 우리 KF 수 | C3 소비 시점 rot 중/p90 ° | C3 trans 중 mm | 우리 최종 포즈 rot ° / trans mm |", "|---|---|---|---|---|---|"] + [f"| {r[0]} | {r[1]} | {r[2]} | {r[3]} | {r[4]} | {r[5]} |" for r in rows]
open("/home/kist/Desktop/BundleSAM3DGS/logs/exp_vggt_probe1/table_C3_ours.md", "w").write("\n".join(out) + "\n"); print("\n".join(out))
