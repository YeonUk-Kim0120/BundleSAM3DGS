"""A3: flip analysis. New-KF E1 (online, from kf.csv) and pair errors (pairs.csv) > 90° / > 150° per sequence; error-rotation axis of
flipped new KFs vs GT mesh PCA axes (mesh brought into the tracker object frame with GroundTruth.align); 10° histograms; pair-type composition."""
import glob, sys
import numpy as np, pandas as pd, trimesh
from pathlib import Path
from p1b_common import *
sys.path.insert(0, "/home/kist/Desktop/BundleSAM3DGS/experiments")
from eval_mesh_cd import load_gt_mesh


def axis_of(R):
    ang = np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))
    if ang < 1e-6: return None
    ax = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]); n = np.linalg.norm(ax)
    if n < 1e-9:   # 180°: eigenvector of eigenvalue 1
        w, v = np.linalg.eigh((R + np.eye(3)) / 2); return v[:, np.argmax(w)]
    return ax / n


def f(x, p=1): return "–" if not np.isfinite(x) else f"{x:.{p}f}"


_ALIGN = {}
def align_of(ds, sq, S):
    """GroundTruth.align (model frame -> tracker object frame) = gt_c2w(frame0) @ g0, g0 = raw GT ob_in_cam of the first frame."""
    if (ds, sq) in _ALIGN: return _ALIGN[(ds, sq)]
    import glob, os
    video = L.REPO / ("datasets/HO3D_v3/evaluation" if ds == "ho3d" else "datasets/YCBInEOAT") / sq
    if ds == "ho3d":
        sys.path.append(str(L.REPO / "BundleTrack/scripts")); from data_reader import Ho3dReader
        g0 = Ho3dReader(str(video)).get_gt_pose(0)
    else:
        g0 = np.loadtxt(sorted(glob.glob(str(video / "annotated_poses/*.txt")))[0]).reshape(4, 4)
    u, _, vt = np.linalg.svd(g0[:3, :3]); R = u @ vt
    if np.linalg.det(R) < 0: u[:, -1] *= -1; R = u @ vt
    g0 = g0.copy(); g0[:3, :3] = R
    A = S.GT[S.kf_ids[0]] @ g0; _ALIGN[(ds, sq)] = A; return A


pairs = pd.read_csv(P1B / "A" / "all_pairs.csv"); doc = ["# A3 표 FL — 뒤집힘 분석", ""]
for model, inp in [("M-V", "I1"), ("M-O", "I1"), ("M-V", "I0")]:
    doc += [f"## {model}/{inp}", "", "| ds | seq | 설정 | 새 KF n | E1 >90° | E1 >150° | 쌍 n | 쌍 >90° | 쌍 >150° | 뒤집힌 쌍 중 far 포함 비율 | 뒤집힌 새 KF 축 vs 메시 PCA 최소각 중앙값 (1축/2축/3축 비율) |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for ds, seqs in [("ho3d", HO3D), ("ycb", YCB)]:
        for sq in seqs:
            for bt in ["ref1-8", "ref5-8"]:
                d = P1 / model / inp / bt / ds / sq
                if not d.exists(): continue
                S = load_seq(ds, sq); kf = read_kf(d)
                try:
                    mesh = load_gt_mesh(ds, sq); V = np.asarray(mesh.vertices); V = V - V.mean(0); _, _, vt = np.linalg.svd(V, full_matrices=False); axes_model = vt[:3]
                    # tracker object frame = align @ model frame (GroundTruth: gt_c2w = align @ inv(g)) -> axes rotate by align R
                    A = align_of(ds, sq, S)[:3, :3]; axes = (A @ axes_model.T).T
                except Exception as e:
                    axes = None; print("mesh fail", ds, sq, e)
                e1 = []; flips = []
                for f_, rows in kf.items():
                    for r in rows:
                        if r["role"] != "new" or r["has_gt"] != "1": continue
                        rot = float(r["rot_online"]); e1.append(rot)
                        if rot > 90 and axes is not None and S.GT.get(f_) is not None:
                            Rest = np.array([float(r[f"c2w_online_{j}"]) for j in range(16)]).reshape(4, 4)[:3, :3]; Rerr = Rest @ S.GT[f_][:3, :3].T; ax = axis_of(Rerr)
                            if ax is not None:
                                angs = [np.degrees(np.arccos(min(1, abs(ax @ a)))) for a in axes]; flips.append((min(angs), int(np.argmin(angs))))
                e1 = np.array(e1); pp = pairs[(pairs.model == model) & (pairs.input == inp) & (pairs.batch_mode == bt) & (pairs.ds == ds) & (pairs.seq == sq)]
                fl = pp[pp.vggt_rot > 90]; far_share = np.mean(fl.pair_type.isin(["new-far", "ref-ref"])) if len(fl) else np.nan
                axtxt = "–"
                if flips:
                    m = np.median([x[0] for x in flips]); cnt = np.bincount([x[1] for x in flips], minlength=3) / len(flips)
                    axtxt = f"{f(m)}° ({f(100*cnt[0],0)}/{f(100*cnt[1],0)}/{f(100*cnt[2],0)} %)"
                doc.append(f"| {ds} | {sq} | {bt} | {len(e1)} | {f(100*np.mean(e1>90),1)} % | {f(100*np.mean(e1>150),1)} % | {len(pp)} | {f(100*np.mean(pp.vggt_rot>90),1)} % | {f(100*np.mean(pp.vggt_rot>150),1)} % | {f(100*far_share,0)} % | {axtxt} |")
    doc.append("")
# histograms (10° bins) of new-KF E1 online for M-V/I1/ref1-8 per sequence
doc += ["## 새 KF E1(online) 히스토그램, 10° 간격, M-V/I1/ref1-8 (HO3D)", "", "| seq | " + " | ".join(f"{a}–{a+10}" for a in range(0, 180, 10)) + " |", "|---|" + "---|" * 18]
for sq in HO3D:
    d = P1 / "M-V/I1/ref1-8/ho3d" / sq; kf = read_kf(d); e = [float(r["rot_online"]) for rows in kf.values() for r in rows if r["role"] == "new" and r["has_gt"] == "1"]
    h, _ = np.histogram(e, bins=np.arange(0, 181, 10)); doc.append(f"| {sq} | " + " | ".join(str(x) for x in h) + " |")
open(LOG / "table_A3_FL.md", "w").write("\n".join(doc) + "\n"); print("A3 done")
