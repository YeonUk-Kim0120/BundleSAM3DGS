"""EXP_BATCH_20260928: input pose files for the joint-refit rounds (arm N in the bundlesdf container, GS arms in the gsplat container).
For a noop run: keyframe order (last cycle's nerf_frames), consumption poses (first poses_before_gs.txt containing the keyframe),
GT poses (GroundTruth first-frame gauge, orthonormalised) and GT-mesh points (tracker frame) for evaluation.
Writes <out>/<ds>_<seq>.json.  Unit-test inputs: --unit-test writes mustard0 K=20 with GT poses and with one keyframe GT+3 deg."""
from __future__ import annotations
import argparse, json, math, sys
from pathlib import Path
import numpy as np
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "experiments")); sys.path.append(str(REPO / "BundleTrack/scripts"))
from exp_feedback_gradient_probe_gtmap import GroundTruth, gt_mesh_in_tracker_frame, load_cycles  # noqa: E402

NOOP = REPO / "outputs/exp_batch_20260927/noop"


def orth(T):
    T = np.asarray(T, dtype=np.float64).copy(); u, _, vt = np.linalg.svd(T[:3, :3]); R = u @ vt
    if np.linalg.det(R) < 0: u[:, -1] *= -1; R = u @ vt
    T[:3, :3] = R; return T


def rodrigues(axis, ang):
    a = np.asarray(axis, float) / np.linalg.norm(axis); K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + math.sin(ang) * K + (1 - math.cos(ang)) * K @ K


def build(ds, seq):
    run = NOOP / ds / seq; video = REPO / ("datasets/HO3D_v3/evaluation" if ds == "ho3d" else "datasets/YCBInEOAT") / seq
    gt = GroundTruth(ds, video, run); cycles = load_cycles(run)
    cons = {}
    for cyc in cycles:
        for fid, P in zip(cyc["ids"], cyc["before"]):
            cons.setdefault(fid, orth(P))
    ids = [f for f in cycles[-1]["ids"] if f in cons]
    cyc_len = [len(c["ids"]) for c in cycles]
    gtp = {f: (orth(gt.gt_c2w(f)).tolist() if gt.gt_c2w(f) is not None else None) for f in ids}
    mesh = gt_mesh_in_tracker_frame(ds, seq, gt, n_samples=20000)
    return dict(dataset=ds, seq=seq, run_dir=str(run), kf_ids=ids, consume=[cons[f].tolist() for f in ids], gt=[gtp[f] for f in ids],
                cycle_lengths=cyc_len, cycle_dirs=[c["dir"] for c in cycles], mesh_points=mesh.astype(np.float32).tolist(), mesh_center=mesh.mean(0).tolist())


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--out", type=Path, default=REPO / "logs/exp_batch_20260928/inputs"); ap.add_argument("--unit-test", action="store_true"); a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    if a.unit_test:
        d = build("ycb", "mustard0"); K = 20; ids = d["kf_ids"][:K]; G = [np.array(g) for g in d["gt"][:K]]
        assert all(g is not None for g in d["gt"][:K])
        rng = np.random.default_rng(3); ax = rng.normal(size=3); R = rodrigues(ax, math.radians(3.0)); c = np.array(d["mesh_center"])
        P = [g.copy() for g in G]; j = 10; Q = np.eye(4); Q[:3, :3] = R @ G[j][:3, :3]; Q[:3, 3] = c + R @ (G[j][:3, 3] - c); P[j] = Q
        for name, poses in (("gt", G), ("gt_pert10", P)):
            json.dump({**{k: v for k, v in d.items() if k not in ("kf_ids", "consume", "gt")}, "kf_ids": ids, "consume": [p.tolist() for p in poses],
                       "gt": [g.tolist() for g in G], "note": f"unit test mustard0 K=20, input = {name} (keyframe index {j} rotated 3 deg about the GT-mesh centre)" if name != "gt" else "unit test mustard0 K=20, input = GT"},
                      open(a.out / f"unit_mustard0_{name}.json", "w"))
        print("unit inputs written")
    else:
        for ds, seq in (("ho3d", "AP12"), ("ycb", "mustard0"), ("ho3d", "SM1"), ("ho3d", "MPM12")):
            d = build(ds, seq); json.dump(d, open(a.out / f"{ds}_{seq}.json", "w")); print(seq, len(d["kf_ids"]), "keyframes,", len(d["cycle_lengths"]), "cycles; first cycle", d["cycle_lengths"][0], flush=True)
