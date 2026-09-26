"""Probe 1b shared helpers (brief section 2).  Imports Probe 1 code read-only; nothing in probe_lib/run_probe is modified.
`python p1b_common.py --build-cache` builds per-sequence pose caches (C1, C2, GT, per-frame snapshots) once."""
from __future__ import annotations
import argparse, csv, glob, json, os, pickle, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import probe_lib as L

P1 = Path("/home/kist/Desktop/BundleSAM3DGS/outputs/exp_vggt_probe1")
P1B = Path("/home/kist/Desktop/BundleSAM3DGS/outputs/exp_vggt_probe1b")
LOG = Path("/home/kist/Desktop/BundleSAM3DGS/logs/exp_vggt_probe1b")
CACHE = P1B / "cache"
HO3D = ["AP10", "AP11", "AP12", "AP13", "AP14", "MPM10", "MPM11", "MPM12", "MPM13", "MPM14", "SB11", "SB13", "SM1"]
YCB = ["bleach0", "bleach_hard_00_03_chaitanya", "cracker_box_reorient", "cracker_box_yalehand0", "mustard0", "mustard_easy_00_02", "sugar_box1", "sugar_box_yalehand0", "tomato_soup_can_yalehand0"]
SWING_BINS = [0, 15, 30, 60, 90, 135, 180.01]; ROLL_BINS = [0, 15, 45, 90, 180.01]
SWING_LABELS = ["[0,15)", "[15,30)", "[30,60)", "[60,90)", "[90,135)", "[135,180]"]; ROLL_LABELS = ["[0,15)", "[15,45)", "[45,90)", "[90,180]"]


# ---------------------------------------------------------------- sequence cache
class SeqCache:
    """C1/C2/GT c2w per keyframe + per-frame snapshots (for R-online), K, kf order."""
    def __init__(self, d): self.__dict__.update(d)
    def snapshot_at(self, kf): return self.snaps[self.frame_index[kf]][1]


def build_cache(ds, sq):
    seq = L.Sequence(ds, sq)
    d = dict(dataset=ds, seq=sq, K=seq.K, kf_ids=seq.kf_ids, C1=seq.C1, C2=seq.C2, GT={k: seq.gt_c2w(k) for k in seq.kf_ids}, snaps=seq.snaps, frame_index=seq.frame_index, run=str(seq.run))
    CACHE.mkdir(parents=True, exist_ok=True); pickle.dump(d, open(CACHE / f"{ds}_{sq}.pkl", "wb")); return d


def load_seq(ds, sq) -> SeqCache:
    p = CACHE / f"{ds}_{sq}.pkl"
    return SeqCache(pickle.load(open(p, "rb")) if p.exists() else build_cache(ds, sq))


# ---------------------------------------------------------------- geometry
def blockdiag(R):
    B = np.eye(4); B[:3, :3] = R; return B


def c2w_batch(b):
    """Section 2: c2w_b,i = inv(w2c_raw_i), translation × s, then · blockdiag(R_virtual_i, 1)."""
    s = b["s"] if (b["s"] is not None and np.isfinite(b["s"])) else 1.0
    out = []
    for w, R in zip(b["w2c_raw"], b["R_virtual"]):
        P = np.linalg.inv(np.array(w, dtype=np.float64)); P[:3, 3] *= s; out.append(P @ blockdiag(np.array(R, dtype=np.float64)))
    return out


def rel(c2w_i, c2w_j): return np.linalg.inv(c2w_i) @ c2w_j


def rot_deg(a, b): return L.rot_deg(a, b)


def tdir_deg(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(np.degrees(np.arccos(np.clip(a @ b / (na * nb), -1, 1)))) if na > 1e-9 and nb > 1e-9 else np.nan


def swing_twist(R):
    """swing = angle(R z, z); twist = |2 atan2(|q_z|, q_w)| of R about z, folded to [0,180]."""
    z = np.array([0, 0, 1.0]); swing = float(np.degrees(np.arccos(np.clip((R @ z) @ z, -1, 1))))
    w = np.sqrt(max(0.0, 1 + np.trace(R))) / 2; qz = (R[1, 0] - R[0, 1]) / (4 * w) if w > 1e-9 else 1.0
    if w < 1e-9:   # 180° rotation: q_w = 0, take |q_z| from the diagonal
        qz = np.sqrt(max(0.0, (R[2, 2] + 1) / 2))
    tw = float(np.degrees(2 * np.arctan2(abs(qz), abs(w)))); tw = tw if tw <= 180 else 360 - tw
    return swing, tw


def bin_index(x, edges):
    for i in range(len(edges) - 1):
        if edges[i] <= x < edges[i + 1]: return i
    return len(edges) - 2


def roles_for_batch(b, batch_mode, k):
    """ref = [first, near × n_near, far × rest] (Probe 1 select_reference order); chain: ref[0] = first."""
    ref, new = b["ref"], b["new"]; roles = {}
    if batch_mode == "chain" or len(ref) == 1:
        roles[ref[0]] = "first"
    else:
        n_near = (k - 1) // 2; roles[ref[0]] = "first"
        for r in ref[1:1 + n_near]: roles[r] = "near"
        for r in ref[1 + n_near:]: roles[r] = "far"
    for n in new: roles[n] = "new"
    return roles


def pair_type(ri, rj):
    s = {ri, rj}
    if "new" in s:
        other = (s - {"new"}).pop() if len(s) == 2 else "new"
        return {"first": "new-first", "near": "new-near", "far": "new-far", "new": "new-new"}[other]
    return "ref-ref"


# ---------------------------------------------------------------- alignment variants (A2)
def align_a(ref_known, ref_est): return L.align(ref_known, ref_est)


def align_variant(mode, ref_known, ref_est, roles_ref):
    """(a) all refs chordal mean; (b) first only; (c) first + near; (d) robust: best single-ref T by median residual of the others,
    then re-align with refs whose rotation residual under it is < 10° (T_r alone if only one)."""
    n = len(ref_known)
    if mode == "a" or n == 1: return L.align(ref_known, ref_est)
    if mode == "b": return L.align(ref_known[:1], ref_est[:1])
    if mode == "c":
        idx = [i for i, r in enumerate(roles_ref) if r in ("first", "near")] or [0]
        return L.align([ref_known[i] for i in idx], [ref_est[i] for i in idx])
    if mode == "d":
        best, best_med, best_res = None, np.inf, None
        for r in range(n):
            T = L.align(ref_known[r:r + 1], ref_est[r:r + 1])
            res = [rot_deg((T @ ref_est[i])[:3, :3], ref_known[i][:3, :3]) for i in range(n) if i != r]
            med = float(np.median(res)) if res else 0.0
            if med < best_med: best, best_med, best_res = T, med, [rot_deg((T @ ref_est[i])[:3, :3], ref_known[i][:3, :3]) for i in range(n)]
        idx = [i for i in range(n) if best_res[i] < 10.0]
        return L.align([ref_known[i] for i in idx], [ref_est[i] for i in idx]) if len(idx) > 1 else best
    raise ValueError(mode)


# ---------------------------------------------------------------- cell iteration
def iter_cells(root=P1):
    for bj in sorted(glob.glob(str(root / "M-*/I*/*/*/*/batches.jsonl"))):
        d = Path(bj).parent; model, inp, bt, ds, sq = d.parts[-5:]
        yield dict(dir=d, model=model, input=inp, batch=bt, ds=ds, seq=sq)


def read_batches(d): return [json.loads(l) for l in open(Path(d) / "batches.jsonl")]


def read_kf(d):
    rows = {}
    for r in csv.DictReader(open(Path(d) / "kf.csv")):
        rows.setdefault(r["kf"], []).append(r)
    return rows   # kf -> rows (a kf can appear in several batches: as ref later)


def batch_k(bt):   # "ref1-8" -> (M=1, k=8); "chain" -> (5, 1)
    if bt == "chain": return 5, 1
    m, k = bt[3:].split("-"); return int(m), int(k)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--build-cache", action="store_true"); a = ap.parse_args()
    if a.build_cache:
        for ds, seqs in [("ho3d", HO3D), ("ycb", YCB)]:
            for sq in seqs:
                if (CACHE / f"{ds}_{sq}.pkl").exists(): continue
                build_cache(ds, sq); print("cached", ds, sq, flush=True)
