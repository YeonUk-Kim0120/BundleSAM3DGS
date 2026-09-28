"""EXP_BATCH_20260928 evaluation helpers: per-keyframe d (mean GT-mesh point displacement, mm), rotation (deg), translation (mm) of
c2w poses against GT (GroundTruth first-frame gauge, precomputed in the input file), improvement / worsening (brief 1)."""
from __future__ import annotations
import json, math
import numpy as np


def rot_angle(R1, R2): return math.degrees(2.0 * math.asin(min(1.0, float(np.linalg.norm(R1 - R2)) / (2.0 * math.sqrt(2.0)))))


class Evaluator:
    def __init__(self, inp):
        if isinstance(inp, str): inp = json.load(open(inp))
        self.ids = inp["kf_ids"]; self.gt = {f: (np.array(g) if g is not None else None) for f, g in zip(inp["kf_ids"], inp["gt"])}
        self.p = np.asarray(inp["mesh_points"], dtype=np.float64)

    def err(self, fid, c2w):
        g = self.gt.get(fid)
        if g is None: return None
        c2w = np.asarray(c2w, dtype=np.float64); A = c2w @ np.linalg.inv(g)
        d = float(np.linalg.norm(self.p @ A[:3, :3].T + A[:3, 3] - self.p, axis=1).mean() * 1000.0)
        return dict(d=d, rot=rot_angle(c2w[:3, :3], g[:3, :3]), trans=float(np.linalg.norm(c2w[:3, 3] - g[:3, 3]) * 1000.0))

    def compare(self, ids, before, after):
        rows = []
        for f, b, a in zip(ids, before, after):
            eb, ea = self.err(f, b), self.err(f, a)
            if eb is None: continue
            thr = max(0.5, 0.1 * eb["d"])
            rows.append(dict(kf=f, d0=eb["d"], d1=ea["d"], rot0=eb["rot"], rot1=ea["rot"], trans0=eb["trans"], trans1=ea["trans"],
                             improved=int(eb["d"] - ea["d"] >= thr), worsened=int(ea["d"] - eb["d"] >= thr)))
        return rows


def summary(rows):
    d0 = np.array([r["d0"] for r in rows]); d1 = np.array([r["d1"] for r in rows])
    return dict(n=len(rows), d0_med=float(np.median(d0)), d1_med=float(np.median(d1)), d0_p90=float(np.percentile(d0, 90)), d1_p90=float(np.percentile(d1, 90)),
                imp=float(np.mean([r["improved"] for r in rows])), wor=float(np.mean([r["worsened"] for r in rows])),
                d_reduction_frac=float((np.median(d0) - np.median(d1)) / max(np.median(d0), 1e-9)))
