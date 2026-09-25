"""Post-hoc analysis (no new VGGT inference): re-align the B-chain outputs with the overlap KF's R-online pose (BSDF
snapshot at the batch's newest new KF) instead of the chained VGGT estimate.  This is what B-ref(5,1) would give.
Reads batches.jsonl (w2c_raw, R_virtual, s, ids) of every M-*/I*/chain cell; writes kf_realigned_online.csv next to it."""
from __future__ import annotations
import csv, glob, json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import probe_lib as L

ROOT = Path("/home/kist/Desktop/BundleSAM3DGS/outputs/exp_vggt_probe1")
res = {}
for bj in sorted(glob.glob(str(ROOT / "M-*/I*/chain/*/*/batches.jsonl"))):
    d = Path(bj).parent; model, inp, _, ds, sq = d.parts[-5:]
    if (d / "kf_realigned_online.csv").exists() and "--force" not in sys.argv:
        rows = [{k: (float(v) if k not in ("kf",) else v) for k, v in r.items()} for r in csv.DictReader(open(d / "kf_realigned_online.csv"))]
        m = lambda k: float(np.median([r[k] for r in rows]))
        res[(f"{model}/{inp}", ds, sq)] = dict(n=len(rows), rot=m("rot_online_snapref"), tr=m("trans_online_snapref"), rot_c1=m("rot_c1"), tr_c1=m("trans_c1"), rot_c2=m("rot_c2"), tr_c2=m("trans_c2"), win=float(np.mean([r["rot_online_snapref"] < r["rot_c1"] for r in rows])))
        continue
    seq = L.Sequence(ds, sq); rows = []
    for b in map(json.loads, open(bj)):
        ids = b["ids"]; n_ref = len(b["ref"]); snap = seq.snapshot_at(b["new"][-1])
        c2w_b = []
        for i in range(len(ids)):
            P = np.linalg.inv(np.array(b["w2c_raw"][i])); P[:3, 3] *= b["s"] if np.isfinite(b["s"]) else 1.0
            B = np.eye(4); B[:3, :3] = np.array(b["R_virtual"][i]); c2w_b.append(P @ B)
        ref_on = [snap.get(r, seq.C1[r]) for r in b["ref"]]
        T = L.align(ref_on, c2w_b[:n_ref])
        for i in range(n_ref, len(ids)):
            g = seq.gt_c2w(ids[i])
            if g is None: continue
            e = L.pose_err(T @ c2w_b[i], g); e1 = L.pose_err(seq.C1[ids[i]], g); e2 = L.pose_err(seq.C2[ids[i]], g)
            rows.append(dict(kf=ids[i], batch=b["batch"], rot_online_snapref=e[0], trans_online_snapref=e[1], rot_c1=e1[0], trans_c1=e1[1], rot_c2=e2[0], trans_c2=e2[1]))
    with open(d / "kf_realigned_online.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); [w.writerow(r) for r in rows]
    m = lambda k: float(np.median([r[k] for r in rows]))
    res[(f"{model}/{inp}", ds, sq)] = dict(n=len(rows), rot=m("rot_online_snapref"), tr=m("trans_online_snapref"), rot_c1=m("rot_c1"), tr_c1=m("trans_c1"), rot_c2=m("rot_c2"), tr_c2=m("trans_c2"), win=float(np.mean([r["rot_online_snapref"] < r["rot_c1"] for r in rows])))
    print(f"{model}/{inp} {ds}/{sq}: n={len(rows)} rot snapref/C1/C2 = {res[(f'{model}/{inp}', ds, sq)]['rot']:.2f}/{m('rot_c1'):.2f}/{m('rot_c2'):.2f}  trans = {m('trans_online_snapref'):.1f}/{m('trans_c1'):.1f}/{m('trans_c2'):.1f}  win {100*res[(f'{model}/{inp}', ds, sq)]['win']:.0f}%", flush=True)
out = ["| 설정 | seq | 새 KF | chain+R-online 기준 rot° | C1 rot° | C2 rot° | trans mm | C1 trans | C2 trans | rot<C1 KF 비율 |", "|---|---|---|---|---|---|---|---|---|---|"]
for (c, ds, sq), r in sorted(res.items()):
    out.append(f"| {c}/chain | {sq} | {r['n']} | {r['rot']:.2f} | {r['rot_c1']:.2f} | {r['rot_c2']:.2f} | {r['tr']:.1f} | {r['tr_c1']:.1f} | {r['tr_c2']:.1f} | {100*r['win']:.0f} % |")
for c in sorted({c for c, _, _ in res}):
    for ds in ["ho3d", "ycb"]:
        v = [r for (cc, d, _), r in res.items() if cc == c and d == ds]
        if v: out.append(f"| **{c}/chain 평균 {ds}** | {len(v)} seq | | {np.mean([r['rot'] for r in v]):.2f} | {np.mean([r['rot_c1'] for r in v]):.2f} | {np.mean([r['rot_c2'] for r in v]):.2f} | {np.mean([r['tr'] for r in v]):.1f} | {np.mean([r['tr_c1'] for r in v]):.1f} | {np.mean([r['tr_c2'] for r in v]):.1f} | {sum(r['rot'] < r['rot_c1'] for r in v)}/{len(v)} seq |")
open("/home/kist/Desktop/BundleSAM3DGS/logs/exp_vggt_probe1/table_chain_realigned_online.md", "w").write("\n".join(out)); print("\n".join(out))
