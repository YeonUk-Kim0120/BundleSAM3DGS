"""A2: re-align new-KF poses of every M-V/I1 and M-O/I1 B-ref cell with alignment variants (a)(b)(c)(d) × {online, oracle}.
(a)-online must reproduce the stored T_online (else abort A2).  Writes realign.csv per cell and table R."""
import csv, sys
import numpy as np, pandas as pd
from p1b_common import *

MODES = ["a", "b", "c", "d"]
rows_all = []; mismatch = []
cache = {}
TABLES_ONLY = "--tables-only" in sys.argv
for c in ([] if TABLES_ONLY else iter_cells()):
    if c["input"] != "I1" or c["batch"] == "chain": continue
    key = (c["ds"], c["seq"]); S = cache.setdefault(key, load_seq(*key))
    bl = read_batches(c["dir"]); M, k = batch_k(c["batch"]); out = P1B / "A" / c["model"] / c["input"] / c["batch"] / c["ds"] / c["seq"]; out.mkdir(parents=True, exist_ok=True)
    rows = []
    for b in bl:
        c2w = c2w_batch(b); n_ref = len(b["ref"]); roles = roles_for_batch(b, c["batch"], k); roles_ref = [roles[r] for r in b["ref"]]
        snap = S.snapshot_at(b["new"][-1]); ref_on = [snap.get(r, S.C1[r]) for r in b["ref"]]
        ok_or = [i for i, r in enumerate(b["ref"]) if S.GT.get(r) is not None]; ref_or = [S.GT[b["ref"][i]] for i in ok_or]
        T_a = align_variant("a", ref_on, c2w[:n_ref], roles_ref)
        d = np.abs(T_a - np.array(b["T_online"])).max()
        if d > 1e-6: mismatch.append((c, b["batch"], d))
        for src in ["online", "oracle"]:
            if src == "online": known, est, rr = ref_on, c2w[:n_ref], roles_ref
            else:
                if not ok_or: continue
                known, est, rr = ref_or, [c2w[i] for i in ok_or], [roles_ref[i] for i in ok_or]
            for mode in MODES:
                T = align_variant(mode, known, est, rr)
                for i in range(n_ref, len(b["ids"])):
                    f = b["ids"][i]; g = S.GT.get(f)
                    if g is None: continue
                    e = L.pose_err(T @ c2w[i], g); e1 = L.pose_err(S.C1[f], g); e2 = L.pose_err(S.C2[f], g)
                    rows.append(dict(model=c["model"], input=c["input"], batch_mode=c["batch"], ds=c["ds"], seq=c["seq"], batch=b["batch"], kf=f, source=src, mode=mode, rot=e[0], trans=e[1], rot_c1=e1[0], rot_c2=e2[0], trans_c1=e1[1]))
    pd.DataFrame(rows).to_csv(out / "realign.csv", index=False); rows_all += rows; print("realign", c["model"], c["batch"], c["ds"], c["seq"], len(rows), flush=True)
if mismatch:
    msg = "A2 (a)-online mismatch vs stored T_online: " + "; ".join(f"{m[0]['model']}/{m[0]['batch']} {m[0]['seq']} b{m[1]} Δ={m[2]:.2e}" for m in mismatch[:10])
    open(LOG / "A2_ABORT.txt", "w").write(msg); print(msg); sys.exit(1)
if TABLES_ONLY: df = pd.read_csv(P1B / "A" / "all_realign.csv")
else: df = pd.DataFrame(rows_all); df.to_csv(P1B / "A" / "all_realign.csv", index=False)


def f(x, p=2): return "–" if not np.isfinite(x) else f"{x:.{p}f}"


def table_R(df, model, bt, ds, seqs):
    d = df[(df.model == model) & (df.batch_mode == bt) & (df.ds == ds)]
    hdr = "| seq | 새 KF | C1 rot 중/p90 | C2 rot 중/p90 | " + " | ".join(f"({m}) {s} rot 중/p90 · trans 중" for s in ["online", "oracle"] for m in MODES) + " |"
    out = [hdr, "|" + "---|" * (hdr.count("|") - 1)]; agg = {}
    for sq in seqs:
        g = d[d.seq == sq]
        if not len(g): continue
        base = g[(g.source == "online") & (g["mode"] == "a")]
        if not len(base): continue
        cells = [f"{f(np.median(base.rot_c1))}/{f(np.percentile(base.rot_c1,90))}", f"{f(np.median(base.rot_c2))}/{f(np.percentile(base.rot_c2,90))}"]
        agg.setdefault("c1", []).append(np.median(base.rot_c1)); agg.setdefault("c2", []).append(np.median(base.rot_c2))
        for s in ["online", "oracle"]:
            for m in MODES:
                x = g[(g.source == s) & (g["mode"] == m)]
                cells.append(f"{f(np.median(x.rot))}/{f(np.percentile(x.rot,90))} · {f(np.median(x.trans),1)}" if len(x) else "–")
                agg.setdefault((s, m), []).append(np.median(x.rot) if len(x) else np.nan); agg.setdefault((s, m, "win"), []).append(np.median(x.rot) < np.median(base.rot_c1) if len(x) else False)
        out.append(f"| {sq} | {len(base)} | " + " | ".join(cells) + " |")
    if "c1" not in agg: return "(행 없음)"
    out.append(f"| **평균** | | {f(np.nanmean(agg['c1']))} | {f(np.nanmean(agg['c2']))} | " + " | ".join(f"{f(np.nanmean(agg[(s,m)]))} (C1보다 나은 seq {sum(agg[(s,m,'win')])}/{len(agg[(s,m)])})" for s in ["online", "oracle"] for m in MODES) + " |")
    return "\n".join(out)


doc = ["# A2 표 R — 정렬 방식 (a) 기존 chordal mean, (b) 첫 자리만, (c) 첫 자리+near, (d) 강건 (× online/oracle 기준 포즈)", ""]
for model in ["M-V", "M-O"]:
    for bt in ["ref1-8", "ref5-8", "ref1-4", "ref5-4"]:
        for ds, seqs in [("ho3d", HO3D), ("ycb", YCB)]:
            if not len(df[(df.model == model) & (df.batch_mode == bt) & (df.ds == ds)]): continue
            doc += [f"## {model} / I1 / {bt} / {ds}", "", table_R(df, model, bt, ds, seqs), ""]
open(LOG / "table_A2_R.md", "w").write("\n".join(doc)); print("A2 done")
