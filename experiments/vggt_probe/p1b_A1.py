"""A1: pairs.csv for every Probe 1 cell (all batch pairs with GT on both frames) + tables P1/P2/P3."""
import csv, sys
import numpy as np, pandas as pd
from p1b_common import *

COLS = ["model", "input", "batch_mode", "ds", "seq", "batch", "kf_i", "kf_j", "role_i", "role_j", "pair_type", "swing", "roll", "vggt_rot", "vggt_tdir", "c2_rot", "c1_rot", "mask_i", "mask_j", "conf_i", "conf_j", "T3"]


def cell_pairs(c, S):
    bl = read_batches(c["dir"]); kf = read_kf(c["dir"]); M, k = batch_k(c["batch"]); rows = []
    for b in bl:
        c2w = c2w_batch(b); roles = roles_for_batch(b, c["batch"], k); ids = b["ids"]
        info = {}
        for f in ids:
            r = [x for x in kf[f] if int(float(x["batch"])) == b["batch"]]; info[f] = r[0] if r else None
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                fi, fj = ids[i], ids[j]; gi, gj = S.GT.get(fi), S.GT.get(fj)
                if gi is None or gj is None: continue
                rg = rel(gi, gj); re_ = rel(c2w[i], c2w[j]); sw, tw = swing_twist(rg[:3, :3])
                r2 = rel(S.C2[fi], S.C2[fj]); r1 = rel(S.C1[fi], S.C1[fj])
                ii, jj = info[fi], info[fj]
                rows.append([c["model"], c["input"], c["batch"], c["ds"], c["seq"], b["batch"], fi, fj, roles[fi], roles[fj], pair_type(roles[fi], roles[fj]), sw, tw,
                             rot_deg(re_[:3, :3], rg[:3, :3]), tdir_deg(re_[:3, 3], rg[:3, 3]), rot_deg(r2[:3, :3], rg[:3, :3]), rot_deg(r1[:3, :3], rg[:3, :3]),
                             float(ii["T6_mask_area"]) if ii else np.nan, float(jj["T6_mask_area"]) if jj else np.nan, float(ii["T5_conf"]) if ii else np.nan, float(jj["T5_conf"]) if jj else np.nan, b.get("T3", np.nan)])
    return rows


def q(x, p): return float(np.percentile(x, p)) if len(x) else np.nan
def f(x, p=1): return "–" if x is None or not np.isfinite(x) else f"{x:.{p}f}"


def table_P1(df, title):
    out = [f"### {title}", "", "| 모델/입력 | 배치 | swing bin | n | VGGT rot 중앙값 | p75 | p90 | >30° | >90° | C2 쌍 중앙값 | C2 p90 |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    df = df.copy(); df["kind"] = np.where(df.batch_mode == "chain", "chain", "B-ref"); df["sb"] = [bin_index(x, SWING_BINS) for x in df.swing]
    for (m, inp, kind), g in df.groupby(["model", "input", "kind"]):
        for sb, gg in g.groupby("sb"):
            v = gg.vggt_rot.values; c2 = gg.c2_rot.values
            out.append(f"| {m}/{inp} | {kind} | {SWING_LABELS[sb]} | {len(v)} | {f(np.median(v))} | {f(q(v,75))} | {f(q(v,90))} | {f(100*np.mean(v>30),0)} % | {f(100*np.mean(v>90),0)} % | {f(np.median(c2))} | {f(q(c2,90))} |")
    return "\n".join(out)


def table_P2(df):
    d = df[(df.model == "M-V") & (df.input == "I1") & (df.batch_mode != "chain")].copy(); d["sb"] = [bin_index(x, SWING_BINS) for x in d.swing]; d["rb"] = [bin_index(x, ROLL_BINS) for x in d.roll]
    out = ["| swing \\ roll | " + " | ".join(ROLL_LABELS) + " |", "|---|" + "---|" * len(ROLL_LABELS)]
    for sb in range(len(SWING_LABELS)):
        cells = []
        for rb in range(len(ROLL_LABELS)):
            v = d[(d.sb == sb) & (d.rb == rb)].vggt_rot.values
            cells.append(f"{f(np.median(v))}° / {f(100*np.mean(v>30),0)} % / n={len(v)}" if len(v) else "–")
        out.append(f"| {SWING_LABELS[sb]} | " + " | ".join(cells) + " |")
    return "\n".join(out)


def table_P3(df):
    d = df[(df.model == "M-V") & (df.input == "I1") & (df.batch_mode != "chain")].copy(); d["sb"] = [bin_index(x, SWING_BINS) for x in d.swing]
    out = ["| 쌍 종류 | swing bin | n | VGGT rot 중앙값 | p90 | >30° | >90° | C2 쌍 중앙값 |", "|---|---|---|---|---|---|---|---|"]
    for pt in ["new-first", "new-near", "new-far", "new-new", "ref-ref"]:
        for sb in range(len(SWING_LABELS)):
            v = d[(d.pair_type == pt) & (d.sb == sb)]; x = v.vggt_rot.values
            if len(x): out.append(f"| {pt} | {SWING_LABELS[sb]} | {len(x)} | {f(np.median(x))} | {f(q(x,90))} | {f(100*np.mean(x>30),0)} % | {f(100*np.mean(x>90),0)} % | {f(np.median(v.c2_rot.values))} |")
    far = d[d.pair_type == "new-far"]; nofar = d[d.pair_type.isin(["new-first", "new-near", "new-new"])]
    out += ["", f"먼 기준이 낀 새 KF 쌍(new-far): n={len(far)}, 중앙값 {f(np.median(far.vggt_rot))}°, >90° {f(100*np.mean(far.vggt_rot>90),0)} %; 낀 것 없음(new-first/near/new): n={len(nofar)}, 중앙값 {f(np.median(nofar.vggt_rot))}°, >90° {f(100*np.mean(nofar.vggt_rot>90),0)} %"]
    return "\n".join(out)


if __name__ == "__main__":
    only = sys.argv[1] if len(sys.argv) > 1 else None
    (P1B / "A").mkdir(exist_ok=True); all_rows = []; cache = {}
    for c in iter_cells():
        key = (c["ds"], c["seq"])
        if key not in cache: cache[key] = load_seq(*key)
        out = P1B / "A" / c["model"] / c["input"] / c["batch"] / c["ds"] / c["seq"]; out.mkdir(parents=True, exist_ok=True)
        rows = cell_pairs(c, cache[key])
        with open(out / "pairs.csv", "w", newline="") as fh:
            w = csv.writer(fh); w.writerow(COLS); w.writerows(rows)
        all_rows += rows; print("pairs", c["model"], c["input"], c["batch"], c["ds"], c["seq"], len(rows), flush=True)
    df = pd.DataFrame(all_rows, columns=COLS); df.to_csv(P1B / "A" / "all_pairs.csv", index=False)
    doc = ["# A1 표 (기선별 정확도)", ""]
    for ds in ["ho3d", "ycb"]:
        d = df[df.ds == ds]
        doc += [f"## 표 P1 — {ds} 합계 (모든 시퀀스 풀)", "", table_P1(d, f"{ds} 합계"), ""]
        for sq in sorted(d.seq.unique()):
            doc += [table_P1(d[d.seq == sq], f"{ds} / {sq}"), ""]
    doc += ["## 표 P2 — M-V/I1 B-ref, swing × roll (중앙값° / >30° 비율 / n), HO3D+YCB", "", table_P2(df), "", "## 표 P2 — HO3D만", "", table_P2(df[df.ds == "ho3d"]), ""]
    doc += ["## 표 P3 — M-V/I1 B-ref, 쌍 종류별 swing bin (HO3D+YCB)", "", table_P3(df), "", "## 표 P3 — HO3D만", "", table_P3(df[df.ds == "ho3d"]), ""]
    open(LOG / "table_A1.md", "w").write("\n".join(doc)); print("A1 done", len(df))
