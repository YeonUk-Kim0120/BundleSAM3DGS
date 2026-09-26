"""Tables for stage B (table B, P-B2, NC) and B6 (table L)."""
import csv, glob, json
import numpy as np, pandas as pd
from pathlib import Path
from p1b_common import *

SEQS = ["AP12", "MPM10", "MPM12", "SB11", "SM1"]; CONDS = ["B1-k8", "B1-k4", "B2", "B1-roll", "B4-black", "B4-gray"]
def f(x, p=2): return "–" if x is None or not np.isfinite(x) else f"{x:.{p}f}"
def q(x, p): return float(np.percentile(x, p)) if len(x) else np.nan


def load_kf(cond, sq):
    p = P1B / "B" / cond / sq / "kf.csv"
    return pd.read_csv(p) if p.exists() else None


def probe1_ref(sq):
    """Probe 1 M-V/I1/ref1-8 new-KF errors (online (a) only) for the comparison row."""
    p = P1 / "M-V/I1/ref1-8/ho3d" / sq / "kf.csv"; d = pd.read_csv(p); d = d[(d.role == "new") & (d.has_gt == 1)]
    return d


def table_B():
    out = ["| 조건 | seq | 새 KF | C1 중/p90 | C2 중/p90 | online (a) 중/p90 | online (d) 중/p90 | oracle (a) 중/p90 | oracle (d) 중/p90 | 쌍 오차 중앙값: new-first / new-near(또는 대역) / new-new | E3 중앙값 |", "|" + "---|" * 11]
    agg = {}
    for cond in CONDS + ["Probe1 ref1-8"]:
        for sq in SEQS:
            if cond == "Probe1 ref1-8":
                d = probe1_ref(sq); on_a = d.rot_online.values; on_d = np.full(1, np.nan); or_a = d.rot_oracle.values; or_d = np.full(1, np.nan)
                pp = pd.read_csv(P1B / "A/M-V/I1/ref1-8/ho3d" / sq / "pairs.csv"); bl = [json.loads(l) for l in open(P1 / "M-V/I1/ref1-8/ho3d" / sq / "batches.jsonl")]; e3 = np.nanmedian([b.get("E3_err", np.nan) for b in bl])
                c1, c2 = d.rot_c1.values, d.rot_c2.values
            else:
                d = load_kf(cond, sq)
                if d is None: continue
                d = d[(d.role == "new") & (d.has_gt == 1)]; on_a, on_d, or_a, or_d = d.rot_a_online.values, d.rot_d_online.values, d.rot_a_oracle.values, d.rot_d_oracle.values; c1, c2 = d.rot_c1.values, d.rot_c2.values
                pp = pd.read_csv(P1B / "B" / cond / sq / "pairs.csv"); bl = [json.loads(l) for l in open(P1B / "B" / cond / sq / "batches.jsonl")]; e3 = np.nanmedian([b.get("E3_err", np.nan) for b in bl])
            pt = lambda t: np.median(pp[pp.pair_type == t].vggt_rot) if (pp.pair_type == t).any() else np.nan
            near = pt("new-near") if (pp.pair_type == "new-near").any() else np.nanmedian([np.median(pp[pp.pair_type == t].vggt_rot) for t in pp.pair_type.unique() if t.startswith("new-band")] or [np.nan])
            out.append(f"| {cond} | {sq} | {len(d)} | {f(np.median(c1))}/{f(q(c1,90))} | {f(np.median(c2))}/{f(q(c2,90))} | {f(np.median(on_a))}/{f(q(on_a,90))} | {f(np.nanmedian(on_d))}/{f(q(on_d[np.isfinite(on_d)],90))} | {f(np.nanmedian(or_a))}/{f(q(or_a[np.isfinite(or_a)],90))} | {f(np.nanmedian(or_d))}/{f(q(or_d[np.isfinite(or_d)],90))} | {f(pt('new-first'),1)} / {f(near,1)} / {f(pt('new-new'),1)} | {f(e3)} |")
            for key, v in [("on_a", np.median(on_a)), ("on_d", np.nanmedian(on_d)), ("or_a", np.nanmedian(or_a)), ("or_d", np.nanmedian(or_d)), ("c1", np.median(c1)), ("c2", np.median(c2))]: agg.setdefault((cond, key), []).append(v)
        if (cond, "on_a") in agg:
            g = lambda key: np.nanmean(agg[(cond, key)])
            out.append(f"| **{cond} 평균** | {len(agg[(cond,'on_a')])} seq | | {f(g('c1'))} | {f(g('c2'))} | {f(g('on_a'))} | {f(g('on_d'))} | {f(g('or_a'))} | {f(g('or_d'))} | | |")
    return "\n".join(out)


def table_PB2():
    out = ["| seq | 대역(기준 역할) | 쌍 n | VGGT 쌍 회전 중앙값 | p90 | >30° | C2 쌍 중앙값 | GT swing 중앙값 |", "|---|---|---|---|---|---|---|---|"]
    allp = []
    for sq in SEQS:
        p = P1B / "B/B2" / sq / "pairs.csv"
        if not p.exists(): continue
        pp = pd.read_csv(p); pp = pp[pp.pair_type.str.startswith("new-")]; allp.append(pp)
        for t in ["new-first", "new-band[15,30)", "new-band[30,60)", "new-band[60,90)", "new-fill"]:
            g = pp[pp.pair_type == t]
            if len(g): out.append(f"| {sq} | {t} | {len(g)} | {f(np.median(g.vggt_rot),1)} | {f(q(g.vggt_rot,90),1)} | {f(100*np.mean(g.vggt_rot>30),0)} % | {f(np.median(g.c2_rot),1)} | {f(np.median(g.swing),0)} |")
    if allp:
        pp = pd.concat(allp)
        for t in ["new-first", "new-band[15,30)", "new-band[30,60)", "new-band[60,90)", "new-fill"]:
            g = pp[pp.pair_type == t]
            if len(g): out.append(f"| **합계** | {t} | {len(g)} | {f(np.median(g.vggt_rot),1)} | {f(q(g.vggt_rot,90),1)} | {f(100*np.mean(g.vggt_rot>30),0)} % | {f(np.median(g.c2_rot),1)} | {f(np.median(g.swing),0)} |")
        # also by GT swing bin of the pair (independent of the selection band)
        out += ["", "| seq | GT swing bin (new–기준 쌍) | n | VGGT 중앙값 | p90 | >30° | C2 중앙값 |", "|---|---|---|---|---|---|---|"]
        pp["sb"] = [bin_index(x, SWING_BINS) for x in pp.swing]
        for sb in range(len(SWING_LABELS)):
            g = pp[pp.sb == sb]
            if len(g): out.append(f"| 합계 | {SWING_LABELS[sb]} | {len(g)} | {f(np.median(g.vggt_rot),1)} | {f(q(g.vggt_rot,90),1)} | {f(100*np.mean(g.vggt_rot>30),0)} % | {f(np.median(g.c2_rot),1)} |")
    return "\n".join(out)


def table_NC():
    out = ["| seq | 배치 수 | 연속 쌍 n | VGGT 상대 회전 크기 중앙값° | GT 상대 회전 크기 중앙값° | VGGT/GT 비 | 참고: 마스크 I1 체인(Probe 1) VGGT 상대 회전 크기 중앙값 |", "|---|---|---|---|---|---|---|"]
    for sq in ["AP12", "SM1"]:
        p = P1B / "B/B5" / sq / "batches.jsonl"
        if not p.exists(): continue
        S = load_seq("ho3d", sq); vm, gm, vm1 = [], [], []
        for b in map(json.loads, open(p)):
            c2w = c2w_batch(b); ids = b["ids"]
            for a in range(len(ids) - 1):
                if S.GT.get(ids[a]) is None or S.GT.get(ids[a + 1]) is None: continue
                vm.append(rot_deg(rel(c2w[a], c2w[a + 1])[:3, :3], np.eye(3))); gm.append(rot_deg(rel(S.GT[ids[a]], S.GT[ids[a + 1]])[:3, :3], np.eye(3)))
        for b in map(json.loads, open(P1 / "M-V/I1/chain/ho3d" / sq / "batches.jsonl")):
            c2w = c2w_batch(b); vm1 += [rot_deg(rel(c2w[a], c2w[a + 1])[:3, :3], np.eye(3)) for a in range(len(b["ids"]) - 1)]
        out.append(f"| {sq} | {sum(1 for _ in open(p))} | {len(vm)} | {f(np.median(vm),2)} | {f(np.median(gm),2)} | {f(np.median(vm)/np.median(gm),2)} | {f(np.median(vm1),2)} |")
    return "\n".join(out)


def table_L():
    p = P1B / "B6/pairs_bench.csv"
    if not p.exists(): return "(B6 미완료)"
    d = pd.read_csv(p); out = []
    def block(g, name):
        rows = [f"### {name} (n={len(g)})", "", "| swing bin | n | LoFTR 성공률(≥5) | 성공률(≥30) | LoFTR 성공 쌍 rot 중/p90 | VGGT 2장 중/p90 | >30° | VGGT 2장+roll 중/p90 | >30° | C2 쌍 중앙값 | roll 중앙값 |", "|---|---|---|---|---|---|---|---|---|---|---|"]
        for lab in SWING_LABELS:
            x = g[g.swing_bin == lab]
            if not len(x): continue
            s = x[x.loftr_success == 1]
            rows.append(f"| {lab} | {len(x)} | {f(100*x.loftr_success.mean(),0)} % | {f(100*x.loftr_success30.mean(),0)} % | {f(np.median(s.loftr_rot),1) if len(s) else '–'}/{f(q(s.loftr_rot,90),1) if len(s) else '–'} | {f(np.median(x.vggt2_rot),1)}/{f(q(x.vggt2_rot,90),1)} | {f(100*np.mean(x.vggt2_rot>30),0)} % | {f(np.median(x.vggt2roll_rot),1)}/{f(q(x.vggt2roll_rot,90),1)} | {f(100*np.mean(x.vggt2roll_rot>30),0)} % | {f(np.median(x.c2_rot),1)} | {f(np.median(x.roll),0)} |")
        return "\n".join(rows)
    out.append(block(d, "5개 시퀀스 합계"))
    for sq in SEQS: out.append(block(d[d.seq == sq], sq))
    return "\n\n".join(out)


if __name__ == "__main__":
    doc = ["# B단계 표", "", "## 표 B — 조건별 새 KF E1 (M-V, I1, M=1; 정렬 (a) 기존 / (d) 강건 × online / oracle) — 5 시퀀스", "", table_B(), "", "## 표 P-B2 — B2 대역별 쌍 오차 (new–기준)", "", table_PB2(), "", "## 표 NC — B5 음성 대조군 (마스크 없음, I0 패딩, 체인)", "", table_NC(), "", "## 표 L — B6 쌍 벤치마크: LoFTR+depth vs VGGT 2장 vs VGGT 2장+roll (GT swing bin별)", "", table_L(), ""]
    open(LOG / "table_B.md", "w").write("\n".join(doc)); print("\n".join(doc))
