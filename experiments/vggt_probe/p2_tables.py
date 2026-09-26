"""Probe 2 tables: A1 (model × estimator, HO3D 13 mean of per-sequence stats), A2 (C1-error bins), A3 (per sequence), occlusion split,
B (success vs delta), S (trust signals), cost, D (depth holes)."""
from __future__ import annotations
import glob, json
import numpy as np, pandas as pd
from scipy.stats import spearmanr
from p2_lib import OUT, LOG, HO3D

EST = ["C1 (초기)", "E-V2", "E-V5b", "E-V5d", "E-V5x2", "E-L", "E-I", "E-VI"]
MODELS = ["MD-gt", "MD-prior", "MD-map"]; BINS = [0, 3, 10, 30, 180.01]; BIN_L = ["[0,3)", "[3,10)", "[10,30)", "[30,180]"]


def f(x, p=1): return "–" if x is None or not np.isfinite(x) else f"{x:.{p}f}"
def pct(x): return "–" if x is None or not np.isfinite(x) else f"{100*x:.0f} %"


def load(stage):
    fs = sorted(glob.glob(str(OUT / stage / "*/*/kf.csv")))
    return pd.concat([pd.read_csv(x, dtype={"kf": str}) for x in fs], ignore_index=True) if fs else pd.DataFrame()


def with_c1(df):
    """Add pseudo-estimator 'C1 (초기)' rows (estimate = initial pose) for comparison."""
    base = df[df.estimator == "E-I"].copy(); base["estimator"] = "C1 (초기)"; base["rot"] = base.rot0; base["trans"] = base.trans0
    base["success"] = ((base.rot0 < 5) & (base.trans0 < 20)).astype(int); base["strict"] = ((base.rot0 < 2) & (base.trans0 < 10)).astype(int); base["improved"] = 0; base["worsened"] = 0
    return pd.concat([base, df], ignore_index=True)


def stats(g):
    return dict(n=len(g), rot_med=g.rot.median(), rot_p90=g.rot.quantile(.9), tr_med=g.trans.median(), tr_p90=g.trans.quantile(.9), succ=g.success.mean(), strict=g.strict.mean(), imp=g.improved.mean(), wor=g.worsened.mean())


def table_A1(df):
    out = ["| 모델 | 추정기 | seq 수 | 회전 중앙값° | 회전 p90° | 이동 중앙값 mm | 이동 p90 mm | 성공률 (5°·20 mm) | 엄격 (2°·10 mm) | 개선률 | 악화율 (+2°) |", "|" + "---|" * 11]
    for m in MODELS:
        d = df[df.model == m]
        if not len(d): continue
        for e in EST:
            per = [stats(g) for _, g in d[d.estimator == e].groupby("seq")]
            if not per: continue
            mean = {k: np.nanmean([p[k] for p in per]) for k in per[0]}
            out.append(f"| {m} | {e} | {len(per)} | {f(mean['rot_med'],2)} | {f(mean['rot_p90'])} | {f(mean['tr_med'])} | {f(mean['tr_p90'])} | {pct(mean['succ'])} | {pct(mean['strict'])} | {pct(mean['imp'])} | {pct(mean['wor'])} |")
    return "\n".join(out)


def table_A2(df):
    out = ["| 모델 | 추정기 | C1 회전 구간 | n | C1 회전 중앙값 | 추정 회전 중앙값° | p90 | 이동 중앙값 mm | 성공률 | 개선률 | 악화율 |", "|" + "---|" * 11]
    d0 = df.copy(); d0["bin"] = pd.cut(d0.rot0, BINS, labels=BIN_L, right=False)
    for m in MODELS:
        for e in EST:
            for b in BIN_L:
                g = d0[(d0.model == m) & (d0.estimator == e) & (d0.bin == b)]
                if not len(g): continue
                s = stats(g); out.append(f"| {m} | {e} | {b} | {s['n']} | {f(g.rot0.median(),2)} | {f(s['rot_med'],2)} | {f(s['rot_p90'])} | {f(s['tr_med'])} | {pct(s['succ'])} | {pct(s['imp'])} | {pct(s['wor'])} |")
    return "\n".join(out)


def table_A3(df, model="MD-prior", ests=("C1 (초기)", "E-V2", "E-V5x2", "E-L", "E-I", "E-VI")):
    d = df[df.model == model]; out = [f"| seq | n | " + " | ".join(f"{e} 회전 중앙값 / 성공률" for e in ests) + " |", "|" + "---|" * (len(ests) + 2)]
    for sq in HO3D:
        g = d[d.seq == sq]
        if not len(g): continue
        cells = []
        for e in ests:
            x = g[g.estimator == e]; cells.append(f"{f(x.rot.median(),2)} / {pct(x.success.mean())}" if len(x) else "–")
        out.append(f"| {sq} | {len(g[g.estimator=='E-I'])} | " + " | ".join(cells) + " |")
    return "\n".join(out)


def table_occ(df, ests=("C1 (초기)", "E-V5x2", "E-L", "E-I", "E-VI")):
    edges = [0, 0.6, 0.8, 0.95, 10]; labels = ["<0.6", "[0.6,0.8)", "[0.8,0.95)", "≥0.95"]
    d = df.copy(); d["occ"] = pd.cut(d.occ_ratio, edges, labels=labels, right=False)
    out = ["| 모델 | 추정기 | 실제 마스크/렌더 실루엣 면적비 | n | 회전 중앙값° | 성공률 | 악화율 |", "|---|---|---|---|---|---|---|"]
    for m in MODELS:
        for e in ests:
            for lab in labels:
                g = d[(d.model == m) & (d.estimator == e) & (d.occ == lab)]
                if len(g): out.append(f"| {m} | {e} | {lab} | {len(g)} | {f(g.rot.median(),2)} | {pct(g.success.mean())} | {pct(g.worsened.mean())} |")
    return "\n".join(out)


def table_B(dfb):
    if not len(dfb): return "(B 미완료)"
    ests = ["E-V2", "E-V5d", "E-V5x2", "E-VI", "E-V5d_r20", "E-V5x2_r20", "E-VI_r20", "E-L", "E-I"]
    out = ["| 모델 | 추정기 | " + " | ".join(f"δ={int(x)}°" for x in sorted(dfb.delta.unique())) + " |", "|---|---|" + "---|" * dfb.delta.nunique()]
    for m in ["MD-gt", "MD-prior"]:
        for e in ests:
            cells = []
            for dl in sorted(dfb.delta.unique()):
                g = dfb[(dfb.model == m) & (dfb.estimator == e) & (dfb.delta == dl)]
                cells.append(f"{pct(g.success.mean())} ({f(g.rot.median())}°)" if len(g) else "–")
            out.append(f"| {m} | {e} | " + " | ".join(cells) + " |")
    out += ["", "시퀀스별 성공률 (E-V5x2 / E-VI / E-L / E-I):", "", "| 모델 | seq | " + " | ".join(f"δ={int(x)}°" for x in sorted(dfb.delta.unique())) + " |", "|---|---|" + "---|" * dfb.delta.nunique()]
    for m in ["MD-gt", "MD-prior"]:
        for sq in sorted(dfb.seq.unique()):
            cells = []
            for dl in sorted(dfb.delta.unique()):
                g = dfb[(dfb.model == m) & (dfb.seq == sq) & (dfb.delta == dl)]
                cells.append(" / ".join(pct(g[g.estimator == e].success.mean()) for e in ("E-V5x2", "E-VI", "E-L", "E-I")))
            out.append(f"| {m} | {sq} | " + " | ".join(cells) + " |")
    return "\n".join(out)


def table_S(df):
    sig = [("T1_rot", +1), ("T1_trans", +1), ("s_spread", +1), ("iou_P0", -1), ("icp_fitness", -1)]
    out = ["| 모델 | 추정기 | 신호 | Spearman ρ (추정 후 회전 오차) | 악화율 전체 | 하위 20 % 버림(P0 유지) 후 악화율 | 버림 후 회전 중앙값 | 버림 전 회전 중앙값 |", "|" + "---|" * 8]
    for m in MODELS:
        for e in ("E-V5d", "E-V5x2", "E-VI"):
            g = df[(df.model == m) & (df.estimator == e)]
            if not len(g): continue
            for s, sgn in sig:
                x = g[s].values; ok = np.isfinite(x)
                if ok.sum() < 20: continue
                rho = spearmanr(x[ok], g.rot.values[ok]).correlation
                bad = sgn * x; thr = np.nanpercentile(bad, 80); drop = np.isfinite(bad) & (bad >= thr)
                rot_after = np.where(drop, g.rot0.values, g.rot.values); wor_after = np.where(drop, 0, g.worsened.values)
                out.append(f"| {m} | {e} | {s} | {f(rho,2)} | {pct(g.worsened.mean())} | {pct(wor_after.mean())} | {f(np.median(rot_after),2)} | {f(g.rot.median(),2)} |")
    return "\n".join(out)


def table_cost(df):
    out = ["| 모델 | 추정기 | 렌더 s | VGGT s | LoFTR s | ICP s | 키프레임 전체 s |", "|---|---|---|---|---|---|---|"]
    for m in MODELS:
        for e in EST[1:]:
            g = df[(df.model == m) & (df.estimator == e)]
            if len(g): out.append(f"| {m} | {e} | {f(g.t_render.median(),3)} | {f(g.t_vggt.median(),3)} | {f(g.t_loftr.median(),3)} | {f(g.t_icp.median(),3)} | {f(g.t_kf.median(),2)} |")
    return "\n".join(out)


def table_D():
    fs = sorted(glob.glob(str(OUT / "D/*/stats.csv")))
    if not fs: return "(D 미완료)"
    st = pd.concat([pd.read_csv(x) for x in fs]); out = ["| seq | 구멍 비율 | 센서 유효 오차 중/p90 mm | VGGT scale 유효 | VGGT scale 구멍 | VGGT affine 유효 | VGGT affine 구멍 | affine 구멍 (conf 상위 50 %) | 센서 유효 (conf 상위 50 %) | 기준: affine top50 구멍 ≤ 2 × 센서 유효 |", "|" + "---|" * 10]
    for sq, g in st.groupby("seq"):
        man = json.load(open(OUT / "D" / sq / "manifest.json")); q = lambda s, p, sub: g[(g.source == s) & (g.pixels == p) & (g.subset == sub)]
        cell = lambda s, p, sub: (f"{f(q(s,p,sub).med_mm.iloc[0])}/{f(q(s,p,sub).p90_mm.iloc[0])}" if len(q(s, p, sub)) else "–")
        a = q("affine", "hole", "top"); sv = q("sensor", "valid", "all")
        verdict = "–" if (not len(a) or man["hole_ratio_pooled"] < 0.05) else ("충족" if a.med_mm.iloc[0] <= 2 * sv.med_mm.iloc[0] else "미충족")
        out.append(f"| {sq} | {pct(man['hole_ratio_pooled'])} | {cell('sensor','valid','all')} | {cell('scale','valid','all')} | {cell('scale','hole','all')} | {cell('affine','valid','all')} | {cell('affine','hole','all')} | {cell('affine','hole','top')} | {cell('sensor','valid','top')} | {verdict} |")
    return "\n".join(out)


if __name__ == "__main__":
    dfa = load("A"); dfb = load("B"); doc = ["# Probe 2 표 (자동 생성: experiments/vggt_probe/p2_tables.py)", ""]
    if len(dfa):
        a = with_c1(dfa)
        doc += ["## 표 A1 — 모델 × 추정기, HO3D 시퀀스 평균", "", table_A1(a), "", "## 표 A2 — C1 회전 오차 구간별 (키프레임 풀)", "", table_A2(a), "",
                "## 표 A3 — 시퀀스별 (MD-prior)", "", table_A3(a), "", "## 표 A3' — 시퀀스별 (MD-gt)", "", table_A3(a, "MD-gt"), "",
                "## 표 A-가림 — 손 가림(실제 마스크/렌더 실루엣 면적비)별", "", table_occ(a), "", "## 표 S — 신뢰 신호", "", table_S(dfa), "", "## 비용 (키프레임당 중앙값)", "", table_cost(dfa), ""]
    doc += ["## 표 B — δ별 성공률 (괄호: 회전 중앙값), AP12·MPM12·SM1 × 40 KF", "", table_B(dfb), "", "## 표 D — depth 구멍", "", table_D(), ""]
    open(LOG / "tables.md", "w").write("\n".join(doc)); print("\n".join(doc)[:3000])
