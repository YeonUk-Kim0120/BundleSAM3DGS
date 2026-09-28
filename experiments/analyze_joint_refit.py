"""EXP_BATCH_20260928 analysis (brief 3-5, 7): mode-S tables, P0/P1, factor decomposition (paired keyframe bootstrap x1000, 2 sigma,
plus the mode-S repeat noise), mode-I per-round curves and final per-keyframe d, cost table.  Missing cells are reported as '-'.
Reads outputs/exp_batch_20260928/<arm>/<ds>/<seq>/<cell>/result.json; writes logs/exp_batch_20260928/analysis/
(tables.md, verdicts.json, per_kf/*.csv, modeI_curves.png, modeI_final_kf.png)."""
from __future__ import annotations
import csv, json, sys
from pathlib import Path
import numpy as np
R = Path("/home/kist/Desktop/BundleSAM3DGS"); E = "exp_batch_20260928"; OUT = R / "outputs" / E; L = R / "logs" / E; A = L / "analysis"
sys.path.insert(0, str(R / "experiments"))
from joint_refit_eval import Evaluator, summary  # noqa: E402

SEQ = [("ho3d", "AP12", ["40", "80", "last"]), ("ycb", "mustard0", ["20", "last"]), ("ho3d", "SM1", ["40", "80", "last"]), ("ho3d", "MPM12", ["40", "80", "last"])]
ARMS = ["N", "G0", "G0B", "G1", "G3"]; STEPS = {"N": [500, 2000], "G0": [500], "G0B": [500], "G1": [500, 2000], "G3": [500, 2000]}
INP = {s: json.load(open(L / "inputs" / f"{ds}_{s}.json")) for ds, s, _ in SEQ}; EV = {s: Evaluator(INP[s]) for s in INP}
NB = 1000; RNG_SEED = 0
_O = json.load(open(L / "analysis" / "original_rounds.json")) if (L / "analysis" / "original_rounds.json").exists() else {}  # experiments/original_round_reference.py
ORIG, ORIGP = _O.get("sam2", {}), _O.get("paper", {})


def load(arm, ds, s, cell):
    f = OUT / arm / ds / s / cell / "result.json"
    return json.load(open(f)) if f.exists() else None


def s_cell(arm, ds, s, K, steps, rep=""):
    r = load(arm, ds, s, f"S_K{K}_s{steps}{rep}")
    if r is None: return None
    rd = r["rounds"][0]; rows = EV[s].compare(rd["kf_ids"], rd["poses_in"], rd["poses_out"])
    return dict(rows=rows, rd=rd, sm=summary(rows), med=lambda k: float(np.median([x[k] for x in rows])))


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)


def boot_diff(ra, rb):
    """paired keyframe bootstrap of median(d1_b) - median(d1_a) over the keyframes both cells share."""
    da = {x["kf"]: x["d1"] for x in ra}; db = {x["kf"]: x["d1"] for x in rb}; ks = [k for k in da if k in db]
    a = np.array([da[k] for k in ks]); b = np.array([db[k] for k in ks]); rng = np.random.default_rng(RNG_SEED)
    idx = rng.integers(0, len(ks), size=(NB, len(ks))); bs = np.median(b[idx], 1) - np.median(a[idx], 1)
    return float(np.median(b) - np.median(a)), float(bs.std()), len(ks)


def fmt(x, p=2): return "-" if x is None else f"{x:.{p}f}"


def main():
    A.mkdir(parents=True, exist_ok=True); md = ["# EXP_BATCH_20260928 분석 표 (자동 생성: experiments/analyze_joint_refit.py)", ""]
    ver = {}
    # ---------------- mode S ----------------
    cells = {}
    for arm in ARMS:
        for steps in STEPS[arm]:
            for ds, s, Ks in SEQ:
                for K in Ks:
                    c = s_cell(arm, ds, s, K, steps); cells[(arm, steps, s, K)] = c
                    if c: write_csv(A / "per_kf" / f"{arm}_{s}_S_K{K}_s{steps}.csv", c["rows"])
    md += ["## 모드 S — arm × 시점 × 시퀀스 (한 라운드; d = GT 메시 점 평균 변위 mm, 개선/악화 = d 변화 ≥ max(0.5 mm, 10 %))", ""]
    md += ["| arm | 스텝 | 시퀀스 | K (n) | 개선 % | 악화 % | d 중앙값 전→후 (감소율) | d p90 전→후 | 회전 중앙값 ° 전→후 | 이동 중앙값 mm 전→후 | 시간 s | 메모리 GB | Gaussian 전→후 |",
           "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for arm in ARMS:
        for steps in STEPS[arm]:
            for ds, s, Ks in SEQ:
                for K in Ks:
                    c = cells[(arm, steps, s, K)]
                    if c is None: md.append(f"| {arm} | {steps} | {s} | {K} | - | - | - | - | - | - | - | - | - |"); continue
                    sm, rd = c["sm"], c["rd"]; g = f"{rd.get('gaussians_before', '-')}→{rd.get('gaussians_after', '-')}" if arm != "N" else "-"
                    md.append(f"| {arm} | {steps} | {s} | {K} ({rd['n']}) | {100*sm['imp']:.0f} | {100*sm['wor']:.0f} | {sm['d0_med']:.2f}→{sm['d1_med']:.2f} ({100*sm['d_reduction_frac']:+.0f} %) | "
                              f"{sm['d0_p90']:.2f}→{sm['d1_p90']:.2f} | {c['med']('rot0'):.2f}→{c['med']('rot1'):.2f} | {c['med']('trans0'):.1f}→{c['med']('trans1'):.1f} | {rd['seconds']:.0f} | {rd['peak_mem_gb']:.1f} | {g} |")
    md += ["", "### 모드 S 요약 행렬 (500 스텝, 칸 = 라운드 후 d 중앙값 mm (감소율) / 개선 %·악화 %)", "", "| 시퀀스 | K | 전 | " + " | ".join(ARMS) + " |", "|---|---|---|" + "---|" * len(ARMS)]
    for ds, s, Ks in SEQ:
        for K in Ks:
            base = next((cells[(a, 500, s, K)] for a in ARMS if cells[(a, 500, s, K)]), None)
            row = f"| {s} | {K} | {fmt(base['sm']['d0_med']) if base else '-'} |"
            for arm in ARMS:
                c = cells[(arm, 500, s, K)]
                row += " - |" if c is None else f" {c['sm']['d1_med']:.2f} ({100*c['sm']['d_reduction_frac']:+.0f} %) / {100*c['sm']['imp']:.0f}·{100*c['sm']['wor']:.0f} |"
            md.append(row)
    # repeat noise
    md += ["", "### 모드 S 반복 잡음 (AP12 마지막 시점, 500 스텝)", "", "| arm | d 중앙값 원 / 반복 | 차이 mm | 키프레임별 |Δd| 중앙값 mm | 시간 s 원 / 반복 |", "|---|---|---|---|---|"]
    rep_noise = {}
    for arm in ("N", "G1"):
        a = cells[(arm, 500, "AP12", "last")]; b = s_cell(arm, "ho3d", "AP12", "last", 500, "_rep1")
        if a and b:
            da = {x["kf"]: x["d1"] for x in a["rows"]}; per = np.median([abs(x["d1"] - da[x["kf"]]) for x in b["rows"] if x["kf"] in da])
            rep_noise[arm] = abs(b["sm"]["d1_med"] - a["sm"]["d1_med"])
            md.append(f"| {arm} | {a['sm']['d1_med']:.2f} / {b['sm']['d1_med']:.2f} | {rep_noise[arm]:.2f} | {per:.2f} | {a['rd']['seconds']:.0f} / {b['rd']['seconds']:.0f} |")
        else: md.append(f"| {arm} | - | - | - | - |")
    ver["repeat_noise_mm"] = rep_noise
    # ---------------- mode I ----------------
    mi = {}; md += ["", "## 모드 I — 증분 재생 (첫 라운드 5개, 이후 새 키프레임 5개마다; 대조 noop = 소비 시점 포즈)", ""]
    for ds, s, _ in SEQ:
        ids = INP[s]["kf_ids"]; cons = [np.array(p) for p in INP[s]["consume"]]
        for arm in ("N", "G1", "G3"):
            r = load(arm, ds, s, "I_s500")
            if r is None: continue
            curve = []
            for rd in r["rounds"]:
                n = rd["n"]; rows = EV[s].compare(ids[:n], rd["poses_in"], rd["poses_out"]); rn = EV[s].compare(ids[:n], cons[:n], cons[:n])
                curve.append(dict(n=n, in_med=float(np.median([x["d0"] for x in rows])), out_med=float(np.median([x["d1"] for x in rows])), out_p90=float(np.percentile([x["d1"] for x in rows], 90)),
                                  noop_med=float(np.median([x["d0"] for x in rn])), noop_p90=float(np.percentile([x["d0"] for x in rn], 90)), seconds=rd["seconds"], peak=rd.get("peak_mem_gb")))
            # per-round split: the round's new keyframes (chained initial poses) vs keyframes already in the set (previous round's output)
            split = dict(new_in=[], new_out=[], old_in=[], old_out=[], new_move=[], old_move=[]); prev_n = 0; P = EV[s].p
            for rd in r["rounds"]:
                n = rd["n"]; rows = EV[s].compare(ids[:n], rd["poses_in"], rd["poses_out"])
                for x in rows:
                    grp = "new" if ids.index(x["kf"]) >= prev_n else "old"; split[f"{grp}_in"].append(x["d0"]); split[f"{grp}_out"].append(x["d1"])
                for i in range(n):  # pose movement within the round (mean GT-mesh point displacement between input and output pose, mm)
                    Aio = np.asarray(rd["poses_out"][i]) @ np.linalg.inv(np.asarray(rd["poses_in"][i]))
                    split["new_move" if i >= prev_n else "old_move"].append(float(np.linalg.norm(P @ Aio[:3, :3].T + Aio[:3, 3] - P, axis=1).mean() * 1000))
                prev_n = n
            last = r["rounds"][-1]; rows = EV[s].compare(ids, cons, last["poses_out"])  # d0 = noop, d1 = arm final
            write_csv(A / "per_kf" / f"{arm}_{s}_I_s500_final.csv", rows)
            mi[(arm, s)] = dict(curve=curve, rows=rows, sm=summary(rows), total_s=float(sum(x["seconds"] for x in curve)), rounds=len(curve), split=split)
    md += ["| arm | 시퀀스 | 라운드 | 최종 d 중앙값 noop → arm (비율) | p90 noop → arm | noop 대비 개선 % · 악화 % | 참고: 원본 온라인 SAM2 / 논문 마스크 최종 d 중앙값 (자기 키프레임) | 총 시간 min | 라운드 평균 s | 최대 메모리 GB |", "|---|---|---|---|---|---|---|---|---|---|"]
    for ds, s, _ in SEQ:
        for arm in ("N", "G1", "G3"):
            m = mi.get((arm, s))
            if m is None: continue
            sm = m["sm"]; pk = max((x["peak"] or 0) for x in m["curve"]); o, op = ORIG.get(s), ORIGP.get(s); ref = f"{o[-1]['after_med']:.2f} / {op[-1]['after_med']:.2f}" if (o and op) else (f"{o[-1]['after_med']:.2f} / -" if o else "-")
            md.append(f"| {arm} | {s} | {m['rounds']} | {sm['d0_med']:.2f} → {sm['d1_med']:.2f} ({sm['d1_med']/sm['d0_med']:.2f}×) | {sm['d0_p90']:.2f} → {sm['d1_p90']:.2f} | {100*sm['imp']:.0f} · {100*sm['wor']:.0f} | {ref} | "
                      f"{m['total_s']/60:.1f} | {m['total_s']/m['rounds']:.1f} | {pk:.1f} |")
    md += ["", "라운드별 d 중앙값 곡선: `modeI_curves.png`, 최종 키프레임별 d: `modeI_final_kf.png`, 라운드별 수치: `verdicts.json`의 `modeI`.", "",
           "### 모드 I 라운드 분해 — 그 라운드에 새로 들어온 키프레임(체인 초기 포즈) vs 이미 있던 키프레임(직전 라운드 출력)", "",
           "| arm | 시퀀스 | 새 키프레임: 라운드 전 → 후 d 중앙값 (라운드 안 감소 비율) | 기존 키프레임: 전 → 후 d 중앙값 (감소 비율) | 라운드 안 포즈 이동량 중앙값 mm 새 / 기존 |", "|---|---|---|---|---|"]
    for ds, s, _ in SEQ:
        for arm in ("N", "G1", "G3"):
            m = mi.get((arm, s))
            if m is None: continue
            sp = {k: np.asarray(v) for k, v in m["split"].items()}
            fr = lambda a, b: f"{100*np.mean(b < a):.0f} %" if len(a) else "-"
            md.append(f"| {arm} | {s} | {np.median(sp['new_in']):.2f} → {np.median(sp['new_out']):.2f} ({fr(sp['new_in'], sp['new_out'])}) | "
                      + (f"{np.median(sp['old_in']):.2f} → {np.median(sp['old_out']):.2f} ({fr(sp['old_in'], sp['old_out'])}) |" if len(sp['old_in']) else "- |")
                      + f" {np.median(sp['new_move']):.2f} / {np.median(sp['old_move']) if len(sp['old_move']) else float('nan'):.2f} |")
    md += ["", "(새 키프레임의 '라운드 전' = 체인 초기 포즈 c2w_corr(prev)·inv(c2w_trk(prev))·c2w_trk(new); 라운드마다 표본을 모두 모은 중앙값, 감소 비율 = 라운드 후 d가 전보다 작은 표본 비율)", ""]
    ver["modeI"] = {f"{a}/{s}": dict(curve=m["curve"], final=m["sm"], total_s=m["total_s"]) for (a, s), m in mi.items()}
    # ---------------- P0 ----------------
    p0i = []
    for ds, s, _ in SEQ:
        c = cells[("N", 500, s, "last")]
        if c: p0i.append(dict(seq=s, red=c["sm"]["d_reduction_frac"], wor=c["sm"]["wor"], ok=bool(c["sm"]["d_reduction_frac"] >= 0.20 and c["sm"]["wor"] <= 0.20)))
    p0ii = []
    for ds, s, _ in SEQ:
        m = mi.get(("N", s))
        if m: p0ii.append(dict(seq=s, ratio=m["sm"]["d1_med"] / m["sm"]["d0_med"], ok=bool(m["sm"]["d1_med"] <= 0.5 * m["sm"]["d0_med"])))
    pass_i = sum(x["ok"] for x in p0i) >= 3; pass_ii = sum(x["ok"] for x in p0ii) >= 2 and any(x["ok"] and x["seq"] == "SM1" for x in p0ii)
    ver["P0"] = dict(i=p0i, ii=p0ii, pass_i=pass_i, pass_ii=pass_ii, passed=pass_i or pass_ii, complete=len(p0i) == 4 and len(p0ii) == 4)
    md += ["## 판정 P0 (재생이 원본 효과를 재현하는가)", "",
           "(i) 모드 S 마지막 시점(500 스텝) N: d 중앙값 감소 ≥ 20 % 그리고 악화률 ≤ 20 %인 시퀀스 ≥ 3/4: " + ", ".join(f"{x['seq']} {100*x['red']:+.0f} %·악화 {100*x['wor']:.0f} % {'○' if x['ok'] else '×'}" for x in p0i) + f" → {'통과' if pass_i else '실패'}",
           "", "(ii) 모드 I N 최종 d 중앙값 ≤ noop의 0.5배인 시퀀스 ≥ 2 (SM1 포함): " + ", ".join(f"{x['seq']} {x['ratio']:.2f}× {'○' if x['ok'] else '×'}" for x in p0ii) + f" → {'통과' if pass_ii else '실패'}",
           "", f"**P0 {'통과' if (pass_i or pass_ii) else '실패'}**" + ("" if ver["P0"]["complete"] else " (셀 일부 미완)"), ""]
    # ---------------- P1 ----------------
    best = json.load(open(L / "best_g.json")) if (L / "best_g.json").exists() else None; ver["best_g"] = best["best"] if best else None
    if best:
        g = best["best"]; per = []
        for ds, s, _ in SEQ:
            cn, cg = cells[("N", 500, s, "last")], cells[(g, 500, s, "last")]
            if cn and cg:
                rn = cn["sm"]["d0_med"] - cn["sm"]["d1_med"]; rg = cg["sm"]["d0_med"] - cg["sm"]["d1_med"]
                per.append(dict(seq=s, n_red_mm=rn, g_red_mm=rg, ok=bool(rn > 0 and rg >= 0.5 * rn), n_wor=cn["sm"]["wor"], g_wor=cg["sm"]["wor"]))
        wor_ok = bool(per) and np.mean([x["g_wor"] for x in per]) <= np.mean([x["n_wor"] for x in per]) + 0.10
        mn, mg = mi.get(("N", "SM1")), mi.get((g, "SM1")); sm1_ok = bool(mn and mg and mg["sm"]["d1_med"] <= 1.5 * mn["sm"]["d1_med"])
        p1 = sum(x["ok"] for x in per) >= 3 and wor_ok and sm1_ok
        ver["P1"] = dict(arm=g, per_seq=per, worsened_ok=wor_ok, sm1_ok=sm1_ok, sm1=[mn["sm"]["d1_med"] if mn else None, mg["sm"]["d1_med"] if mg else None], passed=p1, applicable=ver["P0"]["passed"])
        md += ["## 판정 P1 (GS가 재현하는가) — P0 통과 시에만 효력", "", f"best G = {g} (규칙: `logs/exp_batch_20260928/pick_best_g.py`, 점수 G1 {best['score']['G1']['mean_reduction']:+.3f}, G3 {best['score']['G3']['mean_reduction']:+.3f})", "",
               "| 시퀀스 | N 감소 mm | G 감소 mm | G ≥ 0.5·N (N>0) | 악화 N / G |", "|---|---|---|---|---|"]
        md += [f"| {x['seq']} | {x['n_red_mm']:+.2f} | {x['g_red_mm']:+.2f} | {'○' if x['ok'] else '×'} | {100*x['n_wor']:.0f} / {100*x['g_wor']:.0f} |" for x in per]
        md += ["", f"악화률 평균 G ≤ N + 10 pt: {'○' if wor_ok else '×'}; 모드 I SM1 최종 d G ≤ 1.5·N: {fmt(ver['P1']['sm1'][1])} vs {fmt(ver['P1']['sm1'][0])} mm {'○' if sm1_ok else '×'}",
               "", f"**P1 {'통과' if p1 else '실패'}**" + ("" if ver["P0"]["passed"] else " (P0 실패 → 판정 효력 없음, 표만 남김)"), ""]
    # ---------------- factor decomposition ----------------
    md += ["## 요인 분해 (500 스텝; Δ = 뒤 arm의 라운드 후 d 중앙값 − 앞 arm의 것, 음수 = 뒤 arm이 GT에 더 가까움; σ = 짝지은 키프레임 부트스트랩 1,000회)", "",
           "| 비교 (요인) | 시퀀스 | K | Δ mm | σ mm | 판정 (|Δ| > 2σ) |", "|---|---|---|---|---|---|"]
    fac = {}
    for a, b, what in (("G0", "G0B", "배치 구성"), ("G0B", "G1", "새 단일 층 지도(prior)"), ("G0B", "G3", "새 단일 층 지도(TSDF)"), ("G1", "G3", "prior 유무"), ("N", "G1", "N 대비"), ("N", "G3", "N 대비")):
        res = []
        for ds, s, Ks in SEQ:
            for K in Ks:
                ca, cb = cells[(a, 500, s, K)], cells[(b, 500, s, K)]
                if not (ca and cb): continue
                d, sd, n = boot_diff(ca["rows"], cb["rows"]); v = "뒤가 나음" if d < -2 * sd else ("앞이 나음" if d > 2 * sd else "잡음 이내")
                res.append(dict(seq=s, K=K, delta=d, sigma=sd, n=n, verdict=v)); md.append(f"| {a} → {b} ({what}) | {s} | {K} | {d:+.2f} | {sd:.2f} | {v} |")
        fac[f"{a}->{b}"] = dict(what=what, cells=res, later_better=sum(x["verdict"] == "뒤가 나음" for x in res), earlier_better=sum(x["verdict"] == "앞이 나음" for x in res),
                                noise=sum(x["verdict"] == "잡음 이내" for x in res), mean_delta=float(np.mean([x["delta"] for x in res])) if res else None)
    md += ["", "| 비교 | 셀 수 | 뒤가 나음 | 앞이 나음 | 잡음 이내 | Δ 평균 mm |", "|---|---|---|---|---|---|"]
    md += [f"| {k} ({v['what']}) | {len(v['cells'])} | {v['later_better']} | {v['earlier_better']} | {v['noise']} | {fmt(v['mean_delta'])} |" for k, v in fac.items()]
    md += ["", f"모드 S 반복 잡음(AP12 마지막 시점 d 중앙값 차이): " + ", ".join(f"{k} {v:.2f} mm" for k, v in rep_noise.items()), ""]
    ver["factors"] = fac
    # compact matrix: rows = cells, columns = comparisons; entry = delta mm, (뒤)/(앞) when |delta| > 2 sigma
    keys = list(fac.keys()); md += ["### 요인 분해 압축 행렬 (칸 = Δ mm; '뒤' = 뒤 arm이 2σ 밖으로 나음, '앞' = 앞 arm이 나음, 표시 없음 = 잡음 이내)", "",
                                    "| 시퀀스 | K | " + " | ".join(keys) + " |", "|---|---|" + "---|" * len(keys)]
    for ds, s, Ks in SEQ:
        for K in Ks:
            row = f"| {s} | {K} |"
            for k in keys:
                c = next((x for x in fac[k]["cells"] if x["seq"] == s and x["K"] == K), None)
                row += " - |" if c is None else f" {c['delta']:+.2f}{' (뒤)' if c['verdict'] == '뒤가 나음' else (' (앞)' if c['verdict'] == '앞이 나음' else '')} |"
            md.append(row)
    md += ["", "### 스텝 수 효과 (마지막 시점, 라운드 후 d 중앙값 mm: 500 → 2000, 괄호 = 라운드 전 대비 감소율)", "", "| arm | " + " | ".join(s for _, s, _ in SEQ) + " |", "|---|" + "---|" * len(SEQ)]
    for arm in ("N", "G1", "G3"):
        row = f"| {arm} |"
        for _, s, _ in SEQ:
            a, b = cells[(arm, 500, s, "last")], cells[(arm, 2000, s, "last")]
            row += " - |" if not (a and b) else f" {a['sm']['d1_med']:.2f} ({100*a['sm']['d_reduction_frac']:+.0f} %) → {b['sm']['d1_med']:.2f} ({100*b['sm']['d_reduction_frac']:+.0f} %) |"
        md.append(row)
    md.append("")
    # ---------------- cost ----------------
    md += ["## 비용 (모드 S 500 스텝 마지막 시점의 라운드 시간 s / 최대 메모리 GB; (b)/(c) 결정용)", "", "| arm | " + " | ".join(s for _, s, _ in SEQ) + " |", "|---|" + "---|" * len(SEQ)]
    for arm in ARMS:
        md.append(f"| {arm} | " + " | ".join((lambda c: f"{c['rd']['seconds']:.0f} / {c['rd']['peak_mem_gb']:.1f}" if c else "-")(cells[(arm, 500, s, 'last')]) for _, s, _ in SEQ) + " |")
    (A / "tables.md").write_text("\n".join(md) + "\n"); json.dump(ver, open(A / "verdicts.json", "w"), indent=1, default=float)
    # ---------------- figures ----------------
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        col = {"noop": "k", "N": "tab:blue", "G1": "tab:orange", "G3": "tab:green"}
        fig, ax = plt.subplots(2, 2, figsize=(12, 8))
        for i, (ds, s, _) in enumerate(SEQ):
            a = ax.flat[i]; drawn = False
            for arm in ("N", "G1", "G3"):
                m = mi.get((arm, s))
                if not m: continue
                n = [x["n"] for x in m["curve"]]
                if not drawn:
                    a.plot(n, [x["noop_med"] for x in m["curve"]], color=col["noop"], label="noop median"); a.plot(n, [x["noop_p90"] for x in m["curve"]], color=col["noop"], ls="--", lw=0.8, label="noop p90"); drawn = True
                a.plot(n, [x["out_med"] for x in m["curve"]], color=col[arm], label=f"{arm} median"); a.plot(n, [x["out_p90"] for x in m["curve"]], color=col[arm], ls="--", lw=0.8, label=f"{arm} p90")
            for oo, lab, ls in ((ORIG.get(s), "original online SAM2", "-"), (ORIGP.get(s), "original online paper masks", ":")):
                if oo: a.plot([x["n"] for x in oo], [x["after_med"] for x in oo], color="0.55", lw=1.2, ls=ls, label=f"{lab} median (own keyframes)"); drawn = True
            a.set_title(f"{s}: d over keyframes in the set, after each round"); a.set_xlabel("keyframes in the round"); a.set_ylabel("d (mm)"); a.grid(alpha=0.3)
            if drawn: a.legend(fontsize=7)
        fig.tight_layout(); fig.savefig(A / "modeI_curves.png", dpi=110); plt.close(fig)
        fig, ax = plt.subplots(2, 2, figsize=(12, 8))
        for i, (ds, s, _) in enumerate(SEQ):
            a = ax.flat[i]; drawn = False
            for arm in ("N", "G1", "G3"):
                m = mi.get((arm, s))
                if not m: continue
                x = np.arange(len(m["rows"]))
                if not drawn: a.plot(x, [r["d0"] for r in m["rows"]], color=col["noop"], lw=0.9, label="noop"); drawn = True
                a.plot(x, [r["d1"] for r in m["rows"]], color=col[arm], lw=0.9, label=arm)
            a.set_title(f"{s}: final d per keyframe (mode I)"); a.set_xlabel("keyframe index (with GT)"); a.set_ylabel("d (mm)"); a.grid(alpha=0.3)
            if drawn: a.legend(fontsize=7)
        fig.tight_layout(); fig.savefig(A / "modeI_final_kf.png", dpi=110); plt.close(fig)
    except Exception as e:  # figures are optional
        print("figure error:", e)
    print((A / "tables.md").read_text()[:200]); print("P0", ver["P0"]["passed"], "complete", ver["P0"]["complete"])


if __name__ == "__main__":
    main()
