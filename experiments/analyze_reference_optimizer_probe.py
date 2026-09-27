"""EXP_BATCH_20260927 stage-1 analysis: acceptance margin m (3.5), improvement / worsening rates, premise check and pass rule (3.6).
Input: outputs/exp_batch_20260927/probe_<REF>/<ds>/<seq>/rows.json.  Output: logs/exp_batch_20260927/stage1_tables.md + stage1_verdict.json."""
from __future__ import annotations
import glob, json, math, sys
from pathlib import Path
import numpy as np, pandas as pd
REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "outputs/exp_batch_20260927"; LOG = REPO / "logs/exp_batch_20260927"
TUNE = ["AP12", "mustard0"]; VALID = ["SM1", "MPM12"]; MS = [0.0, 0.05, 0.1, 0.2, 0.3]
REFS = ["R0", "A1", "A2", "A3", "RP", "RG"]; OPTS = ["O1", "O2", "O3", "O4"]; START_KIND = {"tracker": "(i) 트래커", "gt": "(ii) GT", "gt3_0": "(iii) GT+3°", "gt3_1": "(iii) GT+3°", "gt10_0": "(iv) GT+10°", "gt10_1": "(iv) GT+10°"}


def load():
    fs = sorted(glob.glob(str(OUT / "probe_*/*/*/rows.json"))); dfs = []
    for f in fs:
        rows = json.load(open(f))
        if rows: dfs.append(pd.DataFrame(rows).drop(columns=["est"], errors="ignore"))
    d = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()
    if len(d): d["kind"] = d.start.map(START_KIND)
    return d


def apply_m(g, m):
    rho = np.where(g.c0 > 0, (g.c0 - g.c1) / g.c0, 0.0); acc = rho >= m
    d = np.where(acc, g.d1, g.d0); thr = np.maximum(0.5, 0.1 * g.d0)
    return d, (g.d0 - d) >= thr, (d - g.d0) >= thr, acc


def stats(g, m):
    if not len(g): return None
    d, imp, wor, acc = apply_m(g, m)
    return dict(n=len(g), imp=float(imp.mean()), wor=float(wor.mean()), net=float(imp.mean() - wor.mean()), d0_med=float(np.median(g.d0)), d_med=float(np.median(d)),
                d0_p90=float(np.percentile(g.d0, 90)), d_p90=float(np.percentile(d, 90)), accepted=float(acc.mean()))


def pct(x): return "–" if x is None or not np.isfinite(x) else f"{100 * x:.0f} %"
def f(x, p=2): return "–" if x is None or not np.isfinite(x) else f"{x:.{p}f}"


if __name__ == "__main__":
    d = load()
    if not len(d): print("no rows"); sys.exit(0)
    lines = ["# EXP_BATCH_20260927 단계 1 표 (자동 생성: experiments/analyze_reference_optimizer_probe.py)", "",
             "개선 = d 감소 ≥ max(0.5 mm, 10 %), 악화 = d 증가 ≥ max(0.5 mm, 10 %). d = GT 메시 점 평균 변위(mm). 판정 규칙: ρ = (c0 − c1)/c0 ≥ m이면 결과 채택, 아니면 시작 포즈 유지.", ""]
    verdict = {"m": {}, "validation": {}, "premise": {}, "pass": [], "readouts": {}}
    # --- m selection on tuning sequences (tracker-pose probes): mean over the two sequences of (imp - wor)
    lines += ["## 1. 여유값 m 선택 (조정용 AP12·mustard0, 트래커 포즈 probe; 두 시퀀스 (개선률 − 악화률)의 평균)", "", "| 기준 | 최적화기 | " + " | ".join(f"m={m}" for m in MS) + " | 선택 m |", "|---|---|" + "---|" * (len(MS) + 1)]
    for ref in REFS:
        for opt in OPTS:
            g = d[(d.reference == ref) & (d.optimizer == opt) & (d.start == "tracker")]
            if not len(g): continue
            nets = []
            for m in MS:
                v = [stats(g[g.seq == s], m) for s in TUNE if (g.seq == s).any()]
                nets.append(np.mean([x["net"] for x in v]) if v else np.nan)
            best = MS[int(np.nanargmax(nets))] if np.isfinite(nets).any() else 0.0; verdict["m"][f"{ref}/{opt}"] = best
            lines.append(f"| {ref} | {opt} | " + " | ".join(f"{100 * x:+.0f}" if np.isfinite(x) else "–" for x in nets) + f" | {best} |")
    # --- full tables per start kind with the chosen m, tuning / validation separated
    for group, seqs in (("조정용", TUNE), ("검증용", VALID)):
        lines += ["", f"## 2. {group} ({', '.join(seqs)}) — 선택된 m 적용, probe 종류별", "", "| 기준 | 최적화기 | m | probe | seq | n | 개선률 | 악화률 | 개선−악화 | d 중앙값 전 → 후 (mm) | d p90 전 → 후 | 채택률 |", "|" + "---|" * 12]
        for ref in REFS:
            for opt in OPTS:
                m = verdict["m"].get(f"{ref}/{opt}", 0.0)
                for kind in ["(i) 트래커", "(ii) GT", "(iii) GT+3°", "(iv) GT+10°"]:
                    for s in seqs:
                        g = d[(d.reference == ref) & (d.optimizer == opt) & (d.kind == kind) & (d.seq == s)]
                        st = stats(g, m)
                        if st: lines.append(f"| {ref} | {opt} | {m} | {kind} | {s} | {st['n']} | {pct(st['imp'])} | {pct(st['wor'])} | {100 * st['net']:+.0f} | {f(st['d0_med'])} → {f(st['d_med'])} | {f(st['d0_p90'])} → {f(st['d_p90'])} | {pct(st['accepted'])} |")
    # --- raw (m = 0) optimizer behaviour incl. H5 drift at GT and O3 extras
    lines += ["", "## 3. 원 결과 (m = 0, 시퀀스 풀) — (ii) GT 시작의 드리프트, 회전·이동 변화", "", "| 기준 | 최적화기 | probe | n | d 후 중앙값 (mm) | 회전 후 중앙값 (°) | 이동 후 중앙값 (mm) | 개선률 | 악화률 | 시간 중앙값 (s) |", "|" + "---|" * 10]
    for ref in REFS:
        for opt in OPTS:
            for kind in ["(i) 트래커", "(ii) GT", "(iii) GT+3°", "(iv) GT+10°"]:
                g = d[(d.reference == ref) & (d.optimizer == opt) & (d.kind == kind)]
                if not len(g): continue
                st = stats(g, 0.0)
                lines.append(f"| {ref} | {opt} | {kind} | {len(g)} | {f(np.median(g.d1))} | {f(np.median(g.rot1))} | {f(np.median(g.trans1), 1)} | {pct(st['imp'])} | {pct(st['wor'])} | {f(np.median(g.time_s))} |")
    # --- premise (3.6): RG + O3 improves >= 80 % of GT+3 probes and (ii) d median < 0.3 mm
    g3 = d[(d.reference == "RG") & (d.optimizer == "O3") & (d.kind == "(iii) GT+3°")]; g0 = d[(d.reference == "RG") & (d.optimizer == "O3") & (d.kind == "(ii) GT")]
    if len(g3) and len(g0):
        imp3 = stats(g3, 0.0)["imp"]; dr = float(np.median(g0.d1)); ok = imp3 >= 0.8 and dr < 0.3
        verdict["premise"] = dict(RG_O3_gt3_improved=imp3, RG_O3_gt_drift_d_median_mm=dr, ok=ok)
        lines += ["", f"## 4. 전제 확인: RG + O3 GT+3° 개선률 {pct(imp3)} (기준 ≥ 80 %), (ii) GT 시작 d 중앙값 {dr:.3f} mm (기준 < 0.3) → {'통과' if ok else '실패 — 단계 2 금지'}"]
    # --- pass rule on validation sequences (non-RG references)
    lines += ["", "## 5. 통과 조건 (검증용 SM1·MPM12, 트래커 포즈 probe, 선택 m): 두 시퀀스 모두 개선−악화 ≥ +10 pt, 악화 ≤ 15 %, SM1 d 중앙값 감소", "",
              "| 기준 | 최적화기 | m | SM1 개선/악화/순 | SM1 d 전→후 | MPM12 개선/악화/순 | MPM12 d 전→후 | 통과 |", "|---|---|---|---|---|---|---|---|"]
    for ref in [r for r in REFS if r != "RG"] + ["RG"]:
        for opt in OPTS:
            m = verdict["m"].get(f"{ref}/{opt}", 0.0); g = d[(d.reference == ref) & (d.optimizer == opt) & (d.start == "tracker")]
            v = {s: stats(g[g.seq == s], m) for s in VALID}
            if not all(v.values()): continue
            ok = all(x["net"] >= 0.10 and x["wor"] <= 0.15 for x in v.values()) and v["SM1"]["d_med"] < v["SM1"]["d0_med"]
            verdict["validation"][f"{ref}/{opt}"] = dict(m=m, **{s: x for s, x in v.items()}, passed=ok)
            if ok and ref != "RG": verdict["pass"].append(dict(combo=f"{ref}/{opt}", m=m, mean_d_reduction=float(np.mean([x["d0_med"] - x["d_med"] for x in v.values()]))))
            lines.append(f"| {ref} | {opt} | {m} | {pct(v['SM1']['imp'])}/{pct(v['SM1']['wor'])}/{100 * v['SM1']['net']:+.0f} | {f(v['SM1']['d0_med'])}→{f(v['SM1']['d_med'])} | "
                         f"{pct(v['MPM12']['imp'])}/{pct(v['MPM12']['wor'])}/{100 * v['MPM12']['net']:+.0f} | {f(v['MPM12']['d0_med'])}→{f(v['MPM12']['d_med'])} | {'통과' if ok else '–'}{' (RG, 후보 아님)' if ref == 'RG' and ok else ''} |")
    verdict["pass"] = sorted(verdict["pass"], key=lambda x: -x["mean_d_reduction"])
    # --- 6. R0 gradient-direction cosines at the tracker pose (compare with the earlier gradient probe; that one used v1-feedback poses and 4000 init steps)
    lines += ["", "## 6. R0 기울기 방향 (트래커 포즈, probe_view): cos(−∇, 참 보정) > 0 비율과 평균", "", "| seq | n | total 회전 >0 / 평균 | total 이동 >0 / 평균 | depth 회전 >0 | l1 회전 >0 | dssim 회전 >0 |", "|---|---|---|---|---|---|---|"]
    for cf in sorted(glob.glob(str(OUT / "probe_R0/*/*/cos_rows.json"))):
        c = pd.DataFrame(json.load(open(cf)))
        if not len(c): continue
        sq = Path(cf).parent.name; fr = lambda k: (np.nanmean(np.array(c[k], dtype=float) > 0), np.nanmean(np.array(c[k], dtype=float))) if k in c else (np.nan, np.nan)
        t_r, t_t = fr("total_cos_rot"), fr("total_cos_trans")
        lines.append(f"| {sq} | {len(c)} | {pct(t_r[0])} / {f(t_r[1])} | {pct(t_t[0])} / {f(t_t[1])} | {pct(fr('depth_cos_rot')[0])} | {pct(fr('l1_cos_rot')[0])} | {pct(fr('dssim_cos_rot')[0])} |")
    # --- 7. hygiene statistics (A1..A3)
    lines += ["", "## 7. A 계열 후보·승격·폐기 통계 (재생 전체 합계)", "", "| 기준 | seq | 사이클 | 출생 후보 | 승격 | 폐기 (never_good / few_obs / streak / spread / angle) | 승격률 (승격/(승격+폐기)) | 최종 Gaussian | 학습 제외 뷰 (A3, 마지막 사이클) |", "|" + "---|" * 9]
    for hf in sorted(glob.glob(str(OUT / "probe_A*/*/*/hygiene_log.json"))):
        h = json.load(open(hf))
        if not h: continue
        ref = Path(hf).parts[-4].replace("probe_", ""); sq = Path(hf).parent.name
        born = sum(x["born"] for x in h); prom = sum(x["promoted"] for x in h); disc = {k: sum(x["discarded"][k] for x in h) for k in h[0]["discarded"]}
        dt = sum(disc.values()); last = h[-1]
        lines.append(f"| {ref} | {sq} | {len(h)} | {born} | {prom} | {dt} ({' / '.join(str(disc[k]) for k in ('never_good', 'few_obs', 'streak', 'spread', 'angle'))}) | {pct(prom / max(prom + dt, 1))} | {last['gaussians']} | "
                     f"{last['views'] - last['views_active'] if ref == 'A3' else '–'} / {last['views']} |")
    # --- 8. readout guide (brief 3.6), automatic checks
    def se_diff(a, b): return 2 * math.sqrt(max(a["imp"] * (1 - a["imp"]) / a["n"] + b["imp"] * (1 - b["imp"]) / b["n"], 1e-12))
    rd = []
    non_rg = [p_ for p_ in verdict["pass"]]
    rg_pass = any(v.get("passed") for k, v in verdict["validation"].items() if k.startswith("RG/"))
    if not non_rg and rg_pass: rd.append("RG만 통과 → \"쓸 수 있는 기준 없음 = (a) 편향이 원인으로 확정\"")
    if not non_rg and not rg_pass: rd.append("RG도 통과하지 못함 → 통과 조합 없음(RG 포함). (a) 판정 문장은 RG 통과를 전제로 하므로 그대로 적용하지 않고 수치로 보고")
    if any(p_["combo"].startswith("RP/") for p_ in non_rg): rd.append("RP 통과 → \"prior 판사 피드백 성립 후보\"")
    for opt in OPTS:
        base = d[(d.reference == "R0") & (d.optimizer == opt) & (d.start == "tracker") & d.seq.isin(VALID)]
        for ref in ("A1", "A2", "A3"):
            g = d[(d.reference == ref) & (d.optimizer == opt) & (d.start == "tracker") & d.seq.isin(VALID)]
            if len(base) and len(g):
                sb, sg = stats(base, verdict["m"].get(f"R0/{opt}", 0.0)), stats(g, verdict["m"].get(f"{ref}/{opt}", 0.0))
                if sg["imp"] - sb["imp"] > se_diff(sg, sb): rd.append(f"{ref}/{opt} 개선률 {pct(sg['imp'])} vs R0 {pct(sb['imp'])} (검증용, 잡음 2σ 밖) → \"위생 채택 후보\"")
    for ref in REFS:
        a3, a4 = d[(d.reference == ref) & (d.optimizer == "O3") & (d.start == "tracker")], d[(d.reference == ref) & (d.optimizer == "O4") & (d.start == "tracker")]
        if len(a3) and len(a4):
            s3, s4 = stats(a3, 0.0), stats(a4, 0.0); close = abs(s3["net"] - s4["net"]) <= se_diff(s3, s4)
            verdict_txt = 'O3 ≈ O4 ("GN 이득은 최적화기가 아니라 목적함수 덕")' if close else 'O3 ≠ O4'
            rd.append(f"{ref}: O3 순효과 {100 * s3['net']:+.0f} pt vs O4 {100 * s4['net']:+.0f} pt → {verdict_txt}")
    g_rg = d[(d.reference == "RG") & (d.start == "gt")]
    if len(g_rg):
        dr = {o: float(np.median(g_rg[g_rg.optimizer == o].d1)) for o in OPTS if (g_rg.optimizer == o).any()}
        msg = ", ".join(f"{o} {v:.2f} mm" for o, v in dr.items())
        better = [o for o in ("O2", "O3") if o in dr and "O1" in dr and dr[o] < dr["O1"]]
        rd.append(f"RG (ii) GT 시작 드리프트 d 중앙값: {msg} → " + (f"O1 대비 {'·'.join(better)} 감소 → \"H5 해소 수단 확인\"" if better else "O1 대비 감소 없음"))
    verdict["readouts"] = rd
    lines += ["", "## 8. 판독 가이드 자동 점검", ""] + [f"- {x}" for x in rd]
    json.dump(verdict, open(LOG / "stage1_verdict.json", "w"), indent=1, default=float)
    (LOG / "stage1_tables.md").write_text("\n".join(lines) + "\n"); print("\n".join(lines[:60]))
