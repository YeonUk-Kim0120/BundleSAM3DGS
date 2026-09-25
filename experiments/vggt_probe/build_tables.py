"""Tables for EXP_VGGT_PROBE1_RESULTS.md (brief section 10) from outputs/exp_vggt_probe1/<model>/<input>/<batch>/<ds>/<seq>/."""
from __future__ import annotations
import argparse, csv, glob, json
from collections import defaultdict
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr

ROOT = Path("/home/kist/Desktop/BundleSAM3DGS/outputs/exp_vggt_probe1")
HO3D = ["AP10", "AP11", "AP12", "AP13", "AP14", "MPM10", "MPM11", "MPM12", "MPM13", "MPM14", "SB11", "SB13", "SM1"]
YCB = ["bleach0", "bleach_hard_00_03_chaitanya", "cracker_box_reorient", "cracker_box_yalehand0", "mustard0", "mustard_easy_00_02", "sugar_box1", "sugar_box_yalehand0", "tomato_soup_can_yalehand0"]
SIG = ["T1_rot", "T1_trans", "T2_s_cv", "T3_fov_diff", "T4_depth_agree", "T5_conf", "T6_mask_area", "T7_rot", "T7_trans"]
BIG = ["MPM10", "AP11", "SM1", "SB11"]


def load():
    cells = {}
    for kf in glob.glob(str(ROOT / "M-*/I*/*/*/*/kf.csv")):
        d = Path(kf).parent; model, inp, bt, ds, sq = d.parts[-5:]
        rows = []
        for r in csv.DictReader(open(kf)):
            rows.append({k: (v if k in ("kf", "role", "used_ref_ids", "warp") else (float(v) if v not in ("", "nan") else np.nan)) for k, v in r.items()})
        bl = [json.loads(l) for l in open(d / "batches.jsonl")]
        if bt == "chain":
            # (1) pure chain: overlap KF pose = previous batch's VGGT estimate  -> 'online' columns of chain-pure(1)
            #     its 'oracle' columns = (2) overlap KF re-aligned to GT (R-oracle)
            cells[(f"{model}/{inp}/chain-pure(1)", ds, sq)] = (rows, bl)
            # (2) overlap KF re-aligned to the R-online (BSDF snapshot) pose: post-hoc from realign_chain.py
            ra = d / "kf_realigned_online.csv"
            if ra.exists():
                m = {r["kf"]: float(r["rot_online_snapref"]) for r in csv.DictReader(open(ra))}; mt = {r["kf"]: float(r["trans_online_snapref"]) for r in csv.DictReader(open(ra))}
                rows2 = [dict(r, rot_online=m.get(r["kf"], np.nan), trans_online=mt.get(r["kf"], np.nan)) for r in rows]
                cells[(f"{model}/{inp}/chain-realigned(2)", ds, sq)] = (rows2, bl)
        else:
            cells[(f"{model}/{inp}/{bt}", ds, sq)] = (rows, bl)
    return cells


def new_rows(rows): return [r for r in rows if r["role"] == "new" and r["has_gt"] == 1]
def med(xs): xs = [x for x in xs if np.isfinite(x)]; return float(np.median(xs)) if xs else np.nan
def p90(xs): xs = [x for x in xs if np.isfinite(x)]; return float(np.percentile(xs, 90)) if xs else np.nan
def f(x, p=1): return "–" if x is None or not np.isfinite(x) else f"{x:.{p}f}"


def cfg_stats(cells, cfg, ds):
    """per-sequence medians for one config -> dict seq -> dict"""
    out = {}
    for (c, d, sq), (rows, bl) in cells.items():
        if c != cfg or d != ds: continue
        n = new_rows(rows)
        out[sq] = dict(n=len(n), n_kf=len({r["kf"] for r in rows}), rot_on=med([r["rot_online"] for r in n]), rot_on90=p90([r["rot_online"] for r in n]), tr_on=med([r["trans_online"] for r in n]), tr_on90=p90([r["trans_online"] for r in n]),
                       rot_or=med([r["rot_oracle"] for r in n]), rot_or90=p90([r["rot_oracle"] for r in n]), tr_or=med([r["trans_oracle"] for r in n]), tr_or90=p90([r["trans_oracle"] for r in n]),
                       rot_c1=med([r["rot_c1"] for r in n]), rot_c190=p90([r["rot_c1"] for r in n]), tr_c1=med([r["trans_c1"] for r in n]), tr_c190=p90([r["trans_c1"] for r in n]),
                       rot_c2=med([r["rot_c2"] for r in n]), rot_c290=p90([r["rot_c2"] for r in n]), tr_c2=med([r["trans_c2"] for r in n]), tr_c290=p90([r["trans_c2"] for r in n]),
                       win_rot=np.mean([r["rot_online"] < r["rot_c1"] for r in n]) if n else np.nan, e3=med([b.get("E3_err", np.nan) for b in bl]), t2=med([b.get("T2", np.nan) for b in bl]),
                       s=med([b["s"] for b in bl]), s_gt=med([b["s_gt"] for b in bl]), time=med([b["time_s"] for b in bl]), mem=max(b["peak_mem_gb"] for b in bl), nb=len(bl))
    return out


def table_B(cells, ds, seqs):
    cfgs = sorted({c for c, d, _ in cells if d == ds})
    out = ["| 설정 | 완료 seq | online rot° (중앙값 평균) | oracle rot° | C1 rot° | C2 rot° | online trans mm | oracle trans mm | C1 trans mm | C2 trans mm | online<C1 seq 수 | 시퀀스 내 online<C1 KF 비율 |", "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    ranking = []
    for c in cfgs:
        st = cfg_stats(cells, c, ds); v = [st[s] for s in seqs if s in st]
        if not v: continue
        m = lambda k: np.nanmean([x[k] for x in v]); wins = sum(x["rot_on"] < x["rot_c1"] for x in v)
        ranking.append((m("rot_on"), c))
        out.append(f"| {c} | {len(v)}/{len(seqs)} | {f(m('rot_on'),2)} | {f(m('rot_or'),2)} | {f(m('rot_c1'),2)} | {f(m('rot_c2'),2)} | {f(m('tr_on'))} | {f(m('tr_or'))} | {f(m('tr_c1'))} | {f(m('tr_c2'))} | {wins}/{len(v)} | {f(100*m('win_rot'),0)} % |")
    return "\n".join(out), [c for _, c in sorted(ranking)]


def table_A(cells, ds, seqs, top):
    hdr = "| seq | KF | 새 KF(GT) | C1 rot 중/p90 | C2 rot 중/p90 | " + " | ".join(f"{c} online rot 중/p90 | {c} oracle rot 중/p90" for c in top) + " | C1 trans 중/p90 | C2 trans 중/p90 | " + " | ".join(f"{c} online trans 중/p90 | {c} oracle trans 중/p90" for c in top) + " |"
    out = [hdr, "|" + "---|" * (hdr.count("|") - 1)]
    sts = {c: cfg_stats(cells, c, ds) for c in top}
    for sq in seqs:
        base = next((sts[c][sq] for c in top if sq in sts[c]), None)
        if base is None: continue
        cells_r = [f"{f(sts[c][sq]['rot_on'],2)}/{f(sts[c][sq]['rot_on90'],2)} | {f(sts[c][sq]['rot_or'],2)}/{f(sts[c][sq]['rot_or90'],2)}" if sq in sts[c] else "– | –" for c in top]
        cells_t = [f"{f(sts[c][sq]['tr_on'])}/{f(sts[c][sq]['tr_on90'])} | {f(sts[c][sq]['tr_or'])}/{f(sts[c][sq]['tr_or90'])}" if sq in sts[c] else "– | –" for c in top]
        out.append(f"| {sq} | {base['n_kf']} | {base['n']} | {f(base['rot_c1'],2)}/{f(base['rot_c190'],2)} | {f(base['rot_c2'],2)}/{f(base['rot_c290'],2)} | " + " | ".join(cells_r) + f" | {f(base['tr_c1'])}/{f(base['tr_c190'])} | {f(base['tr_c2'])}/{f(base['tr_c290'])} | " + " | ".join(cells_t) + " |")
    return "\n".join(out)


def table_A_chain(cells, ds, seqs):
    """Per-sequence chain rows: (1) pure chain online, (2) overlap re-aligned to R-online, (2) re-aligned to R-oracle, vs C1/C2."""
    cfgs = sorted({c for c, d, _ in cells if d == ds and "chain-pure" in c})
    out = ["| 모델/입력 | seq | 새 KF | (1) 순수 체인 rot 중/p90 | (2) R-online 재정렬 rot 중/p90 | (2) R-oracle 재정렬 rot 중/p90 | C1 rot 중/p90 | C2 rot 중/p90 | (1) trans 중 | (2) R-online trans 중 | (2) R-oracle trans 중 | C1 trans 중 | C2 trans 중 |", "|" + "---|" * 13]
    for c in cfgs:
        pure = cfg_stats(cells, c, ds); rl = cfg_stats(cells, c.replace("chain-pure(1)", "chain-realigned(2)"), ds)
        for sq in seqs:
            if sq not in pure: continue
            p_ = pure[sq]; r_ = rl.get(sq)
            out.append(f"| {c.replace('/chain-pure(1)', '')} | {sq} | {p_['n']} | {f(p_['rot_on'],2)}/{f(p_['rot_on90'],2)} | {(f(r_['rot_on'],2) + '/' + f(r_['rot_on90'],2)) if r_ else '–'} | {f(p_['rot_or'],2)}/{f(p_['rot_or90'],2)} | {f(p_['rot_c1'],2)}/{f(p_['rot_c190'],2)} | {f(p_['rot_c2'],2)}/{f(p_['rot_c290'],2)} | {f(p_['tr_on'])} | {f(r_['tr_on']) if r_ else '–'} | {f(p_['tr_or'])} | {f(p_['tr_c1'])} | {f(p_['tr_c2'])} |")
    return "\n".join(out)


def table_C(cells, ds, top):
    out = ["| seq | 설정 | 새 KF | C1>5° KF 수 | 그 부분집합 C1 rot 중앙값 | 같은 KF의 VGGT online rot 중앙값 | oracle | 그 부분집합 C1 trans | VGGT online trans | 전체 online rot 중앙값 |", "|---|---|---|---|---|---|---|---|---|---|"]
    for sq in BIG:
        for c in top:
            key = (c, ds, sq)
            if key not in cells: continue
            n = new_rows(cells[key][0]); sub = [r for r in n if r["rot_c1"] > 5]
            out.append(f"| {sq} | {c} | {len(n)} | {len(sub)} | {f(med([r['rot_c1'] for r in sub]),2)} | {f(med([r['rot_online'] for r in sub]),2)} | {f(med([r['rot_oracle'] for r in sub]),2)} | {f(med([r['trans_c1'] for r in sub]))} | {f(med([r['trans_online'] for r in sub]))} | {f(med([r['rot_online'] for r in n]),2)} |")
    return "\n".join(out)


def table_D(cells, ds, seqs, cfgs):
    out = ["| 설정 | seq | 배치 수 | s 중앙값 | s_gt 중앙값 | E3 오차 |s/s_gt−1| 중앙값 | T2 s_i 변동계수 중앙값 |", "|---|---|---|---|---|---|---|"]
    for c in cfgs:
        st = cfg_stats(cells, c, ds)
        for sq in seqs:
            if sq in st: out.append(f"| {c} | {sq} | {st[sq]['nb']} | {f(st[sq]['s'],3)} | {f(st[sq]['s_gt'],3)} | {f(st[sq]['e3'],3)} | {f(st[sq]['t2'],3)} |")
    return "\n".join(out)


def table_E(cells, ds, cfg):
    pool = defaultdict(list)
    for (c, d, sq), (rows, bl) in cells.items():
        if c != cfg or d != ds: continue
        for r in new_rows(rows):
            for k in SIG: pool[k].append(r[k])
            pool["rot"].append(r["rot_online"]); pool["trans"].append(r["trans_online"]); pool["batch"].append((sq, r["batch"]))
    rot = np.array(pool["rot"]); tr = np.array(pool["trans"])
    out = [f"설정 {cfg}, 새 KF 풀 n = {len(rot)}", "", "| 신호 | Spearman ρ vs rot E1 (online) | ρ vs trans E1 | n |", "|---|---|---|---|"]
    for k in SIG:
        x = np.array(pool[k]); ok = np.isfinite(x) & np.isfinite(rot)
        out.append(f"| {k} | {f(spearmanr(x[ok], rot[ok]).correlation,2) if ok.sum() > 5 else '–'} | {f(spearmanr(x[ok], tr[ok]).correlation,2) if ok.sum() > 5 else '–'} | {int(ok.sum())} |")
    out += ["", "버림 곡선 (배치 단위, T1 회전 잔차가 큰 배치부터 버림; 첫 배치는 T1 = 0이라 항상 남음):", "", "| 버림 | 남은 새 KF | rot 중앙값° | rot p90° | trans 중앙값 mm | trans p90 mm |", "|---|---|---|---|---|---|"]
    t1 = np.array(pool["T1_rot"]); ok = np.isfinite(t1) & np.isfinite(rot)
    bkeys = sorted({b for b, o in zip(pool["batch"], ok) if o}, key=lambda b: -np.nanmax([t for t, bb in zip(t1, pool["batch"]) if bb == b]))   # worst batches first
    for q in [0, 10, 20, 30]:
        drop = set(bkeys[:int(round(len(bkeys) * q / 100))]); keep = np.array([o and (b not in drop) for b, o in zip(pool["batch"], ok)])
        out.append(f"| {q} % | {int(keep.sum())} | {f(np.median(rot[keep]),2)} | {f(np.percentile(rot[keep],90),2)} | {f(np.median(tr[keep]))} | {f(np.percentile(tr[keep],90))} |")
    return "\n".join(out)


def table_F(cells, ds, seqs, cfgs):
    cfgs = [c for c in cfgs if "realigned" not in c]   # accumulation comparison uses the pure chain (1) only
    out = ["| 설정 | 구간(KF 순서) | 새 KF 수 | online rot 중앙값° | oracle rot° | C1 rot° | online trans mm | C1 trans mm |", "|---|---|---|---|---|---|---|---|"]
    for c in cfgs:
        parts = defaultdict(list)
        for (cc, d, sq), (rows, bl) in cells.items():
            if cc != c or d != ds or sq not in seqs: continue
            n = new_rows(rows); order = {r["kf"]: i for i, r in enumerate(sorted(rows, key=lambda r: r["batch"]))}
            kfs = [r["kf"] for r in rows]; N = len(rows)
            for r in n:
                pos = kfs.index(r["kf"]) / max(N - 1, 1); parts["앞 1/3" if pos < 1 / 3 else ("중간" if pos < 2 / 3 else "뒤 1/3")].append(r)
        for part in ["앞 1/3", "중간", "뒤 1/3"]:
            rs = parts[part]
            out.append(f"| {c} | {part} | {len(rs)} | {f(med([r['rot_online'] for r in rs]),2)} | {f(med([r['rot_oracle'] for r in rs]),2)} | {f(med([r['rot_c1'] for r in rs]),2)} | {f(med([r['trans_online'] for r in rs]))} | {f(med([r['trans_c1'] for r in rs]))} |")
    return "\n".join(out)


def table_cost(cells):
    out = ["| 설정 | 배치 수 | 배치 추론 시간 중앙값 s | 최대 s | 최대 GPU 메모리 GB |", "|---|---|---|---|---|"]
    by = defaultdict(list)
    for (c, d, sq), (rows, bl) in cells.items(): by[c] += bl
    for c in sorted(by): out.append(f"| {c} | {len(by[c])} | {f(med([b['time_s'] for b in by[c]]),2)} | {f(max(b['time_s'] for b in by[c]),2)} | {f(max(b['peak_mem_gb'] for b in by[c]),1)} |")
    return "\n".join(out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default=None); ap.add_argument("--top", type=int, default=2); a = ap.parse_args()
    cells = load(); doc = ["# EXP_VGGT_PROBE1 표 (자동 생성: experiments/vggt_probe/build_tables.py)", ""]
    for ds, seqs in [("ho3d", HO3D), ("ycb", YCB)]:
        if not any(d == ds for _, d, _ in cells): continue
        tB, rank = table_B(cells, ds, seqs)
        top = []
        for model in ["M-V", "M-O"]:
            top += [c for c in rank if c.startswith(model + "/") and "chain" not in c][:a.top]   # top B-ref configs per model (online rot)
        doc += [f"# {ds}", "", "## 표 B. 설정 그리드 평균 (시퀀스별 새 KF 중앙값의 평균; online = R-online 정렬, oracle = GT 기준 정렬). 체인: chain-pure(1) = 겹침 KF 포즈를 직전 배치 VGGT 추정값으로 이어 붙인 순수 체인(online 열), 그 oracle 열 = 겹침 KF를 GT로 재정렬한 변형(2); chain-realigned(2) = 겹침 KF를 R-online(BSDF 스냅샷)으로 재정렬한 변형(oracle 열은 동일)", "", tB, "", f"## 표 A. 시퀀스별 새 KF E1 — 모델별 상위 B-ref 설정 {top}", "", table_A(cells, ds, seqs, top), "", "## 표 A-체인. 시퀀스별 체인 결과: (1) 순수 체인, (2) 겹침 KF 재정렬(R-online / R-oracle)", "", table_A_chain(cells, ds, seqs), ""]
        if ds == "ho3d": doc += ["## 표 C. 큰 오차 시퀀스 상세 (C1 회전 오차 > 5° 부분집합)", "", table_C(cells, ds, top), ""]
        doc += ["## 표 D. 스케일 (E3, T2)", "", table_D(cells, ds, seqs, rank), "", "## 표 E. 신뢰 신호 (상위 설정)", ""]
        for c in top: doc += [table_E(cells, ds, c), ""]
        doc += ["## 표 F. 체인 vs 기준 KF: 새 KF 순서(앞/중간/뒤 1/3)별 E1 (체인은 순수 체인(1) 기준)", "", table_F(cells, ds, seqs, rank), ""]
    doc += ["# 비용", "", table_cost(cells), ""]
    text = "\n".join(doc); print(text)
    if a.out: open(a.out, "w").write(text)
