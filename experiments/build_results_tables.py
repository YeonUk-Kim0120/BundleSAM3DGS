"""Consolidated 22-sequence results tables (2026-09-18, user request: always compare with the original BundleSDF and with the
earlier experiments).  Writes RESULTS_22SEQ.md.  Sources:
  - MILESTONES.md tables (original BundleSDF ADD with paper masks and with SAM2 masks; our SAM2 noop/off = tracker alone)
  - logs/fulleval_20260912: cd_ref_* (original BundleSDF SAM2-mask meshes), cd_refpaper_* (paper-mask meshes),
    add_*/cd_* (GS 2026-09-12 configuration: init 4000, no erosion; meshes re-extracted with density cut 0)
  - logs/exp_fusion_20260915: the B-track 4-sequence configurations and the 22-sequence copy-runner run 'adopt'
  - logs/safety22_20260917: main code c297f6a ('main') and main + shared depth weight 10 ('main_dw10')
usage: python3 experiments/build_results_tables.py [--out RESULTS_22SEQ.md]
"""
from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

import numpy as np

HO = ["AP10", "AP11", "AP12", "AP13", "AP14", "MPM10", "MPM11", "MPM12", "MPM13", "MPM14", "SB11", "SB13", "SM1"]
YC = ["bleach0", "bleach_hard_00_03_chaitanya", "cracker_box_reorient", "cracker_box_yalehand0", "mustard0",
      "mustard_easy_00_02", "sugar_box1", "sugar_box_yalehand0", "tomato_soup_can_yalehand0"]
ALL = HO + YC
DS = {s: "ho3d" for s in HO}; DS.update({s: "ycb" for s in YC})
FOUR = ["mustard0", "cracker_box_yalehand0", "AP12", "MPM12"]
NA = float("nan")


def jload(p):
    return json.load(open(p)) if os.path.exists(p) else None


def add_of(p):
    d = jload(p)
    if d is None: return NA
    r = d[0] if isinstance(d, list) else d
    return float(r["ADD_err_cm"])


def geo_of(p):
    d = jload(p)
    if d is None: return dict(P1=NA, P2=NA, cov=NA)
    return dict(P1=float(d["P1_original"]["chamfer_cm"]), P2=float(d["P2_full_model"]["chamfer_cm"]), cov=100 * float(d["P3_regions"]["unseen_within_5mm_frac"]))


def parse_milestones(path="MILESTONES.md"):
    """Original-BundleSDF ADD tables: HO3D row = XMem on | XMem off | SAM2 on | SAM2 noop (ours) | SAM2 off (ours);
    YCB row = paper on | paper off | SAM2 on (orig) | SAM2 on (ours) | SAM2 noop | SAM2 off."""
    out = {}
    alias = {"bleach_hard_00_03": "bleach_hard_00_03_chaitanya"}
    for line in open(path):
        m = re.match(r"^\| ([A-Za-z0-9_]+) \| (.*)\|\s*$", line.strip())
        if not m: continue
        name = alias.get(m.group(1), m.group(1)); cells = [c.strip() for c in m.group(2).split("|") if c.strip()]
        if name in ALL and name not in out:
            try: vals = [float(c) for c in cells]
            except ValueError: continue
            if name in HO and len(vals) == 5:
                out[name] = dict(paper_on=vals[0], paper_off=vals[1], sam2_on=vals[2], noop=vals[3], off=vals[4])
            elif name in YC and len(vals) == 6:
                out[name] = dict(paper_on=vals[0], paper_off=vals[1], sam2_on=vals[2], sam2_on_ours=vals[3], noop=vals[4], off=vals[5])
    missing = [s for s in ALL if s not in out]
    assert not missing, f"MILESTONES.md rows missing: {missing}"
    return out


def fmt(v, nd=3):
    return "–" if v is None or (isinstance(v, float) and np.isnan(v)) else (f"{v:.{nd}f}" if nd else f"{v:.0f}")


def mean(vals):
    vals = [v for v in vals if not (isinstance(v, float) and np.isnan(v))]
    return float(np.mean(vals)) if vals else NA


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default="RESULTS_22SEQ.md"); args = ap.parse_args()
    B = parse_milestones()
    F, E, S = Path("logs/fulleval_20260912"), Path("logs/exp_fusion_20260915"), Path("logs/safety22_20260917")
    # 22-sequence columns: name -> {seq: ADD}, geometry -> {seq: dict}
    add_cols = {
        "BundleSDF 논문 마스크 on (원본)": {s: B[s]["paper_on"] for s in ALL},
        "BundleSDF SAM2 on (원본)": {s: B[s]["sam2_on"] for s in ALL},
        "트래커 단독 (SAM2 off)": {s: B[s]["off"] for s in ALL},
        "GS 9/12 (init 4000, 침식 0)": {s: add_of(F / f"add_{DS[s]}_{s}.json") for s in ALL},
        "GS adopt 복사본 9/16 (init 500, 침식 2)": {s: add_of(E / f"add_adopt_{DS[s]}_{s}.json") for s in ALL},
        "GS 본 코드 c297f6a 9/17": {s: add_of(S / f"add_main_{DS[s]}_{s}.json") for s in ALL},
        "GS 본 코드 + 깊이 가중치 10": {s: add_of(S / f"add_main_dw10_{DS[s]}_{s}.json") for s in ALL},
    }
    geo_cols = {
        "BundleSDF 논문 마스크 (원본)": {s: geo_of(F / f"cd_refpaper_{DS[s]}_{s}.json") for s in ALL},
        "BundleSDF SAM2 (원본)": {s: geo_of(F / f"cd_ref_{DS[s]}_{s}.json") for s in ALL},
        "GS 9/12": {s: geo_of(F / f"cd_{DS[s]}_{s}_real_world.json") for s in ALL},
        "GS adopt 9/16": {s: geo_of(E / f"cd_adopt_{DS[s]}_{s}_real_world.json") for s in ALL},
        "GS 본 코드 9/17": {s: geo_of(S / f"cd_main_{DS[s]}_{s}_real_world.json") for s in ALL},
        "GS 본 코드 + 깊이 10": {s: geo_of(S / f"cd_main_dw10_{DS[s]}_{s}_real_world.json") for s in ALL},
    }
    L = ["# 22시퀀스 통합 결과표", "", f"생성: `experiments/build_results_tables.py` (2026-09-18). 새 실험이 끝나면 이 스크립트에 열을 추가하고 다시 실행한다.", "",
         "## 읽는 법", "",
         "- **ADD** (cm, 낮을수록 좋음): 첫 프레임 정렬 후 GT 모델 점들의 평균 거리, 시퀀스 전체 프레임 평균 (`experiments/eval_add_ycbineoat.py`). HO3D는 손 조작(가림 큼), YCB는 로봇 팔.",
         "- **P1** (cm): 복원 메시 ↔ GT 중 카메라에 보였던 면의 챔퍼 거리. **P2** (cm): 복원 메시 ↔ GT 전체 모델. **커버리지** (%): GT 중 한 번도 안 보인 영역이 5 mm 이내로 채워진 비율 (`experiments/eval_mesh_cd.py`, 메시 = `mesh_real_world.obj`).",
         "- **원본 BundleSDF**: `~/Desktop/BundleSDF_baseline_outputs`(읽기 전용)의 실제 실행 결과를 같은 채점기로 평가한 값. '논문 마스크'는 데이터셋 제공 마스크(HO3D XMem, YCB 논문 프로토콜), 'SAM2'는 우리와 같은 SAM2 마스크. 원본 저장소와 우리 저장소의 SDF 경로는 같은 입력에서 같은 결과를 냄이 검증됨 (MILESTONES.md 검증 1).",
         "- **트래커 단독**: 우리 파이프라인에서 백엔드 피드백을 끈 것(SAM2 마스크). GS 설정들은 모두 이 트래커 위에 GS 지도 + 프라이어 + 포즈 피드백(v1)을 얹은 것이며 SAM2 마스크를 쓴다.",
         "- **잡음 바닥** (같은 설정 반복 실행의 차이): HO3D ADD ±0.2 cm(불안정 시퀀스 MPM10은 ±0.3), YCB ADD ±0.05, P1/P2 ±0.01, 커버리지 ±2~5 pt. 이보다 작은 차이는 우열이 아니다.",
         "", "## 열 정의", "",
         "| 열 | 무엇 | 실행 | 결과 위치 |", "|---|---|---|---|",
         "| BundleSDF 논문 마스크 on (원본) | 원본 BundleSDF, 논문 마스크, SDF 피드백 on | 2026-09-03 채점 | `logs/add_eval_baseline_*`, MILESTONES.md |",
         "| BundleSDF SAM2 on (원본) | 원본 BundleSDF, SAM2 마스크, SDF 피드백 on | 2026-09-03 | MILESTONES.md, `logs/fulleval_20260912/cd_ref_*` |",
         "| 트래커 단독 | 우리 파이프라인, 피드백 off, SAM2 마스크 | 2026-09-02/03 | `outputs/_archive_records_20260912/fbabl_*_off_*` |",
         "| GS 9/12 | GS 백엔드, 초기화 4000 스텝, 갱신 500, 마스크 침식 0, 밀도 컷 0으로 재추출 | 2026-09-12 | `outputs/fulleval_20260912`, `logs/fulleval_20260912` |",
         "| GS adopt 9/16 | 초기화 500 / 갱신 500 / 침식 2 px, 실험 복사본 러너로 실행 | 2026-09-16 | `outputs/exp_fusion_20260915/adopt`, `logs/exp_fusion_20260915` |",
         "| GS 본 코드 9/17 | 같은 설정을 본 코드(c297f6a) 기본값으로 직접 실행 = 현재 안전지대 | 2026-09-17 | `outputs/safety22_20260917/main`, `logs/safety22_20260917` |",
         "| GS 본 코드 + 깊이 10 | 위 + 깊이 손실 가중치 1→10 (한 값만 다름) | 2026-09-17 | `outputs/safety22_20260917/main_dw10` |", ""]
    # table 1: ADD
    names = list(add_cols)
    L += ["## 표 1. ADD (cm), 22시퀀스", "", "| 시퀀스 | " + " | ".join(names) + " |", "|---|" + "---|" * len(names)]
    for ds, seqs, label in (("ho3d", HO, "HO3D"), ("ycb", YC, "YCB")):
        for s in seqs:
            L.append(f"| {s} | " + " | ".join(fmt(add_cols[n][s]) for n in names) + " |")
        L.append(f"| **{label} 평균 ({len(seqs)})** | " + " | ".join(f"**{fmt(mean([add_cols[n][s] for s in seqs]))}**" for n in names) + " |")
    L.append("| **HO3D 평균, MPM10 제외 (12)** | " + " | ".join(fmt(mean([add_cols[n][s] for s in HO if s != 'MPM10'])) for n in names) + " |")
    L += ["", "읽기: HO3D에서는 원본 SDF 피드백(0.73)이 트래커 단독(1.67)을 크게 구하지만 GS 피드백은 아직 트래커 단독보다 나쁘다(1.87). YCB에서는 GS(1.20)가 원본 SAM2 on(1.49)과 트래커 단독(1.27)보다 낫다. 깊이 가중치 10은 주전자(AP10/AP12)와 MPM11에서 크게 좋아지지만 MPM10에서 무너지고 YCB 전반이 나빠져 평균은 악화된다.", ""]
    # tables 2-4: geometry
    gnames = list(geo_cols)
    for key, title, nd in (("P1", "표 2. P1 — 보인 면 챔퍼 (cm)", 3), ("P2", "표 3. P2 — 전체 모델 챔퍼 (cm)", 3), ("cov", "표 4. 커버리지 — 안 보인 영역 5 mm 이내 복원 (%)", 0)):
        L += [f"## {title}", "", "| 시퀀스 | " + " | ".join(gnames) + " |", "|---|" + "---|" * len(gnames)]
        for ds, seqs, label in (("ho3d", HO, "HO3D"), ("ycb", YC, "YCB")):
            for s in seqs:
                L.append(f"| {s} | " + " | ".join(fmt(geo_cols[n][s][key], nd) for n in gnames) + " |")
            L.append(f"| **{label} 평균 ({len(seqs)})** | " + " | ".join(f"**{fmt(mean([geo_cols[n][s][key] for s in seqs]), nd)}**" for n in gnames) + " |")
        L.append("")
    L += ["읽기: 형상은 모든 GS 설정이 원본 BundleSDF보다 좋다(HO3D P1 0.41 vs 0.48, YCB 0.45 vs 0.72). 커버리지는 프라이어가 못 본 뒷면을 채우기 때문에 원본보다 크게 높다(HO3D 78 % vs 원본 SAM2 53 %·논문 마스크 35 %; YCB 59 % vs 32 %). 깊이 가중치 10은 렌더 깊이는 관측에 더 맞지만(3.7→2.3 mm) GT 대비 P1은 22개 중 17개에서 나빠진다.", ""]
    # table 5: 4-sequence B-track
    cfgs = [("control", "9/12 설정 재실행: 초기화 4000 / 갱신 500 / 침식 0"), ("fusion", "control + 재관측 융합 F (5 mm 대역)"), ("fusion_r2", "F 반복"),
            ("init500", "초기화 500 / 갱신 500"), ("init1000", "초기화 1000 / 갱신 500"), ("init2000", "초기화 2000 / 갱신 500"), ("upd350", "초기화 4000 / 갱신 350"),
            ("init500_upd350", "초기화 500 / 갱신 350"), ("init500_upd350_r2", "같은 설정 반복"), ("lr3_u350", "500/350 + 갱신 위치 lr ×3"), ("lr3_u350_r2", "lr ×3 반복"), ("lr10_u350", "500/350 + lr ×10"),
            ("erode2_u350", "500/350 + 마스크 침식 2 px"), ("erode2_u350_r2", "침식 2 px 반복"), ("erode4_u350", "500/350 + 침식 4 px"),
            ("adopt", "채택 설정: 500/500 + 침식 2 px"), ("adopt_r2", "채택 설정 반복"), ("erode3", "500/500 + 침식 3 px"), ("erode3_r2", "침식 3 px 반복"),
            ("dw10", "채택 + 깊이 가중치 10 (공유)"), ("dw30", "채택 + 깊이 가중치 30 (공유)"), ("map10_pose1", "채택 + 지도 깊이 10 / 포즈 깊이 1"), ("map1_pose10", "채택 + 지도 깊이 1 / 포즈 깊이 10")]
    L += ["## 표 5. B-트랙 4시퀀스 설정 비교 (mustard0, cracker_box_yalehand0, AP12, MPM12; 단일 실행, 반복은 별도 행)", "",
          "| 설정 | 내용 | ADD mustard0 | ADD cracker | ADD AP12 | ADD MPM12 | ADD 평균 | P1 평균 | P2 평균 | 커버리지 평균 |", "|---|---|---|---|---|---|---|---|---|---|"]
    def row(label, desc, adds, geos):
        full = all(not np.isnan(adds.get(s, NA)) for s in FOUR)  # means only over the complete 4-sequence set
        gfull = all(s in geos and not np.isnan(geos[s]["P1"]) for s in FOUR)
        return (f"| {label} | {desc} | " + " | ".join(fmt(adds.get(s, NA)) for s in FOUR)
                + f" | {fmt(mean([adds[s] for s in FOUR])) if full else '– (3개)'} | {fmt(mean([geos[s]['P1'] for s in FOUR])) if gfull else '–'} | {fmt(mean([geos[s]['P2'] for s in FOUR])) if gfull else '–'} | {fmt(mean([geos[s]['cov'] for s in FOUR]), 0) if gfull else '–'} |")
    L.append(row("BundleSDF SAM2 on (원본)", "기준", {s: B[s]["sam2_on"] for s in FOUR}, {s: geo_cols["BundleSDF SAM2 (원본)"][s] for s in FOUR}))
    L.append(row("트래커 단독", "기준", {s: B[s]["off"] for s in FOUR}, {}))
    L.append(row("GS 9/12", "22시퀀스 실행 중 4개", {s: add_cols["GS 9/12 (init 4000, 침식 0)"][s] for s in FOUR}, {s: geo_cols["GS 9/12"][s] for s in FOUR}))
    for cfg, desc in cfgs:
        adds = {s: add_of(E / f"add_{cfg}_{DS[s]}_{s}.json") for s in FOUR}; geos = {s: geo_of(E / f"cd_{cfg}_{DS[s]}_{s}_real_world.json") for s in FOUR if os.path.exists(E / f"cd_{cfg}_{DS[s]}_{s}_real_world.json")}
        if all(np.isnan(v) for v in adds.values()): continue
        L.append(row(cfg, desc, adds, geos))
    L.append(row("main (9/17)", "본 코드 안전지대, 22시퀀스 실행 중 4개", {s: add_cols["GS 본 코드 c297f6a 9/17"][s] for s in FOUR}, {s: geo_cols["GS 본 코드 9/17"][s] for s in FOUR}))
    L.append(row("main_dw10 (9/17)", "본 코드 + 깊이 10, 22시퀀스 실행 중 4개", {s: add_cols["GS 본 코드 + 깊이 가중치 10"][s] for s in FOUR}, {s: geo_cols["GS 본 코드 + 깊이 10"][s] for s in FOUR}))
    # noise floor from repeats
    pairs = [("adopt", "adopt_r2"), ("init500_upd350", "init500_upd350_r2"), ("lr3_u350", "lr3_u350_r2"), ("erode2_u350", "erode2_u350_r2"), ("erode3", "erode3_r2"), ("fusion", "fusion_r2")]
    diffs = {"ADD": [], "P1": [], "P2": [], "cov": []}
    for a, b in pairs:
        for s in FOUR:
            pa, pb = E / f"add_{a}_{DS[s]}_{s}.json", E / f"add_{b}_{DS[s]}_{s}.json"
            if pa.exists() and pb.exists():
                diffs["ADD"].append(abs(add_of(pa) - add_of(pb)))
                ga, gb = geo_of(E / f"cd_{a}_{DS[s]}_{s}_real_world.json"), geo_of(E / f"cd_{b}_{DS[s]}_{s}_real_world.json")
                for k in ("P1", "P2", "cov"):
                    if not (np.isnan(ga[k]) or np.isnan(gb[k])): diffs[k].append(abs(ga[k] - gb[k]))
    L += ["", f"반복 실행 잡음 (같은 설정 두 번, {len(diffs['ADD'])}쌍): |ΔADD| 중앙값 {np.median(diffs['ADD']):.3f} / 최대 {max(diffs['ADD']):.3f} cm, |ΔP1| 중앙값 {np.median(diffs['P1']):.3f} / 최대 {max(diffs['P1']):.3f}, |ΔP2| 중앙값 {np.median(diffs['P2']):.3f}, |Δ커버리지| 중앙값 {np.median(diffs['cov']):.1f} / 최대 {max(diffs['cov']):.1f} pt.", ""]
    # verdicts
    L += ["## 표 6. 실험별 판정", "",
          "| 실험 | 판정 | 근거 | 기록 |", "|---|---|---|---|",
          "| Poisson 밀도 컷 0.05 → 0 | 채택 (본 코드) | 미세 해상도에서 컷이 프라이어가 채운 뒷면을 잘라냄 (AP12 커버리지 7 % → 62 %) | ONLINE_MAP_DEFECTS.md §4, GLOBAL_REFINE_RESULTS.md |",
          "| 재관측 융합 F | 기각 | 가우시안 −41~67 %인데 ADD 동일/악화(MPM12 0.79→0.89), P1 +0.01~0.09 | §7 |",
          "| 초기화 스텝 4000 → 500 | 채택 (본 코드) | 4시퀀스 전부 동등 이상 (평균 ADD 1.639→1.548), 큰 가우시안 −40~75 %; 500<1000<2000<4000 단조 | §7 |",
          "| 갱신 스텝 500 → 350 | 기록 (속도 옵션) | 정확도 잡음 범위, 30 % 빠름; 기본은 500 유지 (사용자 결정) | §7 |",
          "| 갱신 위치 lr ×3 / ×10 | 기록만 | ADD/P1 소폭 이득, 완성도 −2~5 pt 일관 손실 (사용자 결정: 채택 안 함) | §7 |",
          "| 마스크 침식 2 px | 채택 (본 코드) | mustard0 −0.09 ×2 반복, 나머지 동등, 원거리 가우시안 −60~70 %; 3/4 px는 2 px와 동등 | §7, §11 |",
          "| 깊이 가중치 10 (공유) | 기각 (22시퀀스) | 4시퀀스 평균은 개선(1.503→1.447)이었으나 22시퀀스에서 HO3D 1.867→1.911, YCB 1.199→1.299; MPM10 1.46→4.82 드리프트, P1 17/22 악화 | §10, §15 |",
          "| 깊이 가중치 30 | 기각 | MPM12·mustard0 악화 | §10 |",
          "| 지도/포즈 깊이 가중치 분리 | 기각 | 지도 쪽은 mustard0/MPM12 이득·AP12 손실, 포즈 쪽은 반대; 시퀀스마다 방향이 반대 | §12 |",
          "| 프라이어 정합 1/3/5 프레임 | 채택 안 함 | 평균 15.9→14.8→15.6 mm, 주전자만 이득·SM1/mustard0/MPM10 손실; 100~150 스텝이면 수렴 | §14 |",
          "| 빠른 움직임 게이트 (써넣기 생략) | 철회 | 써넣기 2,653건 분석: 득실 50/50이며 움직임·오차 급등·delta·인라이어와 무관 | §13d |",
          "| 프라이어 스케일 처리 | 보류 | YCB 프라이어 4개에 16~33 % 스케일 오차 잔존; 나중에 별도 집중 | §14, Parked |", ""]
    open(args.out, "w").write("\n".join(L) + "\n"); print("wrote", args.out)
    for n in names:
        print(f"{n}: HO3D {mean([add_cols[n][s] for s in HO]):.3f} YCB {mean([add_cols[n][s] for s in YC]):.3f}")


if __name__ == "__main__":
    main()
