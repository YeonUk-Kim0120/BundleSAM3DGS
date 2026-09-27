# EXP_BATCH_20260927 진행 기록 (3차 무인 배치, 2026-09-27 시작)

지시서 `EXP_BATCH_20260927_BRIEF.md` (브랜치 milestone5-v1, HEAD 7966d88). 커밋은 지시서상 금지였으나 9/28 사용자 승인으로 커밋. 본 코드 무수정. 이 파일만 읽고 이어갈 수 있게 단계마다 갱신한다.

## 재개 방법 (요약)
- 컨테이너 `BundleSAM3DGS-gsplat-smoke`, 체인 `logs/exp_batch_20260927/chain.sh <gpu> <jobfile> <name>` (잡 파일 줄 = "<script> <args>"), 진행 로그 `logs/exp_batch_20260927/progress`.
- 산출물 `outputs/exp_batch_20260927/<arm>/<ds>/<seq>`, 매니페스트 `logs/exp_batch_20260927/manifests/`.

## 0. noop 포즈 소스 (9/27 17:45 KST 시작; 컨테이너 date는 다른 시간대로 찍힘 — progress 파일의 시각은 컨테이너 시계)
- 1차 배치 ctrl/noop 산출물은 `ob_in_cam/`만 남아 있고 `color/ depth_filtered/ mask/`와 프레임별 `poses_before_gs.txt`는 승인된 덤프 정리에서 삭제됨(확인: outputs/exp_batch_20260921/ctrl/ho3d/AP12 등 color 0, poses_before_gs 0).
- 따라서 지시서대로 본 코드·같은 설정(1차 ctrl 매니페스트: `--gs_feedback noop --gs_initial_steps 500 --gs_update_steps 500`, 기본 러너 설정 config_gs_2dgs_1mm_lifecycle.yml; 1차는 online_variants 복사본(스위치 없음)을 거쳤고 이번에는 본 코드 진입점 run_ho3d.py / run_custom.py를 직접 호출)으로 noop 온라인 실행 4개를 새로 만든다: `outputs/exp_batch_20260927/noop/{ho3d/AP12, ho3d/SM1, ho3d/MPM12, ycb/mustard0}`. 스크립트 `run_noop.sh`, 잡 `jobs/noop_gpu0.txt`(AP12→SM1), `jobs/noop_gpu1.txt`(MPM12→mustard0).

## 코드 (9/27 17:45–18:10 KST)
- `experiments/exp_contamination_split.py` (단계 0), `experiments/exp_reference_optimizer_probe.py` (단계 1 probe + GN 단위 확인), `experiments/hygiene_runner.py` (A1–A3, HYG_MODE None = 본 코드 동작).
- probe 고정값(`exp_reference_optimizer_probe.P`): 입력 점 최대 5000(고정 시드 무작위 균등), 침식 2 px, 참조 불투명도 ≥ 0.1, PCA k = 16, 대응 최대 2 cm, 이상치 |r| ≤ med + 2.5·MAD(|r|), Huber 2 mm, 감쇠 1e-4, 클립 2 cm / 0.2 rad, GN 최대 8회, 종료 0.1 mm & 1e-3 rad, 채택 유효 비율 ≥ 15 %; O1 Adam 0.01 × 3, O2 Adam 1e-3 × 200, O4 Adam 1e-3 × 200(O3 목적함수, 대응·가중 매 스텝 재계산); d = GT 메시 20,000 샘플.
- 측정 보정: 저장 포즈(float32)의 회전이 |RᵀR − I| ≈ 2e-7이라 arccos 각도가 0.03° 부풀려짐 → probe의 시작·GT 포즈를 SO(3)로 투영하고 각도는 2 asin(|ΔR|_F/2√2)로 계산.

## GN 단위 확인 (3.4) — 구현 통과 (`unit_tests/gn_unit_test.txt`)
- RP 기준(AP12 prior 20,000 중심), 알려진 포즈에서 본 중심점(카메라 좌표) 3개 뷰 × 6 방향, 3°/5 mm 교란: 반복 제한 30이면 18/18이 < 0.001°/0.001 mm로 복구(7–11회), 교란 0 → 이동 0.000000°. 구현 정상.
- 지시서의 최대 8회 제한에서는 14/18이 0.1°/0.3 mm 안, 나머지 4개는 0.04–0.36°/0.6–2.6 mm(원통형 물통의 미끄러짐 방향에서 수렴이 느림). 지시서 값(8회)을 그대로 유지하고 결과 문서에 적는다(판단 지점).

## 위생 러너 단위 확인 (3.3, mustard0, `unit_tests/hygiene_unit_tests.txt`)
- 입력: BSDF SAM2 기준 실행(읽기 전용)의 프레임, 최종 keyframes.yml의 처음 15개(초기 5 + 갱신 10; "키프레임 10개"를 갱신 10개로 해석), GT 포즈, 500/500.
- (1) GT 포즈 A1: 갱신 1–6 출생 후보의 승격률 79.2 % → **통과**(≥ 70 %).
- (2) KF5만 3° 교란: KF5 후보 폐기율 17.4 % → **실패**(≥ 70 %). 다른 KF 승격률 77.3 % (vs 77.9 %). 원인 진단 진행 중(`--diag2`: 3°가 후보 점을 실제로 얼마나 옮기는지, 10°에서는 폐기되는지). 3° 회전은 mustard 표면 점을 수 mm만 옮겨, 지시서 문턱(depth 허용 2 cm, 산포 5 mm) 안에 들어갈 가능성이 크다.
- (3) 스위치 off vs 본 코드: 별도 재생 두 번 비교는 prior Sim(3) 정합 자체가 실행마다 달라(정규화 스케일 7.886–7.898) 본 코드끼리도 Gaussian 42개 차이 → 같은 초기 체크포인트에서 출발하는 방식으로 다시 확인 중(`--only3`).
- 9/27 18:10 KST: stage-1 체인 대기열 등록 — GPU0 `jobs/stage1_gpu0.txt`(RG × 4 → RP × 4), GPU1 `jobs/stage1_gpu1.txt`(R0 × 4), 각 GPU의 noop 체인 종료("CHAIN noop_gpuN DONE") 후 자동 시작(`wait_chain.sh`). A 계열은 단계 0 판정과 위생 단위 확인 뒤 추가.
- (3) 재확인(`unit_tests/hygiene_unit_test3.txt`): 같은 초기 체크포인트에서 갱신 10회 × 500 스텝 — Gaussian 수 10회 모두 동일, 최종 손실 최대 차 7.8e-5(본 코드끼리 1.3e-4) → **통과**.
- (2) 진단(`unit_tests/hygiene_unit_test2_diag.txt`): 3° 교란은 KF5 후보 점을 중앙값 2.6 mm(p90 4.5) 옮기고 폐기율 17.5 %; 10° 교란은 8.7 mm(p90 14.8) 옮겨도 폐기율 23.3 %. 물체 중심 회전 오차는 점을 표면을 따라 미끄러뜨리므로, 뒤 키프레임의 depth는 그 자리에서도 맞고(허용 2 cm) 관측들끼리도 일관(산포 < 5 mm)해 승격된다. 즉 지시서의 A1 규칙은 표면 밖 오염(마스크 누출·경계)은 거르지만 **표면 위 위치 오차(트래커 회전 오차가 만드는 것)는 원리적으로 거르지 못한다**. 구현 오류가 아니라 판정 규칙의 성질로 보고 결과 문서에 적는다. A 계열은 단계 0 판정 범위대로 실행하되 이 한계를 명시한다.

## noop 실행 결과 (9/27 18:13–18:23 KST 완료; progress 시각은 컨테이너 시계)
- AP12 ADD 0.890 / ADD-S 0.363 (1차 noop ctrl 0.820, ctrl_r2 0.894), MPM12 0.536 / 0.231 (1차 0.531), mustard0 0.744 / 0.323 (1차 0.748) → 1차 대조군 재현. SM1 실행 중(1차 noop 4.520, 붕괴 시퀀스).
- 스모크(`--max-cycles 7`, AP12, RP/R0/A1): 세 기준 모두 끝까지 동작, probe 사이클당 약 25 s(6 시작 × 4 최적화기), O3 채택 100 %, 유효 대응 중앙값 0.86.
- 9/27 18:20 KST: GPU1 R0 MPM12 probe 시작(단계 1 체인 자동 시작). GPU0는 SM1 noop 뒤 RG → RP.

## 단계 0 결과 (9/27 18:55 KST, `stage0/contamination_split.json`, `stage0/contamination_split.png`)
| seq | KF | d_abs 중앙값 / p90 (mm) | d_loc 중앙값 / p90 (mm) | d_loc/d_abs (중앙값 비) | d_loc > 3 mm | > 5 mm |
|---|---|---|---|---|---|---|
| AP12 | 166 | 7.36 / 12.34 | 2.89 / 5.11 | 0.39 | 47 % | 12 % |
| mustard0 | 36 | 7.50 / 9.29 | 2.74 / 4.84 | 0.36 | 43 % | 9 % |
| SM1 | 290 | 48.32 / 57.58 | 2.18 / 5.75 | 0.05 | 30 % | 13 % |
| MPM12 | 235 | 4.97 / 7.14 | 1.70 / 3.16 | 0.34 | 12 % | 3 % |
- 결정 규칙: d_loc > 3 mm가 10 % 이상인 시퀀스 4/4 → **A1–A3 모두 실행**.
- A 계열 대기열(18:58 KST): GPU1 `jobs/stage1A_gpu1.txt`(A1 × 4 → A2 × 4, R0 체인 종료 뒤), GPU0 `jobs/stage1A_gpu0.txt`(A3 × 4, RG·RP 체인 종료 뒤). 지시서는 A 전부 GPU1이지만 두 GPU 시간 균형을 위해 A3를 GPU0으로 옮김(실험 내용 동일).
- 단계 1 진행: R0 MPM12 완료(1,848 probe, 77 사이클, 32 분). 이후 progress 파일의 DONE 줄로 확인.

## 단계 1 완료·판정 (9/28 00:40 KST; probe 24개 실패 0, `stage1_tables.md`, `stage1_verdict.json`)
- 전제 확인 실패: RG + O3 GT+3° 개선 36 %(기준 ≥ 80 %), GT 시작 드리프트 0.70 mm(기준 < 0.3) → **단계 2 금지**. 통과 조합 없음(RG 포함; SM1은 48 mm 붕괴라 어떤 기준으로도 16–21 % 이하).
- MPM12(비붕괴 검증) 트래커 probe O3: RG 92 % 개선 / R0 12 % / A1–A3 21–26 % / RP 3 %(악화 95 %). 원인 (a) 편향 재확인.
- 판독 자동 점검: A1·A3 + O3 개선률 12 % vs R0 5 %(2σ 밖) = "위생 채택 후보" 문장 해당하나 통과 기준과 거리 멂; O3 ≈ O4(RP 제외); RG 깨끗한 뷰 드리프트 O3 0.70 mm vs O1 2.42 / O2 3.53 → "H5 해소 수단 확인"(목표 미달).
- 결과 문서 `EXP_BATCH_20260927_RESULTS.md` 작성(9/28 01:00 KST). 커밋 안 함(지시서). 배치 종료.
- 9/28 사용자 승인으로 커밋(원시 run_noop_*.log 제외).
