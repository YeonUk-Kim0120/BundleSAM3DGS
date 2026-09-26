# EXP_VGGT_PROBE2 진행 기록 (2026-09-26 시작)

지시서 `EXP_VGGT_PROBE2_BRIEF.md` (브랜치 milestone5-v1, 시작 커밋 bbce546). 본 코드·기존 probe 파일(`probe_lib.py`, `run_probe.py`, `models.py`, `p1b_*.py`) 무수정(import만), Probe 1·1b·9/22 산출물 읽기 전용. 새 코드 `experiments/vggt_probe/p2_*.py`.
사용자 지시(9/26): 0단계 0-a(렌더 정합성)·0-c(합성 복구)를 통과하기 전에는 다음 단계로 넘어가지 않는다. 결과 문서까지 작성하고, 커밋은 사용자 승인 후.
FoundationPose 가중치: 사용자 메시지의 해당 줄이 템플릿 그대로("<경로 또는 "없음">")였다. 경로가 주어지지 않았으므로 지시서 규칙("사용자가 가중치 경로를 알려주면")에 따라 C단계는 기본적으로 건너뛰는 것으로 보고, 로컬 설치 여부만 확인해 기록한다.

## 모델 준비 (9/26 12:10–)
- MD-gt: `eval_mesh_cd.load_gt_mesh` 텍스처 메시를 `GroundTruth.align`으로 run 물체 좌표계에 옮겨 `offscreen_renderer.ModelRendererOffscreen`(생성자 기본값: ambient 1.0, 검정 배경, znear 0.1, zfar 2, 방향광 없음)으로 렌더. EGL, 640×480, 1.6 ms/장.
- MD-prior: `p2_build_prior.py`가 `exp_spring_anchor.build_runner`(9/22 하네스 초기화)를 그대로 따라 13 시퀀스 prior를 만든다(config_gs_2dgs_1mm_lifecycle.yml, KF4 스냅샷 포즈의 첫 5 KF, KF0에서 Sim(3) 정합, 초기 500 스텝, 첫 append 전 상태). 체크포인트·매니페스트(정합 Sim(3) 포함) `outputs/exp_vggt_probe2/models/prior/<seq>/`. 렌더는 `GaussianRunner._rasterize`(RGB+ED, alpha > 0.5).
- **MD-map 대체(계획 이탈)**: 9/22 reanchor_v2의 `gs_online/checkpoint_final.pt`는 9/24 승인된 덤프 정리에서 삭제됐다. 같은 실행의 `final/gs_online0/checkpoint_global.pt`(run_global_refine steps 0, pose_refine off)가 남아 있고 Gaussian 수가 온라인 체크포인트와 같다(AP12 578,765 = 578,765). 이를 MD-map으로 쓴다.
- FoundationPose: `/home/kist/Desktop/FoundationPose`에 설치·가중치(weights/2023-10-28-18-33-37, 2024-01-11-20-02-45)가 있으나, 사용자 메시지의 가중치 경로 줄이 템플릿 그대로여서 지시서 규칙에 따라 C단계는 건너뛴다.

## 0-a 렌더 정합성 — 통과 (9/26 12:25, `stage0a.txt`, `stage0a.csv`, `stage0_overlay_*.png`)
- AP12 (KF 171): |렌더 depth − 센서 depth| 중앙값 1.30 mm (부호 +0.45 mm), 실루엣–SAM2 마스크 IoU 중앙값 0.945 (p10 0.901).
- SM1 (KF 278): 1.72 mm (+0.13 mm), IoU 0.732 (p10 0.487; 손 가림, 마스크/실루엣 면적비 중앙값 0.754).
- 기준(< 10 mm, > 0.6) 충족. 오버레이에서 렌더가 실제 물체와 겹침을 눈으로 확인.

## 0-b 참고값 (9/26 12:40, `stage0b.txt`)
- MD-prior: AP12 depth 차 3.75 mm (부호 −0.84), IoU 0.806; SM1 2.61 mm (−1.11), IoU 0.605.
- MD-map(0-step global 체크포인트, 필터 없음): AP12 5.51 mm (−5.40), IoU 0.593; SM1 3.12 mm (−2.50), IoU 0.555. 오버레이에서 온라인 지도의 큰 Gaussian·floater 줄무늬가 보임(`stage0_overlay_MD-map_*.png`). 지시서대로 체크포인트를 그대로 렌더한다(필터 적용 안 함).

## 0-c 합성 복구 — 통과 (9/26 13:05, `stage0c.txt`, `stage0c.csv`)
- 실제 = render@GT(MD-gt), 초기 = GT를 모델 중심 주위 무작위 축으로 10°, AP12·SM1 각 20 KF. 추정 후 회전 중앙값(p90): E-V2 0.67 (1.85), E-V5 (b) 0.99 (2.65), (d) 1.02 (2.35), **E-V5×2 0.36 (0.78)**, E-L 0.31 (0.55), E-I 0.02 (0.09), E-V→I 0.01 (0.02). 기준(VGGT 중앙값 < 2°) 충족.
- 1차 실행은 AP12 20 KF 뒤 SM1로 넘어갈 때 pyrender EGL 오류로 중단(이전 시퀀스의 렌더러가 새 렌더러 생성 뒤 GC되며 공유 EGL display를 종료). 렌더러를 프로세스 수명 동안 유지하도록 고쳐 재실행(1회 재시도). 로그 `stage0c_attempt0_eglfail.log`.

## 하이퍼파라미터 고정 (A단계 시작 전, `p2_lib.HP`)
S 518(M-V), 배경 흰색, 공전각 10°(B는 10°·20°), 렌더 실루엣 최소 200 px(미만이면 실패로 P0 유지), (d) 강건 정렬 잔차 임계 10°, 스케일 = Probe 1 `frame_scale`(3 px 침식, depth > 0.1 m, conf 상위 50 %) 풀링 중앙값, LoFTR = Probe 1b B6 설정(400 px gray, thr 0.2, 3점 RANSAC 2000회 5 mm, 성공 ≥ 5 인라이어; 두 장 모두 P0라 면내 회전 0), ICP = point-to-plane, 2 mm voxel, 법선 반경 1 cm·30 이웃, 20→10→5 mm 각 30회. E-V→I의 ICP target은 E-V5×2 결과 포즈에서 새로 렌더한 depth(E-I의 "초기 포즈 렌더" 정의를 따름). 성공 5°·20 mm, 엄격 2°·10 mm, 악화 = 초기 + 2°.

## A·B단계 실행 (9/26 13:10 시작)
- GPU0: A MD-gt(HO3D 13) → A MD-map. GPU1: A MD-prior → B(MD-gt, MD-prior; AP12·MPM12·SM1 각 40 KF × δ 6개, E-V5 공전 10°·20°). 전 키프레임(한 개 거르기 없음). 진행 `progress_A_gpu0.txt`, `progress_A_gpu1.txt`, `progress_B.txt`.
- B 교란의 회전 중심은 GT 메시 중심(두 모델에 같은 초기 포즈), 앵커 공전 중심은 각 모델 자신의 중심.
- 13:20 관찰(판정 아님): A단계 AP10 첫 KF(C1 오차 0°)에서 E-V2가 175.6°. 디버그(`experiments/vggt_probe/p2_debug_ap10.py`, `debug_AP10_kf0_V2_input.png`): 같은 렌더 쌍을 합성 실제(render@GT)로 넣으면 0.006°. 실제 프레임은 손이 손잡이를 가려 물통이 거의 회전대칭으로 보이고, VGGT가 반대편 뷰로 읽음(뒤집힘). 렌더 텍스처와 실제 색도 다르다(도메인 차이). 코드 오류가 아니므로 실행을 계속한다.
- 14:00 D단계를 GPU0의 A(MD-gt → MD-map) 종료 뒤 자동 시작하도록 대기열에 넣음(PID 감시).

## A단계 MD-gt·MD-prior 완료 (9/26 18:20; 13 × 2 시퀀스, 전 키프레임, 실패 0)
- HO3D 13 평균(회전 중앙값 / 성공률; C1 4.09° / 39 %): MD-gt E-I 2.6° / 52 %, E-L 4.3° / 35 %, E-V2 80° / 1 %, E-V5×2 ≈ 21° / 1 %; MD-prior E-I ≈ 7° / 12 %, E-V5×2 ≈ 49° / 1 %. VGGT 계열은 두 모델 모두 C1을 크게 악화(악화율 75–97 %). 상세는 표.
- 이어서: GPU0 MD-map(시작 18:22) → D단계, GPU1 B단계(시작 18:22).

## 진단(지시서 단계 외, 결과 해석용): 렌더–실제 실패의 원인 분리 (9/26 18:45, `experiments/vggt_probe/p2_diag_domain.py`, `diag_domain.txt/.csv`)
MD-gt, AP12·MPM12·SM1 각 15 KF(C1 무관, GT 포즈 사용), VGGT 2장 상대 회전 오차 중앙값:
| seq | 렌더@GT–실제 | 실제(이전 KF)–실제 | 렌더–렌더 | 렌더–(실제 마스크 안을 렌더 색으로 바꾼 실제) |
|---|---|---|---|---|
| AP12 | 85.3° | 1.1° | 0.5° | 0.3° |
| MPM12 | 67.5° | 1.2° | 0.6° | 0.2° |
| SM1 | 35.4° | 1.9° | 1.4° | 0.6° |
→ 손 가림·실루엣·배경·코드 경로는 원인이 아니다(실루엣은 실제 그대로 두고 색만 렌더로 바꾸면 0.2–0.6°). 원인은 렌더(ambient 1.0 평면 조명 텍스처)와 실제 사진 사이의 외관 차이다. VGGT는 같은 도메인 쌍(실제–실제, 렌더–렌더)에서만 정확하다.

## B·MD-map·D 완료, 결과 문서 (9/26 20:30–21:10)
- B단계(AP12·MPM12·SM1 × 40 KF × δ 6개 × MD-gt·MD-prior, 공전 10°·20°) 완료, 실패 0. δ = 30°에서 MD-gt 성공률 E-I 60 %, E-L 36 %, E-V5×2 0 %, E-V→I 7 %.
- A단계 MD-map 13 시퀀스 완료: E-V2 14.3°(MD-gt 80.4°, MD-prior 105.7°), 그래도 C1(4.0°)보다 나쁨.
- D단계 7 시퀀스 완료: 구멍 비율 1–9 %; 대상(≥ 5 %)은 MPM10 하나, affine conf 상위 50 % 구멍 4.4 mm vs 센서 유효 3.3 mm → 충족.
- 판독 수치 `verdicts.txt`, 표 `tables.md`, 매니페스트 색인 `manifests/index.json`(78개), 결과 문서 `EXP_VGGT_PROBE2_RESULTS.md`.
- 커밋하지 않음(사용자 승인 대기).
- 9/26 21:20 사용자 승인으로 커밋.
