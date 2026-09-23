# 무인 실험 배치 2차 지시서 (2026-09-22, 약 20시간)

브랜치 `milestone5-v1` (`5c28f99`). 1차 배치(`EXP_BATCH_20260921_RESULTS.md`)의 결론 위에서 두 가지를 잰다.
질문할 수 없으니 아래 규칙과 결정 규칙대로 스스로 진행한다. 시간이 더 걸려도 된다.

**이번 배치의 두 질문**
- Q1 (C1 생사 실험): 포즈를 통제했을 때 SAM3D prior가 복원에 얼마나 기여하는가, 그리고 prior를 그냥 꽂는 것(frozen)과 lifecycle로 화해하는 것(full)의 차이가 있는가.
- Q2 (시스템 결정): 하이브리드(포즈 = 원본 NOF, 복원 = GS)를 온라인 조건으로 돌리면 4.6의 상한(HO3D P1 0.320)에 얼마나 가까운가. GS는 키프레임을 소비하는 순간의 포즈로 append하고, NOF는 그 뒤에도 포즈를 고치기 때문이다.

---

## 0. 규칙 (1차 배치와 동일)

**허용**: `experiments/` 아래 새 파일; `experiments/online_variants/` 복사본의 스위치 뒤 수정(스위치 없으면 기존과 동일); `logs/exp_batch_20260922/` 아래 설정·스크립트·로그·채점 JSON; `outputs/exp_batch_20260922/<arm>/<ds>/<seq>` 새 디렉터리; 결과 문서 `EXP_BATCH_20260922_RESULTS.md` 1개.
**금지**: 본 코드 수정(`gaussian_runner.py`, `prior_lifecycle.py`, `gaussian_global.py`, `sam3d_prior.py`, `bundlesdf.py`, `run_*.py`, `BundleTrack/`, 기존 `config*.yml`, 기존 `experiments/*.py`), commit/push/`build.sh`, 기존 산출물 삭제·덮어쓰기. `~/Desktop/BundleSDF_baseline_outputs/`는 읽기 전용. 본 코드 수정이 필요해 보이면 하지 말고 결과 문서 "승인 필요"에 적는다.
**운영**: 1차 배치의 `run_bsdfpose.sh`/`chain_bsdfpose.sh`를 템플릿으로 복사해 `nohup` 체인으로 띄우고 폴링. `logs/exp_batch_20260922/PROGRESS.md`를 단계마다 갱신(컨텍스트 압축 후 이 파일만 읽고 이어갈 수 있게). 실행마다 매니페스트(명령, 환경변수, HEAD, 복사본 diff, 시작/종료). 실패 시 1회 재시도 후 건너뜀, 실패 산출물 보존. 디스크 150 GB 미만이면 새 실행 금지. `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`. 컨테이너 `BundleSAM3DGS-gsplat-smoke`, `cd /home/kist/Desktop/BundleSAM3DGS`.

---

## 1. 공통 프로토콜

- **포즈**: 원본 BundleSDF SAM2-on 실행(`~/Desktop/BundleSDF_baseline_outputs/full_eval_sam2_<ds>/<ds>/<seq>`)의 키프레임 포즈. 영상은 그 실행의 `color/ depth_filtered/ mask/`. 1차 배치 4.6과 동일.
- **재생 엔진**: `experiments/exp_replay_bsdf_poses.py`의 로직(온라인과 같은 Sim(3) prior 정합 on keyframe 0, 초기 500 + 갱신 500, noop). 이 스크립트는 수정하지 말고, 이를 import·확장하는 **새 스크립트**를 쓴다. 러너 변형은 `BirthRunner`처럼 `GaussianRunner` 서브클래스로 구현한다(복사본 러너를 건드릴 필요가 없으면 건드리지 않는다).
- **채점**: 실행마다 메시 2종(global 2000 스텝 `final/gs`, 학습 0 `final/gs_online0`)을 `eval_mesh_cd.py`로. 정렬 게이지 `--run-dir` = 해당 BundleSDF 실행. YCB는 `logs/exp_batch_20260921/common_seen/<seq>.yml`을 `--keyframes-yml`로. `eval_map_quality.py`도 실행(global 체크포인트).
- **대조군 `full`** = 1차 배치의 `outputs/exp_batch_20260921/bsdfpose/<ds>/<seq>` (재실행하지 않는다). 채점 JSON `logs/exp_batch_20260921/cd_bsdfpose_*`.
- **잡음 바닥**: P1/P2 ±0.03 cm, seen ±0.03, unseen 커버리지 ±5 pt. 이 안은 "차이 없음".
- 22시퀀스 = HO3D 13 (AP10–14, MPM10–14, SB11, SB13, SM1) + YCB 9. HO3D가 우선이다.

---

## 2. Q1 — prior 4단 ablation (재생, 포즈 통제)

새 스크립트 `experiments/exp_prior_ablation_replay.py --prior-mode {none,frozen,initonly}` (`full`은 대조군 재사용).

| arm | 정의 |
|---|---|
| **none** | prior 없음. 첫 5 키프레임의 depth 점으로 초기화(본 코드의 no-prior 초기화 경로, `run_gaussian_incremental.py`가 쓰는 것과 같은 함수·정규화). lifecycle은 관측 계열만. |
| **frozen** | prior를 그대로 꽂기만 함. surfel은 온라인 Sim(3) 정합 후 **영원히 UNSEEN**(학습 없음 = frozen rows, SUSPECT/CONTRADICTED 전이 없음, 제거 없음). 관측 append는 평소대로. 구현: `classify_lifecycle`이 no-op인 서브클래스. 추출은 기존대로(UNSEEN 포함). |
| **initonly** | prior를 관측처럼 취급. 초기화 직후 prior 전부 VERIFIED(자유 학습), 이후 전이·제거 없음(`classify_lifecycle` no-op). |
| full | 현재 lifecycle (= 1차 4.6). |

- 실행 전 단위 확인(mustard0, 키프레임 10개까지만): frozen에서 prior 상태 카운트가 끝까지 전부 UNSEEN이고 Gaussian 수 = prior + append; initonly에서 전부 VERIFIED; none에서 lineage prior가 0개. 세 arm 모두 `full`과 같은 키프레임 수·같은 포즈를 읽었는지 매니페스트로 확인.
- 순서: **none × HO3D 13 → frozen × HO3D 13 → none × YCB 9 → frozen × YCB 9 → initonly × HO3D 13 → initonly × YCB 9**. initonly는 시간이 남을 때만.
- 판독(결과 문서에 반드시): arm별 HO3D/YCB 평균 P1·P2·seen·unseen 커버리지(메시 2종), `full` 대비 시퀀스별 차이와 잡음 밖 개수. 별도 표로 **prior가 틀린 시퀀스 6개**(cracker_box_yalehand0, sugar_box_yalehand0, bleach_hard, tomato: 스케일 오차 / MPM10, AP12: 배치 오차)를 따로 본다. 여기서 frozen vs full의 차이가 C1의 존재 증명이다.
- 결정 규칙(판정은 사용자가, 표는 정확히): (a) none vs full — 커버리지·P2에서 full이 이기면 "prior 기여 확인"; P1(seen 영역)에서 none이 이기면 그것도 그대로 적는다. (b) frozen vs full — 틀린 prior 6개에서 full이 P1/seen 기준 잡음 밖으로 이기고 나머지 16개에서 지지 않으면 "화해 기여 확인"; 22개 전부 잡음 범위면 "lifecycle이 현재 지표로는 무효"라고 그대로 적는다(이 경우 C1 설계를 다시 봐야 하므로 숨기지 말 것).

---

## 3. Q2 — 온라인 현실 조건 하이브리드 재생 (HO3D 13)

새 스크립트 `experiments/exp_hybrid_realistic_replay.py --arm {consume,reanchor}`. `exp_spring_anchor.BirthRunner`(출생 뷰 id)와 그 bake 로직을 재사용한다.

- **포즈 소스**: BundleSDF 실행의 **프레임별** `<frame>/keyframes.yml` 스냅샷(4.6은 마지막 스냅샷만 썼다). 키프레임 j의 "소비 시점 포즈" = j를 처음 포함하는 스냅샷 안의 j 포즈. "현재 스냅샷" = 키프레임 j를 처리하는 시점의 그 스냅샷.
- **consume**: 모든 키프레임을 소비 시점 포즈로 append하고 이후 절대 갱신하지 않음(가장 나쁜 경우 = 통합만 하고 아무것도 안 했을 때).
- **reanchor**: 키프레임 j를 처리하기 직전, 현재 스냅샷에서 이미 저장된 키프레임 i의 포즈가 저장값과 다르면(회전 > 0.01° 또는 이동 > 0.01 mm) `T_i = c2w_new @ inv(c2w_old)`(metric 물체 좌표계)를 i에서 태어난 Gaussian의 means·quats와 `observed_points_metric`의 해당 점들에 적용하고(정규화 좌표계로 변환해서: `T_norm = N T N⁻¹`, N은 등방 스케일+이동), 뷰 포즈를 c2w_new로 바꾼다. prior 계열은 움직이지 않는다. lifecycle 필드는 그대로. 그다음 j를 append.
- **건전성 검사(필수, AP12)**: (1) reanchor로 끝까지 재생한 뒤 마지막 스냅샷으로 한 번 더 재앵커하면 모든 뷰 포즈가 4.6의 포즈와 일치(최대 1e-4)하고, 그 지도의 online0 메시 P1이 4.6의 online0 P1과 잡음 범위 안이어야 한다. (2) 스냅샷 간 포즈 변화량 통계(키프레임별 첫 소비 포즈 → 최종 포즈의 회전°/이동 mm 중앙값·p90)를 시퀀스별로 기록한다. 이 값이 작으면 Q2는 자동으로 "통합만 하면 됨"이다.
- 판독: 시퀀스별·평균 P1/P2/seen/커버리지를 `own poses(fulleval 9/12)`, `consume`, `reanchor`, `full(4.6 상한)` 네 열로. reanchor − full 이 잡음 범위면 "온라인 하이브리드 = 상한". consume − full 이 크면 재앵커링이 통합에 필수라는 뜻.
- (옵션, 시간 남으면) `refreshonly`: 뷰 포즈만 갱신하고 층은 두는 arm = 현재 본 코드 `refresh_view_poses`가 하는 일.

---

## 4. 채우기 작업 (GPU가 빌 때만)

- **0/0 스텝 under BSDF poses** (HO3D 13, `initial_opacity 0.5`, 초기 0/갱신 0, 나머지는 `full`과 동일): 하이브리드에서 학습이 필요한지 결정. 실행당 몇 분이라 저비용. 판독은 online0 메시 P1·커버리지와 사이클 시간.
- initonly(2절)와 refreshonly(3절).

---

## 5. 순서와 예산

참고: 4.6 재생 1회 ≈ 25~35분(500/500 + global 2000 + 채점). GPU 2장.

1. 즉시: Q1 `none` 구현(가장 단순) → 단위 확인 → **GPU0에 none × HO3D 13** 시작. 동시에 `frozen` 구현(no-op 서브클래스) → **GPU1에 frozen × HO3D 13**.
2. 두 체인이 도는 동안 Q2 스크립트 작성 + 건전성 검사(1)을 AP12 1개로(공유 GPU).
3. GPU1 frozen HO3D 끝나면 **Q2 consume 13 → reanchor 13**. GPU0 none HO3D 끝나면 **none YCB 9 → frozen YCB 9**.
4. 남는 시간: 0/0 HO3D 13 → initonly → refreshonly.

시간이 모자라면 아래에서부터 생략: refreshonly → initonly → 0/0 → frozen YCB → none YCB. **none HO3D, frozen HO3D, Q2 consume/reanchor HO3D는 반드시 끝낸다.** 새 실험을 임의로 추가하지 않는다.

---

## 6. 결과 문서 `EXP_BATCH_20260922_RESULTS.md`

1. 맨 위 한국어 1페이지 요약: Q1 답(prior 기여 / 화해 기여, 각각 근거 수치 2개), Q2 답(온라인 하이브리드 P1과 상한과의 차이, 포즈 변화량 통계), 사용자가 결정할 것.
2. 2절·3절·4절 표 전부(시퀀스별, 평균, `full` 대비 차이, 잡음 표시), 틀린 prior 6개 별도 표.
3. 계획 이탈·실패·미완 항목과 재개 방법.
4. "승인 필요": 본 코드에 반영할 가치가 있는 변경(예: reanchor 로직을 `refresh_view_poses`에 넣는 diff)을 제안만.
5. 산출물 색인(실행 디렉터리, JSON, 스크립트).
