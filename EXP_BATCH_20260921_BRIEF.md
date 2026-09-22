# 무인 실험 배치 지시서 (2026-09-21, 약 40시간)

브랜치 `milestone5-v1` 기준. 사용자는 약 40시간 자리를 비운다. 이 문서가 그동안의 **유일한 지시**이며, 질문할 수 없으니 아래 규칙과 결정 규칙대로 스스로 진행한다. 시간이 더 걸려도 된다(출근 직후 GPU를 쓸 일이 없음).

---

## 0. 사전 승인 범위 (CLAUDE.md의 "수정 전 승인" 규칙을 이 범위에 한해 대체)

**허용 (추가 승인 없이 진행)**
- `experiments/` 아래 **새 파일** 작성.
- `experiments/online_variants/`의 실험 복사본(`gaussian_runner.py`, `prior_lifecycle.py`) 수정. 단 모든 변경은 `GS_ONLINE_VARIANTS` 스위치 뒤에 두고, **스위치가 없으면 지금과 완전히 같게 동작**해야 한다. 필요하면 `gaussian_global.py`의 실험 복사본을 같은 디렉터리에 새로 둔다.
- `logs/exp_batch_20260921/` 아래 설정 복사본(yml), 실행 스크립트, 진행 로그, 채점 JSON.
- `outputs/exp_batch_20260921/<cfg>/<ds>/<seq>` 형태의 **새** 출력 디렉터리.
- 레포 루트에 결과 문서 **새 파일 1개**: `EXP_BATCH_20260921_RESULTS.md`.

**금지**
- 본 코드 수정: `gaussian_runner.py`, `prior_lifecycle.py`, `gaussian_global.py`, `sam3d_prior.py`, `bundlesdf.py`, `run_*.py`, `BundleTrack/`, `mycuda/`, 기존 `config*.yml`, 기존 `experiments/*.py`(채점기 포함 — 비교 가능성 유지).
- commit / amend / tag / push, `build.sh` 실행.
- 기존 `outputs/`, `logs/`, 체크포인트의 삭제·덮어쓰기. `~/Desktop/BundleSDF_baseline_outputs/`는 읽기 전용.
- 본 코드 수정이 필요해 보이는 일을 만나면 **하지 말고** 결과 문서의 "승인 필요" 절에 적고 다음 항목으로 넘어간다.

**무인 운영 규칙**
- 사용자 입력을 기다리며 멈추지 않는다. 실패하면 원인을 기록하고 1회 재시도, 또 실패하면 건너뛴다(실패 산출물은 증거로 보존).
- GPU 작업은 `logs/exp_fusion_20260915/`의 `run_one.sh`, `chain_gpu*.sh`를 **템플릿으로 복사**해 `nohup` 체인으로 띄우고 폴링한다(에이전트 세션이 멈춰도 GPU 작업은 계속되도록).
- `logs/exp_batch_20260921/PROGRESS.md`를 각 단계 시작/종료마다 갱신한다(무엇을 했고, 무엇이 돌고 있고, 다음은 무엇인지, 결정 규칙의 판정 결과). 컨텍스트가 압축돼도 이 파일만 읽고 이어갈 수 있게 쓴다.
- 실행마다 매니페스트(정확한 명령, 환경변수, git HEAD, 실험 복사본의 diff, 설정 yml, 시작/종료 시각)를 남긴다.
- GPU: HO3D 큰 시퀀스(AP10, AP12)는 GPU1. 같은 GPU에 HO3D 실행 2개를 동시에 올리지 않는다. `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`. OOM이면 다른 GPU에서 1회 재시도.
- 디스크: 각 실행 전 여유 공간 확인. 150 GB 미만이면 새 실행을 띄우지 말고 PROGRESS에 기록 후 CPU 전용 항목만 진행(삭제로 해결하지 않는다). 템플릿이 쓰던 debug level·덤프 설정을 그대로 따른다.
- 컨테이너: GS 작업은 `BundleSAM3DGS-gsplat-smoke`, `docker exec` 안에서 `cd /home/kist/Desktop/BundleSAM3DGS`.

---

## 1. 배경 — 검증하려는 가설

- **H1 (방향 초기화)**: `gaussian_runner._new_splat_values`가 append되는 관측 Gaussian의 quat을 `[1,0,0,0]`으로 둔다. 3DGS(등방) 시절엔 무해했지만 2DGS에서는 disk 법선 = 회전의 3번째 축이므로 모든 관측 disk가 물체 좌표계 z축(≈첫 카메라 광축)을 향한 채 태어난다. prior 쪽은 `quats_from_normals`로 올바르게 초기화된다. update의 quat lr은 1e-4/step이라 큰 각도는 사실상 교정되지 않을 것이다. 예상 결과: 관측 Gaussian의 opacity<0.1 비율 73~80%(B6), 렌더 depth 오차, 회전이 큰 HO3D에서의 지도 품질 저하. **먼저 코드에서 이 초기화가 실제로 그런지, 다른 곳에서 방향이 덮어써지지 않는지 확인할 것.** 아니라면 그 사실을 기록하고 H1 항목은 확인 분석(3.1)만 수행.
- **H2 (학습의 기여)**: update에서 means는 0.28 mm밖에 못 움직이므로 지도의 기하는 사실상 "역투영 점 + prior"다. 학습(500/500 스텝)이 메쉬에 기여하는 양을 모른다.
- **H3 (이중층)**: 관측 영역에서 VERIFIED prior(opacity 0.9)와 관측층(출생 opacity 0.1)이 2~5 mm 간격으로 공존한다.
- **H4 (스프링)**: 키프레임 k의 포즈를 고쳐도 k에서 태어난 Gaussian은 출생 위치에 남는다(`_bake_pose_deltas`, `refresh_view_poses`는 뷰 포즈만 변경, 출생 뷰 id 없음). k 자신의 층이 출생 포즈에서 k의 영상을 완벽히 설명하므로 포즈 보정을 되돌리는 힘으로 작용한다. 지도가 포즈의 함수가 아니어서 BA가 성립하지 않는다.
- **H5 (Adam 잡음 바닥)**: v1은 스텝당 뷰 1개, `[N,6]` 텐서에 dense Adam(lr 0.01), 사이클마다 0에서 재시작. 순수 잡음 기울기만으로도 사이클당 2~3°, 2~3 mm 이동이 나온다(시뮬레이션). 실제 로그 이동량과 같은 자릿수.

---

## 2. 공통 프로토콜

- **B-트랙 4시퀀스**: `mustard0`, `cracker_box_yalehand0`(YCB), `AP12`, `MPM12`(HO3D).
- **파이프라인**: 공식 진입점(`run_custom.py` / `run_ho3d.py`)을 `experiments/online_variants/launch.py`로 실행(09-15 배치와 동일, 배선 검증됨). 러너 설정 `config_gs_2dgs_1mm_lifecycle.yml`, `--gs_initial_steps 500 --gs_update_steps 500`, global 2000 스텝, SAM2 마스크, SAM3D prior 경로는 템플릿과 동일.
- **1단계(지도 실험)는 전부 `--gs_feedback noop`**. 이유: 지금까지 지도 변경을 "v1 피드백 on 상태의 ADD"로 판정했는데 피드백 루프가 고장난 상태라 결과가 섞인다. noop이면 트래커 포즈가 지도와 무관하므로 ADD는 arm 간에 잡음 범위에서 같아야 한다(**같지 않으면 배선 오류로 보고 조사**). 판정은 지도·메쉬 지표로만 한다.
- **2단계(포즈 실험)는 `--gs_feedback on`**, 판정은 ADD.
- **실행마다 채점**: `eval_add_ycbineoat.py`(ADD/ADD-S), `eval_mesh_cd.py`(P1/P2/P3), `eval_map_quality.py`, `analyze_map_layers.py`. 09-15 배치의 채점 스크립트를 그대로 재사용.
- **메쉬는 실행당 2개를 채점**: (a) 온라인 최종 지도에서 **추가 학습 없이** 추출한 메쉬(`gs_online/checkpoint_final.pt` → `extract_meshes`; steps 0이 막혀 있으면 experiments 스크립트에서 직접 호출), (b) 표준 global 2000 스텝 후 메쉬.
- **비교는 같은 배치의 대조군하고만** 한다. 과거 표(fulleval 등)는 참고 열로만.
- **잡음 바닥**: ADD HO3D ±0.2 cm, YCB ±0.05 cm, P1/P2 ±0.03 cm, unseen 커버리지 ±5 pt. 이 안의 차이는 "차이 없음"으로 적는다.
- 단일 변수 원칙: arm 하나는 스위치 하나만 다르다.

---

## 3. 0단계 — 읽기 전용 확인 (GPU 거의 불필요, 가장 먼저)

### 3.1 관측 Gaussian 방향 분석 (H1)
새 스크립트 `experiments/analyze_observed_normals.py`.
- 입력: `outputs/fulleval_20260912`의 mustard0, AP12, MPM12 실행. 체크포인트 두 개씩: `gs_online/checkpoint_final.pt`, `final/gs/checkpoint_global.pt`.
- 대상: 관측 계열 Gaussian(lifecycle lineage=observed). opacity ≥0.1 / <0.1로 나눠 보고. 비교 기준으로 prior 계열 VERIFIED도 같은 분석.
- 법선 n = `quat→R`의 3번째 열(`_splat_normals`와 같은 정의). disk는 양면이므로 각도는 `acos(|a·b|)`.
- 기준 법선 n_gt: `eval_map_quality.py` / `analyze_map_layers.py`와 같은 정렬(첫 온라인 포즈 + ICP 2 cm)로 GT 메쉬를 지도 좌표계로 옮긴 뒤 최근접 GT 면의 법선. GT 표면에서 5 mm 이내 Gaussian만.
- 계산: e0 = angle(ẑ, n_gt) ("한 번도 학습 안 됐다면 가졌을 오차"), e1 = angle(n, n_gt) (현재 오차), a_z = angle(n, ẑ).
- 보고: e0 구간(0–15, 15–30, 30–45, 45–60, 60–90°)별 개수, e1 중앙값/p25/p75, opacity<0.1 비율. 전체 a_z 히스토그램.
- **판정 규칙**: e0>45° 구간에서 e1 중앙값 >30°인 시퀀스가 3개 중 2개 이상이면 "H1 확정", e1 중앙값 <15°이면 "학습이 교정함(H1 기각)", 그 사이면 "부분". 또 e0가 클수록 opacity<0.1 비율이 단조 증가하는지 적는다. **어떤 판정이든 4.1은 실행한다**(초기화가 옳아서 손해 볼 일은 없으므로). 판정은 4.1의 기대 효과 크기 해석에 쓴다.

### 3.2 평가 정비 (코드 수정 없이)
- YCB의 seen/unseen 라벨은 `eval_mesh_cd.py`가 `--run-dir`의 `keyframes.yml` 키프레임 GT 포즈로 계산한다. 실행마다 키프레임 집합이 달라 두 방법의 "unseen" 분모가 다를 수 있다. **공통 라벨**: 시퀀스별로 GT 포즈가 있는 전 프레임(많으면 stride 5)을 `keyframe_<id>` 키로 나열한 yml을 `logs/exp_batch_20260921/common_seen/<seq>.yml`로 만들고 `--keyframes-yml`로 넘긴다.
- YCB 9개에 대해 우리 메쉬(`fulleval_20260912`의 `final/gs`)와 BundleSDF SAM2 참조 메쉬를 공통 라벨로 재채점. 기존 JSON은 건드리지 않고 새 JSON으로.
- 기존 `cd_ref_*` JSON에서 BundleSDF의 seen 영역(GT→pred seen) 값을 22개 전부 표로 뽑아 우리 값과 나란히 둔다(HO3D는 기존 JSON 그대로, YCB는 공통 라벨 재채점 값).
- 먼저 확인: 기존 `cd_ref_*` 채점 때 어떤 keyframes.yml이 쓰였는지(JSON의 `seen_source` 필드). 결과 문서에 기록.

---

## 4. 1단계 — 지도 바로잡기 (noop)

순서대로. 4.1의 판정이 이후 arm의 **기반(base)**을 정한다.

### 4.1 법선 초기화 (H1) — 스위치 `normalinit`
- 실험 복사본에서 append 경로만 변경: 새로 append되는 점의 quat을 표면 법선으로 초기화.
- 법선 추정: 해당 update의 **novelty 필터 이전 전체 후보 점군**(`_frames_to_cloud` 결과)을 이웃 탐색용으로 쓰고, append될 novel 점마다 반경 4 mm(최소 6개 이웃) PCA 법선. 이웃 부족 시 점→카메라 중심 방향으로 대체. 부호는 출처 프레임의 카메라 중심(물체 좌표계)을 향하게. 변환은 `sam3d_prior.quats_from_normals` 재사용(+z → 법선, wxyz).
- 좌표계 주의: 점은 metric 물체 좌표계, 정규화는 등방 스케일+이동이라 법선 방향은 그대로.
- 단위 확인(실행 전): 합성 평면/구 점군으로 추정 법선 오차 <5°, 생성 quat의 3번째 축이 법선과 일치, 스위치 off일 때 기존과 비트 단위로 같은 출력.
- 실행: 대조군(스위치 없음) + `normalinit` × 4시퀀스 = 8실행, noop.
- 지표: 관측 opacity<0.1 비율, 렌더 depth 잔차(중앙값/p90), 중심→GT, big/far 개수, Gaussian 수, P1/P2/seen/unseen(메쉬 2종), 3.1 분석을 새 체크포인트에 재실행(e1 분포).
- **기반 채택 규칙**: 4시퀀스 중 3개 이상에서 [관측 opacity<0.1 비율 10 pt 이상 감소 **또는** 렌더 depth 잔차 중앙값 10% 이상 감소]이고, 어떤 시퀀스에서도 P1(global 메쉬)이 대조군보다 0.03 넘게 나빠지지 않고 unseen 커버리지가 5 pt 넘게 떨어지지 않으면 **이후 모든 arm의 기반 = `normalinit`**. 아니면 기반 = 현재 코드(그래도 결과는 전부 보고).

### 4.2 sh_degree 0
- 설정 복사본 `logs/exp_batch_20260921/config_*_sh0.yml`(`sh_degree: 0`). 근거: `shN`은 0으로 초기화되고 update lr 1.25e-5, init 500스텝 동안 active degree 0이라 끝까지 거의 0 — 메모리(Gaussian당 59→14 파라미터)만 쓴다.
- 먼저 mustard0 스모크(shN 크기 0 텐서로 optimizer·strategy·체크포인트·PLY 내보내기가 도는지). 실패하면 원인 기록 후 건너뜀(본 코드 수정 금지, 복사본에서 사소하게 해결되면 스위치 뒤에).
- 실행: 기반 + sh0 × (AP12, mustard0), noop. 지표는 4.1과 같고 추가로 GPU 최대 메모리, 사이클 시간. 판정: 모든 지표가 잡음 범위 안이면 "채택 권고".

### 4.3 스텝 ablation (H2)
- 0/0에서는 관측 Gaussian의 opacity가 출생값 0.1에 머물고 추출 필터가 `opac > 0.1`(strict)이라 관측층이 전부 빠진다. 그래서 이 묶음은 `initial_opacity: 0.5` 설정 복사본을 공통으로 쓴다.
- arm(모두 noop, 기반 위): **S-ref** = 500/500, opacity 0.1(=4.1의 기반 실행 재사용) / **S-A** = 500/500, o=0.5 / **S-B** = 500/100, o=0.5 / **S-C** = 0/0, o=0.5. 4시퀀스.
- 0/0에서도 lifecycle 분류와 append는 그대로 돈다(`train(steps=0)`은 허용됨). 돌지 않으면 복사본에 스위치로 "학습 호출만 생략".
- 핵심 판독은 **추가 학습 없는 온라인-최종 메쉬**(2절 (a)). S-C의 (a)가 "학습 0인 시스템"의 성적이다. S-A vs S-ref는 출생 opacity 효과를 noop에서 깨끗하게 다시 잰 것(과거 0.5 기각은 피드백 on의 ADD 기준이었음).
- 보고: arm별 P1/P2/seen/unseen(메쉬 2종), 지도 지표, 사이클 시간, 전체 실행 시간.

### 4.4 추출 단계 이중층 테스트 (H3, CPU)
새 스크립트 `experiments/exp_extract_supersede.py`. 학습 0, 기존 체크포인트만 사용.
- `gaussian_global.gaussian_surfels`의 필터를 스크립트 안에서 재현하고(본 코드 수정 금지) 조건 하나를 추가: prior 계열 VERIFIED surfel 중, 반경 r 안에 "살아 있는" 관측 계열 Gaussian(opacity ≥0.1, 반지름 ≤10 mm, 거리 ≤1.25)이 k개 이상이면 제외. (r, k) = (3 mm, 3), (5 mm, 3) 두 가지.
- 이후 Poisson·후처리는 현재 추출과 동일한 함수/설정.
- 대상: `fulleval_20260912` 22개의 `final/gs/checkpoint_global.pt` + 4.1의 기반 실행 4개. 같은 체크포인트의 **현재 추출을 같은 스크립트로 다시 뽑은 것**이 대조군(재현 확인: 기존 표와 ±0.01 이내).
- 지표: P1/P2, pred→GT, GT→pred seen/unseen, 제외된 surfel 수, 구멍(메쉬 성분 수, seen 5 mm내 비율 하락 여부).
- GPU 작업과 병렬로 CPU에서 돌린다.

### 4.5 scale 상한 — 스위치 `scaleclamp`
- 복사본 `train()`에서 `optimizer.step()` 직후 `splats["scales"].data[:, :2].clamp_(max=log(cap))`, cap = 정규화 단위로 환산한 면내 반지름 5 mm(전 Gaussian 공통). 삭제가 아니라 clamp.
- 실행: 기반 + `scaleclamp` × 4시퀀스, noop. 지표: big 개수(0이어야 함), P1/P2/seen/unseen, 렌더 depth 잔차, opacity 분포.

### 4.6 BundleSDF 포즈 재생 (GPU가 빌 때 채우는 작업, 우선순위 낮음)
- 목적: "좋은 포즈가 주어지면 우리 지도의 메쉬가 어디까지 가는가" = 하이브리드(포즈는 NOF, 재구성은 GS) 시스템의 성적표.
- `run_gaussian_incremental.py`(`--prior-*`, lifecycle 지원)로 `~/Desktop/BundleSDF_baseline_outputs/`의 SAM2-on 실행 22개를 재생. 포즈는 각 실행의 **마지막 `keyframes.yml`**(NOF가 고친 최종 키프레임 포즈). 입력 영상은 그 실행의 `color/ depth_filtered/ mask/`; 없으면 데이터셋 원본 rgb/depth + SAM2 마스크로 대체하고 "depth_filtered 아님"을 교란 요인으로 표시.
- 기반 설정(4.1 판정 반영), 500/500, global 2000, 메쉬 2종 채점. 정렬 게이지용 `--run-dir`은 해당 BundleSDF 실행.
- 22개가 부담되면 HO3D 13개 우선.

---

## 5. 2단계 — 지도를 포즈의 함수로 (H4, H5)

### 5.1 오프라인 스프링 테스트 (관문)
새 스크립트 `experiments/exp_spring_anchor.py`(`exp_c_sim_recovery.py`, `exp_feedback_gradient_probe_gtmap.py`의 재생·주입·GT 포즈 로딩을 재사용).
- 재생 소스: `outputs/gsfb_v1_{ycb,ho3d}_<seq>_20260907`(`color/ depth_filtered/ mask/` 보유) 중 mustard0, AP12.
- 깨끗한 기준을 위해 **모든 키프레임에 GT 포즈**를 쓴다. 단 표적 키프레임 k\* 하나에만 **append 이전에** 오차를 주입한다(회전 3°, 이동 5 mm, 무작위 방향, 시드 고정). 즉 k\*의 점들은 틀린 포즈에서 태어난다. **주입을 append 뒤에 하면 자기 층이 옳은 자리에 남아 복구를 도와 결과가 부풀려지므로 금지.**
- k\*: 키프레임 10, 20, 40번째(시퀀스가 짧으면 가능한 것만). k\* 이후 10개 키프레임을 더 재생한 뒤(앞뒤 층이 겹치도록) 포즈 최적화 1사이클(500스텝; 추가로 1500스텝도).
- arm 2×2:
  - **current**: 지금 구조(Gaussian은 월드에 고정, 뷰 포즈 델타만).
  - **anchored**: 관측 Gaussian마다 출생 뷰 id `b`를 둔다. 렌더에 쓰는 유효 위치 `means_eff = R_Δ[b]·means + t_Δ[b]`, 유효 quat = `q_Δ[b] ⊗ q`(정규화 물체 좌표계; 포즈 델타는 `c2w' = T_i @ c2w`로 같은 좌표계에서 왼쪽 곱이므로 같은 T_i를 그대로 적용). prior 계열과 첫 키프레임 출생분은 항등에 앵커. 최적화 후 bake: 출생 그룹별로 저장된 means/quats에 변환 적용, novelty DB(`observed_points_metric`)에도 같은 변환(점별 출생 id 병렬 배열). 제거(`_remove_contradicted`)·append 때 id 배열을 같이 유지.
  - × {지도 고정(포즈만), joint(지도+포즈)}.
- **anchored 구현의 필수 건전성 검사**(통과 못 하면 결과를 쓰지 말 것):
  1. 델타 0에서 anchored 렌더 = current 렌더(최대 절대차 <1e-5).
  2. 지도에 뷰 k의 자기 층만 있을 때(다른 층·prior 숨김) 뷰 k의 loss의 Δ_k에 대한 기울기 ≈ 0(다른 층이 있을 때의 기울기 노름 대비 <1e-3).
  3. bake 후 재렌더 = bake 전 델타 적용 렌더.
- 지표: k\*의 회전·이동 복구율(1 − 이후 오차/주입 오차), 깨끗한 키프레임들의 드리프트(중앙값, 최댓값), 시드 2개 × k\* × 시퀀스별 표.
- **관문**: joint 조건에서 anchored의 회전 복구율 중앙값 ≥50% **이고** current ≤20% **이고** 깨끗한 키프레임 드리프트 중앙값 <0.3° / 0.5 mm → 5.2의 `anchor` arm 진행. 아니면 `anchor` arm은 실행하지 않고 원인 분석(어떤 조건에서 복구가 되는지/안 되는지)을 보고.

### 5.2 온라인 6시퀀스 (`--gs_feedback on`)
- 시퀀스: AP12, AP10, SM1, MPM11(HO3D), mustard0, tomato_soup_can_yalehand0(YCB). AP10·AP12는 GPU1.
- arm(모두 4.1에서 정한 기반 위, 500/500):
  - **ctrl**: v1 그대로.
  - **poselr001**: 설정 복사본에서 `pose_feedback.lr` 0.01 → 0.001만 변경(H5). **5.1 관문과 무관하므로 GPU가 비는 즉시(1단계 중이라도) 돌려도 된다.**
  - **anchor**: 5.1을 온라인 복사본에 이식(스위치 `anchor`). 관문 통과 시에만.
  - **anchor+poselr001**: 시간이 남으면.
- 지표: ADD/ADD-S(시퀀스별, HO3D 평균, 비붕괴/붕괴 분리), 같은 배치의 noop ADD(4.1 기반 실행 또는 필요 시 추가 noop 실행)를 "트래커 단독" 기준선으로, `feedback_log.json`의 사이클당 이동량(회전·이동 중앙값/p90), 메쉬 P1/P2.
- 판독 기준(판정은 사용자가 함, 표만 정확히): 비붕괴 시퀀스(AP12, MPM11)가 noop 대비 ±0.2 안인가, 붕괴 시퀀스(SM1, AP10)가 개선되는가, YCB 손실이 ±0.05 안인가. H5: poselr001에서 ctrl 대비 해악(ADD − noop)이 줄어드는가, 이동량이 lr에 비례해 줄었는가.

---

## 6. 우선순위와 시간 예산

참고 시간: mustard0 약 7분, HO3D 31~61분/시퀀스(온라인), global+채점 추가. 4시퀀스 1세트 ≈ 2.5 GPU-시간, GPU 2장.

1. 3.1 → 3.2 (CPU 위주, 즉시)
2. 4.1 (8실행) — 끝나면 기반 판정을 PROGRESS에 기록
3. 4.4 (CPU, GPU 작업과 병렬) / 5.1 코드 작성·건전성 검사(GPU 대기 시간에)
4. 4.3, 4.5, 4.2
5. 5.1 실행 → 관문 판정
6. 5.2 (`poselr001`은 GPU가 비면 앞당겨도 됨)
7. 4.6 (남는 GPU 시간)

시간이 모자라면 아래에서부터 생략: 4.6 → 5.2의 `anchor+poselr001` → 4.2. 3.1, 4.1, 4.4, 5.1은 반드시 끝낸다. 모든 항목이 끝났는데 시간이 남으면 4.1·4.3의 HO3D arm을 1회 반복해 잡음을 잰다(새 실험을 임의로 만들지 않는다).

---

## 7. 최종 보고 — `EXP_BATCH_20260921_RESULTS.md`

1. **맨 위에 한국어 1페이지 요약**: 가설 H1~H5 각각 "확정/기각/불명확"과 근거 수치 1~2개, 채택 권고 목록, 사용자가 결정해야 할 것.
2. 항목별 표(모든 arm·시퀀스, 대조군 대비 차이, 잡음 범위 표시), 결정 규칙의 판정 과정.
3. 계획에서 벗어난 점과 이유, 실패·건너뛴 실행과 원인, 미완 항목과 재개 방법.
4. "승인 필요" 절: 본 코드에 반영할 가치가 있어 보이는 변경을 diff 수준으로 제안(적용은 하지 않음).
5. 산출물 경로 색인(실행 디렉터리, 채점 JSON, 스크립트) — `OUTPUTS_INDEX.md` 형식으로, 단 그 파일 자체는 수정하지 말고 이 문서 안에.
