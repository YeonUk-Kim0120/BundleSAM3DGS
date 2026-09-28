# EXP_BATCH_20260928 진행 기록 (4차 무인 배치, 2026-09-28 시작)

지시서 `EXP_BATCH_20260928_BRIEF.md` (브랜치 milestone5-v1, HEAD 8096fd0). 본 코드 무수정, 커밋 금지(지시서). 이 파일만 읽고 이어갈 수 있게 단계마다 갱신한다.

## 재개 방법 (요약)
- 컨테이너: GS arm = `BundleSAM3DGS-gsplat-smoke`; N arm = `bundlesdf`(9/2 SDF 피드백 공정 ablation에 쓴 컨테이너, /home 마운트, `/usr/bin/python3`, torch 2.6.0+cu124). 9/28에 `docker start bundlesdf`로 재기동.
- 입력: `outputs/exp_batch_20260927/noop/<ds>/<seq>` (읽기 전용). 산출물 `outputs/exp_batch_20260928/<arm>/<ds>/<seq>`, 로그·매니페스트 `logs/exp_batch_20260928/`.

## 계획 이탈 (시작 시점)
- **사이클별 `gs_state.ply` 없음**: 지시서 입력 목록의 사이클별 `gs_state.ply`는 9/28 사용자 승인 덤프 정리(`logs/cleanup_20260928`)에서 삭제됐다(이 배치 지시서 수령 전). 또 이 파일은 상태 표시용 PLY라 러너 상태(SH·최적화기·뷰)를 복원할 수 없다. 대신 G0/G0B의 "사이클 K 직후 온라인 지도"는 3차 R0와 같은 재생(`init_runner_like_online` + 사이클마다 noop 포즈 갱신 + `update(500)`, 초기화 500)으로 다시 만들고 해당 사이클에서 체크포인트로 저장해 쓴다. 온라인 지도와는 GPU 비결정성만큼 다르다.
- 원본 기준 실행의 라운드별 `nerf/config.yml`은 HO3D에만 있고(예: AP12 `<frame>/nerf/config.yml`), YCB(mustard0)는 최종 `final/nerf/config.yml`과 실행 수준 `config_nerf.yml`만 있다 → mustard0는 `final/nerf/config.yml` 값을 쓴다(경로 키 제외).

## 1단계: 구현·단위 확인 (9/28 01:10–01:40)
스크립트(전부 새 파일, 본 코드 import만): `experiments/exp_round_inputs.py`(입력 JSON), `experiments/exp_nerf_round.py`(N 하네스, bundlesdf 컨테이너), `experiments/exp_gs_joint_refit.py`(G0/G0B/G1/G3 + 지도 재생, gsplat 컨테이너), `experiments/joint_refit_eval.py`(d·회전·이동, 개선/악화), `experiments/joint_refit_unit_eval.py`(단위 판정).
- N 하네스 변환 검증: `lrate_pose=0`으로 한 라운드 → 입력=출력 최대 |Δ| 5.5e-8 (sc_factor·translation·GL/CV 변환·첫 키프레임 고정 정확).
- 입력 포즈 선택 확인: 원본 `run_nerf`는 라운드 시점 트래커의 전 키프레임 현재 포즈(`p_dict['cam_in_obs']`)를 쓴다. 우리 입력 `consume`(키프레임이 처음 들어온 사이클의 `poses_before_gs`)과 사이클 K의 포즈 차이는 전 시퀀스·전 K에서 중앙값 0.00–0.01 mm, 최대 0.19 mm → 동일하다고 보고 그대로 쓴다.

단위 확인 (mustard0, 키프레임 20개, 500 스텝; `logs/exp_batch_20260928/unit_tests/unit_results.txt`):

| arm | (1) GT 입력 → d 중앙값 (기준 < 1 mm) | (2) KF10 GT+3° d 전→후 (기준: 감소) | (2) 자기 (1) 결과 대비 KF10 차이 2.82 → | 나머지 (2) vs (1) | 시간 |
|---|---|---|---|---|---|
| N | 3.51 mm 실패 | 2.82 → 2.34 통과 | 0.40 mm (86 % 제거) | 3.73 vs 3.51 | 15 s, 1.7 GB |
| G0 | 1.49 mm 실패 | 2.82 → 2.00 통과 | 0.90 mm (68 %) | 1.61 vs 1.57 | 6 s |
| G0B | 2.43 mm 실패 | 2.82 → 2.23 통과 | 0.14 mm (95 %) | 2.64 vs 2.45 | 32 s |
| G1 | 5.05 mm 실패 | 2.82 → 6.42 실패 | 0.37 mm (87 %) | 4.91 vs 4.99 | 25 s |
| G3 | 4.23 mm 실패 | 2.82 → 3.79 실패 | 0.27 mm (90 %) | 4.18 vs 4.29 | 26 s |

- G0/G0B 단위 지도: GT 포즈로 같은 재생을 해서 만든 지도(`outputs/exp_batch_20260928/unit/g0map_gt_mustard0/K20.pt`, 20뷰, 40,945 Gaussian).
- **진단 1 (N (1) 실패 원인)**: GT 마스크(`datasets/YCBInEOAT/mustard0/gt_mask`) + 원시 depth로 같은 (1)을 다시 돌려도 d 중앙값 3.42 mm(p90 4.32)로 그대로다(`unit/Ndiag_gt_gtmask_rawdepth`). 저장된 키프레임 depth는 SAM2 마스크 밖이 0이라 원시 depth를 같이 바꿨다(하네스에 진단 전용 `--mask-dir/--depth-dir` 추가, 기본 동작 불변). → 마스크 오차가 원인이 아니고, 원본 방법 자체가 GT에서 한 라운드에 약 3.5 mm 떨어진 고정점으로 간다(GT 포즈·depth 사이 불일치 또는 방법 고유 편향).
- **진단 2 (G1·G3 (2) 실패 원인)**: 각 arm의 (1) 결과(교란 없음)를 기준으로 재면 모든 arm이 3° 교란의 68–95 %를 제거한다. G1·G3의 (2) 실패는 교란을 못 고쳐서가 아니라 모든 키프레임이 같이 4–5 mm 밀리는 공통 이동 때문이다. G1 새 지도 중심→GT 메시 중앙값 3.90 mm(prior 모양·정합 오차), G3 TSDF 지도 2.33 mm(뷰 5 렌더 depth 잔차 2.39 mm).
- **판단**: 기준 (1)은 원본 N도 통과하지 못하므로 구현 결함 검출 기준으로 쓸 수 없다. 하네스 변환은 정확히 검증됐고, 모든 arm이 교란된 키프레임을 자기 고정점 쪽으로 되돌리므로 공동 재적합 메커니즘은 동작한다. 단위 확인 "실패"를 그대로 기록하고 본 실행을 진행한다(결과 문서의 계획 이탈·판단 항목에 적는다). 공통 이동(고정점 편향)은 본 실행에서 d 증가로 나타날 것이다.
- 연기 확인(100 스텝, `outputs/exp_batch_20260928/smoke/`): N·G1·G3 모드 I 경로가 mustard0 8라운드를 끝까지 돈다. 참고로 N은 100 스텝에서도 라운드마다 d가 늘었다(7.50 → 13.25 mm, noop 7.50).

## 2단계: 본 실행 (9/28 01:42 시작)
- 실행기: `logs/exp_batch_20260928/run_job.sh`(셀 1개, 매니페스트 `manifests/<tag>.json`, 로그 `runs/<tag>.log`, 디스크 150 GB 미만이면 중지, 미완 출력은 `outputs/exp_batch_20260928/_failed/`로 이동), `chain.sh`(실패 1회 재시도), `wait_chain.sh`(선행 체인 DONE 대기 후 실행), 진행 기록 `logs/exp_batch_20260928/progress`(KST).
- GPU0 (`bundlesdf`): `jobs/gpu0_N.txt` = N 모드 S 500(11셀) → 2000(11셀) → 모드 I(SM1 → AP12 → MPM12 → mustard0) → 반복(AP12 last 500 rep1).
- GPU1 (gsplat): G0 지도 재생 4개 병렬(`jobs/gpu1_maps_<seq>.txt`, noop 포즈, 출력 `outputs/exp_batch_20260928/g0maps/<ds>/<seq>/K<k>.pt` + `maps.json`) → `jobs/gpu1_S500.txt`(G0 → G0B → G1 → G3, 500 스텝 44셀) → `pick_best_g.py`가 best G를 고르고 `jobs/gpu1_after.txt`를 만든다: best G 모드 I SM1·AP12 → G1 2000(11셀) → best G 모드 I MPM12 → G1 반복 → best G 모드 I mustard0 → G3 2000(11셀) → 두 번째 G 모드 I(4개). 순서는 지시서 6의 생략 순서를 거꾸로 한 우선순위.
- **best G 규칙(결과를 보기 전에 고정)**: G1·G3 중 모드 S 500 스텝 11셀의 d 중앙값 감소율 평균이 큰 쪽; 차이 < 0.01이면 악화률 평균이 낮은 쪽. G0·G0B는 지시서대로 모드 I에서 제외.
- 셀 출력: `outputs/exp_batch_20260928/<arm>/<ds>/<seq>/S_K<K>_s<steps>[_rep1]/result.json`, 모드 I는 `I_s500/`.
- 재개: `progress`의 DONE/FAIL을 보고, 끝나지 않은 줄만 남긴 job 파일로 `docker exec -d -e CONTAINER=<c> <c> bash -c "cd /home/kist/Desktop/BundleSAM3DGS && nohup bash logs/exp_batch_20260928/chain.sh <gpu> <jobfile> <name> > /dev/null 2>&1"`. 완료 셀은 run_job.sh가 SKIP한다.
- 첫 판독(N 모드 S 500): 마지막 시점 d 중앙값 감소 AP12 0 %, mustard0 +3 %, SM1 −1 %, MPM12 +4 % → P0(i) 기준(20 % 이상, 3/4)에는 크게 못 미친다. AP12 p90은 12.34 → 9.17 mm로 꼬리를 줄인다.

## 2단계 경과 (9/28 02:40)
- 지도 재생 완료: mustard0 01:50, AP12 02:28, MPM12 02:34, SM1 02:39 (`g0maps/<ds>/<seq>/maps.json`; Gaussian 수 AP12 K40 187,837 / K80 292,717 / last 451,301). 02:39부터 `gpu1_S500` 체인(G0 → G0B → G1 → G3) 진행 중.
- N 모드 S 22셀 완료(01:43–02:00). 마지막 시점 500스텝 d 중앙값 감소 AP12 0 %, mustard0 +3 %, SM1 −1 %, MPM12 +4 % (2000스텝 +7/+6/−1/+9 %) → P0(i) 실패.
- **N 모드 I SM1(02:19)·AP12(02:30) 완료**: SM1 최종 d 중앙값 noop 48.32 → N 4.93 mm (0.10×, p90 57.58 → 9.07, noop 대비 개선 99 %·악화 0 %). AP12 7.36 → 4.57 mm (0.62×, 개선 75 %·악화 11 %). SM1 곡선: 붕괴 점프가 체인 입력으로 들어와 n=20에서 22 mm까지 올라가지만 라운드마다 전체가 당겨져 n=55 17 mm, n=150 9.4 mm, 끝 5.2 mm. P0(ii)는 MPM12·mustard0 결과 대기(현재 SM1만 ○).
- 참고(원본 온라인 실행 채점, `experiments/original_round_reference.py` → `analysis/original_rounds.json`): 원본은 새 키프레임마다 라운드를 돌며 라운드당 d 중앙값 변화 평균 ≈ 0(SM1 −0.01, AP12 +0.00, MPM12 +0.01 mm). 원본 온라인 키프레임 d(마지막 라운드 후): SM1 4.25, AP12 5.10, MPM12 10.69 mm (noop 48.32 / 7.36 / 4.97). SM1에서 원본은 붕괴 자체가 없다(n=5 라운드가 60 % 개선, 이후 1–6 mm 유지). → 오프라인 재생 N 모드 I의 SM1 최종값 4.93 mm는 원본 온라인 수준과 같다.

## 2단계 경과 (9/28 02:50) — N 체인 완료, P0 규칙 판정 = 실패
- N 모드 I 최종 d 중앙값 (noop → N): SM1 48.32 → 4.93 (0.10×), AP12 7.36 → 4.57 (0.62×), MPM12 4.97 → 6.34 (1.28×, 악화 56 %), mustard0 7.50 → 11.11 (1.48×, 악화 97 %).
- P0 (i) 실패(마지막 시점 한 라운드 감소 −1…+4 %), (ii) 실패(0.5× 이하는 SM1 하나). 지시서대로 "오프라인 재생으로는 원본 효과가 재현되지 않음(규칙상)" + P1 판정 없이 표만 + 비용 표.
- 단, 시퀀스별 방향은 원본과 같다: 원본 SAM2 on vs 트래커 단독 ADD가 SM1 0.551 vs 4.361, AP12 0.454 vs 0.888(도움), MPM12 0.777 vs 0.457, mustard0 1.369 vs 0.748(해악). 원본 온라인 키프레임 d(SM1 4.25, AP12 5.10, MPM12 10.69 mm)와 재생 N(4.93, 4.57, 6.34)도 같은 수준·방향. → 결과 문서에 "규칙 판정"과 "원본 대비 재현 정도"를 나눠 적고, P0 해석은 사용자 결정 항목으로 올린다.
- 반복 잡음: N AP12 last 7.38 / 7.37 mm (차이 0.01, 키프레임별 |Δd| 중앙값 0.01).
- G0·G0B 모드 S 22셀 완료(02:39–02:48): 거의 무변화(마지막 시점 −2…+1 %). G1 진행 중(AP12 K40 4.20 → 6.88 mm, 악화 82 %).
- N 체인 끝(02:48) → GPU0 대기자(`wait_gpu0_after.sh`)는 `gpu1_after` 시작(= best G 선택) 후 `gpu0_after`를 돈다.

## 2단계 경과 (9/28 02:58) — 모드 S 500 전부 완료, best G = G3
- `gpu1_S500` 44셀 완료(02:39–02:57, 실패 0). `pick_best_g.py`(사전 고정 규칙): 모드 S 500 11셀 d 중앙값 감소율 평균 G1 −0.114, G3 −0.034 (악화률 평균 G1 0.43, G3 0.24) → **best G = G3**, 두 번째 = G1 (`logs/exp_batch_20260928/best_g.json`).
- G1(prior 단독 새 지도): mustard0에서만 뚜렷이 개선(last 7.50 → 5.45 mm, 개선 72 %), AP12·MPM12 K40/K80은 악화 81–85 %, SM1 +1…+3 %.
- 체인: GPU1 `gpu1_after` = G3 모드 I SM1 → AP12 → G1 2000(11셀) → G3 모드 I MPM12 → G1 반복 → G3 모드 I mustard0. GPU0 `gpu0_after`(gsplat 컨테이너, 02:58 시작) = G3 2000(11셀) → G1 모드 I SM1 → AP12 → MPM12 → mustard0.
- 참고 추가: 원본 논문 마스크 실행(`full_eval`)도 같은 척도로 채점(SM1 4.79, AP12 6.02, MPM12 11.76 mm; `analysis/original_rounds.json`의 "paper").
- 모드 I 라운드 분해(N): AP12 새 키프레임 5.39 → 4.46 mm(77 % 개선), 기존 키프레임 4.29 → 4.29 → 이득은 새 키프레임을 매 라운드 전체와 함께 맞춰 체인 전파를 막는 데서 나온다. SM1 새 11.09 → 9.37 / 기존 6.78 → 6.60, MPM12 중립, mustard0 악화.

## 2단계 경과 (9/28 03:20) — G3 모드 I SM1: 붕괴 회복 없음
- G3 모드 I SM1(02:57–03:17): 최종 d 중앙값 48.22 mm (noop 48.32, 1.00×; N 4.93). 라운드 안 포즈 이동량은 N과 비슷(새 키프레임 1.93 vs 2.27 mm)한데 방향이 GT 쪽이 아니다(새 키프레임 개선 47 % vs N 76 %, 기존 13 % vs 71 %).
- 첫 라운드(키프레임 5개) 차이: noop 트래커 오차 0/1.1/2.2/4.9/9.0 mm → N 0/1.2/1.3/1.2/1.7, G3 0/1.0/1.2/3.3/5.3. 이후 G3는 좋은 앞 키프레임까지 나쁜 쪽으로 끌려간다(1.0 → 5.1 mm, 4라운드). 가설(검증 안 함): 현재 포즈로 만든 세밀한 TSDF 지도가 어긋남을 이미 담고 있어, 처음부터 거칠게→세밀하게 배우는 NeRF의 정렬 효과가 없다.
- G3 2000스텝 모드 S 11셀 완료(GPU0, 02:58–03:13). GPU0은 G1 모드 I 진행 중, GPU1은 G3 모드 I AP12 진행 중.
- (03:23) G1 모드 I SM1 진행 중 중간값: n=20 22.0, n=80 8.8, n=155 5.6 mm (같은 n에서 N보다 낮음). 첫 라운드 KF3/KF4 4.9/9.0 → 2.4/3.7 mm(G3 3.3/5.3, N 1.2/1.7). 모드 S로 고른 best G(G3)가 아니라 두 번째 G(G1)가 SM1 회복을 재현하고 있다 — 두 번째 G 모드 I까지 돌리는 순서가 유지되어 있다.

## 완료 (9/28 04:05) — 결과 문서 작성
- 실행 106개(지도 재생 4 + 셀 102) 전부 완료, 실패·재시도 0. 체인 끝: gpu0_N 02:48, gpu1_S500 02:57, gpu0_after 04:01, gpu1_after 04:04.
- 모드 I 최종 d 중앙값(noop → arm): N SM1 0.10× / AP12 0.62× / MPM12 1.28× / mustard0 1.48×; G1 0.11× / 2.97× / 2.56× / 1.13×; G3 1.00× / 5.53× / 4.03× / 1.63×.
- 판정: P0 규칙상 실패((i) 한 라운드 효과 ≈ 0, (ii) SM1만 0.5× 이하) → P1 판정 없음(계산값 G3·G1 모두 실패). 원본 온라인 대비로는 N 재생이 시퀀스별 방향·수준을 재현(사용자 결정 항목).
- 결과 문서 `EXP_BATCH_20260928_RESULTS.md` (조립: 스크래치 초안 + `analysis/tables.md`). 표 다시 만들기: `/usr/bin/python3 experiments/analyze_joint_refit.py`.
- 본 코드 무수정 확인(git status: 새 파일만). 커밋 안 함(지시서).
