# EXP_VGGT_PROBE1 진행 기록 (2026-09-25 시작)

지시서: `EXP_VGGT_PROBE1_BRIEF.md`. 본 코드 무수정, 커밋 없음. 코드: `experiments/vggt_probe/` (models.py, probe_lib.py, run_probe.py, stage0_checks.py).

## 환경
- venv `/home/kist/venvs/vggt_probe` — **편차**: 컨테이너 파이썬에 ensurepip가 없어 `python -m venv` 대신 `virtualenv --system-site-packages`로 생성(효과 동일: torch 2.6.0+cu124 등 컨테이너 패키지 재사용, 추가 설치는 venv 안에만). 컨테이너 기본 환경에는 pip install 없음.
- 패키지: `vggt` (GitHub facebook/vggt main), `vggt_omega` (pip --no-deps). 두 저장소 코드 무수정.
- M-V 체크포인트: `facebook/VGGT-1B` model.pt 4794 MB, sha256 `d15bf50a8615c8225ed48b51ea5cac673d82442ec0309036df555a053253afe0`. 13장@518 추론 2.07 s, 최대 10.0 GB (3090).
- **M-Ω 사용 불가**: `facebook/vggt-omega` 저장소가 gated이며 현재 HF 계정이 승인 목록에 없음(403 GatedRepoError, 9/25 20:40 및 재시도 동일). 지시서 §2에 따라 **M-V 단독 진행**. 계정 승인 후 `vggt_omega_1b_512.pt` 다운로드만 하면 `models.OmegaWrapper`로 바로 실행 가능.
- 실행: `docker exec -e HF_HOME=/home/kist/.cache/huggingface -e CUDA_VISIBLE_DEVICES=<g> BundleSAM3DGS-gsplat-smoke bash -c "source /home/kist/venvs/vggt_probe/bin/activate && cd .../experiments/vggt_probe && python run_probe.py ..."`.
- 입력: `~/Desktop/BundleSDF_baseline_outputs/full_eval_sam2_{ho3d,ycb}/{ho3d,ycb}/<seq>` (읽기 전용) — color/depth_filtered/mask/cam_K.txt + 프레임별 keyframes.yml 스냅샷(R-online). GT: `experiments/exp_feedback_gradient_probe.GroundTruth` (평가기의 첫 프레임 정렬 gauge, C1/C2와 동일 기준).

## 구현 요약 (지시서 대응)
- I1: 마스크 중심 광선 r을 z축으로 보내는 최소 회전 R_i, 공통 f_v = 0.4·S / max(회전 후 마스크 정규화 좌표 최대치), 역매핑 remap(RGB 쌍선형, 마스크/깊이 최근접), 바깥·비마스크 = 흰색. 포즈 복원 c2w_i = c2w_v,i · blockdiag(R_i,1).
- I0: 마스크(흰 배경) → 긴 변 S, 짧은 변 중앙 흰 패딩.
- 배치: B-chain(직전 새 KF 1장 기준), B-ref(M,k): 후보 = 이전 KF 전부, 기준 포즈 = 새 KF 중 마지막 KF 프레임의 BSDF 스냅샷(R-online), 새 KF 포즈 = C1; 선택 = 새 KF 광축과 평균 각도 최소 1장 + 나머지 절반 최근접 + 절반 farthest-point. 첫 배치 KF0 기준 + KF1–4.
- 스케일: 원본 마스크 3 px 침식, 센서 깊이 > 0.1 m, 마스크 내 conf 상위 50 %, s_i = median(ρ_sensor/ρ_vggt)(광선 거리), 유효 200 px 미만 제외, s_batch = median s_i. s_gt(E3) = GT 상대 이동에 최소자승 적합한 스케일(진단).
- 정렬: 기준 KF 알려진 포즈(첫 배치 C1, 이후 추정 포즈)로 R_T = proj_SO3(Σ R_r R̂_rᵀ), t_T = 성분별 median.
- 지표: E1(새 KF의 GT 대비 회전°/이동 mm, C1/C2 열 동반), E2(모든 쌍 상대 회전 오차·상대 이동 방향 오차 중앙값), E3(s vs s_gt), T1(마스크 내 conf 중앙값) T2/T3(정렬 잔차) T4(스케일 후 깊이 일치 상대오차 중앙값) T5(s_i IQR 비) T6(마스크 면적) T7(새–기준 최소 광축 각).
- 출력: `outputs/exp_vggt_probe1/<stage>/<seq>_<input>[...]/{kf.csv, batches.jsonl, snapshots/<frame>/keyframes.yml, manifest.json, run.log}`.

## 0단계 단위 검사
- 0-a 합성 왕복(스케일 0.37, 임의 gauge, 8 포즈, 3장 기준 정렬): 스케일 오차 0, 정렬 오차 5.6e-16, s_gt 오차 5.6e-17 → **통과** (기준 1e-6).
- 0-b AP12 첫 20 KF, f_v = 628: 워프→역워프 마스크 IoU 최소 0.9996, 평균 0.9998, 마스크 최대 범위 0.400·S, 중심 오프셋 ≤ 7 px → **통과** (기준 0.98).
- 0-c 연기 시험(AP12, mustard0 × I1/I0 × 첫 2 배치): 진행 중.
- 0-c 연기 시험 (M-V × I1 × 첫 2 배치, 결과 `outputs/exp_vggt_probe1/stage0c/`): 코드 정상 완주. s 건전성: AP12 s = 0.402 vs KF0 마스크 평균 광선거리 0.373 m, mustard0 s = 0.861 vs 0.909 m → 같은 자릿수(±50 %) **통과**. s_i 범위 AP12 0.377–0.413, mustard0 0.733–0.902; T3(FoV 차) AP12 0.6–0.8°, mustard0 6–12°. 첫 배치(KF0 기준 1장) 새 KF E1: AP12 회전 8.2° / 62 mm (C1 4.2°), mustard0 2.9° / 79 mm (C1 4.2°). 두 번째 배치(기준 5장): AP12 3.4° / 21 mm (C1 1.9°), mustard0 1.9° / 47 mm (C1 6.8°). E3: AP12는 s/s_gt ≈ 1.3 (depth 기반 스케일과 VGGT 이동 스케일이 30 % 불일치), mustard0는 0.9–0.96.
- **계획 이탈(구현 오류 정정)**: 첫 1단계 실행(21:01 시작)은 기준 KF의 '알려진 포즈'로 VGGT의 이전 추정값을 연쇄 사용하고 있었음(지시서 §6·§7의 R-online/R-oracle 정의 위반). AP10·AP11 2셀 완료 후 중단, 산출물 삭제, 진행 기록은 `progress_stage1_aborted_chainedref.txt`로 보관. 러너를 지시서대로 재작성: VGGT 1회 추론 후 R-online(새 KF 중 마지막 KF 프레임의 BSDF 스냅샷; B-chain은 직전 배치 추정값)과 R-oracle(GT) 정렬을 각각 계산, 배치 s = 전체 유효 픽셀 median, 신호 T1–T7 지시서 번호대로, 출력 경로 `outputs/exp_vggt_probe1/<model>/<input>/<batch>/<ds>/<seq>/`. 위 0-c 수치는 재작성 후 값.
- 0-c 초기 버전 산출물은 `outputs/exp_vggt_probe1/stage0c_v1/`에 보존(연쇄 기준 버전, 참고용).

## 1·2단계 실행 (2026-09-25 21:20 시작)
- GPU1: 1단계 M-V × I1 × B-ref(5,8) × HO3D 13 (`progress_stage1.txt`).
- GPU0: 2단계 필수 셀 M-V × I1 × {B-chain, B-ref(1,8)} × HO3D 13 (`progress_stage2a.txt`) — 1단계와 독립(튜닝 없음)이라 병렬 시작.
- 셀당 약 2–3분(대부분 CPU: 스냅샷 로딩·워프; GPU 추론은 배치당 0.75–1.35 s, 최대 10 GB).
- 대기열: GPU1 = 1단계 → stage2b (I1 × {ref1-4, ref5-4}, I0 × {chain, ref5-8}); GPU0 = stage2a (I1 × {chain, ref1-8}) → stage2c (I0 × {ref1-8, ref1-4, ref5-4}). HO3D 13 전체 그리드 = 10 설정 × 13 = 130 셀, 예상 종료 9/26 02–03시. 이후 3단계(YCB 9: 상위 2 설정 + B-chain), 4단계(I2 3시퀀스).
- 진행 중 관찰(AP10·AP11, 판정 아님): B-chain의 oracle 정렬(겹침 1장 GT)은 새 KF 회전 2.5° 수준으로 C1(6–7°)보다 낮음 → 연속 6장 안에서 VGGT 상대 포즈는 정확(E2 1.3–3.3°). 반면 B-ref(5,8)은 기준 KF가 먼 시점을 포함해 배치 내 상대 회전 오차(E2)가 6–40°로 커지고 oracle 정렬도 20° 수준 → 넓은 기선 배치에서 VGGT 자체가 불일치. 체인 online은 누적(68°). 스케일 s/s_gt도 넓은 배치에서 1.3–3.9로 벌어짐(체인 0.92–1.13). 표 F·B에서 확인 예정.

## 1단계 완료 (9/25 21:46; 13/13, 실패 0) — 1차 표 `table_stage1_partial.md`
- M-V × I1 × B-ref(5,8), HO3D 13: 새 KF 회전 중앙값(시퀀스 평균) **online 18.5° / oracle 18.0° vs C1 4.0° / C2 4.4°**, 이동 133 / 125 vs 31 / 35 mm. online < C1인 시퀀스 0/13, KF 단위 비율 10 %. online ≈ oracle → 기준 포즈 품질이 병목이 아님. 원인 후보는 넓은 기선 기준 KF(farthest-point 포함)를 넣은 배치에서 VGGT 자체의 상대 포즈가 불일치(E2 6–40°, 표 D의 s/s_gt 1.4–6.4)하는 것. 같은 시퀀스의 B-chain oracle(연속 6장, 겹침 1장 GT)은 2.7°로 C1보다 낮음(5 seq 기준).
- Q1 1차 답: "B-ref(5,8) 그대로는 VGGT 포즈가 BSDF C1보다 4배 이상 나쁨". 판정은 사용자. 2단계 그리드(배치 구성 축)가 이 원인을 가른다.
- 9/25 21:58 대기열 사고: stage2b/2c 대기 셸이 `pgrep -f`로 자기 자신을 매칭해 영원히 대기(GPU1 12분 유휴). 대기 셸 두 개를 PID로 종료하고 stage2b를 GPU1에 즉시 시작, stage2c는 stage2a PID(/proc) 감시로 재대기. 실행 중이던 stage2a는 영향 없음.

## M-Ω 추가 (9/25 22:30, 사용자 지시: 접근 승인됨)
- 체크포인트 `/home/kist/checkpoints/vggt_omega/vggt_omega_1b_512.pt` 4,576,706,117 B, sha256 `c02da418b18bb01d0392598d3f6147366bcde1bb70fd08a5e3bf7925b0667934` (토큰은 1회 환경변수로만 사용, 어디에도 저장 안 함). 컨테이너/venv에서 같은 경로로 보임. `models.OmegaWrapper`는 이 로컬 파일에서 strict 로드(누락/잉여 키 0), 포즈 복호는 패키지의 `encoding_to_camera`, S = 512, patch 16, 자체 bf16 autocast.
- CPU 기능 검사(AP12 3장): 첫 프레임 ≈ 항등(최대 7e-4), K f≈622(가상 f 604), s_0 = 0.440 vs 평균 광선거리 0.373 → 자릿수 일치.
- GPU 계획(M-V와 같은 GPU에 겹치지 않음, M-V 실행은 그대로): GPU1 = M-V stage2b 종료 후 → M-Ω 0-c 스모크(AP12·mustard0, 자동 건전성 판정 → `omega_smoke.txt`, 통과 시 `SMOKE_PASS_omega`) → omegaA (I1 × {ref5-8, chain, ref1-8} → {ref1-4, ref5-4}); GPU0 = M-V stage2c 종료 후 스모크 통과 확인 → omegaB (I0 × 5 배치). 기준 KF 선택·정렬·스케일 규칙은 M-V와 동일(코드 공유).
- **M-Ω 0-c 스모크 통과** (9/26 00:12, GPU1, `omega_smoke.txt`, 산출물 `outputs/exp_vggt_probe1/stage0c_omega/`): AP12 s = 0.304 vs 평균 광선거리 0.373 m, mustard0 s = 0.865 vs 0.909 m (둘 다 ±50 % 안). s_i AP12 0.277–0.466(M-V 0.377–0.413보다 산포 큼), mustard0 0.818–0.930. T3(FoV 차) 9–11°(M-V 1–12°). 첫 2 배치 새 KF E1 회전 online/oracle/C1/C2: AP12 2.95/5.27/2.05/2.73°, mustard0 1.88/1.88/6.40/4.12°. 배치 0.72 s, 최대 7.2 GB. → omegaA(I1 × 5 배치 × HO3D 13) GPU1에서 자동 시작. M-V stage2b 완료(52/52), stage2c 19/39 진행 중(GPU0).

## M-V HO3D 그리드 완료 (9/26 01:36; 10 설정 × 13 = 130셀, 실패 0) — 표 `table_grid_MV_ho3d.md`
- 표 B 요약(새 KF 회전 중앙값의 시퀀스 평균, online/oracle vs C1 4.02 / C2 4.42): I1 ref1-8 14.6/13.8, ref5-8 18.5/18.0, ref1-4 18.6/17.6, ref5-4 23.0/21.9; I0는 각각 +1–1.7° 나쁨. B-ref 전 설정에서 online < C1인 시퀀스 0/13. 체인 순수(1) I1 47.6°(앞 1/3 18° → 뒤 1/3 57°로 누적), 체인 재정렬 R-oracle(2) 3.15°(C1보다 낮음, 13/13 중 11 ≤ C2), 재정렬 R-online(2) 5.6°.
- M-V 상위 2 설정(online 회전 기준) = I1/ref1-8, I1/ref5-8(18.52 < ref1-4 18.55). 3단계(YCB 9)는 이 둘 + I1/chain, 4단계 I2(AP12·MPM10·mustard0)는 ref1-8 + chain으로 실행.
- 대기열(GPU1, omegaA 종료 후 자동): M-V stage3_ycb → M-V stage4_I2 → M-Ω stage3_ycb(같은 설정) → M-Ω stage4_I2. GPU0: omegaB(I0 × 5 배치) 진행 중.
- M-Ω 중간(omegaA 31/65): I1 ref1-8 18.9°(11 seq), ref5-8 22.2°(12 seq), 체인 oracle 3.3° → M-V와 같은 양상, 약간 나쁨.

## 종료 (9/26 07:50)
- 전 셀 완료: HO3D 그리드 M-V 130 + M-Ω 130, YCB 27 + 27, I2 6 + 6 = 326 셀, 실패 0. M-Ω I1 그리드 프로세스가 35/39에서 traceback 없이 종료(원인 미상) → SB13 ref1-8, SM1 ×3 재실행(`progress_omegaA_rerun.txt`).
- 사후 분석: 체인 재정렬(2) `realign_chain.py` → 각 체인 셀의 `kf_realigned_online.csv`; C3 `c3_ours.py` → `table_C3_ours.md`; 매니페스트 색인 `manifests/index.json`, `manifests/checkpoints.json`.
- 표 `table_all.md`(build_tables.py, 모델별 상위 B-ref 2 설정 + 체인 (1)/(2) 분리, 표 F는 순수 체인(1) 기준), 결과 문서 `EXP_VGGT_PROBE1_RESULTS.md`.
- 요약: B-ref 20 설정 모두 HO3D에서 C1보다 나쁨(최상 M-V/I1/ref1-8 14.6° vs C1 4.0°, 0/13), online ≈ oracle; 연속 체인 상대 포즈는 정확(GT 앵커 3.15°)하나 앵커 오차를 물려받음(R-online 5.5°), 순수 체인 발산(47.6°); I1 > I0, M-V > M-Ω; 신뢰 신호 T1 ρ 0.74·T7 0.92, 20 % 버림에 p90 −33 %; 스케일 불일치 E3 0.6–1.2; YCB에서는 C1과 동급(4/9). 우리 시스템 C3보다는 HO3D 7/13에서 낮음.
- 커밋 없음(지시서). 승인 필요 사항은 결과 문서 §8.
