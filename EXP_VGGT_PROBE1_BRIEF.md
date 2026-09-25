# VGGT 포즈 정확도 probe 지시서 (Probe 1, 2026-09-25)
브랜치 `milestone5-v1` (`8d027e9`). 본 코드를 건드리지 않는 오프라인 실험이다. 질문할 수 없으면 아래 규칙과 판독 기준대로 스스로 진행하고, 판단이 필요했던 지점은 결과 문서에 적는다.
0. 배경 (왜 하는가)

* 9/21 4.6: BundleSDF 최종 포즈로 우리 지도를 만들면 HO3D P1이 0.320이고, 우리 트래커 포즈로는 0.459다. HO3D 복원의 병목은 포즈다.
* 9/22 Q2: 포즈가 온라인으로 들어오고 나중에 수정돼도, reanchor(출생 층을 포즈 변화만큼 강체 이동)를 하면 상한과 같다(0.319 vs 0.320). 지도 쪽 메커니즘은 준비됐고, 남은 건 키프레임 포즈 품질이다.
* 검토 중인 구조: 백엔드에서 VGGT 계열 feed-forward 모델로 키프레임 포즈를 추정한다. 여러 장을 한 번에 넣으면 카메라 포즈와 depth를 내는 모델이다. 목적은 드리프트와 큰 오차를 끊는 것이다. 이후 센서 depth로 정밀화하고, 확정된 포즈만 지도에 넣는다(VGGT-SLAM 2에서 착안).
* 위험 1: VGGT는 정적 장면(카메라가 움직임)으로 학습됐다. 우리 데이터는 카메라 고정·물체 이동이라 배경을 지우고 물체만 넣어야 하고, 이 조건의 정확도는 알려져 있지 않다.
* 위험 2: VGGT의 정밀도(벤치마크상 도 단위)가 트래커 오차와 같은 자릿수일 수 있다. 트래커의 소비 시점→최종 변화량은 회전 중앙값 1.7°, 이동 14 mm다.

이번 probe의 세 질문

* Q1. 마스크한 물체 영상에서 VGGT 계열 포즈는 GT 대비 얼마나 정확한가? 트래커가 지도에 넣는 포즈(소비 시점)보다 나은가? 특히 큰 오차 시퀀스에서는 어떤가?
* Q2. 어떤 입력·배치 방식이 좋은가? 비교 축은 모델, 가상 카메라 warp 여부, 1장 겹침 체인 vs 기준 KF 배치, 한 번에 넣는 새 KF 수 M, 기준 KF 수 k다.
* Q3. 온라인에서 계산할 수 있는 신뢰 신호가 실제 오차를 예측하는가? 즉 확정 판정이 성립하는가?

하지 않는 것: 본 코드 통합, depth 정밀화(다음 probe), 지도 학습·메시 채점, 모델 fine-tuning.
1. 규칙
허용

* `experiments/vggt_probe/` 아래 새 파일.
* `logs/exp_vggt_probe1/` 아래 설정·스크립트·로그·표.
* `outputs/exp_vggt_probe1/...` 새 디렉터리.
* 결과 문서 `EXP_VGGT_PROBE1_RESULTS.md` 1개.
* 별도 venv `/home/kist/venvs/vggt_probe`에 `vggt`, `vggt-omega` 설치. 만들 때 `python -m venv --system-site-packages`를 써서 컨테이너의 torch를 재사용한다.

금지

* 본 코드 수정: `gaussian_runner.py`, `prior_lifecycle.py`, `gaussian_global.py`, `sam3d_prior.py`, `bundlesdf.py`, `run_*.py`, `BundleTrack/`, 기존 `config*.yml`, 기존 `experiments/*.py`.
* 컨테이너 base 환경에 pip install.
* VGGT·VGGT-Ω 저장소 코드 수정. 필요한 건 래퍼에서 해결한다.
* commit/push/`build.sh`.
* 기존 산출물 삭제·덮어쓰기. 출력은 `exist_ok=False`로 만든다.
* `~/Desktop/BundleSDF_baseline_outputs/`에 쓰기. 읽기 전용이다.
* 본 코드 수정이 필요해 보이면 하지 말고 결과 문서 "승인 필요"에 적는다.

운영

* `logs/exp_vggt_probe1/PROGRESS.md`를 단계마다 갱신한다. 컨텍스트가 압축된 뒤 이 파일만 읽고 이어갈 수 있어야 한다.
* 실행마다 매니페스트를 남긴다: 명령, HEAD, 모델 체크포인트 파일명·sha256, 전체 설정, 시작/종료 시각.
* 실패하면 1회 재시도 후 건너뛰고, 실패 산출물은 보존한다.
* 디스크 150 GB 미만이면 새 실행을 하지 않는다.
* `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.
* 컨테이너 `BundleSAM3DGS-gsplat-smoke`, 작업 위치 `cd /home/kist/Desktop/BundleSAM3DGS`. GPU 2장.

시퀀스별 튜닝 금지

* 모든 하이퍼파라미터는 9절 2단계 전에 고정한다.
* 디버깅은 AP12와 mustard0에서만 하고, 그렇게 했다고 결과 문서에 적는다.

2. 용어

* 새 KF: 이번 배치에서 포즈를 추정할 키프레임.
* 기준 KF: 새 KF보다 먼저 생긴 키프레임 중 배치에 같이 넣는 것. 포즈를 안다고 가정하고 두 가지에 쓴다.
   * 좌표 기준: 배치 첫 자리.
   * 검산: VGGT가 다시 추정한 포즈와 알려진 포즈의 잔차.
   * SAM3D prior나 reanchor의 "앵커"와는 무관하다.
* 기준 포즈 소스:
   * `R-online`: 그 시점 BSDF 스냅샷의 포즈. 온라인에서 알 수 있는 수준의 대용이다.
   * `R-oracle`: GT 포즈. 상한이다.
* C1 (소비 시점 포즈): 키프레임 j를 처음 포함하는 스냅샷 안의 j 포즈. 9/22 Q2와 같은 정의다.
* C2 (최종 포즈): 마지막 스냅샷의 포즈.
* batch frame: VGGT 출력 좌표계. 배치 첫 프레임 카메라가 항등이다.

3. 데이터·비교 기준·GT

* 시퀀스: 22개. HO3D 13(AP10–14, MPM10–14, SB11, SB13, SM1)이 우선이고, 다음이 YCB 9다.
* 경로: 영상 폴더와 BSDF 실행 폴더는 `logs/exp_batch_20260922/manifests/*.json`에서 가져온다.
* 입력 영상·키프레임 목록: BSDF baseline SAM2 실행 `~/Desktop/BundleSDF_baseline_outputs/full_eval_sam2_<ds>/<ds>/<seq>`를 쓴다.
   * 영상은 이 폴더의 `color/ depth_filtered/ mask/`와 `cam_K`.
   * 키프레임 목록은 마지막 `<frame>/keyframes.yml`에서 읽고, 순서는 생성 순서를 따른다.
* C1, C2: `experiments/exp_hybrid_realistic_replay.py`의 `load_snapshots`로 프레임별 스냅샷을 읽어 2절 정의대로 만든다.
* C3 (있으면): 우리 시스템 `outputs/fulleval_20260912`의 소비 시점 포즈. 키프레임 집합이 BSDF와 다르므로 같은 행에서 직접 비교하지 말고 별도 행으로 둔다. 프레임별 스냅샷이 없으면 없다고 적는다.
* GT·게이지: `experiments/exp_feedback_gradient_probe.GroundTruth(dataset, video_dir, run_dir)`를 import한다.
   * 사용법은 `eval_global_poses.py`와 같다: `gt_c2w(fid)`, `rot_error`.
   * 이동 오차는 물체 좌표계에서 카메라 중심 거리(mm)다.
   * GT가 없는 프레임은 제외하고 개수를 기록한다.

4. 모델

* M-V: VGGT-1B.
   * 저장소 `facebookresearch/vggt` 최신 main(2026-05 메모리 수정 포함).
   * 체크포인트 `facebook/VGGT-1B`.
   * patch 14, 입력 한 변 518.
* M-Ω: VGGT-Ω 1B-512.
   * 저장소 `facebookresearch/vggt-omega`.
   * 체크포인트 `vggt_omega_1b_512.pt`. Hugging Face 접근 승인이 필요하다.
   * patch 16, 입력 한 변 512.
   * 접근이 안 되면 M-V만 진행하고 결과 문서에 적는다. 이 때문에 중단하지 않는다.
* 공통 래퍼 `experiments/vggt_probe/models.py`:
   * 인터페이스는 `infer(images: float[N,3,S,S] in [0,1]) -> dict`다.
   * 반환 항목:
      * `w2c[N,4,4]`: 첫 프레임 항등, OpenCV 규약
      * `K[N,3,3]`
      * `depth[N,S,S]`: z-depth
      * `conf[N,S,S]`
      * `time_s`
      * `peak_mem_gb`
   * 포즈 변환은 각 저장소의 함수를 그대로 쓴다: `pose_encoding_to_extri_intri`(VGGT), `encoding_to_camera`(Ω).
   * 정규화와 크기 제약은 각 저장소의 `load_fn`을 읽어 맞춘다. 파일 로더 자체는 쓰지 않고, 우리가 warp한 텐서를 넣는다.
   * 추론은 각 README 방식의 autocast(bf16)로 한다.
* 두 모델 모두 FoV만 예측하고 주점은 이미지 중앙이라고 가정한다(`pose_enc.py`에서 확인). 5절 warp가 필요한 이유다.

5. 입력 방식
I0 (기본 사용법, 기준선)

* 원본 이미지에 물체 마스크를 적용하고, 마스크 밖은 흰색(1.0)으로 채운다.
* 긴 변을 S에 맞추고, 짧은 변은 중앙 기준으로 흰색 패딩해 S×S로 만든다. K도 같은 변환으로 갱신한다.
* 원본 주점이 이미 중앙 근처라 편향이 작다는 가정을 확인하는 기준선이다.

I1 (가상 카메라 warp, 주력)
프레임 i마다 다음을 한다.

1. 물체 중심 픽셀 `c_i`를 마스크 무게중심으로 잡고, 광선 `r_i = normalize(K⁻¹[c_i;1])`을 구한다.
2. `R_i`는 `r_i`를 z축으로 보내는 최소 회전이다. 축은 `r_i × ẑ`, 각은 `acos(r_i·ẑ)`, Rodrigues 공식으로 만든다. `R_i r_i = ẑ`가 성립해야 한다.
3. `K_v = [[f_v,0,S/2],[0,f_v,S/2],[0,0,1]]`로 둔다. `f_v`는 배치 안에서 공통이다. 회전 후 모든 프레임의 마스크 경계가 이미지 중심에서 `0.4·S` 안에 들어오는 최대값으로 정한다.
4. 역매핑 `p = K R_iᵀ K_v⁻¹ p_v`로 샘플링한다. RGB는 bilinear, 마스크와 depth는 nearest. 마스크 밖과 이미지 밖은 흰색이다.

* 포즈 복원: VGGT 가상 카메라의 `c2w_v,i = inv(w2c_i)`에서 실제 카메라 포즈를 `c2w_i = c2w_v,i · blockdiag(R_i, 1)`로 구한다.
* VGGT가 예측한 FoV와 `K_v`의 FoV 차이를 신호 T3으로 기록한다.
* 선례: FMOV(Shi et al., 2024, arXiv 2405.05858)가 물체 중심을 향하는 가상 카메라를 썼다.

I2 (진단, AP12·MPM10·mustard0 3개만)

* bbox 중심으로 잘라 확대하되 주점 보정 없이 넣는다.
* 주점 가정 위반이 만드는 편향 크기를 확인하는 용도다. 결과는 참고용이다.

6. 배치 구성
B-chain (사용자 원안)

* 첫 배치는 KF0–KF4이고 KF0이 기준이다.
* 이후 배치는 [직전 배치의 마지막 KF] + [다음 새 KF 5장]이다.
* 좌표는 겹치는 한 장으로 이어 붙인다. 그 장의 포즈는 직전 배치에서 추정된 값이다.
* 스케일은 배치마다 센서 depth로 새로 구한다(7절).
* 목적: 체인 누적 오차 측정.

B-ref(M, k)

* 키프레임을 생성 순서로 M장씩 묶어 새 KF로 두고, 그보다 먼저 생긴 키프레임 중 k장을 기준 KF로 같이 넣는다. 배치 크기 N = M + k.
* 선택 규칙: 온라인에서 알 수 있는 정보만 쓴다. 기준 KF의 시점은 R-online 포즈로, 새 KF의 시점은 C1 포즈로 계산한다. 시점 방향은 물체 좌표계에서 카메라 광축(`R[:,2]`)이다.
   * 첫 자리: 새 KF들과의 평균 광축 각이 가장 작은 기준 KF.
   * 나머지 k−1장 중 절반: 새 KF에 가까운 순.
   * 나머지 절반: 이미 고른 것들에서 먼 순(farthest point sampling).
   * 기준 후보가 k보다 적으면 있는 만큼만 넣는다.
   * 첫 배치는 KF0–KF4이고, KF0만 기준이다.
* 그리드: M ∈ {1, 5}, k ∈ {4, 8}.
* 기준 KF 선택은 항상 R-online 기준이라 VGGT 입력은 R-online과 R-oracle에서 같다. VGGT는 한 번만 돌리고, 정렬(7절)만 두 소스로 따로 한다.

7. metric 변환과 정렬
스케일 s

* 프레임마다 유효 픽셀은 다음 세 조건을 모두 만족하는 픽셀이다.
   * 원본 해상도에서 3px 침식한 마스크 안
   * 센서 depth > 0.1 m
   * VGGT conf가 그 프레임 마스크 안에서 상위 50%
* z-depth가 아니라 광선거리로 비교한다. 가상 카메라 회전으로 z가 바뀌기 때문이다.
   * 센서: `ρ_s = D_s(p)·‖K⁻¹[p;1]‖`
   * VGGT: `ρ_v = z_v(p_v)·‖K_v⁻¹[p_v;1]‖` (I0에서는 `K_v` 대신 갱신된 K)
* `s_i = median(ρ_s/ρ_v)`, 배치 스케일 `s`는 모든 유효 픽셀의 median이다.
* 유효 픽셀이 200개 미만인 프레임은 s 계산에서 빼고 기록한다.
* batch frame 포즈의 이동 성분에 s를 곱한다.

정렬 T ∈ SE(3) (batch → 물체 좌표계)

* 기준 KF r의 알려진 포즈를 `c2w_r`, 스케일을 적용한 batch 포즈를 `ĉ2w_r`이라 한다.
* `R_T = proj_SO3(Σ_r R_r R̂_rᵀ)`
* `t_T = median_r(t_r − R_T t̂_r)` (성분별 median)
* 새 KF 포즈는 `T·ĉ2w_new`다.
* B-chain은 겹치는 한 장으로 T를 구한다.
* 정렬 후 기준 KF 잔차(회전°, 이동 mm)를 신호 T1로 기록한다.

8. 지표
새 KF마다 (주 지표)

* E1: GT 대비 회전 오차(°)와 이동 오차(mm). 같은 KF의 C1, C2 오차를 같은 행에 둔다.

배치마다

* E2 (게이지 무관): 배치 안 모든 쌍의 상대 회전 오차(°)와 상대 이동 방향 오차(°), GT 대비.
* E3 (스케일): `s_gt = argmin_s Σ‖s·t̂_ij − t^gt_ij‖²`(배치 안 쌍, 닫힌 해)로 구하고, 오차 `|s/s_gt − 1|`을 기록한다.

신뢰 신호 (온라인에서 계산 가능한 것만)

* T1: 기준 KF 잔차 중앙값(회전, 이동)
* T2: s_i 변동계수
* T3: |FoV_vggt − FoV_Kv| 평균
* T4: depth 일치도, `median |s·ρ_v − ρ_s| / ρ_s`
* T5: 마스크 안 VGGT conf 평균
* T6: 새 KF 마스크 면적(px)
* T7: 새 KF의 VGGT 포즈와 C1의 차이(회전°, 이동 mm)

신호 유용성

* 각 T와 새 KF E1 회전 오차의 Spearman ρ. 배치 단위 신호는 그 배치의 새 KF들에 공유한다.
* T1 기준으로 배치 하위 10/20/30%를 버렸을 때, 남은 새 KF의 E1 중앙값과 p90.

비용

* 배치당 추론 시간과 최대 GPU 메모리.

9. 순서
0단계. 준비·단위 확인 (필수, 결과는 PROGRESS.md에)

* a. 변환 검증: GT 포즈로 가상 카메라 출력을 합성해 5·7절 변환에 넣었을 때, GT가 1e-6 안에서 복원되어야 한다. 합성할 때 임의 스케일 0.37을 쓰고 배치 첫 프레임을 기준으로 둔다.
* b. warp 검증: I1 warp 후 역warp한 마스크의 IoU가 0.98보다 커야 한다(AP12 한 배치).
* c. 스모크 실행: M-V × I1 × KF0–KF4(AP12, mustard0).
   * E1, s, s_i, T3을 출력한다.
   * 건전성 확인: s가 KF0 마스크 점의 평균 광선거리와 같은 자릿수(±50%)여야 한다. VGGT 학습이 평균 거리 1로 정규화하기 때문이다.
* 0-a를 통과하기 전에는 어떤 결과도 내지 않는다.

1단계. M-V × I1 × B-ref(5,8) × HO3D 13. 1차 표를 만든다(Q1의 첫 답).
2단계. HO3D 13 전체 그리드

* 모델 {M-V, M-Ω} × 입력 {I0, I1} × 배치 {B-chain, B-ref(1,4), B-ref(1,8), B-ref(5,4), B-ref(5,8)}.
* 각 실행에서 R-online, R-oracle 정렬을 둘 다 계산한다.
* GPU 2장에 모델별로 나눠 병렬로 돌린다.

3단계. YCB 9: 2단계 상위 2개 설정과 B-chain.
4단계. I2 진단 3시퀀스.
시간 관리

* 부족하면 이 순서로 생략한다: I2 → YCB → B-ref(1,4)와 B-ref(5,4) → I0.
* 다음은 반드시 끝낸다: M-V/M-Ω × I1 × {B-chain, B-ref(5,8), B-ref(1,8)} × HO3D 13.
* 새 실험을 임의로 추가하지 않는다.

10. 산출물
`outputs/exp_vggt_probe1/<model>/<input>/<batch>/<ds>/<seq>/`

* `kf.csv`: 새 KF당 한 행. 열은 다음과 같다.
   * kf id, frame id, 배치 id
   * 추정 c2w 16값 × {online, oracle}
   * E1 × {online, oracle}
   * C1·C2 오차
   * T1–T7
* `batches.jsonl`: 배치마다 한 줄. 구성(기준·새 KF id와 순서), s, s_i, s_gt, E2, T1–T5, 시간, 메모리.
* `snapshots/<frame>/keyframes.yml`: 다음 단계 재생용. R-online 정렬 기준이다.
   * 배치마다 그 시점까지의 모든 키프레임 포즈를 적는다. 이미 추정된 것은 VGGT 추정값을 쓰고, 아직 없는 것은 뺀다.
   * 형식은 BSDF 스냅샷과 같다(`keyframe_<id>: {cam_in_ob: 16 floats}`). BSDF 실행의 파일을 열어 똑같이 맞추고, `load_snapshots`로 읽히는지 확인한다.

`logs/exp_vggt_probe1/`

* PROGRESS.md, manifests, 표 md.

결과 문서 `EXP_VGGT_PROBE1_RESULTS.md`

1. 한 페이지 요약: Q1–Q3의 답과 수치.
2. 표 A: HO3D 시퀀스별 새 KF E1(회전 중앙값/p90, 이동 중앙값/p90). 열은 C1, C2, 상위 설정 2개(online), 같은 설정(oracle).
3. 표 B: 설정 그리드 평균(HO3D, online/oracle).
4. 표 C: 큰 오차 시퀀스(MPM10, AP11, SM1, SB11) 상세. C1 오차가 5°를 넘는 새 KF만 모은 부분집합에서의 VGGT 오차도 같이.
5. 표 D: 스케일(E3, T2).
6. 표 E: 신뢰 신호 Spearman과 버림 곡선.
7. 표 F: 체인 vs 기준 KF. 새 KF 순서(시퀀스 앞 1/3, 중간, 뒤)에 따른 E1 추이.
8. 시간·메모리, 실패·계획 이탈, 승인 필요 사항.

판독 기준 (잠정)

* 판정은 사용자가 한다. 표는 정확히 만든다.
* "VGGT 우위": 최상 설정(online)의 새 KF E1 회전 중앙값이 C1보다 낮은 시퀀스가 HO3D 13개 중 8개 이상.
* "큰 오차 복구": C1 오차가 5°를 넘는 부분집합에서 VGGT 오차 중앙값이 C1의 절반 이하.
* "정밀도 한계": 최상 설정의 중앙값이 C2보다 크면 그대로 적는다. 예상되는 결과이고, depth 정밀화 probe가 필요하다는 근거가 된다.
* "신뢰 신호 유효": |ρ| ≥ 0.4인 신호가 하나 이상 있고, T1 기준으로 20%를 버렸을 때 p90이 25% 이상 줄어든다.
* online과 oracle의 차이가 크면 "기준 포즈 품질이 병목"이라고 적는다.

11. 주의

* VGGT extrinsic은 world→camera(OpenCV)이고 첫 프레임이 항등이다.
* VGGT depth는 (가상) 카메라의 z-depth다. 센서와는 광선거리로 비교한다.
* 주점이 중앙에서 벗어난 입력은 I2 진단 외에는 넣지 않는다.
* 대칭이거나 무늬 없는 물체에서 뒤집힘(약 180° 오차)이 보이면 시퀀스별로 개수를 세서 적는다. 평균이 몇 개의 뒤집힘에 지배되지 않도록 중앙값과 p90을 주로 보고, 평균은 참고로만 쓴다.
* 키프레임 id ↔ 프레임 id 매핑은 스냅샷 키를 기준으로 한다.
* 배치의 첫 자리가 게이지를 정하므로, 배치 구성 순서를 매니페스트에 남긴다.
* VGGT-Ω 체크포인트는 두 종류다(512는 in-the-wild 권장, 416-reproduce는 벤치마크용). 이번에는 512만 쓴다.
