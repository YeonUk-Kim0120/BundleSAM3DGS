# BundleSAM3DGS 프로젝트 인수인계

> 최종 정리일: 2026-08-20 (Asia/Seoul)
>
> 저장소: `/home/kist/Desktop/BundleSAM3DGS`
>
> 이 문서는 새 환경의 에이전트가 과거 대화, 코드 변경, 실험 설계와 결과를 이어받기 위한 현재 기준 문서다.

## 1. 가장 먼저 지켜야 할 작업 원칙

- **코드를 수정하기 전에 반드시 사용자에게 먼저 설명하고 명시적 승인을 받아야 한다.** 설명에는 수정 목적, 대상 파일, 구체적인 구현 방향, 영향을 받는 동작, 검증 방법을 포함한다.
- 사용자는 승인하지 않은 코드 변경을 매우 강하게 금지했다. 읽기 전용 조사와 결과 분석은 가능하지만, 사소한 한 줄 수정이나 테스트 추가도 먼저 승인받아야 한다.
- 승인 이후 범위가 달라지면 멈추고 다시 승인받는다. 관련 없는 정리, 리팩터링, 포맷 변경을 함께 하지 않는다.
- 실험 결과와 체크포인트는 연구 기록이다. 기존 결과 디렉터리를 덮어쓰거나 삭제하지 않는다. 삭제가 필요해 보여도 정확한 대상과 보존된 요약을 먼저 제시하고 승인을 받는다.
- 커밋, amend, tag, push를 임의로 하지 않는다. 의미 있는 이정표에 도달하면 사용자에게 커밋을 권하고 메시지만 제안한다.
- **이 문서 작성 승인은 문서 한 파일에만 해당한다.** 아래에 적은 split 수치 안정성 수정 등은 아직 승인되지 않았다.

## 2. 현재 상태 한눈에 보기

1. 기존 BundleSDF tracking/SDF 파이프라인은 실행 가능한 상태로 보존되어 있다. SAM2로 미리 만든 mask를 reader가 읽도록 연결했고, 추적 재현성을 해치던 실험적 rematch/역투영 수정은 baseline 방식으로 되돌렸다.
2. `gaussian_runner.py`와 gsplat 실험 도구는 구현되어 있지만, **아직 `bundlesdf.py` 온라인 파이프라인에 연결되지 않았다.** 현재 GS 결과는 완료된 tracking log의 고정 pose를 replay한 오프라인 검증 결과다.
3. RGB-D 신규점은 매 키프레임마다 기존 Gaussian에 append한다. 전체 Gaussian을 매번 다시 초기화하지 않는다. mustard에서는 1 cm가 너무 거칠어 1 mm voxel/novelty 거리로 결정했다.
4. threshold 기반 DefaultStrategy 대신 exact-rank percentage topology를 진단했다. 3% repeated 실험에서 prune은 비교적 안전했고 duplicate는 RGB를 개선하지만 geometry를 악화했으며, split은 표현을 크게 흔들고 결국 opacity `+inf`로 실패했다.
5. 최우선 미해결 문제는 split child opacity 계산의 float32 포화다. 원인은 확인했지만 수정은 승인 전이며 코드에 반영되지 않았다.

## 3. 저장소와 Git 상태

- branch: `main`
- 현재 HEAD: `711744f4dacd7426c00f009736b24751c8f26193` (`backup`, 2026-08-18)
- 이 문서를 만들기 직전 worktree는 clean이었다.
- 중요한 커밋 흐름:

| 커밋 | 의미 |
| --- | --- |
| `e90d32722d99d4d88ec1ed59d154770ff7b34af7` | 초기 추적 버그 수정, SAM2 mask reader/CLI, README, focused tests 등을 추가했다. 이때 rematch 및 intrinsics 수정도 잠시 포함됐다. |
| `0113d63` | ADD/ADD-S가 두 번째 키프레임부터 baseline과 달라진 원인을 비교한 뒤, rematch와 CUDACache 역투영 처리를 baseline과 같게 원복했다. |
| `711744f` | standalone GS runner, topology 실험 코드, 설정과 테스트를 저장한 현재 이정표다. 실험 당시 manifest는 `0113d63 + dirty`로 기록되지만, 당시 GS 파일이 현재 이 커밋에 포함되어 있다. |

`datasets/`, `logs/`, `*.ckpt`, LoFTR weights 등은 Git에 들어가지 않는다. 새 머신에 저장소만 clone하면 데이터와 실험 근거가 사라지므로 별도로 복사하거나 마운트하고 아래 SHA256을 확인해야 한다.

## 4. 기존 BundleSDF 파이프라인과 데이터 흐름

현재 `run_custom.py --mode run_video`의 큰 흐름은 다음과 같다.

1. `YcbineoatReader`가 RGB, millimeter depth, precomputed mask, `cam_K.txt`를 읽는다.
2. `BundleSdf`와 native `BundleTrack`이 프레임을 등록하고 LoFTR correspondence, RANSAC, pose optimization을 수행한다.
3. keyframe과 pose가 reconstruction process의 `NerfRunner`로 전달된다.
4. `run_video` 종료 후 `run_one_video_global_nerf()`가 자동 호출되어 global SDF refinement까지 수행된다.
5. pose, tracking log, SDF mesh 등이 output에 저장된다.

중요한 현재 사실:

- `bundlesdf.py`는 여전히 `nerf_runner.py`의 `NerfRunner`만 생성한다. GS backend 선택 분기나 `GaussianRunner` import는 없다.
- `run_gaussian_incremental.py`는 의도적으로 `bundlesdf.py`와 독립적이다. 저장된 `color/`, `depth_filtered/`, `mask/`, `cam_K.txt`, `keyframes.yml`을 입력으로 삼는다.
- mustard GS 실험에 사용한 저장 pose에는 기존 SDF reconstruction update의 pose refinement가 이미 반영되어 있다. 따라서 현재 결과는 GS 표현/학습/data path 검증이며, 순수 tracker-only pose나 완전한 online GS 통합 성능이 아니다.
- SAM2 모델을 이 저장소에서 online inference하는 것이 아니다. `masks_sam2/*.png` 또는 HO3D `masks_SAM2/`에 이미 생성된 mask를 읽는다.

## 5. 기존 코드 문제 검토와 최종 결정

과거에는 문제를 번호로 논의했지만, 중요한 것은 “한때 적용했다가 원복한 내용”과 “현재 유지된 내용”을 구분하는 것이다.

### 5.1 현재 유지된 수정

#### Matching FAIL 검사와 reference fallback

- 원래는 correspondence가 없을 때 `_matches[(frame, ref)]`를 먼저 읽어 `KeyError`가 날 수 있고, 중간 `pdb.set_trace()`에서 실행이 멈췄다.
- `find_corres()`가 match count를 반환하게 하고, caller는 `_matches`를 신뢰하기 전에 `frame._status == FAIL`을 먼저 확인한다. FAIL frame은 즉시 `forgetFrame()` 처리한다.
- 최초 reference의 match가 부족하면 동일 reference를 건너뛰고 covisibility 순서로 다른 keyframe을 시도한다. 모든 후보가 실패한 뒤에만 최종 FAIL 처리한다.
- RANSAC 최소 입력보다 match가 적은 pair는 빈 match로 만들고 GPU RANSAC 입력에서 제외한다.

FAIL 검사를 먼저 해야 하는 이유는 FAIL을 만든 경로가 `_matches` 항목을 반드시 생성한다는 보장이 없기 때문이다. 상태를 확인하지 않고 dictionary를 읽으면 정상적인 추적 실패가 Python 예외로 바뀐다.

#### CUDA pose host buffer

- `BundleTrack/src/cuda/LossGPU.cpp`에서 이미 `n_frames` 크기로 만든 `transforms_cpu`에 `clear()`를 호출한 뒤 CUDA copy와 indexing을 하던 한 줄을 제거했다.
- 목적은 내용만 지우고 capacity를 쓰려는 것이 아니다. `std::vector::clear()`는 size를 0으로 만들기 때문에, capacity가 남아 있어도 `data()`에 복사 후 `operator[]`로 읽는 것은 유효한 vector element 접근이 아니며 undefined behavior다.
- `clear()`를 제거해 vector size와 실제 element lifetime을 유지하는 방향을 사용자가 승인했다.

#### HO3D benchmark CLI

- `benchmark_ho3d.py`가 parsing한 `args`를 곧바로 `args = []`로 덮던 한 줄을 제거했다.
- 이 줄이 있으면 이후 `args.out_dir` 같은 접근이 실패한다. 사용자는 5번 문제로 이 한 줄 제거만 승인했다.

#### SAM2 mask 연결

- `YcbineoatReader`는 `mask_dir`를 받고 기본값 `masks_sam2`를 사용할 수 있다. `get_mask_file()`이 RGB와 같은 basename의 mask를 찾고 누락 시 명확한 `FileNotFoundError`를 낸다.
- `Ho3dReader`는 dataset root와 `masks_SAM2` 경로를 상대/절대 경로로 처리한다. 과거 하드코딩된 `HO3D_ROOT` 의존을 제거했다.
- `run_custom.py --mask_dir` 기본값은 `masks_sam2`, `run_ho3d.py --mask_dir` 기본값은 `masks_SAM2`다.
- 이것은 precomputed mask wiring이며 SAM2 network를 tracking process 안에서 실행하는 기능은 아니다.

#### README와 trimesh 호환

- `readme.md`는 `run_video`가 global refinement를 자동 실행한다는 실제 동작, SAM2 mask 디렉터리와 선택적 global refine 재실행 명령을 반영한다.
- `nerf_runner.py`의 deprecated `mesh.remove_duplicate_faces()`를 `mesh.update_faces(mesh.unique_faces())`로 바꿨다.

### 5.2 실험 후 baseline으로 되돌린 수정

#### Raw-match cache rematch

- 한때 `_raw_matches`만 있고 `_matches`가 없는 pair도 raw cache에서 3D correspondence/RANSAC을 다시 생성하도록 바꿨다.
- 논리적으로는 cache 상태 불일치를 복구하는 수정이지만 실제 mustard 추적은 baseline과 두 번째 keyframe부터 달라졌고 ADD/ADD-S가 악화됐다.
- 사용자 결정에 따라 `0113d63`에서 baseline rematch 흐름으로 되돌렸다. 현재는 FeatureManager가 `query_pairs`로 반환한 새 pair만 `rawMatchesToCorres()`한다.
- 따라서 “raw cache가 있으면 filtered `_matches`까지 항상 복구된다”는 invariant는 현재 보장하지 않는다. 이것은 모르는 버그가 아니라 재현성을 우선해 의도적으로 남긴 상태다.

#### CUDACache intrinsics/backprojection

- Eigen column-major와 CUDA matrix layout을 의심해 4x4 intrinsics inverse를 행/열 기준으로 명시 복사하는 수정을 시험했다.
- 실제 결과가 baseline보다 나빠졌고 추적 경로가 두 번째 keyframe부터 달라졌다. rematch와 재투영을 완전히 baseline과 같게 비교하자는 사용자 결정에 따라 원복했다.
- 현재 `BundleTrack/src/cuda/CUDACache.cpp`의 legacy reinterpret cast는 깔끔한 설계라서 유지한 것이 아니라 **기준 결과 재현을 위한 baseline-compatible 선택**이다.

### 5.3 낡은 문서와 테스트 주의

- `FIXES_SUMMARY.md`는 rematch 재생성과 명시적인 intrinsics 행/열 복사가 현재도 적용됐다고 적어 두었지만, 두 내용은 `0113d63`에서 원복됐다. 이 문서의 현재 설명을 우선한다.
- `tests/test_matching_flow.py`와 `tests/test_matching_gpu_integration.py`도 cached raw match에서 `_matches`를 다시 만든다는 원복 전 기대를 포함한다. 현재 deliberate baseline behavior에서는 해당 assertion이 실패할 수 있다.
- 새 에이전트는 matching tests의 실패를 보고 즉시 코드를 “고치면” 안 된다. baseline 재현과 cache semantic 중 무엇을 우선할지 사용자와 다시 합의해야 한다.
- 과거 번호 7·8 문제는 검토 후 수정하지 않기로 했고, 저장소의 확정 변경으로 남은 것은 없다. 커밋에 근거가 없는 세부 내용을 추측해 복원하지 않는다.

## 6. GS 전환에서 합의한 설계 방향

### 6.1 통합 전략

- 기존 `nerf_runner.py`를 직접 뜯어고치지 않고 보존한다.
- 새로운 `gaussian_runner.py`를 별도 backend로 유지한다.
- 충분히 검증한 뒤 `bundlesdf.py`에서 SDF와 GS를 명확히 분기하는 것이 목표다.
- tracker, keyframe 선정, pose 흐름 등 필수적이지 않은 기존 파이프라인은 최대한 건드리지 않는다.
- 온라인 분기는 아직 구현하지 않았다. 새 환경에서 이를 다음 작업이라고 보고 바로 연결해서는 안 되며, 먼저 standalone의 미해결 사항과 interface 설계를 사용자에게 확인해야 한다.

### 6.2 매 키프레임의 표현 갱신

두 선택지 중 다음을 채택했다.

- 선택하지 않은 방식: 매 키프레임마다 전체 Gaussian을 새 point cloud에서 다시 초기화한다.
- 채택한 방식: 기존 Gaussian parameter를 유지하고, 새 키프레임 RGB-D 중 과거에 관측하지 않은 점만 append한다.

물체 뒷면처럼 새 표면이 나타나면 Gaussian 수가 증가하는 것이 의도다. topology를 count-neutral하게 설계한 것은 **append 증가량을 다시 없애기 위한 것이 아니다.** append가 끝난 현재 수 `N` 안에서 grow K와 prune K를 교환해 topology 연산 자체의 수 변화만 0으로 만들기 위함이다.

### 6.3 좌표계 계약

- tracker/save pose의 공개 입력은 metric OpenCV camera-to-object/world, 즉 `c2w_cv`다.
- point는 카메라 좌표에서 RGB-D backprojection한 뒤 `c2w_cv`로 object metric 좌표로 보낸다.
- scene normalization은 `x_normalized = scale * (x_metric + translation)`이다. pose translation도 같은 변환을 적용한다.
- gsplat `viewmats`에는 normalized CV c2w를 inverse한 object/world-to-camera (`w2c_cv`)를 준다.
- SDF interface에서 사용하는 OpenGL 축 flip을 GS renderer 직전에 추가하지 않는다.
- 첫 5 frame의 BundleSDF `compute_scene_bounds()`를 재사용할 때만 기존 함수 계약 때문에 CV c2w를 임시 GL c2w로 변환한다. runner 내부의 gsplat 좌표 계약은 계속 CV다.
- crop을 하면 `cx`, `cy`에서 crop origin을 빼고, resize/downsample하면 `fx`, `fy`, `cx`, `cy`를 새 해상도 비율로 조정한다.

이 방향은 “tracker pose는 CV c2w이고 gsplat은 CV w2c viewmat을 받는다”는 외부 검토 의견과 코드의 projection test를 함께 확인해 채택했다.

### 6.4 RGB-D seed와 신규점 선택

`rgbd_to_point_cloud()`의 흐름:

1. object mask가 true이고 depth가 finite이며 설정 범위 안인 pixel만 선택한다.
2. `K`로 CV camera point를 backproject한다.
3. `c2w_cv`로 object metric point로 바꾼다.
4. deterministic voxel averaging으로 point/color를 downsample한다.

첫 5 keyframe cloud로 초기 Gaussian을 만들고, 이후 keyframe cloud는 `observed_points_metric` KD-tree와의 최근접 거리가 `novelty_distance`보다 큰 점만 append한다. 사용자와 논의한 복잡한 다단계 가시성/거리 정책 중 우선 **이 한 가지 거리 기준만** 적용했다.

`observed_points_metric`은 학습 참여 횟수가 아니다. 과거 RGB-D에서 seed 후보로 처리한 metric 위치의 누적 기억이다. 현재 live Gaussian과 분리되어 있어 Gaussian을 prune해도 해당 위치는 observed history에 남는다. 그러므로 나중에 같은 위치가 보여도 novelty filter 때문에 재-append되지 않을 수 있는 stale-memory 한계가 있다.

### 6.5 1 cm에서 1 mm로 바꾼 이유

- 1 cm 설정은 첫 5 KF에서 410개, 36 KF에서 478개밖에 남지 않아 새 keyframe이 거의 표현을 추가하지 못했다.
- 1 mm 설정은 첫 5 KF에서 13,345개, KF6 append 후 14,836개, 36 KF에서 45,369개였다.
- 사용자는 1 cm가 너무 coarse하다고 판단했고, 동일 조건 비교 후 voxel size와 novelty distance 모두 `0.001 m`를 채택했다.
- 기본 `config_gs.yml`은 아직 1 cm다. mustard 연구 경로에서는 `config_gs_1mm*.yml`을 명시해야 한다.

### 6.6 깊이 범위

- 현재 GS point seed는 `0.1 <= depth <= 2.0 m`만 사용한다.
- 이 제한은 원래 `nerf_runner.py`의 동일 정책을 그대로 옮긴 것이 아니라 GS 초기 구현에 넣은 유효 depth gate다.
- mustard의 737 mask frame을 조사했을 때 masked depth는 모두 2 m보다 작았고 최대값은 대략 1 m 이하였다. 현재 실험에는 영향을 주지 않는다.
- 더 먼 물체가 있는 다른 데이터셋에서는 문제가 될 수 있다. 사용자는 이 값을 지금은 유지하되 보류 사항으로 두었다. 임의로 일반화하거나 고정 상수의 정당성을 주장하지 않는다.

### 6.7 신규 Gaussian 초기값

- mean: normalized RGB-D point
- scale: 가까운 이웃 최대 4개의 RMS 거리에서 만든 isotropic log scale. 최소값은 `0.1 * voxel_size * scene_scale`이다.
- rotation: identity quaternion
- opacity: `0.1`의 logit
- color: `(RGB - 0.5) / SH_C0`를 SH degree 0 coefficient에 넣고 나머지 SH coefficient는 0

### 6.8 학습과 아직 없는 기능

- 매 step 현재까지 저장된 view 중 하나를 균등 random sampling한다.
- loss는 object mask 내부 RGB L1 + DSSIM이며 `ssim_weight=0.2`다.
- renderer는 RGB, alpha, expected depth를 만들고 depth는 metric camera-z로 되돌려 평가한다.
- 현재 없는 것: depth loss, silhouette/background loss, pose delta/pose optimization, mesh extraction/return, GUI 및 online `p_dict` 연동.
- mesh는 “실패했을 때 fallback”이 아니라 아직 구현하지 않은 기능이며 사용자가 일단 미루기로 했다.
- RGB-only masked loss 때문에 RGB MAE가 좋아지면서 depth, alpha IoU, outside alpha가 나빠질 수 있다. repeated duplicate 결과가 이를 실제로 보여 주었다.

### 6.9 append 후 optimizer/state 처리

- 기존 splat tensor 값은 보존해 새 seed를 이어 붙인다.
- 그러나 append 때 `ParameterDict`를 다시 만들고 모든 Adam optimizer와 DefaultStrategy state를 다시 생성한다.
- 즉 기존 Gaussian도 parameter 값은 이어가지만 Adam moment와 topology 통계는 잃는다. update pose delta가 있다면 identity에서 다시 시작한다는 과거 표현은 이런 reconstruction update 경계를 뜻했지만, 현재는 pose delta 자체가 없다.
- private optimizer state를 직접 resize하는 복잡성을 피한 초기의 안전한 선택이다. 성능/수렴상 한계로 기록하되 승인 없이 바꾸지 않는다.

## 7. GS 관련 파일 역할

| 파일 | 역할 |
| --- | --- |
| `gaussian_runner.py` | 고정 pose incremental Gaussian state, 좌표/normalization, RGB-D seed, append, train/render, checkpoint, PLY export |
| `gaussian_budget_strategy.py` | threshold-free percentage candidate 선택과 manual duplicate/split/prune 적용 |
| `run_gaussian_incremental.py` | 저장된 tracking log를 첫 5 KF initialize + 이후 KF append/update로 replay |
| `run_gaussian_budget_ab.py` | KF6의 단일 budget-neutral mixed event와 1/2/5% 비교 |
| `run_gaussian_operator_ablation.py` | 공통 KF6 상태에서 A/P/D/S/M 연산 분리, immediate/+50/+250/+500 평가 |
| `run_gaussian_long_followup.py` | KF6에서 한 번 연산한 branch를 추가 topology 없이 KF36까지 추적 |
| `run_gaussian_repeated_topology.py` | KF6–36 매 keyframe exact 3% 연산을 강제하는 mustard 전용 stress test |
| `config_gs.yml` | gsplat 1.5.3, 기본 1 cm, initial 30k/update 500, 일반 DefaultStrategy |
| `config_gs_1mm.yml` | voxel/novelty만 1 mm로 override |
| `config_gs_1mm_append_only.yml` | 1 mm이며 automatic topology 시작을 100000으로 미뤄 사실상 비활성화 |
| `config_gs_1mm_update_first_event.yml` | update refine window 100–500 |
| `config_gs_1mm_update_single_event.yml` | update window 100–201로 첫 event 하나만 허용 |
| `tests/test_gaussian_geometry.py` | 좌표계, normalization, K crop/resize, RGB-D 역투영, transaction rollback |
| `tests/test_gaussian_budget_strategy.py` | rank/visibility/tail 보호/operator 분리/count-neutral/opacity compositing |
| `tests/test_gaussian_gpu_integration.py` | 선택적 gsplat CUDA initialize/append/render/checkpoint/PLY/topology 통합 검사 |

`BundleSAM3D`라는 확장자 없는 tracked 파일은 GS entry point가 아니다. 다른 Harvest 실행 명령과 개인 메모가 섞인 scratch file이므로 source of truth로 사용하지 않는다.

## 8. Percentage topology의 설계와 의미

### 8.1 왜 threshold-only에서 percentage로 갔는가

기본 DefaultStrategy threshold는 데이터와 학습 상태에 따라 Gaussian 수가 크게 달라지고 조절 변수가 많다. 이 프로젝트는 매 keyframe RGB-D seed를 대량 append하므로 일반적인 3DGS densification과 조건도 다르다. 사용자는 topology가 append 성장까지 상쇄하는 것을 원하지 않았고, 현재 표현 안에서 불필요한 점을 제한된 비율만큼 교체하는 실험을 제안했다.

처음에는 1/2/5%를 비교했고, 이후 operator 원인을 분리할 때 최대 교체 비율 3%를 선택했다. Percentage가 최종 알고리즘으로 확정된 것은 아니며, 현재 코드는 통제된 진단 도구다.

### 8.2 현재 selection과 연산 순서

- `K = floor(fraction * N)`이다.
- grow 후보는 prefix 학습 중 `count > 0`인, 즉 적어도 한 번 rasterization에서 통계를 받은 visible Gaussian만 사용한다.
- grow rank는 mean screen-space gradient `grad2d / count`가 높은 순이다.
- prune rank는 opacity가 낮은 순이다.
- prune 후보는 grow와 겹치지 않으며, 방금 append된 tail은 같은 event의 prune에서만 보호한다.
- 후보가 부족하면 selector는 K를 축소할 수 있다. repeated runner는 승인된 exact 3%를 만족하지 못하면 silent 진행하지 않고 assert로 중단한다.

모드:

- `prune_only` (P): grow 없이 K개만 prune하므로 수가 줄어든다.
- `duplicate_only` (D): grow 상위 K개 duplicate 후 prune K개, 수 유지.
- `split_only` (S): grow 상위 K개 split 후 prune K개, 수 유지.
- `mixed` (M): grow K를 scale로 정렬해 작은 절반은 duplicate, 큰 절반은 split한 뒤 prune K개, 수 유지.
- append-only (A): topology event가 없지만 동일한 총 학습 step을 수행하는 대조군.

manual event 후 `grad2d`, `count`, `radii`를 0으로 초기화한다. D/S/M의 수 유지 기준은 event 직전, 즉 append가 이미 끝난 `N`이다. 따라서 새 표면으로 증가한 전체 Gaussian 수는 보존된다.

## 9. 실험 공통 조건과 metric 정의

### 9.1 고정 입력

- tracking root: `logs/mustard0_bundlesam3dgs_exact_baseline_rematch_backprojection_20260811`
- 정확한 36-KF snapshot: `logs/mustard0_bundlesam3dgs_exact_baseline_rematch_backprojection_20260811/1581120495188295656/keyframes.yml`
- YAML SHA256: `12abca70687e2473ce85179e06cd23c3d8b6f802972d7e8ba5e87e216d0da62b`
- intrinsic:

```text
[[319.58200,   0.00000, 320.21500],
 [  0.00000, 417.11868, 244.34866],
 [  0.00000,   0.00000,   1.00000]]
```

- 첫 5 frame normalization: actual scale `7.5682218500`, metric translation `[0.00612262, 0.00494046, -0.00628344]`
- runtime manifest: PyTorch `2.6.0+cu124`, gsplat `1.5.3`

tracking root에는 timestamp snapshot이 많이 있으므로 `find_latest_keyframe_snapshot()`에 맡기지 말고 위 YAML을 명시해야 한다.

### 9.2 metric

- view별 값을 비가중 평균한다.
- RGB MAE/PSNR: object mask 내부 RGB
- depth MAE: mask 내부이며 fixed target depth가 유효한 pixel, 단위는 아래 표에서 mm
- alpha IoU: alpha threshold 0.5
- outside alpha: crop 안의 object mask 바깥 평균 alpha

현재 GS 평가는 saved pose render metric이다. BundleSDF baseline의 ADD/ADD-S pose metric과 같은 지표가 아니므로 직접 숫자를 비교하면 안 된다.

## 10. 실험 연대기와 결론

### 10.1 1 cm 대 1 mm (2026-08-11)

| 설정 | 결과 경로 | 첫 5 KF | KF6 | KF36 | 결론 |
| --- | --- | ---: | ---: | ---: | --- |
| 1 cm | `logs/mustard0_gaussian_fixed_pose_36kf_smoke_20260811` | 410 | +1 → 411 | 478 | 지나치게 coarse |
| 1 mm | `logs/mustard0_gaussian_fixed_pose_1mm_36kf_smoke_20260811` | 13,345 | +1,491 → 14,836 | 45,369 | 이후 채택 |

1 cm 최초 5+1 경로 `...fixed_pose_5plus1_smoke_20260811`은 manifest가 stale `running`이며 불완전하다. 성공본은 `...fixed_pose_5plus1_smoke_retry1_20260811`이다. 1 mm 5+1 성공본은 `...fixed_pose_1mm_5plus1_smoke_20260811`이다.

### 10.2 DefaultStrategy event와 회복 길이

- append-only는 651 initial +251 또는 +451 step 뒤에도 `N=14,836`을 유지했다.
- update step 200에서 DefaultStrategy event를 허용하면 `N=18,234` 또는 `18,239`로 한 번에 약 3.4k가 늘었다.
- topology 직후 250 step 회복은 표현 안정화에 부족해 보였다. event 뒤 500 step을 학습하는 설계로 옮겼다.
- 이 결과가 threshold-based automatic densification을 그대로 최종 채택하지 않고 exact-rank 실험으로 전환한 배경이다.

관련 경로:

- `logs/mustard0_gaussian_1mm_5plus1_append_only_651_251_gpu0_20260811`
- `logs/mustard0_gaussian_1mm_5plus1_append_only_651_451_gpu0_20260811`
- `logs/mustard0_gaussian_1mm_5plus1_update_first_event_651_251_gpu1_20260811`
- `logs/mustard0_gaussian_1mm_5plus1_single_event_recovery250_651_451_gpu1_20260811`

### 10.3 단일 budget-neutral 비율 sweep

공통 KF6 상태 `N=14,836`에서 prefix 201 step 뒤 mixed grow/prune를 수행하고 500 step 회복했다.

| 비율 | K | duplicate/split | 최종 training loss |
| ---: | ---: | ---: | ---: |
| A, event 없음 | 0 | 0/0 | 0.049385 |
| 1% | 148 | 74/74 | 0.050896 |
| 2% | 296 | 148/148 | 0.052543 |
| 5% | 741 | 371/370 | 0.054316 |

같은 수를 prune해 D/S/M의 N을 유지했다. 강제 교체 비율이 클수록 loss가 악화했고, pruning 자체보다 duplicate/split 중 무엇이 표현을 흔드는지 분리할 필요가 생겼다.

경로: `logs/mustard0_gaussian_1mm_5plus1_budget_neutral_2pct_recovery500_shared_ckpt_20260813`

### 10.4 KF6 one-event operator ablation, exact 3%

경로: `logs/mustard0_gaussian_1mm_5plus1_operator_ablation_3pct_recovery500_shared_ckpt_20260814`

공통 `N=14,836`, exact `K=445`, newest KF6 tail 1,491개는 same-event prune에서 보호했다. A/P/D/S/M을 event 직후와 +50/+250/+500 step에 평가했다.

| branch | event 직후 RGB / depth mm | +500 RGB / depth mm | 핵심 |
| --- | --- | --- | --- |
| A | 0.037287 / 3.19758 | 0.035815 / 3.20875 | 대조군 |
| P | 0.037286 / 3.19775 | 0.035823 / 3.20895 | 사실상 A와 동일 |
| D | 0.037730 / 3.27223 | 0.035636 / 3.24670 | 즉시 충격 작지만 depth 악화 |
| S | 0.045527 / 3.47594 | 0.038088 / 3.34772 | 즉시 가장 크게 파괴 |
| M | 0.044276 / 3.39989 | 0.037528 / 3.29039 | split 영향이 크게 나타남 |

A 대비 event 직후 S는 RGB `+22.10%`, depth `+8.71%`; M은 RGB `+18.74%`, depth `+6.33%` 악화했다. D는 각각 `+1.19%`, `+2.33%`였다. **현재 topology의 즉시 표현 파괴는 split이 주원인**이라는 결론을 얻었다.

### 10.5 한 번의 3% event 후 KF36까지 append-only

경로: `logs/mustard0_gaussian_1mm_single_event_long36_20260814`

KF6에서만 각 operator를 적용하고 KF7–36은 새 점 append +500 step, topology 없음으로 진행했다. 모든 branch가 완료됐다.

| branch | N | RGB MAE | depth mm | alpha IoU | outside alpha |
| --- | ---: | ---: | ---: | ---: | ---: |
| A | 45,369 | 0.0426430 | 3.08349 | 0.614083 | 0.294707 |
| P | 44,924 | 0.0426269 | 3.07654 | 0.614310 | 0.294410 |
| D | 45,369 | 0.0424665 | 3.10710 | 0.613930 | 0.294934 |
| S | 45,369 | 0.0425290 | 3.10850 | 0.615855 | 0.289243 |
| M | 45,369 | 0.0424970 | 3.14975 | 0.617794 | 0.287865 |

한 번의 split 충격은 새 view와 장기 학습으로 RGB가 회복됐지만 depth penalty가 남았다. 단일 445개 변경은 최종 45k 중 1% 미만으로 희석되므로 반복 적용의 안전성을 증명하지는 않는다.

### 10.6 매 keyframe exact 3% repeated stress

경로: `logs/mustard0_gaussian_1mm_repeated_topology_3pct_20260814`

KF6–36마다 다음 순서를 적용했다.

```text
RGB-D 신규점 append
  -> 201-step prefix 학습/통계 수집
  -> exact 3% topology event
  -> 500-step recovery
```

A도 event 없이 동일한 총 701 step을 수행했다. 자동 DefaultStrategy topology/reset은 비활성화했고 milestone 6/12/18/24/30/36에서 full-view metrics와 checkpoint를 저장했다.

여기서 milestone 간격은 저장/평가 시점일 뿐 topology 주기가 아니다. long36은 KF6에서 한 번만 event를 적용했고, repeated stress는 KF6부터 **매 keyframe** event를 적용했다. 별도의 “5 keyframe마다 topology” 완료 실험으로 해석하면 안 된다.

완료한 KF36 결과:

| branch | N | 누적 연산 | RGB MAE | depth mm | alpha IoU | outside alpha |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| A | 45,369 | 없음 | 0.0386843 | 3.18125 | 0.603123 | 0.308558 |
| P | 24,555 | prune 20,814 | 0.0391960 | 3.19177 | 0.602417 | 0.309332 |
| D | 45,369 | duplicate/prune 29,447 | 0.0375284 | 3.40078 | 0.580849 | 0.336563 |

해석:

- P는 A보다 Gaussian이 45.9% 적지만 RGB `+1.32%`, depth `+0.01052 mm` 정도만 악화했다. mustard에서 현재 low-opacity pruning 기준은 비교적 안전해 보인다.
- D는 RGB를 `-2.99%` 개선했지만 depth를 `+0.21953 mm` 악화했고 IoU가 낮아지며 outside alpha가 늘었다. RGB-only loss가 appearance를 개선하면서 geometry/silhouette를 훼손한 증거다.
- S는 KF30까지 완료 후 KF31 event에서 실패했다. KF30에서 `N=40,978`, RGB `0.0378893`, depth `3.56901 mm`였다. 같은 시점 A보다 RGB는 1.79% 좋지만 depth는 `+0.37566 mm` 나빴다.
- M은 KF33까지 `N=43,233`으로 진행 후 KF34 event에서 실패했다. KF30 depth는 `3.51176 mm`로 A보다 `+0.31841 mm` 나빴다.
- 기록된 S 25회와 M 28회 event 모두 latest-view RGB를 event 직후 악화시켰다.

**주의:** long36 A는 KF7–36에 update당 500 step, repeated A는 KF6–36에 update당 701 step이다. 서로 다른 실험의 절대 metric을 직접 비교하지 말고 각 실험 내부의 동시 A 대비만 비교한다. GPU별 배경 부하가 달랐으므로 elapsed time도 알고리즘 속도 비교 근거로 쓰지 않는다.

## 11. 최우선 미해결 문제: split opacity `+inf`

### 11.1 확인된 원인

`gaussian_budget_strategy.py`의 split path가 gsplat `split(..., revised_opacity=True)`를 호출한다. 반복 학습으로 parent opacity logit이 매우 커지면 다음 과정이 발생한다.

- S의 KF31 선택 parent logit: 약 `16.664110`
- M의 KF34 선택 parent logit: 약 `16.659229`
- float32 `sigmoid(logit)`이 정확히 `1.0`으로 포화
- gsplat의 revised child opacity 계산 결과도 probability `1.0`
- 다시 logit으로 바꾸며 `logit(1.0) = +inf`
- topology finite-value guard가 `FloatingPointError: Non-finite values in opacities after topology`로 안전하게 중단

실패 디렉터리와 manifest는 이 원인을 증명하는 artifact이므로 보존한다. S는 KF30 이후, M은 KF33 이후의 final PLY/milestone이 없는 것이 정상이다.

### 11.2 논의된 좁은 수정안 — 아직 미승인

다음 방향은 제안만 했고 구현하지 않았다.

1. selected split parent의 child opacity logit을 probability 왕복 없이 logit-domain의 수치 안정적인 식으로 계산한다.
2. 해당 값을 parent slot에 설정하고 필요한 optimizer state를 일관되게 처리한다.
3. gsplat `split()`은 `revised_opacity=False`로 호출해 내부의 불안정한 재계산을 피한다.
4. grow/prune candidate, K=3%, child position/scale, loss, append 정책은 바꾸지 않는다.
5. 극단 high-logit에서도 finite이고 parent와 children의 composite opacity가 보존되는 unit regression을 추가한다.
6. 기존 실패 결과를 건드리지 않고 S/M만 새로운 output directory에서 같은 조건으로 재실행한다.

새 에이전트는 이 방향이 타당해 보여도 바로 수정하지 말고, 대상 파일과 exact formula/test를 사용자에게 다시 제시해 승인을 받아야 한다.

## 12. 환경과 의존성

### 12.1 과거에 사용한 환경

- 환경 분리의 의도는 과거 BundleSDF 작업 폴더/환경을 그대로 변경하는 것이 아니라, 필요한 image와 dependency만 현재 `BundleSAM3DGS`용 환경에 재사용하는 것이었다. 기존 BundleSDF 쪽은 독립적으로 남겨 둔다.
- 컨테이너 이름: `BundleSAM3DGS-gsplat-smoke`
- base image: `nvcr.io/nvidian/bundlesdf:latest`
- mount: `/home:/home`, `/tmp:/tmp`
- options: `--gpus all`, `--network=host`, `--ipc=host`
- Python: `/usr/bin/python3`
- PyTorch: `2.6.0+cu124`
- gsplat: `1.5.3`
- 두 CUDA device를 병렬 실험에 사용했다. 과거 작업 중 RTX 3090 두 장으로 확인한 기록이 있으나 artifact manifest에는 model name이 없으므로 새 환경에서 `nvidia-smi`로 다시 확인한다.
- 컨테이너 기본 workdir는 repo가 아니므로 `docker exec` 안에서 항상 `cd /home/kist/Desktop/BundleSAM3DGS`가 필요하다.

### 12.2 2026-08-20 현재 상태

- 위 컨테이너는 `Exited (255)` 상태다.
- host `nvidia-smi`는 NVIDIA driver와 통신하지 못했다.
- 따라서 이 인수인계 작성 시점에는 GPU test나 실험을 재실행하지 않았다.

새 환경의 첫 확인:

```bash
docker ps -a --filter name=BundleSAM3DGS-gsplat-smoke
nvidia-smi
docker start BundleSAM3DGS-gsplat-smoke
docker exec BundleSAM3DGS-gsplat-smoke bash -lc '
  cd /home/kist/Desktop/BundleSAM3DGS &&
  /usr/bin/python3 -c "import torch, gsplat; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.device_count(), gsplat.__version__)"
'
```

기존 컨테이너가 없으면 `docker/dockerfile.gsplat`로 base image에 gsplat 1.5.3만 추가한 독립 image를 만든다.

```bash
docker build -f docker/dockerfile.gsplat -t bundlesam3dgs:gsplat .
docker run --gpus all --network host --ipc host \
  --name BundleSAM3DGS-gsplat-smoke \
  -v /home:/home -v /tmp:/tmp \
  -d bundlesam3dgs:gsplat sleep infinity
```

`bash build.sh`는 `mycuda`와 `BundleTrack/build`를 지우고 native extension을 다시 빌드한다. 새 이미지/환경 검증이나 native code 변경 때만 container 안에서 실행하고, 무조건적인 첫 단계로 반복하지 않는다.

## 13. 데이터, 가중치, 보존할 artifact

### 13.1 데이터와 mask

- mustard: `datasets/YCBInEOAT/mustard0`, 약 4.2 GB
- RGB 737장과 `masks_sam2` 737장은 basename set이 일치한다.
- `depth/`에는 1,332개로 여분 파일이 있지만 실험 frame list는 저장된 keyframe ID를 따른다.
- HO3D: `datasets/HO3D_v3`, 약 6.7 GB, `masks_SAM2` 포함
- YCB model: `datasets/YCB_Video_Models/models/006_mustard_bottle`
- LoFTR 주요 weight: `BundleTrack/LoFTR/weights/outdoor_ds.ckpt`
  - size: 46,341,978 bytes
  - SHA256: `21f5bec5968178e8bc8b7633441836fe5de4f47d861dd2cd7dc38e271b0479ec`

새 mask를 만들어야 할 때 사용할 수 있는 외부 SAM2 설치/체크포인트가 과거 머신의 `/home/kist/Desktop/sam2`와 `/home/kist/Desktop/sam2/checkpoints/`에 있다. 그러나 현재 repo에 mask 생성 pipeline은 연결되어 있지 않으며 기존 실험 재현에는 PNG mask만 필요하다.

### 13.2 절대 보존할 tracking 입력과 checkpoint

| artifact | 경로/검증 |
| --- | --- |
| exact tracking log | `logs/mustard0_bundlesam3dgs_exact_baseline_rematch_backprojection_20260811` |
| exact 36-KF pose YAML | 위 root의 `1581120495188295656/keyframes.yml`, SHA256 `12abca70687e2473ce85179e06cd23c3d8b6f802972d7e8ba5e87e216d0da62b` |
| common 5-KF checkpoint | `logs/mustard0_gaussian_1mm_5plus1_budget_neutral_2pct_recovery500_shared_ckpt_20260813/checkpoint_initial.pt`, SHA256 `996848108ac6a99af90321cf340f0ecaf178a89fb5b4169c5f0e63905a9cf3a7` |
| common KF6 append+201 checkpoint | 같은 root의 `checkpoint_common_step200.pt`, SHA256 `2a37b113f3d060d229219895db2f44fa9e3dccb0d7b0055b2d3e1859424d4088` |

`checkpoint_initial.pt` 상태는 5 views, `N=13,345`, `update_index=0`, `total_steps=strategy_step=651`, voxel/novelty 1 mm다. `checkpoint_common_step200.pt`는 KF6의 1,491개 append 뒤 `N=14,836`, prefix 201 step 상태이며 protected tail 시작 index는 13,345다.

### 13.3 보존할 결과 root

- `logs/mustard0_gaussian_1mm_5plus1_operator_ablation_3pct_recovery500_shared_ckpt_20260814`
- `logs/mustard0_gaussian_1mm_single_event_long36_20260814`
- `logs/mustard0_gaussian_1mm_repeated_topology_3pct_20260814`
- 전체 `logs/`는 정리 당시 약 3.8 GB다.
- 기존 BundleSDF 비교 결과: `/home/kist/Desktop/BundleSDF_baseline_outputs`

baseline root에는 서로 다른 provenance의 평가 suite가 여러 개 있어 숫자를 섞으면 안 된다. 예를 들어 `mustard0_sdf_stride10`, `mustard0_sdf_stride10_rerun`, `full_eval_sam2_ycb/gpu/ycb_mustard0_{online,global}.csv`가 있다. 각 suite의 `provenance.json` 또는 input config를 확인해 같은 조건끼리 비교한다.

## 14. 검증 명령과 재실행 template

### 14.1 CPU/경량 검사

```bash
cd /home/kist/Desktop/BundleSAM3DGS
/usr/bin/python3 -m unittest tests.test_gaussian_geometry tests.test_gaussian_budget_strategy
```

과거 결과는 geometry 7개, budget strategy 6개가 통과했다. 현재 split extreme-logit regression은 아직 없다.

### 14.2 선택적 GPU integration

```bash
cd /home/kist/Desktop/BundleSAM3DGS
RUN_GSPLAT_GPU_INTEGRATION=1 GAUSSIAN_TEST_DEVICE=cuda:0 \
  /usr/bin/python3 -m unittest tests.test_gaussian_gpu_integration
```

과거 2개가 통과했다. gsplat 1.5.3과 CUDA가 없으면 skip되므로 “실행 성공”과 “실제로 2개 수행”을 구분한다.

### 14.3 standalone 36-KF replay template

```bash
cd /home/kist/Desktop/BundleSAM3DGS
/usr/bin/python3 run_gaussian_incremental.py \
  --track-dir logs/mustard0_bundlesam3dgs_exact_baseline_rematch_backprojection_20260811 \
  --keyframes-yaml logs/mustard0_bundlesam3dgs_exact_baseline_rematch_backprojection_20260811/1581120495188295656/keyframes.yml \
  --output-dir logs/<새로운-고유-이름> \
  --config config_gs_1mm_append_only.yml \
  --device cuda:0 \
  --max-keyframes 36
```

`output-dir`는 존재하면 안 된다. script가 `exist_ok=False` 정책으로 중단하는 것은 기존 결과 보호를 위한 의도다.

### 14.4 repeated topology 재실행 template

```bash
cd /home/kist/Desktop/BundleSAM3DGS
/usr/bin/python3 run_gaussian_repeated_topology.py \
  --checkpoint logs/mustard0_gaussian_1mm_5plus1_budget_neutral_2pct_recovery500_shared_ckpt_20260813/checkpoint_initial.pt \
  --branch-name <고유-branch-name> \
  --topology-mode <none|prune_only|duplicate_only|split_only|mixed> \
  --track-dir logs/mustard0_bundlesam3dgs_exact_baseline_rematch_backprojection_20260811 \
  --keyframes-yaml logs/mustard0_bundlesam3dgs_exact_baseline_rematch_backprojection_20260811/1581120495188295656/keyframes.yml \
  --output-dir logs/<새로운-고유-이름> \
  --device cuda:0 \
  --max-keyframes 36 \
  --budget-fraction 0.03 \
  --prefix-steps 201 \
  --recovery-steps 500 \
  --lr-decay-horizon-steps 701 \
  --milestones 6 12 18 24 30 36
```

split CUDA RNG는 device global RNG의 영향을 받는다. operator 공정성을 엄밀히 보려면 S/M을 같은 GPU에서 순차 실행하거나 event마다 canonical device seed 정책을 별도 승인받아 구현해야 한다. GPU0/1에서 동시에 돌린 runtime은 비교하지 않는다.

## 15. 다음 작업의 권장 순서

1. 새 환경에서 Git HEAD, 데이터/weight/checkpoint 존재 여부와 SHA256을 확인한다.
2. NVIDIA driver, 두 GPU visibility, container, PyTorch/gsplat 1.5.3 import를 확인한다.
3. 이 문서와 `AGENTS.md`를 읽고, 현재 코드가 standalone fixed-pose 단계임을 다시 확인한다.
4. split opacity 안정화의 exact formula, 대상 `gaussian_budget_strategy.py`, 추가할 test, S/M 재실행 계획을 사용자에게 제시한다.
5. **명시적 승인 후에만** 좁은 수정과 CPU/GPU focused test를 수행한다.
6. 실패한 S/M directory는 유지하고 새 output directory에서 S/M을 재실행한다. A/P/D는 같은 조건의 기존 결과가 있으므로 불필요하게 다시 돌리지 않는다.
7. 결과를 A/P/D 및 aligned KF와 비교해 split의 표현 충격, depth/IoU, finite stability를 판단한다.
8. 그 다음에 depth/silhouette loss, stale observed history, optimizer state 보존, topology 주기/최대 비율을 각각 독립 실험으로 논의한다.
9. standalone 정책이 안정된 뒤에야 `bundlesdf.py`의 SDF/GS backend 분기와 online interface를 설계하고 별도 승인을 받는다.

## 16. 열린 연구 질문

- repeated prune-only가 Gaussian 수를 거의 절반 줄이고도 mustard 품질을 유지한 이유가 다른 object/sequence에도 일반화되는가?
- duplicate의 RGB 개선과 depth/IoU 악화를 depth loss나 alpha/background loss가 완화하는가?
- split의 수치 안정성을 고친 뒤에도 geometric displacement 자체가 표현을 과도하게 파괴하는가?
- exact percentage를 매 frame 적용하는 대신 최대 비율, adaptive 주기, event trigger를 어떻게 정의할 것인가?
- prune된 Gaussian 위치를 `observed_points_metric`에서 언제 제거해 재관측 append를 허용할 것인가?
- append 때 전체 optimizer/strategy state를 reset하는 현재 단순 정책이 장기 수렴에 얼마나 손해인가?
- `0.1–2.0 m` depth gate를 dataset geometry에 맞게 설정 또는 자동화해야 하는가?
- fixed saved pose 검증을 통과한 뒤 online GS가 tracker pose update와 어떻게 동기화될 것인가?
- mesh가 필요한 기존 BundleSDF consumer에 GS 결과를 어떤 interface로 제공할 것인가?

## 17. 문서 신뢰 우선순위

현재 상태가 충돌할 때 다음 순서로 판단한다.

1. 현재 코드와 해당 실험의 JSON manifest/checkpoint
2. 이 `PROJECT_HANDOFF.md`
3. Git commit history, 특히 `0113d63`
4. `AGENTS.md`의 협업 규칙
5. `FIXES_SUMMARY.md` 및 `BundleSAM3D` scratch note

`FIXES_SUMMARY.md`는 과거 의도를 이해하는 데는 유용하지만 rematch/intrinsics의 현재 상태에는 틀리다. 어떤 기록이 애매하면 코드를 바꾸지 말고 read-only로 확인한 뒤 사용자에게 근거와 선택지를 제시한다.
