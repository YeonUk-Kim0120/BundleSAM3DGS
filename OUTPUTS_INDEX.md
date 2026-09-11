# 실험 산출물 색인 (`outputs/`, `logs/`) — 2026-09-11 기준

`outputs/`(git 무시)의 실행 디렉터리 218개가 무엇이고, 그 결론이 어디에 기록돼 있는지 정리한 색인이다.
디스크 정리(덤프 삭제) 전에 남겨야 할 기록이 모두 있는지 확인하는 기준 문서이며, 정리 후에도 이 문서와 아래 JSON·결과 문서만으로
모든 수치를 다시 찾을 수 있어야 한다. 채점 JSON의 `run_dir`가 곧 실행 디렉터리 이름이다.

## 실행 디렉터리 안에서 "기록"에 해당하는 파일 (모든 실행 공통)

| 파일 | 역할 | 쓰는 스크립트 |
|---|---|---|
| `ob_in_cam/*.txt` | 프레임별 온라인 포즈. **ADD/ADD-S 채점의 유일한 입력** | `experiments/eval_add_ycbineoat.py` |
| `<cycle>/poses_before_gs.txt`, `poses_after_gs.txt` (SDF 실행은 `_nerf`), `nerf_frames.txt` | 재구성 사이클마다 트래커가 준 포즈 / 되돌려준 포즈 / 키프레임 id | `experiments/exp_feedback_gradient_probe.py`, `logs/gsfb_batch_20260907/diag_*.py` |
| `<frame>/keyframes.yml`, `opt_*_poses.txt` | 트래커의 프레임별 키프레임 포즈 스냅샷 | (진단 예비) |
| `gs_online/feedback_log.json`, `lifecycle_log.json` | 사이클별 보정 크기(뷰별 회전·이동), lifecycle 상태 통계 | 결과 문서 §7.5의 보정 크기 통계 |
| `ablation_manifest.json`, `config_*.yml`, `cam_K.txt` | 조건·마스크·시드·설정 | 재현 |
| `color/ depth_filtered/ mask/` | 키프레임 재생 입력 (v1 실행에서만 필요) | `run_gaussian_incremental.load_frame` ← 기울기 프로브 |
| `gs_online/checkpoint_final.pt`, `splats_*.ply` | 최종 지도 (재구성 평가 후보) | 아직 없음 |

기록이 아닌 것(용량의 95%): 사이클별 지도 덤프 `<cycle>/gs_state.ply`(289 GB), 트래커 이미지 덤프 `depth/ normal/ color_segmented/ depth_vis/`,
LoFTR 매칭 덤프 `*_uvs.txt`(53 GB, 1,390만 파일), 사이클별 SDF 메쉬·octree PLY. 어떤 스크립트도 읽지 않는다.

## `outputs/` 실행 패밀리

ADD는 cm, 마스크는 별도 표기 없으면 SAM2(`masks_sam2` / `masks_SAM2`). "문서"는 결론이 적힌 곳, "JSON"은 채점 파일.

### SDF 백엔드 (원본 BundleSDF 피드백 공정 평가, 컨테이너 `bundlesdf`)

| 패밀리 | 개수 | 조건 | 문서 | JSON |
|---|---|---|---|---|
| `ablation_fb_{on,noop,off}_r{1,2}_20260902b` | 6 | mustard0, 조건별 2회 반복 (on 1.380/1.379, noop 0.760/0.754, off 0.743/0.752) | MILESTONES ⑤ "공정 평가", `ONLINE_GS_BACKEND_RESULTS.md` | `logs/add_eval_sdf_feedback_ablation_20260902.json` |
| `ablation_fb_{noop,on}_r1_20260902` | 2 → 0 | **미완주**(90프레임, gsplat 컨테이너에서 SDF hang) — 2026-09-11 폴더째 삭제 | 메모리(컨테이너 hang) | 없음 |
| `diag_T1_sdf_on_bundlesdf_container_20260902` | 1 → 0 | **미완주** 108프레임 컨테이너 스모크 — 2026-09-11 폴더째 삭제 | – | 없음 |
| `diag_mustard0_{on,noop,off}_datasetmasks_20260903` | 3 | 논문 마스크(`masks/`) 프로토콜 (0.742 / 1.212 / 1.053) | MILESTONES ⑤ "검증 2" 표 | `logs/add_eval_diag_mustard0_masks_pristine_20260903.json` (2026-09-11 채점) |
| `diag_mustard0_{on,off}_tracker_pristine_20260903` | 2 | 트래커 함수를 원본으로 교체(`experiments/pristine_tracker_patch.py`), on 1.347 | MILESTONES ⑤ "검증 1" | 같은 JSON |
| `diag_ho3d_smoke_SM1_20260903` | 1 | **미완주** 66프레임 스모크 (덤프만 정리, 폴더 잔존) | – | 없음 |
| `fbabl_ho3d_<seq>_{noop,off}_20260903` (13 seq) + `fbabl_ho3d_SM1_on_20260903` | 27 | HO3D 13개 noop/off (SAM2 on은 원본 baseline 폴더 결과 사용); SM1 on 0.553 = 원본 0.551 재현 | MILESTONES ⑤ HO3D 표 (on 0.728 / noop 1.757 / off 1.674) | `logs/add_eval_ours_ho3d_<seq>_{noop,off}_20260903.json`, `logs/add_eval_ours_ho3d_SM1_on_20260903.json` |
| `fbabl_ycb_<seq>_{on,noop,off}_2026090{2,3}` | 31 → 24 | YCB 9개 × 3조건. 9/2의 bleach0×3, cracker_box_yalehand0×3, tomato on은 빈 마스크 크래시로 **미완주** → 9/3 재실행이 완주본 (미완주 7개는 2026-09-11 폴더째 삭제) | MILESTONES ⑤ YCB 표 (on 1.489 / noop 1.309 / off 1.275) | `logs/add_eval_ours_ycb_<seq>_2026090{2,3}.json`, `logs/add_eval_ours_ycb_tomato_{on,noop_off}_…json` |

원본 BundleSDF 자체 결과(논문 마스크 on/off, SAM2 on)는 `~/Desktop/BundleSDF_baseline_outputs/full_eval*`(읽기 전용)이며
채점은 `logs/add_eval_baseline_*_20260903.json`.

### GS 백엔드 (④ 온라인 백엔드 + ⑤ 포즈 피드백, 컨테이너 `BundleSAM3DGS-gsplat-smoke`)

공통: 드라이버 `experiments/run_sdf_feedback_ablation.py --backend gaussian`, 러너 설정 `config_gs_2dgs_1mm_lifecycle.yml`(prior 0.9) 또는
`logs/gsfb_batch_20260908/config_gs_prior_opacity01.yml`(prior 0.1), 초기 4000 + 사이클 500스텝, SAM3D prior `~/Desktop/sam-3d-objects/output_*_sam2mask_mesh/`.
실행 스크립트·체인·진행 로그·표 스크립트는 `logs/gsfb_batch_2026090{7,8,9}/`.

| 패밀리 | 개수 | 조건 | 문서 | JSON |
|---|---|---|---|---|
| `mustard0_gs_online_20260901` | 1 | ④ 첫 온라인 완주, no-op 피드백 (ADD 0.758) | `ONLINE_GS_BACKEND_RESULTS.md` | `logs/add_eval_all_mustard0_runs_20260902.json` |
| `gsfb_noop_ho3d_SM1_20260907` | 1 | GS 백엔드 noop (4.363 = SDF noop 4.449; 백엔드 자체는 정상) | RESULTS §2 | `logs/gsfb_batch_20260907/add_eval_gsfb_noop_*.json` |
| `gsfb_v1_{ycb,ho3d}_<seq>_20260907` | 22 | ⑤ 2차 v1 = 원본 순서 PoseArray 이식 (YCB 1.214 / HO3D 2.407). **기울기 프로브의 재생 기준 실행** (`color/ depth_filtered/ mask/` + 사이클 포즈 필요) | RESULTS §1–§4 | `logs/gsfb_batch_20260907/add_eval_gsfb_v1_*.json` |
| `gsfb_v2_…_20260907` | 23 → 22 | v2 = append 전 pose-only 150 + 조인트 350 (YCB 1.259 / HO3D 3.191). `v2_ho3d_AP10`은 OOM **미완주**(933/1607)로 2026-09-11 삭제, `_retry`가 완주본 | RESULTS §2 | 같은 폴더 `add_eval_gsfb_v2_*.json` |
| `gsfb_v3_…_20260907` | 5 → 4 | v3 = v2 + 사전정제 lr 0.05 (전부 v2보다 나쁨). `v3_ho3d_SM1`은 OOM **미완주**(833/895)로 2026-09-11 삭제, `_retry`가 완주본 | RESULTS §2 | `add_eval_gsfb_v3_*.json` |
| `gsfb_po01_…_20260908` | 6 | C = prior 불투명도 0.1 (MPM10 4.99→1.27/1.69) | RESULTS §6.1–6.2 | `logs/gsfb_batch_20260908/add_eval_gsfb_po01_*.json` |
| `gsfb_quota5_…_20260908` | 6 | D = 새 키프레임 quota (AP10 5.69→3.30/4.40) | RESULTS §6.1–6.2 | `add_eval_gsfb_quota5_*.json` |
| `gsfb_gtmesh_…_20260908`, `gsfb_sam3dgt_…_20260908` | 5+4 | B = 오라클 prior (GT 메쉬@GT 포즈 / SAM3D 메쉬 ICP→GT) | RESULTS §6 B, §6.1 | `add_eval_gsfb_{gtmesh,sam3dgt}_*.json` |
| `gsfb_po01quota5_…_2026090{8,9}` | 20 | **C+D = 현 최선 조건** (YCB 8개 1.106, HO3D 9개 1.89; MPM12·13·14·SB11·tomato 미실행) | RESULTS §6.2, **§7.5** | `logs/gsfb_batch_2026090{8,9}/add_eval_gsfb_po01quota5_*.json` |
| `gsfb_{depth,weighted,normsum,gate}_…_20260909` | 6+6+6+8 | 1단계 포즈 기울기 정책 E1/E2/E3/F (HO3D-4 평균 6.70 / 4.26 / 4.87 / 3.14, 전부 기각·F는 참조) | RESULTS §7–7.3 | `logs/gsfb_batch_20260909/add_eval_gsfb_{depth,weighted,normsum,gate}_*.json` |
| `gsfb_{depthnoinit,weightednoinit,gatenoinit,v1noinit,cdnoinit}_…_20260909` | 1+1+2+6+9 | `in_initial=false` (초기 사이클 피드백 제외) 변형 | RESULTS §7.4 | `add_eval_gsfb_*noinit_*.json` |
| `gsfb_gatecd_…_20260909` | 6 | F + C + D 조합 (F 단독과 같음, 기각) | RESULTS §7.4 | `add_eval_gsfb_gatecd_*.json` |
| `_failed_launch/` | 3 | YAML 중복 키·prior 경로 오류로 시작 실패한 잔재 (10 MB, 증거로 보존) | RESULTS §1 | – |

## `logs/` 주요 디렉터리

| 디렉터리 | 내용 |
|---|---|
| `logs/mustard0_bundlesam3dgs_exact_baseline_rematch_backprojection_20260811` | 오프라인 GS 실험의 고정 트래킹 로그 (`PROJECT_HANDOFF.md` §9·§13, SHA256 기록) — 보존 |
| `logs/mustard0_bundlesam3dgs_{,repro_,legacy_backprojection_,…gpu0_repro_}20260811` | 8/27 노이즈 바닥 감사에 쓴 트래킹 로그(원본 코드·exact baseline·수정 변형·재현; ADD 1.369~1.395 = 실행 간 잡음 범위). 채점 `logs/add_eval_all_mustard0_runs_20260902.json` |
| `logs/mustard0_gaussian_*_2026081{1,3,4}` | ① 이전 토폴로지 실험 (`PROJECT_HANDOFF.md` §10, 보존 체크포인트 §13.2) |
| `logs/*_sam3d_align*_20260831`, `*_exp_colortransfer_*` | ② 정합·색 이식 ablation (`SAM3D_ALIGNMENT_RESULTS.md`) |
| `logs/mustard0_{priornolc,lifecycle*}_full_20260901`, `ycb_crackeryale_lifecycle_full_20260901`, `ho3d_MPM10_*` | ③ lifecycle GT 검증·showcase (`PRIOR_LIFECYCLE_RESULTS.md`) |
| `logs/mustard0_lifecycle_mainpath_20260901` | ④ 포팅 동치 검증 |
| `logs/mustard0_{poseonly_probe,probe_gtcheck,c_sim_recovery*}_20260902` | ⑤ 재설계 프로브 (MILESTONES ⑤) |
| `logs/add_eval_*.json` | SDF 공정 평가·baseline 채점 JSON (2026-09-02~03, 일부 2026-09-11 추가 채점) |
| `logs/gsfb_batch_20260907` | v1/v2/v3/noop 배치: `run_one*.sh`, `chain_gpu*.sh/.progress`, 채점 JSON, 진단 스크립트 `diag_*.py`, `kbins.csv`, 코드 스냅샷 `attempt2_worktree.patch` |
| `logs/gsfb_batch_20260908` | B/C/D 배치, `config_gs_prior_opacity01.yml`, `summary.py` |
| `logs/gsfb_batch_20260909` | 1단계 정책·no-init·C+D 전체 패스 배치, `summary.py`(전 조건 표), `config_gs_*noinit*.yml` |
| `logs/gradprobe_20260908*`, `gradprobe_20260909_all`, `gradprobe_20260911_allkf` | 오프라인 기울기 프로브 (RESULTS §6 A, §6.3, §7.1–7.2; 9/11은 키프레임별 전수, `make_report.py`) |
| `logs/oracle_priors_20260908` | 오라클 prior 생성물 (`experiments/make_oracle_prior.py`) |
| `logs/gtmap_probe_20260911` | ⑤-4 GT-map vs tracker-map 프로브 (`experiments/exp_feedback_gradient_probe_gtmap.py`, 다른 세션이 2026-09-11 실행) |
| `logs/disk_cleanup_20260911` | 아래 정리 작업의 스크립트·manifest·로그 |

## 정리 기록 (2026-09-11)

사용자 승인(덤프 정리 / 미완주 13개 폴더째 삭제 / 매칭 덤프는 개수 CSV로 압축 후 삭제) 아래 실행. 도구는 `logs/disk_cleanup_20260911/`의
`prune_outputs.py`(기본 dry-run, `--apply`로 삭제, 그룹 A~D 정책은 파일 머리에 기술)와 `condense_matches.py`, 실행 전후 manifest
`prune_manifest_{dryrun,apply}.json`, 로그 `prune_apply.log`·`condense.log`, 정리 전 전수 스캔 `outputs_scan.json`.

- **삭제 551.5 GB** (`outputs/` 643 GB → 67 GB, 디스크 86% → 54%): 사이클별 `gs_state.ply` 284 GB, 트래커 이미지 덤프 122 GB
  (v1·C+D·noop·④ 실행은 재생 입력 `color/ depth_filtered/ mask/` 유지), 기각 조건의 최종 지도 체크포인트·PLY 75 GB, LoFTR 매칭 txt 52 GB
  (1,390만 파일 → 실행당 `match_counts.csv`: cycle·frame·ref·kind(BA/before_ransac/after_ransac)·n_lines), 사이클별 SDF 메쉬·octree PLY 3.3 GB
  (마지막 메쉬 1개 유지). 오류 0건.
- **폴더째 삭제 13개** (모두 재실행 완주본 존재): `gsfb_v2_ho3d_AP10_20260907`, `gsfb_v3_ho3d_SM1_20260907`,
  `fbabl_ycb_{bleach0,cracker_box_yalehand0}_{on,noop,off}_20260902`, `fbabl_ycb_tomato_soup_can_yalehand0_on_20260902`,
  `ablation_fb_{noop,on}_r1_20260902`, `diag_T1_sdf_on_bundlesdf_container_20260902`.
- **남긴 것 54 GB**: v1·C+D·noop·④의 최종 지도 21 GB, 재생 이미지 20 GB, 사이클별 포즈 txt/yml 10 GB, `ob_in_cam/`, 피드백·lifecycle 로그,
  manifest·설정, `match_counts.csv`. 채점 JSON과 `summary.py` 표는 정리 전과 동일하게 재현됨.
- **미처리 2개**: `gsfb_v1_ho3d_{AP12,SB13}_20260907`(합 10 GB)는 동시에 돌던 ⑤-4 프로브가 읽고 있어 제외 → 프로브 종료 후
  `prune_outputs.py --apply --only gsfb_v1_ho3d_AP12_20260907 gsfb_v1_ho3d_SB13_20260907`로 같은 정책 적용.
- **재발 방지 미적용**: 289 GB의 원인은 `bundlesdf.py` `run_gaussian`이 SPDLOG≥2에서 매 사이클 `gs_state.ply`를 쓰는 것. 이를 opt-in으로 바꾸는
  것은 코드 수정이라 별도 승인 후 진행 (포즈 덤프 `poses_before/after_gs.txt`는 진단에 필요하므로 유지).
