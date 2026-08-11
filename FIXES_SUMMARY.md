# 수정 사항 요약

## 매칭 실패 및 reference fallback
- 이유: 매칭 부족 시 없는 `_matches`를 읽어 `KeyError`가 발생하고 디버거에서 멈췄습니다.
- 변경: 후보별 실패와 프레임 최종 실패를 분리하고, 모든 reference 실패 후에만 `FAIL` 처리합니다.

## NeRF 이후 correspondence 재생성
- 이유: `_raw_matches`만 남은 pair가 LoFTR도 재처리도 하지 않아 `_matches`가 복구되지 않았습니다.
- 변경: raw cache는 재사용하고 3D correspondence와 RANSAC만 다시 수행합니다.

## GPU pose 복사 버퍼
- 이유: `vector.clear()` 후 CUDA 결과를 복사하고 인덱싱해 undefined behavior가 발생했습니다.
- 변경: `clear()`를 제거해 `n_frames` 크기의 host buffer를 계속 유지합니다.

## 카메라 intrinsics 변환
- 이유: Eigen column-major 행렬을 CUDA row-major 행렬로 강제 변환해 역투영 좌표가 틀렸습니다.
- 변경: 16개 원소를 행·열 기준으로 명시적으로 복사하도록 변경했습니다.

## HO3D benchmark CLI
- 이유: 파싱한 `args`를 빈 list로 덮어써 `args.out_dir` 등의 접근이 실패했습니다.
- 변경: 불필요한 `args = []` 한 줄을 제거했습니다.

## SAM2 mask 연결
- 이유: 기존 reader가 `masks` 또는 XMem 절대경로만 읽어 추가된 SAM2 mask를 사용하지 못했습니다.
- 변경: YCB `masks_sam2`, HO3D `masks_SAM2` 경로를 CLI와 reader에서 선택하도록 연결했습니다.

## 실행 문서
- 이유: README는 `global_refine`을 별도 필수 단계로 안내했지만 `run_video`가 이미 자동 실행했습니다.
- 변경: 자동 global refinement, SAM2 폴더 구조, 선택적 재실행 명령을 실제 동작에 맞게 정리했습니다.

## 독립 실행 환경
- 이유: 기존 BundleSDF 작업 환경과 현재 변경 사항을 분리할 필요가 있었습니다.
- 변경: 현재 저장소 전용 `BundleSAM3DGS` 컨테이너에서 CUDA 모듈과 LoFTR를 빌드했습니다.

## 검증
- 이유: matching, native CUDA, SAM2 경로의 회귀 여부를 최소 범위에서 확인해야 했습니다.
- 변경: focused 테스트와 YCB 2-frame smoke로 raw 재생성, optimizer, 역투영 수치를 확인했습니다.
