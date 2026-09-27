
## 4. 계획 이탈·실패·미완 항목

1. **포즈 소스 새로 만듦**: 1차 배치 noop 산출물에 `color/ depth_filtered/ mask/`와 프레임별 `poses_before_gs.txt`가 없어(승인된 덤프 정리) 지시서대로 noop 온라인 실행 4개를 새로 만들었다. 1차는 online_variants 복사본(스위치 없음)을 거쳤고 이번은 본 코드 진입점을 직접 호출했다. ADD는 1차와 잡음 범위에서 같다.
2. **GN 단위 확인**: 지시서 반복 상한 8에서는 18개 중 14개만 허용오차 안(구현 확인은 상한 30으로 통과). 상한은 바꾸지 않았다. 전제 확인 실패의 한 원인일 수 있다(RG + O3 GT+3°의 반복 수 중앙값 8 = 상한).
3. **위생 단위 확인 (2) 실패**: 표면 위 위치 오차를 거르지 못하는 판정 규칙의 성질(§3.2). A 계열은 단계 0 결정대로 실행했다.
4. **위생 단위 확인 (3) 방식 변경**: 별도 재생 두 번 비교 대신 같은 초기 체크포인트에서 출발(prior 정합의 비재현성 때문).
5. **GPU 배치 변경**: A3를 GPU0으로 옮겨 두 GPU 시간을 맞췄다(실험 내용 동일).
6. **단계 2 미실행**: 전제 확인 실패(지시서 3.6).
7. 실패·재시도: 없음(noop 4, probe 24 전부 1회에 완료). 스모크용 짧은 실행 3개는 `/tmp`에만 썼다.
8. 컨테이너 시계가 호스트와 다른 시간대로 찍혀 `progress` 파일의 시각은 컨테이너 기준이다(시작 = 호스트 9/27 17:45 KST).

재개 방법: 모든 실행이 끝났다. 표만 다시 만들려면 컨테이너에서 `/usr/bin/python3 experiments/analyze_reference_optimizer_probe.py`. 개별 probe 재실행은 `GPU=<n> bash logs/exp_batch_20260927/run_probe.sh <REF> <ds> <seq>`(완료된 것은 건너뜀).

## 5. 승인 필요

* **본 코드 반영 권고: 없음.** 통과 조합이 없고 단계 2가 금지됐다. 위생 규칙(A1–A3)은 반영을 권하지 않는다(표면 위 오차를 못 거르고 지도만 13–29 % 커짐).
* 선택 후속(새 실행 필요):
  1. GN 반복 상한 8 → 30으로 RG × 4만 다시 돌려 전제(GT+3° 개선 ≥ 80 %, 드리프트 < 0.3 mm)를 재확인. 약 3시간. 전제가 서면 GN + 채택 판정을 "해악 줄이기" 수단으로 단계 2에서 볼 수 있다.
  2. (a) 방향을 계속한다면, 트래커와 독립인 정확한 기준이 먼저다. 예: 스케일·형상이 맞는 prior(이번 RP는 MPM12에서 95 % 악화), 또는 원본처럼 전 키프레임 공동 정제.
* 정리: `outputs/exp_batch_20260927/noop` 12 GB(사이클별 `gs_state.ply` 포함). 삭제는 승인 후.
* 커밋: 지시서상 하지 않았다. 커밋할 때 대상은 결과 문서, `experiments/` 새 스크립트 5개, `logs/exp_batch_20260927/`(`-f`)이다.

## 6. 산출물 색인

* 실행: `outputs/exp_batch_20260927/noop/{ho3d/AP12, ho3d/SM1, ho3d/MPM12, ycb/mustard0}`; `outputs/exp_batch_20260927/probe_{R0,A1,A2,A3,RP,RG}/<ds>/<seq>/`(`rows.json` probe별 행, `cos_rows.json` R0 기울기, `hygiene_log.json` A 계열 사이클 통계, `lifecycle_tail.json`, `args.json`, `done.json`).
* 로그·JSON: `logs/exp_batch_20260927/`: `PROGRESS.md`, `progress`, `manifests/`(실행별 명령·환경·HEAD·스크립트 sha1), `stage0/contamination_split.{json,png}`, `stage1_tables.md`, `stage1_verdict.json`, `unit_tests/`(GN·위생 단위 확인과 진단), `run_noop_*.log`, `probe_*.log`, `jobs/`, 스크립트 `run_noop.sh`, `run_probe.sh`, `chain.sh`, `wait_chain.sh`.
* 스크립트: `experiments/exp_contamination_split.py`(단계 0), `experiments/exp_reference_optimizer_probe.py`(단계 1 probe, GN 단위 확인), `experiments/hygiene_runner.py`(A1–A3), `experiments/hygiene_unit_tests.py`, `experiments/analyze_reference_optimizer_probe.py`(표·판정).
