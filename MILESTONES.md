# BundleSAM3DGS 마일스톤 기록

최종 갱신: 2026-09-02.
마일스톤 ①~⑤의 목적·구현·검증 결과를 한눈에 잇는 요약 문서다. 상세 근거는 각 결과 문서(문서 끝 표)와 `PROJECT_HANDOFF.md`를 따르고, 이 문서가 코드/실험 JSON 산출물과 충돌하면 코드·산출물이 우선한다.

## 전체 그림

BundleSDF(미지 물체 6-DoF 트래킹 + 신경 재구성, CVPR 2023)의 SDF 백엔드를 Gaussian Splatting으로 교체하되, 단순 교체가 아니라 두 가지를 더한다:

1. **SAM3D 첫 프레임 전체 형상 prior**로 시작해, 손/그리퍼에 가려 끝까지 관측되지 않는 부분까지 완전한 메쉬를 만든다. BundleSDF는 그 부분에 구멍이 남는다 — 이것이 헤드라인 차별점.
2. GS 지도가 트래커 포즈를 되고쳐주는 **양방향 루프**("Bundle")를 완성한다.

전략: 전체 파이프라인을 먼저 연결하고, 성능/기여 다듬기는 그 다음.

```
① 표현(2DGS)  →  ② 시작점(SAM3D prior 정합)  →  ③ 충돌 관리(lifecycle)  →  ④ 온라인 결합  →  ⑤ 피드백 루프
    완료              완료                           완료                       완료            1차 시도 롤백, 재설계 논의 중
```

## ① 2DGS 렌더러 + 기하 감독 — 완료

- **왜**: 최종 산출물이 메쉬(기하)다. 3DGS는 사진 재현은 좋지만 표면이 정의되지 않고, 2DGS(납작한 surfel형 가우시안)는 표면·법선이 잘 정의된다(6DoPE-GS와 같은 선택).
- **구현**: `gaussian_runner.py`에 렌더러 스위치(gsplat 1.5.3 `rasterization_2dgs`), 손실 확장 — 마스크드 RGB(L1+DSSIM)에 depth Huber(δ=3cm, 정규화 공간 변환) + 2DGS 법선 일관성(w0.05@7k) 추가. distortion 손실은 직접 depth 감독과 겹치면 유해하여 w=0.
- **검증**: 동일 조건 비교에서 2DGS+기하 감독이 3DGS 기준선보다 기하 우수 → 기본 채택 (`config_gs_2dgs_1mm.yml`).
- 커밋: `16ad988`. 주의: 비현재 CUDA 디바이스에서 2DGS backward가 illegal memory access — `GaussianRunner.__init__`의 `torch.cuda.set_device` 가드로 해결(재발 주의).

## ② SAM3D prior + 첫 프레임 Sim(3) 정합 — 완료

- **왜**: 관측만으로는 가려진 부분이 영원히 구멍. SAM3D가 첫 프레임 한 장으로 전체 형상 추측을 주지만 R,T,s가 부정확해 정밀 정합이 필요.
- **구현**: SAM3D는 오프라인 전처리(SAM2 마스크 + GT cam_K를 양쪽 주입 지점에 사용, 실제 depth pointmap). prior 표현은 gaussian이 아닌 **mesh** 채택(raw `output["mesh"][0]`, canonical ±0.5; GLB의 y-up 회전 함정 주의) → surfel 샘플링(radius ×0.75) → 첫 프레임 RGB-D에 render-and-compare로 R,T,s 정제(`run_sam3d_alignment.py`).
- **색 이식(C안, 기본값)**: 메쉬 정점색이 조악 → SAM3D gaussian PLY의 색을 k-NN 역거리가중으로 surfel에 이식 + SSIM photo 손실 w1.0. YCB 9 + HO3D 13 시퀀스 A/C 비교에서 C 승 → 채택 (`4c30380`).
- **검증**: 초기 포즈 오차 4.4~46mm → 정제 후 **2.7~8mm**. depth 대상은 raw가 기본(시퀀스별 최적 depth는 결과 문서에 기록).
- 커밋: `37e3922`, `8874fe2`, `4c30380`. 이 2.7~8mm가 우리 render-and-compare 기계의 검증된 정밀도 바닥이며 ⑤에서 다시 중요해진다.

## ③ Prior lifecycle (상태기계) — 완료

- **왜**: prior는 추측. 보이는 곳은 관측이 이기고, 안 보이는 곳은 prior를 지켜야 한다. 이 충돌을 가우시안 단위로 자동 관리.
- **구현**: `prior_lifecycle.py` — 4-state(UNSEEN → VERIFIED / SUSPECT → CONTRADICTED). 뷰별 depth 증거 분류(support / free-space 위반 / behind-occluded / behind-miss), 비스듬한 시선 관대화(÷|cos|, cap 3×), 독립 뷰(10°) 2/3표 전이. VERIFIED만 학습, CONTRADICTED는 opacity 감쇠 후 제거, **UNSEEN 동결**. v1은 densify/reset off.
- **검증(GT 메쉬 대조)**: 뒷면 처리로 UNSEEN 동결이 최선 — 동결 4.88mm < 자유 학습 5.56mm < 바이어스 외삽(③b) 5.80mm. ③b는 홀드아웃 CV에선 이기는 듯했으나 GT에서 최악 → 기각(뷰 지표는 오염을 이득으로 오채점할 수 있음 — 교훈). 손 가림 보존 showcase: cracker_box_yalehand0 (MPM10은 재파지+GT 등록 오차로 부적합 판정).
- 커밋: `2acfc23`, `40ef639`.

## ④ 온라인 GS 백엔드 — 완료

- **왜**: ①~③은 저장된 트래킹 로그의 오프라인 리플레이 검증. "온라인 방법" 주장에는 트래킹과 동시 실행이 필요.
- **구현**: `bundlesdf.py`에 `run_nerf`와 동일 계약(공유 풀, 키프레임 5개부터, 초기 4000스텝 후 사이클당 500스텝)의 `run_gaussian` 프로세스. 첫 배치에서 prior 로드 + C안 색 이식 + 온라인 Sim(3) 정합 + `initialize_from_prior`; 이후 배치마다 트래커 갱신 포즈로 `refresh_view_poses` + 신규점 append + 학습. 실행: `run_custom.py --backend gaussian`. 트래커(C++)는 무수정.
- **검증(mustard0 온라인 완주)**: 피드백 없이(no-op) **ADD 0.758cm**, SDF 백엔드(자체 포즈 피드백 포함) **1.369cm**. 즉 SDF의 포즈 피드백은 트래커를 해치고 있었고, ⑤의 성공 기준은 "no-op 0.758cm를 이겨라"로 설정됨.
- 커밋: `38a3473`, `cb8f634`. 이월 항목(백로그): 다중 시퀀스 온라인 평가, `run_ho3d` 연결, occ_mask 온라인 적용, 글로벌 GS 정제.

## ⑤ GS→트래커 포즈 피드백 — 1차 시도 롤백, 재설계 논의 중

- **목표**: 역방향 화살표 — GS 지도가 render-and-compare로 트래커 포즈 보정량을 계산해 돌려준다. 정량 목표는 온라인 ADD < no-op 0.758cm.
- **설계 합의**: tracker-authority — 트래커가 포즈의 주인, GS는 매 사이클 0에서 시작하는 델타만 계산. 보정 클램프 3mm/3°(트래커 rematch 발동선 5mm/5° 미만 → 트래킹을 무너뜨릴 수 없음). 오프라인 교란-복구 실험으로 게이트 후 온라인 A/B. (대안 b: SDF식 pose-array 소유는 fallback.)

### 1차 시도 (v1~v3, 2026-09-01 롤백)

교란-복구 실험: 키프레임 포즈에 2.5mm/1.5° 오차 주입, 1사이클(500스텝) 후 복구율과 무교란 뷰 드리프트 측정. 통과 기준: 이동·회전 모두 양의 복구 + 드리프트 <1mm.

| 버전 | 설정 | 이동 복구 | 회전 복구 | 깨끗한 뷰 드리프트 |
|---|---|---|---|---|
| v1 | 단일 lr, 무정규화 | −32% | +27% | 진성 드리프트 ~4.3mm |
| v2 | lr분리(1e-4/1e-5)+정규화+워밍업100 | −1.2% | +23% | ~0.06mm |
| v3 | v2에서 이동만 완화(lr 5e-5, reg 10) | −11.8% | +24% | 0.37mm |

**교훈**:
1. 드리프트 방어(델타 정규화 + rot/trans lr 분리 + 워밍업)는 검증 완료 — lr 5배에도 0.37mm.
2. 회전 피드백은 사이클당 ~24% 복구로 세 설정 모두 일관.
3. **이동 실패의 원인은 회전-이동 커플링 + 맵 흡수**: 물체 거리 0.75m에서 회전 1.5°는 화면상 ~19.5mm(이동 2.5mm의 8배 지렛대) → 이동이 회전 잔차를 대신 메꾸는 방향으로 끌려감(프레임별 역상관 + lr 자유도 비례 오버슈트로 확인). 또한 조인트 학습 중 맵이 포즈 오차를 흡수 경쟁(새 프레임이 틀린 포즈로 append·학습 참여).
4. 미해결 의문: 2.5mm 정밀 복구는 ②의 검증 바닥 2.7~8mm와 같은 급 — 목표 자체가 무리였을 가능성. outlier(5mm+) 교정으로 재정의할지, 회전 전용으로 갈지, 맵 동결 pose-only 단계를 둘지는 미결.

**보존 위치**: 코드는 `milestone5-feedback-attempt1` 브랜치(`04c785c` 구현 → `f2aa94c` 보완 → `2d28635` v3 스크립트), 실험 로그는 `logs/mustard0_pose_recovery{,_v2,_v3}_20260901/`. main은 `aa7a673`(④ 완료 상태)로 리셋.

## 산출물 위치

| 항목 | 위치 |
|---|---|
| ② 정합 결과 상세 | `SAM3D_ALIGNMENT_RESULTS.md` |
| ③ lifecycle 결과 상세 | `PRIOR_LIFECYCLE_RESULTS.md` |
| ④ 온라인 백엔드 결과 상세 | `ONLINE_GS_BACKEND_RESULTS.md` |
| ⑤ 1차 시도 코드 | git 브랜치 `milestone5-feedback-attempt1` |
| ⑤ 1차 시도 로그 | `logs/mustard0_pose_recovery{,_v2,_v3}_20260901/` |
| 기본 설정 | `config_gs_2dgs_1mm.yml`, `config_gs_2dgs_1mm_lifecycle.yml` (mustard 연구는 1mm 계열 필수) |
| 권위 문서 | `PROJECT_HANDOFF.md` (충돌 시 코드·실험 산출물 우선) |

## 주요 수치 한눈에

- ② 정합: 초기 4.4~46mm → 2.7~8mm (YCB 9 + HO3D 13)
- ③ 뒷면 GT: UNSEEN 동결 4.88mm < 자유 학습 5.56 < ③b 5.80
- ④ mustard0 온라인 ADD: GS no-op **0.758cm** < SDF 1.369cm ← ⑤의 기준선
- ⑤ 커플링 상수: 물체 거리 0.75m에서 1.5° ≈ 화면상 19.5mm
