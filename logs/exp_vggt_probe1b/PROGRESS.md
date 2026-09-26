# EXP_VGGT_PROBE1B 진행 기록 (2026-09-26 시작)

지시서 `EXP_VGGT_PROBE1B_BRIEF.md`. 기존 probe 파일 무수정(import만), Probe 1 산출물 읽기 전용. 새 코드는 `experiments/vggt_probe/p1b_*.py`.
사용자 지시(9/26): A0 재현 확인과 B3 roll 정규화 단위 확인을 통과하기 전에는 다음 단계로 넘어가지 않는다. 끝나면 결과 문서 작성 후 커밋(사용자 사전 승인).

## A0 재현 확인 — 통과 (9/26 09:05, `A0.txt`)
무작위 3셀(seed 0): M-V/I0/ref1-8 MPM13(새 KF 221), M-V/I0/ref5-8 AP10(177), M-O/I0/ref1-4 MPM12(235). `batches.jsonl`의 w2c_raw·R_virtual·s·T_online·T_oracle만으로 재구성한 새 KF c2w가 `kf.csv` 저장값과 online/oracle 모두 최대 |Δ| = 0 (기준 1e-6). A2(a)에서 T 자체도 기준 포즈로부터 재계산해 대조한다.
- 준비: 시퀀스별 포즈 캐시(C1/C2/GT/프레임별 스냅샷) `outputs/exp_vggt_probe1b/cache/` 생성 중(22 seq, ~20분).

## B3 roll 정규화 단위 확인 — 통과 (9/26 09:20, `B3_checks.txt`, `warp_roll_AP12.png`)
- (a) 합성 왕복(roll 포함 R_v = R_z(ψ)·R_c, 스케일 0.37, 임의 gauge, 8 포즈, 3장 정렬): 스케일 오차 0, 포즈 오차 5.6e-16 (기준 1e-6).
- (b) AP12 KF0–8, roll 포함 warp→역warp 마스크 IoU 최소 0.9997, 평균 0.9999 (기준 0.98), 마스크 범위 0.400·S. ψ = 0…14°, 투영 길이 0.91–0.99(플래그 0).
- (c) ψ 기하: 첫 자리 기준의 위 방향이 가상 이미지의 −y로 정확히 정렬(잔차 2e-16).

## 실행 (9/26 09:40 시작)
- A단계(CPU): 캐시 완료 후 A1 → A2 → A3 → A4 자동 연쇄(`A1.log`…`A4.log`).
- B단계(GPU, M-V, 하이퍼파라미터 고정: k=8/4, B2 대역 [15,30)/[30,60)/[60,90) 2장씩, ψ 투영 임계 0.2, 배경 1.0/0.0/0.5, 정렬 (a)/(d), (d) 잔차 임계 10°): GPU0 = AP12, MPM10 × 7조건 → B6; GPU1 = MPM12 × 6조건 → (캐시 완료 후) SB11, SM1 × 7조건. B5(무마스크 체인)는 AP12, SM1만. 진행: `progress_B_gpu0.txt`, `progress_B_gpu1.txt`.
- B6: LoFTR 전처리는 `FeatureManager.cpp processImagePair` 복제(뒤 프레임 A, 앞 프레임 B; B를 A로 카메라 z축 성분만큼 회전 — 포즈는 online: A = C1, B = A 시점 스냅샷; ROI = 마스크 bbox + 10 px; 공통 max_dim 스케일; 400×400 gray warpPerspective; LoFTR match_coarse.thr 0.2 outdoor_ds). 대응점 필터: 두 마스크 안 + depth > 0.1 m(법선 검사는 생략, 이탈로 기록). RANSAC: 3점 Procrustes, 2000회, inlier 5 mm (`config_ho3d.yml ransac`), 성공 = 인라이어 ≥ min_match_with_ref 5 (≥ 30도 병기).

## A단계 완료 (9/26 10:20; A1 `table_A1.md`, A2 `table_A2_R.md`, A3 `table_A3_FL.md`, A4 `table_A4.md`; 쌍 836,848행 `outputs/exp_vggt_probe1b/A/all_pairs.csv`, 재정렬 188,640행 `all_realign.csv`)
- A2 (a)-online은 저장된 T_online과 전 배치에서 1e-6 안에서 일치(중단 조건 없음).
- A1 (M-V/I1 B-ref, 전체 쌍): swing [0,15) 중앙값 3.6°, [15,30) 7.2°, [30,60) 24.1°, [60,90) 64.5°, [90,135) 100°, [135,180] 133° (C2 쌍 1.8–6.4°). 체인 셀은 [0,15) 2.4°, [15,30) 3.8°, [30,60) 6.3°. 먼 기준이 낀 새 KF 쌍 중앙값 59.9°(>90° 41 %) vs 낀 것 없는 쌍 4.4°(5 %).
- A2 (HO3D 13 평균, 새 KF 회전 중앙값; C1 4.02, C2 4.42): M-V/I1/ref1-8 online (a) 14.61 → (b) 첫 자리만 5.53 / (c) 5.99→6.99 / (d) 강건 6.79; oracle (a) 13.76 → (b) 2.76 (C1보다 나은 seq 9/13) / (c) 4.39 / (d) 4.26. ref1-4: (b) online 5.48, oracle 2.66 (10/13). M-Ω ref1-8: (b) online 5.29, oracle 2.60 (10/13). → 판독 기준 "(나) 정렬 오염 주원인"(30 % 이상 감소) 충족: (b) −62 %, (d) −54 %.
- A3: 새 KF E1 >90°는 MPM12 11.4 %, MPM14 10.7 %, MPM13 8.0 %, AP10 5.1 %; 쌍 오차 >90°는 MPM10–14 44–55 %; 뒤집힌 쌍의 91–99 %가 far 기준을 포함. 뒤집힌 새 KF의 오차 회전축 vs 메시 PCA 주축 최소각 중앙값 7–32°.
- A4: 쌍 오차 vs swing ρ = (표 참조), roll·마스크·conf·T3는 swing 통제 후 |ρ| ≤ 0.3 수준.
- 버그 수정 기록: A2 표 생성에서 pandas `g.mode`(메서드) 오용 → `g["mode"]`; 재정렬 데이터 자체는 영향 없음(표만 재생성).

## B단계 완료 (9/26 10:44; 34 셀, 실패 0; `table_B.md`, `table_B_roll.md`)
- 표 B 5 시퀀스 평균(새 KF 회전 중앙값; C1 4.09, C2 4.27): B1-k8 online (a) 10.35 / (d) 8.09, oracle 8.70 / 5.23; B1-k4 7.44 / 6.57, oracle 4.77 / 3.64; B2 20.59 / 13.76; B1-roll 7.97 / 6.67, oracle 5.49 / 3.85; B4-black 11.91 / 9.31; B4-gray 10.16 / 7.88; Probe 1 ref1-8 (a) 15.68.
- (가) 판독: B1-k8 oracle (a) 8.70 / (d) 5.23 vs Probe 1 체인 (2) R-oracle 같은 5 시퀀스 평균 3.90 → +1° 안에 못 듦. 가까운 8장만 써도 체인 수준이 안 됨(가까운 8장의 new–ref swing p90이 21–42°, roll > 45° 쌍이 17 %). B1-k4 oracle (d) 3.64는 체인 수준.
- (라) 판독: roll > 45° 쌍 오차 중앙값 B1-k8 85.2° → B1-roll 23.3° (−73 %), roll ≤ 45° 쌍도 4.5 → 3.5° (−23 %) → 기준(20 %) 충족.
- 배경색: 회색 −0.19°(영향 없음), 검정 +1.56° / oracle +1.9°(나쁨).
- NC(B5): 마스크 없는 원본 영상 체인에서 VGGT 연속 프레임 상대 회전 중앙값 0.07° vs GT 6–7° (마스크 I1 체인 5.7–6.8°) → 배경 고정 시 VGGT는 카메라가 정지했다고 봄. 마스킹 필수 확인.
- B6 1차 시도 실패(LoFTR 래퍼가 1채널 입력 요구; 3채널로 넣어 conv 오류) → 입력을 gray 1채널로 수정해 재실행(1회 재시도 규칙). 로그 `B6_attempt0_failed.log`, `B6.log`.

## B6 완료·종료 (9/26 11:12)
- B6: 5 시퀀스 1,740쌍(bin당 ≤ 60, seed 0), LoFTR 성공률(≥5) [0,15) 99 % → [90,135) 41 %; VGGT 2장 중앙값 4.7 / 9.2 / 36.4 / 67.8 / 81.9 / 100.4°, +roll 3.6 / 7.4 / 17.0 / 35.8 / 40.8 / 29.9°; LoFTR 성공 쌍 2.3 / 3.4 / 6.7 / 32.9 / 94 / 161°. 핵심 질문 답: LoFTR가 끊기는 bin([90,135))에서 VGGT도 불가; VGGT 사용 가능 swing 상한 = [0,15).
- 결과 문서 `EXP_VGGT_PROBE1B_RESULTS.md` 작성. 매니페스트 색인 `manifests/index.json`(B 34 + B6 1). 커밋(사용자 사전 지시).
