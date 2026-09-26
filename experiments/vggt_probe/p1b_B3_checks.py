"""B3 unit checks (mandatory before stage B): (a) synthetic round trip with roll-normalised virtual rotation R_v = Rz(psi) R_c,
scale 0.37, arbitrary gauge, alignment on 3 refs -> GT within 1e-6; (b) warp/unwarp mask IoU > 0.98 with R_v on AP12 batch 0;
(c) psi correctness: after rotation the first-reference up vector projects to image up (-y) with |x| < 1e-9."""
import sys
import numpy as np
from p1b_common import *
from p1b_B_lib import roll_psi, Rz, warp_I1_bg

lines = []
# (a) synthetic
rng = np.random.default_rng(1); s_true = 0.37
def rand_pose():
    R = L.proj_so3(rng.normal(size=(3, 3))); P = np.eye(4); P[:3, :3] = R; P[:3, 3] = rng.normal(size=3) * 0.3; return P
gt = [rand_pose() for _ in range(8)]; G = rand_pose(); K = np.array([[600, 0, 320], [0, 600, 240], [0, 0, 1.0]])
Rc = [L.centre_rotation(K, rng.uniform([100, 100], [540, 380])) for _ in gt]
Rv = []
for i, P in enumerate(gt):
    psi, proj, flag = roll_psi(gt[0][:3, :3], P[:3, :3], Rc[i]); Rv.append(Rz(psi) @ Rc[i])
est = []
for P, R in zip(gt, Rv):    # what VGGT would output: virtual cameras in an unknown gauge, translation scaled by 1/s
    Q = np.linalg.inv(G) @ P @ np.linalg.inv(blockdiag(R)); Q[:3, 3] /= s_true; est.append(Q)
rho_s = rng.uniform(0.3, 1.0, 5000); rho_v = rho_s / s_true; s = float(np.median(rho_s / rho_v))
rec = []
for Q, R in zip(est, Rv):
    P = Q.copy(); P[:3, 3] *= s; rec.append(P @ blockdiag(R))
T = L.align(gt[:3], rec[:3]); err = max(np.abs(T @ P - g).max() for P, g in zip(rec, gt))
ok_a = abs(s - s_true) < 1e-6 and err < 1e-6; lines.append(f"B3-a synthetic round trip with roll: scale err {abs(s-s_true):.2e}, pose err {err:.2e} -> {'PASS' if ok_a else 'FAIL'}")
# (c) psi geometry on the synthetic poses
worst = 0.0
for i, P in enumerate(gt):
    u = -gt[0][:3, 1]; u_v = Rv[i] @ (P[:3, :3].T @ u); proj = np.hypot(u_v[0], u_v[1])
    if proj >= 0.2: worst = max(worst, abs(u_v[0]) / proj, (u_v[1] / proj + 1))
ok_c = worst < 1e-9; lines.append(f"B3-c up-vector -> image up: worst |x|/proj, 1+y/proj = {worst:.2e} -> {'PASS' if ok_c else 'FAIL'}")
# (b) warp IoU on AP12 with roll-normalised rotation (first reference = KF0 R-online)
S_ = load_seq("ho3d", "AP12"); seq = L.Sequence("ho3d", "AP12"); ids = seq.kf_ids[:9]; Kc = seq.K; S = 518
frames = [seq.frame(f) for f in ids]; Rc = [L.centre_rotation(Kc, L.mask_centroid(m)) for _, _, m in frames]
R_first = S_.C1[ids[0]][:3, :3]; Rv = []; psis = []
for i, f in enumerate(ids):
    Ri = S_.C1[f][:3, :3]; psi, proj, flag = roll_psi(R_first, Ri, Rc[i]); Rv.append(Rz(psi) @ Rc[i]); psis.append((np.degrees(psi), proj, flag))
fv = L.virtual_focal(Kc, [m for _, _, m in frames], Rv, S); Kv = np.array([[fv, 0, S / 2], [0, fv, S / 2], [0, 0, 1.0]]); ious = []; ext = []
for (rgb, d, m), R in zip(frames, Rv):
    img, dv, mv, _ = warp_I1_bg(rgb, d, m, Kc, R, Kv, S); back = L.unwarp_mask(mv, Kc, R, Kv, S, *m.shape); ious.append((back & m).sum() / (back | m).sum())
    ys, xs = np.nonzero(mv); ext.append(max(np.abs(xs - S / 2).max(), np.abs(ys - S / 2).max()) / S)
ok_b = min(ious) > 0.98; lines.append(f"B3-b AP12 KF0-8 warp IoU with roll: min {min(ious):.4f} mean {np.mean(ious):.4f}, max extent {max(ext):.3f}S, f_v {fv:.1f}; psi(deg)/proj/flag = " + ", ".join(f"{p:.0f}/{q:.2f}/{int(fl)}" for p, q, fl in psis) + f" -> {'PASS' if ok_b else 'FAIL'}")
ok = ok_a and ok_b and ok_c; lines.append("B3 checks " + ("PASS" if ok else "FAIL")); print("\n".join(lines)); open(LOG / "B3_checks.txt", "w").write("\n".join(lines) + "\n")
import cv2; grid = np.concatenate([warp_I1_bg(rgb, d, m, Kc, R, Kv, S)[0] for (rgb, d, m), R in zip(frames[:5], Rv[:5])], 1)
cv2.imwrite(str(LOG / "warp_roll_AP12.png"), cv2.cvtColor((cv2.resize(grid, (0, 0), fx=0.4, fy=0.4) * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
sys.exit(0 if ok else 1)
