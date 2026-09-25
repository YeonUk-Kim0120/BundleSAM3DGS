"""Stage 0 unit checks (brief section 10): (a) synthetic scale/alignment round trip, (b) warp/unwarp mask IoU on AP12."""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np, cv2
sys.path.insert(0, str(Path(__file__).resolve().parent))
import probe_lib as L


def check_a():
    rng = np.random.default_rng(0); s_true = 0.37
    def rand_pose():
        R = L.proj_so3(rng.normal(size=(3, 3))); P = np.eye(4); P[:3, :3] = R; P[:3, 3] = rng.normal(size=3) * 0.3; return P
    gt = [rand_pose() for _ in range(8)]                     # object-frame GT c2w
    G = rand_pose()                                           # unknown gauge of the "vggt" batch frame
    est = []
    for P in gt:                                              # vggt frame = G^-1 * P, translation scaled by 1/s
        Q = np.linalg.inv(G) @ P; Q[:3, 3] /= s_true; est.append(Q)
    # scale from synthetic ray distances: rho_v = rho_s / s
    rho_s = rng.uniform(0.3, 1.0, 5000); rho_v = rho_s / s_true
    s = float(np.median(rho_s / rho_v)); assert abs(s - s_true) < 1e-6, s
    scaled = [Q.copy() for Q in est]
    for Q in scaled: Q[:3, 3] *= s
    T = L.align(gt[:3], scaled[:3])
    err = max(np.abs(T @ Q - P).max() for Q, P in zip(scaled, gt)); assert err < 1e-6, err
    # virtual-rotation recovery: c2w_v = c2w @ blockdiag(R,1)^-1, recover c2w = c2w_v @ blockdiag(R,1)
    R = L.rodrigues([0.3, -0.5, 0.1], 0.7); B = np.eye(4); B[:3, :3] = R
    P = gt[0]; Pv = P @ np.linalg.inv(B); assert np.abs(Pv @ B - P).max() < 1e-12
    # s_gt on the unscaled estimates must recover s_true
    sg = L.scale_gt([np.linalg.inv(G) @ P * 1 for P in gt], gt, T[:3, :3])  # translations unscaled: est_t = G^-1 P -> s_gt = 1 in this gauge
    est_un = [np.linalg.inv(G) @ P for P in gt]
    for Q in est_un: Q[:3, 3] /= s_true
    sg = L.scale_gt(est_un, gt, T[:3, :3]); assert abs(sg - s_true) < 1e-6, sg
    print(f"0-a PASS: scale err {abs(s - s_true):.2e}, align err {err:.2e}, s_gt err {abs(sg - s_true):.2e}")


def check_b(seq_name="AP12", S=518, n=20):
    seq = L.Sequence("ho3d", seq_name); ids = seq.kf_ids[:n]; K = seq.K
    frames = [seq.frame(f) for f in ids]
    Rs = [L.centre_rotation(K, L.mask_centroid(m)) for _, _, m in frames]
    fv = L.virtual_focal(K, [m for _, _, m in frames], Rs, S); Kv = np.array([[fv, 0, S / 2], [0, fv, S / 2], [0, 0, 1.0]])
    ious, edge, cen = [], [], []
    for (rgb, d, m), R in zip(frames, Rs):
        img, dv, mv, _ = L.warp_I1(rgb, d, m, K, R, Kv, S)
        back = L.unwarp_mask(mv, K, R, Kv, S, *m.shape)
        ious.append((back & m).sum() / (back | m).sum())
        ys, xs = np.nonzero(mv); edge.append(max(np.abs(xs - S / 2).max(), np.abs(ys - S / 2).max()) / S); cen.append((xs.mean() - S / 2, ys.mean() - S / 2))
    print(f"0-b {seq_name}: f_v={fv:.1f} IoU min {min(ious):.4f} mean {np.mean(ious):.4f}; max mask extent {max(edge):.3f}*S (limit 0.4); centroid offsets max |dx|,|dy| = {max(abs(c[0]) for c in cen):.1f},{max(abs(c[1]) for c in cen):.1f} px")
    assert min(ious) > 0.98, min(ious)
    cv2.imwrite(str(Path(sys.argv[1]) / f"warp_{seq_name}_{ids[0]}.png"), cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)) if len(sys.argv) > 1 else None
    print("0-b PASS")


if __name__ == "__main__":
    check_a(); check_b()
    if len(sys.argv) > 1: check_b("mustard0" if False else "AP12")
