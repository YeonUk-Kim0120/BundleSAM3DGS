import sys, math, numpy as np
sys.path.insert(0, "/home/kist/Desktop/BundleSAM3DGS/experiments")
import exp_reference_optimizer_probe as X
from pathlib import Path
from exp_feedback_gradient_probe_gtmap import load_cycles, SAM3D_ROOT
from gaussian_runner import GaussianFrame
from run_gaussian_incremental import load_frame
rd = Path("outputs/exp_batch_20260927/noop/ho3d/AP12"); K = np.loadtxt(rd / "cam_K.txt").reshape(3, 3).astype(np.float32); cyc = load_cycles(rd)[0]
pp = {"mesh_npz": SAM3D_ROOT / "output_HO3D_sam2mask_mesh/AP12_mesh_depth.npz", "pose_json": SAM3D_ROOT / "output_HO3D_sam2mask_mesh/AP12_mesh_depth.json", "gaussian_ply": SAM3D_ROOT / "output_HO3D_sam2mask_mesh/AP12_splat_depth.ply"}
frames = [GaussianFrame(fid, *(lambda f: (f.rgb, f.depth, f.mask, f.K, f.c2w_cv))(load_frame(rd, fid, K, cyc["before"][i].astype(np.float32)))).validated() for i, fid in enumerate(cyc["ids"])]
runner = X.build_first("RP", X.REPO / "config_gs_2dgs_1mm_lifecycle.yml", "cuda:0", pp, frames, cyc["before"][0].astype(np.float32), 0); ref = X.SurfelReference(runner)
def orth(T):
    u, _, vt = np.linalg.svd(T[:3, :3]); R = u @ vt
    if np.linalg.det(R) < 0: u[:, -1] *= -1; R = u @ vt
    T = T.copy(); T[:3, :3] = R; return T
def ang(R1, R2): return math.degrees(2 * math.asin(min(1.0, np.linalg.norm(R1 - R2) / (2 * math.sqrt(2)))))
T_raw = cyc["before"][1].astype(np.float64); print("orthonormality error of saved pose", np.abs(T_raw[:3, :3].T @ T_raw[:3, :3] - np.eye(3)).max())
rng = np.random.default_rng(0)
for k in range(3):
    T_true = orth(cyc["before"][k].astype(np.float64)); w2c = np.linalg.inv(T_true); Xc = ref.mu @ w2c[:3, :3].T + w2c[:3, 3]
    u = Xc[:, 0] / Xc[:, 2] * K[0, 0] + K[0, 2]; v = Xc[:, 1] / Xc[:, 2] * K[1, 1] + K[1, 2]
    pts = Xc[(Xc[:, 2] > 0.05) & (u >= 0) & (u < 640) & (v >= 0) & (v < 480)]; pts = pts[rng.choice(len(pts), min(5000, len(pts)), replace=False)]
    for trial in range(6):
        ax = rng.normal(size=3); ax /= np.linalg.norm(ax); td = rng.normal(size=3); td /= np.linalg.norm(td)
        Ts = X.left_update(T_true, np.concatenate((td * 0.005, ax * math.radians(3.0))))
        res = []
        for mi in (8, 30):
            X.P["gn_max_iter"] = mi; Te, c0, c1, it, ratio, acc = X.gn_refine(ref, Ts, pts); res.append(f"max{mi}: {ang(Te[:3,:3], T_true[:3,:3]):.4f} deg {np.linalg.norm(Te[:3,3]-T_true[:3,3])*1000:.3f} mm it {it}")
        X.P["gn_max_iter"] = 8
        print(k, trial, " | ".join(res))
    Te, *_ = X.gn_refine(ref, T_true, pts); print(k, "zero:", f"{ang(Te[:3,:3], T_true[:3,:3]):.6f} deg")
