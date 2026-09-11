"""EXPERIMENT ⑤-B: oracle priors for the pose-feedback cause hunt.

Hypothesis (1) of MILESTONE5_FEEDBACK_RESULTS.md: the persistent GS map's
object frame is biased by the SAM3D prior (alignment 2.7-8 mm + some rotation),
so the pose gradient is not GT-consistent.  This script builds priors whose
placement is *known* (GT is used only here, offline), in the exact file
format ``bundlesdf.run_gaussian`` consumes, so the online run differs from
the normal one only in the prior:

- ``gtmesh``  : the dataset GT mesh, placed with the GT object pose of frame 0
                (removes prior SHAPE and PLACEMENT error);
- ``sam3dgt`` : the SAM3D mesh with its ② refined Sim(3) pose corrected by a
                scaled ICP onto the GT mesh (removes PLACEMENT error, keeps
                the SAM3D shape).  The measured ICP correction is also the
                residual misalignment of our ② alignment vs GT.

Both are written as ``<seq>_<variant>.npz`` (+ ``.json``) under --out-dir and
must be run online with ``prior.skip_alignment`` (the pose json is exact) and,
for ``gtmesh``, ``prior.skip_color_transfer`` (its canonical frame is not the
SAM3D one).  Conventions follow ``sam3d_prior``: canonical -> PyTorch3D camera
is x_p3d = (s*x) @ R_row + T, and OpenCV camera negates x, y.

  python3 experiments/make_oracle_prior.py --dataset ho3d \
    --video-dir datasets/HO3D_v3/evaluation/AP12 \
    --run-dir outputs/gsfb_v1_ho3d_AP12_20260907 --out-dir logs/oracle_priors_20260908
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
import trimesh

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "experiments"))
sys.path.append(str(REPO / "BundleTrack/scripts"))

from sam3d_prior import (  # noqa: E402
    Sim3Pose,
    SurfelSet,
    load_mesh_prior,
    load_sam3d_pose_or_refined,
    matrix_to_quat_wxyz,
    quat_wxyz_to_matrix,
    sample_surfels,
    transform_surfels_canonical_to_cv_camera,
)

SAM3D_ROOT = Path("/home/kist/Desktop/sam-3d-objects")
D_FLIP = np.diag([-1.0, -1.0, 1.0])
HO3D_OBJECTS = {"AP": "019_pitcher_base", "MPM": "010_potted_meat_can", "SB": "021_bleach_cleanser",
                "SM": "006_mustard_bottle"}
YCB_OBJECTS = {"bleach0": "021_bleach_cleanser", "bleach_hard_00_03_chaitanya": "021_bleach_cleanser",
               "cracker_box_reorient": "003_cracker_box", "cracker_box_yalehand0": "003_cracker_box",
               "mustard0": "006_mustard_bottle", "mustard_easy_00_02": "006_mustard_bottle",
               "sugar_box1": "004_sugar_box", "sugar_box_yalehand0": "004_sugar_box",
               "tomato_soup_can_yalehand0": "005_tomato_soup_can"}


def rot_deg(R: np.ndarray) -> float:
    return float(np.degrees(np.arccos(np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0))))


def load_gt_frame0(dataset: str, video_dir: Path) -> np.ndarray:
    if dataset == "ho3d":
        from data_reader import Ho3dReader

        g = Ho3dReader(str(video_dir)).get_gt_pose(0)
    else:
        g = np.loadtxt(sorted(glob.glob(str(video_dir / "annotated_poses/*.txt")))[0]).reshape(4, 4)
    if g is None:
        raise RuntimeError("frame 0 has no GT pose")
    return np.asarray(g, dtype=np.float64)


def load_gt_mesh(dataset: str, seq: str) -> trimesh.Trimesh:
    if dataset == "ho3d":
        obj = HO3D_OBJECTS[[k for k in HO3D_OBJECTS if seq.startswith(k)][0]]
        root = REPO / "datasets/HO3D_v3/models" / obj
    else:
        obj = YCB_OBJECTS[seq]
        root = REPO / "datasets/YCB_Video_Models/models" / obj
    for name in ("textured_simple.obj", "textured.obj"):
        if (root / name).exists():
            mesh = trimesh.load(str(root / name), force="mesh", process=False)
            break
    else:
        raise FileNotFoundError(f"no textured mesh under {root}")
    if not isinstance(mesh, trimesh.Trimesh):
        raise RuntimeError(f"unexpected mesh type {type(mesh)}")
    return mesh


def mesh_vertex_colors(mesh: trimesh.Trimesh) -> np.ndarray:
    try:
        colors = np.asarray(mesh.visual.to_color().vertex_colors, dtype=np.float64)[:, :3] / 255.0
        if colors.shape[0] == len(mesh.vertices) and np.isfinite(colors).all():
            return np.clip(colors, 0.0, 1.0).astype(np.float32)
    except Exception as exc:  # noqa: BLE001
        print(f"[oracle] texture -> vertex colors failed ({exc}); using gray")
    return np.full((len(mesh.vertices), 3), 0.5, dtype=np.float32)


def pose_json(R_row: np.ndarray, T: np.ndarray, scale: float, extra: dict) -> dict:
    quat = matrix_to_quat_wxyz(torch.from_numpy(R_row.astype(np.float32))[None])[0]
    # round-trip check of the quaternion convention
    R_back = quat_wxyz_to_matrix(quat[None])[0].numpy().astype(np.float64)
    assert np.abs(R_back - R_row).max() < 2e-3, f"quaternion round trip failed ({np.abs(R_back - R_row).max():.2e})"
    return {"mapping": "sam3d_canonical_to_pytorch3d_camera", "pose": {
        "rotation": [float(v) for v in quat], "translation": [float(v) for v in T],
        "scale": [float(scale)] * 3}, **extra}


def cv_from_pose(points: np.ndarray, pose: Sim3Pose) -> np.ndarray:
    n = len(points)
    surf = SurfelSet(means=torch.from_numpy(points.astype(np.float32)),
                     normals=torch.tensor([[0.0, 0.0, 1.0]]).repeat(n, 1),
                     radii=torch.ones(n), colors=torch.full((n, 3), 0.5), opacities=torch.full((n,), 0.9))
    return transform_surfels_canonical_to_cv_camera(surf, pose).means.numpy().astype(np.float64)


def depth_residual_mm(points_cv: np.ndarray, run_dir: Path) -> dict:
    """Median |z - depth| of a z-buffered projection of the points into frame 0 (sanity check)."""
    import cv2

    K = np.loadtxt(run_dir / "cam_K.txt").reshape(3, 3)
    first = sorted(os.listdir(run_dir / "ob_in_cam"))[0].replace(".txt", "")
    depth = cv2.imread(str(run_dir / "depth_filtered" / f"{first}.png"), -1).astype(np.float64) / 1000.0
    mask = cv2.imread(str(run_dir / "mask" / f"{first}.png"), -1) > 0
    h, w = depth.shape
    z = points_cv[:, 2]
    ok = z > 0.05
    u = np.round(K[0, 0] * points_cv[ok, 0] / z[ok] + K[0, 2]).astype(int)
    v = np.round(K[1, 1] * points_cv[ok, 1] / z[ok] + K[1, 2]).astype(int)
    inside = (u >= 0) & (u < w) & (v >= 0) & (v < h)
    zbuf = np.full((h, w), np.inf)
    np.minimum.at(zbuf, (v[inside], u[inside]), z[ok][inside])
    valid = np.isfinite(zbuf) & mask & (depth > 0.05)
    diff = np.abs(zbuf[valid] - depth[valid]) * 1000.0
    return {"pixels": int(valid.sum()), "median_mm": float(np.median(diff)) if len(diff) else None,
            "p90_mm": float(np.percentile(diff, 90)) if len(diff) else None,
            "projected_inside_mask_ratio": float((mask[v[inside], u[inside]]).mean()) if inside.any() else None}


def sim3_icp(source: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, dict]:
    """Scaled point-to-point ICP (open3d), coarse to fine.  Returns 4x4 (sR | t) and stats."""
    import open3d as o3d

    src = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(source))
    tgt = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(target))
    T = np.eye(4)
    stats = []
    for thr in (0.05, 0.02, 0.01, 0.005):
        reg = o3d.pipelines.registration.registration_icp(
            src, tgt, thr, T,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(with_scaling=True),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=300))
        T = np.asarray(reg.transformation)
        stats.append({"threshold_m": thr, "fitness": float(reg.fitness), "inlier_rmse_mm": float(reg.inlier_rmse * 1000)})
    return T, {"stages": stats}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=("ycb", "ho3d"), required=True)
    ap.add_argument("--video-dir", type=Path, required=True)
    ap.add_argument("--run-dir", type=Path, required=True, help="a finished run dir (cam_K, frame-0 depth/mask)")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--prior-root", type=Path, default=SAM3D_ROOT)
    ap.add_argument("--refined-json", type=Path, default=None,
                    help="② refined pose cache for the ICP start (default: logs/*_exp_colortransfer_C_20260831)")
    ap.add_argument("--variants", nargs="+", default=["gtmesh", "sam3dgt"])
    args = ap.parse_args()
    seq = args.video_dir.name
    args.out_dir.mkdir(parents=True, exist_ok=True)
    sub = "output_YCBInEOAT_sam2mask_mesh" if args.dataset == "ycb" else "output_HO3D_sam2mask_mesh"
    sam_npz = args.prior_root / sub / f"{seq}_mesh_depth.npz"
    sam_json = args.prior_root / sub / f"{seq}_mesh_depth.json"
    refined = args.refined_json or (
        REPO / "logs" / f"{'ho3d_' if args.dataset == 'ho3d' else ''}{seq}_exp_colortransfer_C_20260831"
        / "sam3d_rts_refined.json")

    gt0 = load_gt_frame0(args.dataset, args.video_dir)            # ob_in_cam of frame 0 (OpenCV)
    R_gt, t_gt = gt0[:3, :3], gt0[:3, 3]
    u_, _, vt_ = np.linalg.svd(R_gt)                              # annotated rotations are not exactly orthonormal
    R_ortho = u_ @ vt_
    if np.linalg.det(R_ortho) < 0:
        u_[:, -1] *= -1.0
        R_ortho = u_ @ vt_
    report_ortho_err = float(np.abs(R_ortho - R_gt).max())
    R_gt = R_ortho
    report: dict = {"seq": seq, "dataset": args.dataset, "gt_rotation_orthonormalization_max_abs": report_ortho_err}
    gt_mesh = load_gt_mesh(args.dataset, seq)
    gt_pts_cv = gt_mesh.vertices @ R_gt.T + t_gt[None, :]
    report["gt_mesh"] = {"vertices": int(len(gt_mesh.vertices)), "faces": int(len(gt_mesh.faces)),
                         "depth_residual_frame0": depth_residual_mm(gt_pts_cv, args.run_dir)}
    print(f"[oracle] GT mesh @ GT pose, frame-0 depth residual: {report['gt_mesh']['depth_residual_frame0']}")

    if "gtmesh" in args.variants:
        # want x_cv = R_gt x + t  ->  R_row = R_gt^T D, T = D t, s = 1
        R_row = R_gt.T @ D_FLIP
        T = D_FLIP @ t_gt
        pose = Sim3Pose(scale=torch.ones(3), R_row=torch.from_numpy(R_row.astype(np.float32)),
                        T=torch.from_numpy(T.astype(np.float32)))
        check = cv_from_pose(gt_mesh.vertices, pose)
        err = np.abs(check - gt_pts_cv).max()
        assert err < 1e-3, f"gtmesh pose convention check failed: {err}"
        colors = mesh_vertex_colors(gt_mesh)
        np.savez(args.out_dir / f"{seq}_gtmesh.npz", vertices=gt_mesh.vertices.astype(np.float32),
                 faces=gt_mesh.faces.astype(np.int64), vertex_colors=colors, success=True)
        with open(args.out_dir / f"{seq}_gtmesh.json", "w") as fh:
            json.dump(pose_json(R_row, T, 1.0, {"oracle": "gt_mesh_at_gt_pose_frame0", "gt_ob_in_cam_frame0": gt0.tolist()}), fh, indent=1)
        # make sure the online loader reproduces the placement exactly
        prior = load_mesh_prior(args.out_dir / f"{seq}_gtmesh.npz")
        loaded = load_sam3d_pose_or_refined(args.out_dir / f"{seq}_gtmesh.json")
        surf = transform_surfels_canonical_to_cv_camera(sample_surfels(prior, 20000, seed=0, radius_multiplier=0.75), loaded)
        report["gtmesh"] = {"convention_check_max_err_m": float(err), "surfel_depth_residual_frame0": depth_residual_mm(surf.means.numpy().astype(np.float64), args.run_dir),
                            "colored_vertices": bool(colors.std() > 0.01)}
        print(f"[oracle] gtmesh written; surfel frame-0 depth residual {report['gtmesh']['surfel_depth_residual_frame0']}")

    if "sam3dgt" in args.variants:
        prior = load_mesh_prior(sam_npz)
        start = load_sam3d_pose_or_refined(refined if refined.exists() else sam_json)
        report["sam3dgt"] = {"icp_start": str(refined if refined.exists() else sam_json)}
        surf0 = sample_surfels(prior, 20000, seed=0, radius_multiplier=0.75)
        src_cv = transform_surfels_canonical_to_cv_camera(surf0, start).means.numpy().astype(np.float64)
        report["sam3dgt"]["start_depth_residual_frame0"] = depth_residual_mm(src_cv, args.run_dir)
        tgt_pts = np.asarray(trimesh.sample.sample_surface(gt_mesh, 60000)[0]) @ R_gt.T + t_gt[None, :]
        T_icp, icp_stats = sim3_icp(src_cv, tgt_pts)
        sigma = float(np.cbrt(np.linalg.det(T_icp[:3, :3])))
        R_icp = T_icp[:3, :3] / sigma
        t_icp = T_icp[:3, 3]
        # compose onto the row-convention pose: x_cv' = sigma R_icp x_cv + t_icp
        s0 = float(start.scale.mean())
        R_row0 = start.R_row.numpy().astype(np.float64)
        T0 = start.T.numpy().astype(np.float64)
        R_row1 = R_row0 @ D_FLIP @ R_icp.T @ D_FLIP
        T1 = D_FLIP @ (sigma * R_icp @ (D_FLIP @ T0) + t_icp)
        s1 = sigma * s0
        pose1 = Sim3Pose(scale=torch.full((3,), float(s1)), R_row=torch.from_numpy(R_row1.astype(np.float32)),
                         T=torch.from_numpy(T1.astype(np.float32)))
        check = cv_from_pose(surf0.means.numpy().astype(np.float64), pose1)
        expected = src_cv @ (sigma * R_icp).T + t_icp[None, :]
        err = np.abs(check - expected).max()
        assert err < 1e-3, f"sam3dgt composition check failed: {err}"
        shutil.copy(sam_npz, args.out_dir / f"{seq}_sam3dgt.npz")
        with open(args.out_dir / f"{seq}_sam3dgt.json", "w") as fh:
            json.dump(pose_json(R_row1, T1, s1, {"oracle": "sam3d_mesh_icp_to_gt_mesh_frame0", "icp": icp_stats,
                                                 "correction": {"rotation_deg": rot_deg(R_icp), "translation_mm": float(np.linalg.norm(t_icp) * 1000),
                                                                "scale": sigma}}), fh, indent=1)
        report["sam3dgt"].update({"icp": icp_stats, "correction_vs_refined": {"rotation_deg": rot_deg(R_icp), "translation_mm": float(np.linalg.norm(t_icp) * 1000), "scale": sigma},
                                  "composition_check_max_err_m": float(err),
                                  "final_depth_residual_frame0": depth_residual_mm(check, args.run_dir)})
        print(f"[oracle] sam3dgt: ② refined -> GT correction rot {rot_deg(R_icp):.2f} deg, trans {np.linalg.norm(t_icp) * 1000:.1f} mm, "
              f"scale {sigma:.3f}; icp {icp_stats['stages'][-1]}; depth residual {report['sam3dgt']['final_depth_residual_frame0']}")

    with open(args.out_dir / f"{seq}_oracle_report.json", "w") as fh:
        json.dump(report, fh, indent=1)


if __name__ == "__main__":
    main()
