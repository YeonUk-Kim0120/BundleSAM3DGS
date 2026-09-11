"""Reconstruction evaluation with three Chamfer protocols (stage 1, 2026-09-11).  Works for any mesh (ours or
BundleSDF's ``textured_mesh.obj``), so both methods are scored by the same code.

Alignment (identical to the original benchmark): mesh -> first online pose (``ob_in_cam/<first>.txt``) ->
inverse GT pose of frame 0, i.e. into the GT object frame; crop to GT bbox + 0.3 m; largest component; sample
99 999 surface points; 5 mm voxel downsample; point-to-point ICP (2 cm) onto the protocol-1 GT cloud.  The ICP
transform found on protocol 1 is reused by protocols 2 and 3.

P1 "original":  GT = HO3D ``visible_mesh.ply`` (seen parts only; 5 mm downsampled) / YCB full model (99 999
                samples, as the baseline folder's YCB benchmark).  Mutual Chamfer = 0.5*(mean pred->GT + mean GT->pred), cm.
P2 "full":      GT = the complete CAD model (99 999 samples).  Mutual Chamfer.  Holes in the prediction and
                surfaces the prediction invents both count here.
P3 "regions":   the full-model GT samples are split into SEEN / UNSEEN.  HO3D: a GT face is seen if all its
                vertices are finite in visible_mesh.obj (same topology as the model).  YCB: a GT sample is seen if,
                from at least one keyframe GT pose, it is inside the image and not self-occluded (Open3D ray cast;
                hand/gripper occlusion is not modelled).  Reported: GT->pred mean distance per region (how well
                each region is covered), pred->GT overall, and the region sizes.

  python3 experiments/eval_mesh_cd.py --dataset ho3d --video-dir datasets/HO3D_v3/evaluation/AP12 \
      --run-dir outputs/gsfb_v1_ho3d_AP12_20260907 --mesh outputs/gsfb_v1_ho3d_AP12_20260907/final/gs/mesh_tsdf_virtual.obj \
      --out-json logs/.../cd_AP12_tsdf_virtual.json
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from pathlib import Path

import numpy as np
import open3d as o3d
import trimesh
from scipy.spatial import cKDTree

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.append(str(REPO / "BundleTrack/scripts"))

HO3D_OBJECTS = {"AP": "019_pitcher_base", "MPM": "010_potted_meat_can", "SB": "021_bleach_cleanser", "SM": "006_mustard_bottle"}
YCB_OBJECTS = {"bleach0": "021_bleach_cleanser", "bleach_hard_00_03_chaitanya": "021_bleach_cleanser",
               "cracker_box_reorient": "003_cracker_box", "cracker_box_yalehand0": "003_cracker_box",
               "mustard0": "006_mustard_bottle", "mustard_easy_00_02": "006_mustard_bottle",
               "sugar_box1": "004_sugar_box", "sugar_box_yalehand0": "004_sugar_box",
               "tomato_soup_can_yalehand0": "005_tomato_soup_can"}
N_SAMPLES = 99999
VOXEL = 0.005
ICP_THRES = 0.02


def load_gt_mesh(dataset: str, seq: str) -> trimesh.Trimesh:
    if dataset == "ho3d":
        obj = HO3D_OBJECTS[[k for k in HO3D_OBJECTS if seq.startswith(k)][0]]
        root = REPO / "datasets/HO3D_v3/models" / obj
    else:
        root = REPO / "datasets/YCB_Video_Models/models" / YCB_OBJECTS[seq]
    for name in ("textured_simple.obj", "textured.obj"):
        if (root / name).exists():
            return trimesh.load(str(root / name), force="mesh", process=False)
    raise FileNotFoundError(root)


def load_gt_poses(dataset: str, video_dir: Path):
    if dataset == "ho3d":
        from data_reader import Ho3dReader
        reader = Ho3dReader(str(video_dir))
        poses = [reader.get_gt_pose(i) for i in range(len(reader.id_strs))]
        ids = list(reader.id_strs)
        K = reader.K.copy()
        import cv2
        H, W = cv2.imread(reader.color_files[0]).shape[:2]
    else:
        files = sorted(glob.glob(str(video_dir / "annotated_poses/*.txt")))     # 0000000.txt ... (frame index)
        poses = [np.loadtxt(f).reshape(4, 4) for f in files]
        rgb = sorted(glob.glob(str(video_dir / "rgb/*.png")))                   # <timestamp>.png = frame id
        ids = [os.path.splitext(os.path.basename(f))[0] for f in rgb]
        poses += [None] * (len(ids) - len(poses))
        K = np.loadtxt(video_dir / "cam_K.txt").reshape(3, 3)
        import cv2
        H, W = cv2.imread(sorted(glob.glob(str(video_dir / "rgb/*.png")))[0]).shape[:2]
    return ids, poses, K, (H, W)


def mutual_chamfer(a: np.ndarray, b: np.ndarray) -> tuple[float, float, float]:
    d_ab = cKDTree(b).query(a)[0]; d_ba = cKDTree(a).query(b)[0]
    return 0.5 * (d_ab.mean() + d_ba.mean()), d_ab.mean(), d_ba.mean()


def to_o3d(pts: np.ndarray) -> o3d.geometry.PointCloud:
    p = o3d.geometry.PointCloud(); p.points = o3d.utility.Vector3dVector(np.asarray(pts, dtype=np.float64)); return p


def prepare_pred(mesh: trimesh.Trimesh, gt_pts: np.ndarray) -> trimesh.Trimesh:
    """Original benchmark's cleaning: crop to GT bbox + 0.3 m, clean, largest component (min_edge 1000, then 3)."""
    from Utils import trimesh_split

    def trimesh_clean(m: trimesh.Trimesh) -> trimesh.Trimesh:
        # Utils.trimesh_clean written for the old trimesh API; same operations with the current API
        m.merge_vertices()
        m.update_faces(m.nondegenerate_faces())
        m.update_faces(m.unique_faces())
        m.remove_infinite_values()
        m.remove_unreferenced_vertices()
        return m
    max_coord = gt_pts.max(axis=0).reshape(1, 3) + 0.3
    min_coord = gt_pts.min(axis=0).reshape(1, 3) - 0.3
    bad = (mesh.vertices > max_coord).any(axis=-1) | (mesh.vertices < min_coord).any(axis=-1)
    mesh.vertices[bad] = np.inf
    mesh = trimesh_clean(mesh)
    components = trimesh_split(mesh, min_edge=1000)
    if len(components) == 0:
        components = trimesh_split(mesh, min_edge=3)
    best, best_size = None, 0
    for c in components:
        if np.linalg.norm(c.vertices, axis=-1).min() > 0.1:
            continue
        if len(c.vertices) > best_size:
            best, best_size = c, len(c.vertices)
    if best is None:
        raise RuntimeError("no mesh component near the object")
    return best


def seen_labels_ycb(gt_mesh: trimesh.Trimesh, samples: np.ndarray, kf_poses_gt: list[np.ndarray], K: np.ndarray, hw) -> np.ndarray:
    """A sample is SEEN if from some keyframe GT pose (ob_in_cam) it projects inside the image and the first
    mesh hit along the camera ray is (within 3 mm of) the sample itself."""
    scene = o3d.t.geometry.RaycastingScene()
    legacy = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(np.asarray(gt_mesh.vertices, dtype=np.float64)),
                                       o3d.utility.Vector3iVector(np.asarray(gt_mesh.faces, dtype=np.int32)))
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(legacy))
    H, W = hw
    seen = np.zeros(len(samples), dtype=bool)
    homo = np.concatenate([samples, np.ones((len(samples), 1))], 1)
    for T in kf_poses_gt:                      # object -> camera
        pc = (T @ homo.T).T[:, :3]
        z = pc[:, 2]
        ok = z > 0.05
        u = K[0, 0] * pc[:, 0] / np.where(ok, z, 1) + K[0, 2]
        v = K[1, 1] * pc[:, 1] / np.where(ok, z, 1) + K[1, 2]
        ok &= (u >= 0) & (u < W) & (v >= 0) & (v < H)
        if not ok.any():
            continue
        cam_obj = np.linalg.inv(T)[:3, 3]      # camera centre in object frame
        dirs = samples[ok] - cam_obj
        dist = np.linalg.norm(dirs, axis=1); dirs /= dist[:, None]
        rays = o3d.core.Tensor(np.concatenate([np.repeat(cam_obj[None], len(dirs), 0), dirs], 1).astype(np.float32))
        hit = scene.cast_rays(rays)["t_hit"].numpy()
        vis = np.abs(hit - dist) < 0.003
        idx = np.nonzero(ok)[0][vis]
        seen[idx] = True
    return seen


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=("ycb", "ho3d"), required=True)
    ap.add_argument("--video-dir", type=Path, required=True)
    ap.add_argument("--run-dir", type=Path, required=True, help="run with ob_in_cam/ (first online pose = alignment gauge)")
    ap.add_argument("--mesh", type=Path, required=True)
    ap.add_argument("--out-json", type=Path, required=True)
    ap.add_argument("--keyframes-yml", type=Path, default=None, help="YCB P3 visibility: keyframe ids (default: last <run-dir>/*/keyframes.yml)")
    ap.add_argument("--save-clouds", action="store_true")
    args = ap.parse_args()
    seq = args.video_dir.name
    ids, gt_poses, K, hw = load_gt_poses(args.dataset, args.video_dir)
    gt_mesh = load_gt_mesh(args.dataset, seq)
    if gt_poses[0] is None:
        raise RuntimeError("frame 0 has no GT pose")
    first = sorted(os.listdir(args.run_dir / "ob_in_cam"))[0]
    pred0 = np.loadtxt(args.run_dir / "ob_in_cam" / first).reshape(4, 4)

    # ---- GT clouds
    full_pts, full_faces = trimesh.sample.sample_surface(gt_mesh, N_SAMPLES, seed=0)
    full_pts = np.asarray(full_pts, dtype=np.float64)
    if args.dataset == "ho3d":
        vis = o3d.io.read_point_cloud(str(args.video_dir / "visible_mesh.ply")).voxel_down_sample(VOXEL)
        p1_gt = np.asarray(vis.points).copy()
        vmesh = trimesh.load(str(args.video_dir / "visible_mesh.obj"), force="mesh", process=False)
        finite = np.isfinite(vmesh.vertices).all(1)
        seen = finite[gt_mesh.faces[np.asarray(full_faces)]].all(1)
        seen_source = "visible_mesh.obj faces"
    else:
        p1_gt = full_pts
        kf_yml = args.keyframes_yml or sorted(glob.glob(str(args.run_dir / "*/keyframes.yml")))[-1]
        import yaml
        kf_ids = [k.replace("keyframe_", "") for k in yaml.safe_load(open(kf_yml)).keys()]
        kf_poses = [gt_poses[ids.index(f)] for f in kf_ids if f in ids and gt_poses[ids.index(f)] is not None]
        seen = seen_labels_ycb(gt_mesh, full_pts, kf_poses, K, hw)
        seen_source = f"ray cast from {len(kf_poses)} keyframe GT poses ({os.path.basename(os.path.dirname(kf_yml))})"

    # ---- prediction: align, clean, sample, ICP (original protocol)
    mesh = trimesh.load(str(args.mesh), force="mesh", process=False)
    n_raw = len(mesh.vertices)
    mesh.apply_transform(pred0)
    mesh.apply_transform(np.linalg.inv(gt_poses[0]))
    mesh = prepare_pred(mesh, p1_gt)
    pred_pts, _ = trimesh.sample.sample_surface(mesh, N_SAMPLES, seed=0)
    pred_pts = np.asarray(pred_pts, dtype=np.float64)
    pcd_pred = to_o3d(pred_pts).voxel_down_sample(VOXEL)
    reg = o3d.pipelines.registration.registration_icp(pcd_pred, to_o3d(p1_gt), ICP_THRES, np.eye(4),
                                                      o3d.pipelines.registration.TransformationEstimationPointToPoint())
    T_icp = np.asarray(reg.transformation)
    pred_icp = (T_icp @ np.concatenate([pred_pts, np.ones((len(pred_pts), 1))], 1).T).T[:, :3]

    out = {"dataset": args.dataset, "seq": seq, "mesh": str(args.mesh), "run_dir": str(args.run_dir),
           "mesh_vertices_raw": int(n_raw), "mesh_vertices_used": int(len(mesh.vertices)), "icp_fitness": float(reg.fitness),
           "icp_rmse_m": float(reg.inlier_rmse), "icp_rotation_deg": float(np.degrees(np.arccos(np.clip((np.trace(T_icp[:3, :3]) - 1) / 2, -1, 1)))),
           "icp_translation_mm": float(np.linalg.norm(T_icp[:3, 3]) * 1000)}
    cd, p2g, g2p = mutual_chamfer(pred_icp, p1_gt)
    out["P1_original"] = {"gt": "visible_mesh.ply" if args.dataset == "ho3d" else "full model", "chamfer_cm": cd * 100,
                          "pred_to_gt_cm": p2g * 100, "gt_to_pred_cm": g2p * 100, "gt_points": int(len(p1_gt))}
    cd, p2g, g2p = mutual_chamfer(pred_icp, full_pts)
    out["P2_full_model"] = {"chamfer_cm": cd * 100, "pred_to_gt_cm": p2g * 100, "gt_to_pred_cm": g2p * 100, "gt_points": int(len(full_pts))}
    d_g2p = cKDTree(pred_icp).query(full_pts)[0]
    out["P3_regions"] = {"seen_source": seen_source, "seen_frac": float(seen.mean()),
                         "gt_to_pred_seen_cm": float(d_g2p[seen].mean() * 100) if seen.any() else None,
                         "gt_to_pred_unseen_cm": float(d_g2p[~seen].mean() * 100) if (~seen).any() else None,
                         "unseen_within_5mm_frac": float((d_g2p[~seen] < 0.005).mean()) if (~seen).any() else None,
                         "seen_within_5mm_frac": float((d_g2p[seen] < 0.005).mean()) if seen.any() else None,
                         "pred_to_gt_cm": out["P2_full_model"]["pred_to_gt_cm"]}
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_json, "w") as fh:
        json.dump(out, fh, indent=1)
    if args.save_clouds:
        stem = args.out_json.with_suffix("")
        o3d.io.write_point_cloud(f"{stem}_pred_icp.ply", to_o3d(pred_icp))
        gt_c = to_o3d(full_pts); gt_c.colors = o3d.utility.Vector3dVector(np.where(seen[:, None], [0.2, 0.8, 0.2], [0.9, 0.2, 0.2]))
        o3d.io.write_point_cloud(f"{stem}_gt_regions.ply", gt_c)
    p1, p2, p3 = out["P1_original"], out["P2_full_model"], out["P3_regions"]
    def f(v, spec=".3f"):
        return "n/a" if v is None else format(v, spec)
    print(f"{seq} {args.mesh.name}: P1 {f(p1['chamfer_cm'])} cm (pred->gt {f(p1['pred_to_gt_cm'])}, gt->pred {f(p1['gt_to_pred_cm'])}) | "
          f"P2 {f(p2['chamfer_cm'])} (pred->gt {f(p2['pred_to_gt_cm'])}, gt->pred {f(p2['gt_to_pred_cm'])}) | "
          f"P3 seen {p3['seen_frac']:.0%}: gt->pred seen {f(p3['gt_to_pred_seen_cm'])}, unseen {f(p3['gt_to_pred_unseen_cm'])} cm, "
          f"unseen<5mm {f(p3['unseen_within_5mm_frac'], '.0%')} | icp {out['icp_rotation_deg']:.1f} deg / {out['icp_translation_mm']:.1f} mm")

if __name__ == "__main__":
    main()
