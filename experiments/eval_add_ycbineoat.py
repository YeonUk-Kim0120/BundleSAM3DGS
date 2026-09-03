"""ADD / ADD-S evaluation of YCBInEOAT online-tracking runs.

Reproduces the milestone-4 ADD gate protocol (which itself mirrors
``benchmark_ho3d.py``): per-frame ``ob_in_cam/*.txt`` predictions are
first-frame aligned to GT (``pred_i @ inv(pred_0) @ gt_0`` — right-multiply;
the left-multiplied variant gives ~50 cm garbage), then ADD / ADD-S mean
errors (cm) and AUC@0.1 m are computed on the YCB model vertices.

  python3 experiments/eval_add_ycbineoat.py \
    --video-dir datasets/YCBInEOAT/mustard0 \
    --run-dirs outputs/run_a outputs/run_b
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np
import trimesh

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from Utils import add_err, adi_err, compute_auc  # noqa: E402

VIDEO_TO_OBJECT = {
    "bleach0": "021_bleach_cleanser",
    "bleach_hard_00_03_chaitanya": "021_bleach_cleanser",
    "cracker_box_reorient": "003_cracker_box",
    "cracker_box_yalehand0": "003_cracker_box",
    "mustard0": "006_mustard_bottle",
    "mustard_easy_00_02": "006_mustard_bottle",
    "sugar_box1": "004_sugar_box",
    "sugar_box_yalehand0": "004_sugar_box",
    "tomato_soup_can_yalehand0": "005_tomato_soup_can",
}


def evaluate(video_dir: Path, run_dir: Path) -> dict:
    color_files = sorted(glob.glob(str(video_dir / "rgb" / "*.png")))
    gt_files = sorted(glob.glob(str(video_dir / "annotated_poses" / "*")))
    pose_files = sorted(glob.glob(str(run_dir / "ob_in_cam" / "*.txt")))
    if len(pose_files) < len(color_files):
        raise RuntimeError(
            f"{run_dir}: {len(pose_files)} poses < {len(color_files)} frames"
        )
    if len(gt_files) != len(color_files):
        raise RuntimeError(f"{video_dir}: GT count != frame count")
    gt_poses = np.array([np.loadtxt(f).reshape(4, 4) for f in gt_files])
    pred_poses = np.array(
        [np.loadtxt(f).reshape(4, 4) for f in pose_files[: len(color_files)]]
    )
    pred_poses = pred_poses @ np.linalg.inv(pred_poses[0]) @ gt_poses[0]

    obj = VIDEO_TO_OBJECT[video_dir.name]
    mesh = trimesh.load(
        str(REPO / "datasets/YCB_Video_Models/models" / obj / "textured_simple.obj")
    )
    model_pts = np.asarray(mesh.vertices, dtype=np.float64)

    adi = np.array([adi_err(p, g, model_pts.copy())
                    for p, g in zip(pred_poses, gt_poses)])
    add = np.array([add_err(p, g, model_pts.copy())
                    for p, g in zip(pred_poses, gt_poses)])
    return {
        "run_dir": str(run_dir),
        "frames": int(len(pred_poses)),
        "ADDS_err_cm": float(adi.mean() * 100),
        "ADD_err_cm": float(add.mean() * 100),
        "ADDS_AUC": float(compute_auc(adi) * 100),
        "ADD_AUC": float(compute_auc(add) * 100),
        "ADD_err_cm_median": float(np.median(add) * 100),
        "ADD_err_cm_p90": float(np.percentile(add, 90) * 100),
    }


def evaluate_ho3d(video_dir: Path, run_dir: Path) -> dict:
    """Pose part of ``benchmark_ho3d.benchmark_one_video`` (GT from meta pkl,
    frames without GT skipped, first-frame alignment, YCB model vertices)."""
    sys.path.append(str(REPO / "BundleTrack/scripts"))
    from data_reader import Ho3dReader  # noqa: E402

    reader = Ho3dReader(str(video_dir))
    pose_files = sorted(glob.glob(str(run_dir / "ob_in_cam" / "*.txt")))
    if len(pose_files) < len(reader.color_files):
        raise RuntimeError(
            f"{run_dir}: {len(pose_files)} poses < {len(reader.color_files)} frames"
        )
    gt_poses, ids = [], []
    for i in range(len(reader.color_files)):
        gt = reader.get_gt_pose(i)
        if gt is None:
            continue
        gt_poses.append(gt)
        ids.append(i)
    gt_poses = np.array(gt_poses)
    pred_poses = np.array([np.loadtxt(pose_files[i]).reshape(4, 4) for i in ids])
    pred_poses = pred_poses @ np.linalg.inv(pred_poses[0]) @ gt_poses[0]
    model_pts = np.asarray(reader.get_gt_mesh().vertices, dtype=np.float64)

    adi = np.array([adi_err(p, g, model_pts.copy())
                    for p, g in zip(pred_poses, gt_poses)])
    add = np.array([add_err(p, g, model_pts.copy())
                    for p, g in zip(pred_poses, gt_poses)])
    return {
        "run_dir": str(run_dir),
        "frames": int(len(pred_poses)),
        "ADDS_err_cm": float(adi.mean() * 100),
        "ADD_err_cm": float(add.mean() * 100),
        "ADDS_AUC": float(compute_auc(adi) * 100),
        "ADD_AUC": float(compute_auc(add) * 100),
        "ADD_err_cm_median": float(np.median(add) * 100),
        "ADD_err_cm_p90": float(np.percentile(add, 90) * 100),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video-dir", type=Path,
                        default=REPO / "datasets/YCBInEOAT/mustard0")
    parser.add_argument("--dataset", choices=("ycb", "ho3d"), default="ycb")
    parser.add_argument("--run-dirs", type=Path, nargs="+", required=True)
    parser.add_argument("--out-json", type=Path, default=None)
    args = parser.parse_args()

    results = []
    for run_dir in args.run_dirs:
        if args.dataset == "ho3d":
            result = evaluate_ho3d(args.video_dir, run_dir)
        else:
            result = evaluate(args.video_dir, run_dir)
        results.append(result)
        print(
            f"{run_dir.name}: ADD-S {result['ADDS_err_cm']:.3f} cm | "
            f"ADD {result['ADD_err_cm']:.3f} cm | "
            f"ADD-S AUC {result['ADDS_AUC']:.2f} | ADD AUC {result['ADD_AUC']:.2f} "
            f"| ADD median {result['ADD_err_cm_median']:.3f} p90 "
            f"{result['ADD_err_cm_p90']:.3f} ({result['frames']} frames)"
        )
    if args.out_json is not None:
        with args.out_json.open("w") as f:
            json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
