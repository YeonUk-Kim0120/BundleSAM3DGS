"""Fair pose-feedback ablation on the ORIGINAL SDF backend (no repo edits).

Three conditions, identical backend / configs / masks / seed; the only
variable is what the tracker receives back from the reconstruction process:

  on   : BundleSDF as shipped — SDF-refined keyframe poses are written back
         (PoseArray clamp 2 cm / 20 deg) and keyframes become ``_nerfed``
         (frozen in the tracker's bundle adjustment, Bundler.cpp:919).
  noop : the tracker's own poses are written back unchanged (PoseArray
         max_trans = max_rot = 0 -> exact identity deltas), keyframes still
         become ``_nerfed``. Isolates the effect of the pose refinement.
  off  : nothing is written back and nothing becomes ``_nerfed`` — the
         tracker never sees ``optimized_cvcam_in_obs`` (hidden by a dict
         proxy in the tracker process). Isolates the ``_nerfed`` freezing.

Mirrors ``run_custom.run_one_video`` overrides exactly, but refuses an
existing out folder and skips the offline global refinement (online
``ob_in_cam`` poses are what the ADD gate evaluates).

  CUDA_VISIBLE_DEVICES=1 python3 experiments/run_sdf_feedback_ablation.py \
    --feedback noop --out_folder outputs/ablation_fb_noop_r1_<date>
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
code_dir = str(REPO)

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import yaml  # noqa: E402

from bundlesdf import BundleSdf, YcbineoatReader, set_seed  # noqa: E402


class HiddenKeyDict:
    """Tracker-side view of the shared p_dict with one key made invisible."""

    def __init__(self, inner, hidden_key):
        self._inner = inner
        self._hidden = hidden_key

    def __contains__(self, key):
        return False if key == self._hidden else (key in self._inner)

    def __getitem__(self, key):
        if key == self._hidden:
            raise KeyError(key)
        return self._inner[key]

    def __setitem__(self, key, value):
        self._inner[key] = value

    def __delitem__(self, key):
        del self._inner[key]

    def get(self, key, default=None):
        if key == self._hidden:
            return default
        return self._inner.get(key, default)

    def __getattr__(self, name):
        return getattr(self._inner, name)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_dir", type=str,
                        default=f"{code_dir}/datasets/YCBInEOAT/mustard0")
    parser.add_argument("--out_folder", type=str, required=True)
    parser.add_argument("--feedback", choices=("on", "noop", "off"),
                        required=True)
    parser.add_argument("--mask_dir", type=str, default=None,
                        help="default: masks_sam2 (ycb) / masks_SAM2 (ho3d)")
    parser.add_argument("--dataset", choices=("ycb", "ho3d"), default="ycb",
                        help="ycb mirrors run_custom.run_one_video, ho3d "
                             "mirrors run_ho3d.run_one_video")
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--debug_level", type=int, default=2)
    parser.add_argument("--tracker", choices=("ours", "pristine"), default="ours",
                        help="pristine: runtime-swap find_corres/process_new_frame "
                             "for the upstream BundleSDF versions "
                             "(experiments/pristine_tracker_patch.py)")
    args = parser.parse_args()
    if args.mask_dir is None:
        args.mask_dir = "masks_sam2" if args.dataset == "ycb" else "masks_SAM2"

    out_folder = args.out_folder
    if os.path.exists(out_folder):
        raise FileExistsError(f"refusing to overwrite {out_folder}")
    os.makedirs(out_folder)
    t_start = time.time()

    if args.tracker == "pristine":
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        import pristine_tracker_patch  # noqa: E402
        pristine_tracker_patch.apply()

    set_seed(0)

    if args.dataset == "ho3d":
        run_ho3d_video(args, out_folder)
    else:
        run_ycb_video(args, out_folder)
    write_manifest(args, out_folder, t_start)


def apply_feedback_condition(args, cfg_nerf, tracker_factory):
    """Shared condition wiring: returns the constructed tracker."""
    if args.feedback == "noop":
        # PoseArray deltas are tanh(x) * max -> exact identity write-back.
        cfg_nerf["max_trans"] = 0.0
        cfg_nerf["max_rot"] = 0.0
    tracker = tracker_factory(cfg_nerf)
    if args.feedback == "off":
        # The reconstruction process keeps publishing poses into the shared
        # dict; the tracker simply never sees the key (spawned child holds
        # the real proxy, so this cannot leak into it).
        tracker.p_dict = HiddenKeyDict(tracker.p_dict, "optimized_cvcam_in_obs")
    return tracker


def run_ho3d_video(args, out_folder):
    sys.path.append(f"{code_dir}/BundleTrack/scripts")
    from data_reader import Ho3dReader  # noqa: E402

    out_folder = out_folder.rstrip("/") + "/"  # run_ho3d: trailing slash required
    reader = Ho3dReader(args.video_dir, mask_dir=args.mask_dir)

    # ---- identical to run_ho3d.run_one_video ----------------------------
    cfg_bundletrack = yaml.load(
        open(f"{code_dir}/BundleTrack/config_ho3d.yml", "r"), Loader=yaml.Loader
    )
    cfg_bundletrack["data_dir"] = args.video_dir
    cfg_bundletrack["SPDLOG"] = 2
    cfg_bundletrack["depth_processing"]["zfar"] = 1
    cfg_bundletrack["debug_dir"] = out_folder
    cfg_track_dir = f"{out_folder}/config_bundletrack.yml"
    yaml.dump(cfg_bundletrack, open(cfg_track_dir, "w"))

    cfg_nerf = yaml.load(open(f"{code_dir}/config.yml", "r"), Loader=yaml.Loader)
    cfg_nerf["trunc_start"] = 0.01
    cfg_nerf["trunc"] = 0.01
    cfg_nerf["down_scale_ratio"] = 1
    cfg_nerf["far"] = cfg_bundletrack["depth_processing"]["zfar"]
    cfg_nerf["datadir"] = f"{out_folder}/nerf_with_bundletrack_online"
    cfg_nerf["save_dir"] = cfg_nerf["datadir"]
    # ---------------------------------------------------------------------

    def factory(cfg):
        cfg_nerf_dir = f"{out_folder}/config_nerf.yml"
        yaml.dump(cfg, open(cfg_nerf_dir, "w"))
        return BundleSdf(cfg_track_dir=cfg_track_dir, cfg_nerf_dir=cfg_nerf_dir,
                         start_nerf_keyframes=5, use_gui=False)

    tracker = apply_feedback_condition(args, cfg_nerf, factory)
    for i, color_file in enumerate(reader.color_files):
        color = cv2.imread(color_file)
        H, W = color.shape[:2]
        depth = reader.get_depth(i)
        mask = reader.get_mask(i)
        mask = cv2.resize(mask, (W, H), interpolation=cv2.INTER_NEAREST)
        tracker.run(color, depth, reader.K, reader.id_strs[i], mask=mask,
                    occ_mask=None)
    tracker.on_finish()
    args._n_step = cfg_nerf["n_step"]
    args._max_trans = cfg_nerf["max_trans"]
    args._max_rot = cfg_nerf["max_rot"]


def run_ycb_video(args, out_folder):
    # ---- identical to run_custom.run_one_video ---------------------------
    cfg_bundletrack = yaml.load(
        open(f"{code_dir}/BundleTrack/config_ho3d.yml", "r"), Loader=yaml.Loader
    )
    cfg_bundletrack["SPDLOG"] = int(args.debug_level)
    cfg_bundletrack["depth_processing"]["percentile"] = 95
    cfg_bundletrack["erode_mask"] = 3
    cfg_bundletrack["debug_dir"] = out_folder + "/"
    cfg_bundletrack["bundle"]["max_BA_frames"] = 10
    cfg_bundletrack["bundle"]["max_optimized_feature_loss"] = 0.03
    cfg_bundletrack["feature_corres"]["max_dist_neighbor"] = 0.02
    cfg_bundletrack["feature_corres"]["max_normal_neighbor"] = 30
    cfg_bundletrack["feature_corres"]["max_dist_no_neighbor"] = 0.01
    cfg_bundletrack["feature_corres"]["max_normal_no_neighbor"] = 20
    cfg_bundletrack["feature_corres"]["map_points"] = True
    cfg_bundletrack["feature_corres"]["resize"] = 400
    cfg_bundletrack["feature_corres"]["rematch_after_nerf"] = True
    cfg_bundletrack["keyframe"]["min_rot"] = 5
    cfg_bundletrack["ransac"]["inlier_dist"] = 0.01
    cfg_bundletrack["ransac"]["inlier_normal_angle"] = 20
    cfg_bundletrack["ransac"]["max_trans_neighbor"] = 0.02
    cfg_bundletrack["ransac"]["max_rot_deg_neighbor"] = 30
    cfg_bundletrack["ransac"]["max_trans_no_neighbor"] = 0.01
    cfg_bundletrack["ransac"]["max_rot_no_neighbor"] = 10
    cfg_bundletrack["p2p"]["max_dist"] = 0.02
    cfg_bundletrack["p2p"]["max_normal_angle"] = 45
    cfg_track_dir = f"{out_folder}/config_bundletrack.yml"
    yaml.dump(cfg_bundletrack, open(cfg_track_dir, "w"))

    cfg_nerf = yaml.load(open(f"{code_dir}/config.yml", "r"), Loader=yaml.Loader)
    cfg_nerf["continual"] = True
    cfg_nerf["trunc_start"] = 0.01
    cfg_nerf["trunc"] = 0.01
    cfg_nerf["mesh_resolution"] = 0.005
    cfg_nerf["down_scale_ratio"] = 1
    cfg_nerf["fs_sdf"] = 0.1
    cfg_nerf["far"] = cfg_bundletrack["depth_processing"]["zfar"]
    cfg_nerf["datadir"] = f"{cfg_bundletrack['debug_dir']}/nerf_with_bundletrack_online"
    cfg_nerf["notes"] = ""
    cfg_nerf["expname"] = "nerf_with_bundletrack_online"
    cfg_nerf["save_dir"] = cfg_nerf["datadir"]
    cfg_nerf["backend"] = "nerf"
    # ---------------------------------------------------------------------

    def factory(cfg):
        cfg_nerf_dir = f"{out_folder}/config_nerf.yml"
        yaml.dump(cfg, open(cfg_nerf_dir, "w"))
        return BundleSdf(cfg_track_dir=cfg_track_dir, cfg_nerf_dir=cfg_nerf_dir,
                         start_nerf_keyframes=5, use_gui=False)

    tracker = apply_feedback_condition(args, cfg_nerf, factory)

    reader = YcbineoatReader(video_dir=args.video_dir, shorter_side=480,
                             mask_dir=args.mask_dir)
    for i in range(0, len(reader.color_files), args.stride):
        color = cv2.imread(reader.color_files[i])
        depth = reader.get_depth(i)
        H, W = depth.shape[:2]
        color = cv2.resize(color, (W, H), interpolation=cv2.INTER_NEAREST)
        depth = cv2.resize(depth, (W, H), interpolation=cv2.INTER_NEAREST)
        mask = reader.get_mask(i)
        mask = cv2.resize(mask, (W, H), interpolation=cv2.INTER_NEAREST)
        if cfg_bundletrack["erode_mask"] > 0:
            kernel = np.ones((cfg_bundletrack["erode_mask"],
                              cfg_bundletrack["erode_mask"]), np.uint8)
            mask = cv2.erode(mask.astype(np.uint8), kernel)
        tracker.run(color, depth, reader.K.copy(), reader.id_strs[i],
                    mask=mask, occ_mask=None, pose_in_model=np.eye(4))
    tracker.on_finish()
    args._n_step = cfg_nerf["n_step"]
    args._max_trans = cfg_nerf["max_trans"]
    args._max_rot = cfg_nerf["max_rot"]


def write_manifest(args, out_folder, t_start):
    try:
        git_rev = subprocess.check_output(
            ["git", "-C", code_dir, "rev-parse", "--short", "HEAD"], text=True
        ).strip()
    except Exception:  # noqa: BLE001
        git_rev = None
    with open(f"{out_folder}/ablation_manifest.json", "w") as f:
        json.dump({
            "feedback": args.feedback,
            "tracker": args.tracker,
            "dataset": args.dataset,
            "video_dir": args.video_dir,
            "mask_dir": args.mask_dir,
            "stride": args.stride,
            "git_rev": git_rev,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "n_step": args._n_step,
            "max_trans": args._max_trans,
            "max_rot": args._max_rot,
            "elapsed_s": time.time() - t_start,
        }, f, indent=2)


if __name__ == "__main__":
    main()
