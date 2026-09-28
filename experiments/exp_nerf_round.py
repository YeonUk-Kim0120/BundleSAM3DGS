"""EXP_BATCH_20260928 arm N: the original BundleSDF NeRF round replayed offline (run in the `bundlesdf` container; main code imported only).
Per round (bundlesdf.run_nerf, continual branch): first round compute_scene_bounds -> sc_factor x 0.7, translation, octree cloud = its
real-scale cloud; later rounds: previous cloud + new keyframes' back-projection (compute_scene_bounds_worker), 1 cm voxel, biggest DBSCAN
cluster.  preprocess_data, then a NEW NerfRunner with all keyframes of the round (fresh network + fresh PoseArray, first keyframe fixed),
train(), get_optimized_poses_in_real_world (first-keyframe anchored).  nerf config = the baseline run's round config (HO3D) or final config
(YCB); only datadir / save_dir / n_step / sc_factor / translation are set here.
Mode S: one round on the first K keyframes.  Mode I: 5 keyframes, then every 5 new keyframes (remainder included); existing keyframes start
from the previous round's corrected poses; a new keyframe starts at c2w_corr(prev) inv(c2w_trk(prev)) c2w_trk(new).
Input: logs/exp_batch_20260928/inputs/<file>.json ('consume' = input poses).  Output: <out>/result.json."""
from __future__ import annotations
import argparse, copy, glob, json, os, shutil, sys, time
from pathlib import Path
import numpy as np, cv2, yaml, torch
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from nerf_runner import NerfRunner  # noqa: E402
from tool import compute_scene_bounds, compute_scene_bounds_worker, find_biggest_cluster  # noqa: E402
from nerf_helpers import preprocess_data  # noqa: E402
from Utils import get_optimized_poses_in_real_world, glcam_in_cvcam, toOpen3dCloud  # noqa: E402

BASE = Path("/home/kist/Desktop/BundleSDF_baseline_outputs")
VOX_RES = 0.01


def base_config(ds, seq):
    root = BASE / f"full_eval_sam2_{ds}" / ds / seq
    rounds = sorted(glob.glob(str(root / "*/nerf/config.yml")))
    path = rounds[-1] if rounds else str(root / "final/nerf/config.yml")
    cfg = yaml.safe_load(open(path))
    for k in ("sc_factor", "translation"): cfg.pop(k, None)
    return cfg, path


def load_frames(run_dir, ids, mask_dir=None, depth_dir=None):
    """mask_dir / depth_dir: unit-check diagnostic only (e.g. dataset gt_mask + raw depth in mm); default = the run's saved keyframe files."""
    rgbs, depths, masks = [], [], []
    for f in ids:
        rgbs.append(cv2.cvtColor(cv2.imread(f"{run_dir}/color/{f}.png"), cv2.COLOR_BGR2RGB))
        depths.append(cv2.imread(f"{depth_dir}/{f}.png" if depth_dir else f"{run_dir}/depth_filtered/{f}.png", -1).astype(np.float64) / 1e3)
        masks.append(cv2.imread(f"{mask_dir}/{f}.png" if mask_dir else f"{run_dir}/mask/{f}.png", -1))
    return rgbs, depths, masks


class NerfRounds:
    def __init__(self, cfg, K, work):
        self.cfg0 = cfg; self.K = K; self.work = Path(work); self.sc = self.tr = None; self.prev_pcd = None; self.tf = None

    def round(self, rgbs, depths, masks, cam_in_obs, n_new, n_step):
        """cam_in_obs: [N,4,4] metric OpenCV c2w of all keyframes of the round (object frame).  Returns refined [N,4,4]."""
        t0 = time.time(); torch.cuda.reset_peak_memory_stats()
        cfg = copy.deepcopy(self.cfg0); d = self.work / "round"; shutil.rmtree(d, ignore_errors=True); d.mkdir(parents=True)
        cfg["datadir"] = cfg["save_dir"] = str(d); cfg["n_step"] = int(n_step)
        glcam = np.asarray(cam_in_obs, dtype=np.float64) @ glcam_in_cvcam
        if self.sc is None:
            sc, tr, pcd_real, _ = compute_scene_bounds(None, glcam, self.K, use_mask=True, base_dir=str(d), rgbs=np.array(rgbs), depths=np.array(depths), masks=np.array(masks),
                                                         eps=cfg["dbscan_eps"], min_samples=cfg["dbscan_eps_min_samples"])
            self.sc = float(sc) * 0.7; self.tr = np.asarray(tr)
            self.tf = np.eye(4); self.tf[:3, 3] = self.tr; t1 = np.eye(4); t1[:3, :3] *= self.sc; self.tf = t1 @ self.tf
            pcd_all = pcd_real
        else:
            pcd_all = copy.deepcopy(self.prev_pcd)
            for i in range(len(rgbs) - n_new, len(rgbs)):
                pts, colors = compute_scene_bounds_worker(None, self.K, glcam[i], use_mask=True, rgb=rgbs[i], depth=depths[i], mask=masks[i])
                pcd_all += toOpen3dCloud(pts, colors)
            pcd_all = pcd_all.voxel_down_sample(VOX_RES)
            _, keep = find_biggest_cluster(np.asarray(pcd_all.points), eps=cfg["dbscan_eps"], min_samples=cfg["dbscan_eps_min_samples"])
            pcd_all = pcd_all.select_by_index(np.arange(len(np.asarray(pcd_all.points)))[keep])
        cfg["sc_factor"] = self.sc; cfg["translation"] = self.tr
        pcd_norm = copy.deepcopy(pcd_all); pcd_norm.transform(self.tf)
        r, dp, m, _, poses = preprocess_data(np.array(rgbs).copy(), np.array(depths).copy(), np.array(masks).copy(), normal_maps=None, poses=glcam.copy(), sc_factor=self.sc, translation=self.tr)
        nerf = NerfRunner(cfg, r, dp, m, normal_maps=None, poses=poses, K=self.K, occ_masks=None, build_octree_pcd=pcd_norm)
        nerf.train()
        opt, _ = get_optimized_poses_in_real_world(poses, nerf.models["pose_array"], self.sc, self.tr)
        self.prev_pcd = pcd_all.voxel_down_sample(VOX_RES)
        stats = dict(seconds=time.time() - t0, peak_mem_gb=torch.cuda.max_memory_allocated() / 1e9, n_frames=len(rgbs), sc_factor=self.sc)
        del nerf; torch.cuda.empty_cache(); shutil.rmtree(d, ignore_errors=True)
        return np.asarray(opt, dtype=np.float64), stats


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--input", type=Path, required=True); ap.add_argument("--mode", choices=("S", "I"), required=True)
    ap.add_argument("--K", default="last", help="mode S: keyframe count (first cycle with >= K keyframes) or 'last'"); ap.add_argument("--n-step", type=int, default=500)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--mask-dir", default=None, help="unit diagnostic only: keyframe masks from this folder"); ap.add_argument("--depth-dir", default=None, help="unit diagnostic only: depth (mm png) from this folder")
    a = ap.parse_args()
    inp = json.load(open(a.input)); a.out.mkdir(parents=True, exist_ok=False)
    ids = inp["kf_ids"]; P_trk = [np.array(p) for p in inp["consume"]]; K_cam = np.loadtxt(f"{inp['run_dir']}/cam_K.txt").reshape(3, 3)
    cfg, cfg_path = base_config(inp["dataset"], inp["seq"]); runner = NerfRounds(cfg, K_cam, a.out)
    res = dict(input=str(a.input), mode=a.mode, n_step=a.n_step, config_source=cfg_path, mask_dir=a.mask_dir, depth_dir=a.depth_dir, rounds=[])
    if a.mode == "S":
        if a.K == "last": n = len(ids)
        else:
            k = int(a.K); lens = inp.get("cycle_lengths") or list(range(1, len(ids) + 1)); n = next((L for L in lens if L >= k), len(ids))
        n = min(n, len(ids)); rgbs, depths, masks = load_frames(inp["run_dir"], ids[:n], a.mask_dir, a.depth_dir)
        out, st = runner.round(rgbs, depths, masks, P_trk[:n], n, a.n_step)
        res["rounds"].append(dict(n=n, kf_ids=ids[:n], poses_in=[p.tolist() for p in P_trk[:n]], poses_out=out.tolist(), **st))
        print(f"[N] {inp['seq']} S K={n} steps {a.n_step}: {st['seconds']:.1f}s peak {st['peak_mem_gb']:.1f} GB", flush=True)
    else:
        rgbs, depths, masks = load_frames(inp["run_dir"], ids, a.mask_dir, a.depth_dir); cur = []; n = 0
        while n < len(ids):
            n_new = 5 if n == 0 else min(5, len(ids) - n)
            if n == 0: cur = [P_trk[i].copy() for i in range(5)]
            else:
                prev = n - 1; rel = cur[prev] @ np.linalg.inv(P_trk[prev])
                cur = cur + [rel @ P_trk[i] for i in range(n, n + n_new)]
            n += n_new
            out, st = runner.round(rgbs[:n], depths[:n], masks[:n], np.array(cur), n_new, a.n_step)
            res["rounds"].append(dict(n=n, poses_in=[p.tolist() for p in cur], poses_out=out.tolist(), **st)); cur = [p for p in out]
            print(f"[N] {inp['seq']} I round {len(res['rounds'])} n={n}: {st['seconds']:.1f}s", flush=True)
            json.dump(res, open(a.out / "result.partial.json", "w"))
        res["kf_ids"] = ids
    json.dump(res, open(a.out / "result.json", "w")); (a.out / "result.partial.json").unlink(missing_ok=True)


if __name__ == "__main__":
    main()
