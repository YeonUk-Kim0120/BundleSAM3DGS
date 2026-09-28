"""EXP_BATCH_20260928 GS arms of the joint re-fit round (gsplat container).  Main code imported only.
  G0  : online noop map at cycle K (rebuilt by replay, see --build-maps), keeps training, 1 view / step  = GaussianRunner.train (v1 pose deltas).
  G0B : same map, 8 distinct views / step (loss averaged over the 8 views, one optimizer step).
  G1  : NEW map = SAM3D prior surfels after the online Sim(3) alignment only (init_runner_like_online, 0 steps), no append, all surfels trainable.
  G3  : NEW map = TSDF fusion of the input poses (gaussian_global.tsdf_fuse, 2 mm / 1 cm) -> vertex colours by nearest back-projected keyframe
        pixel -> sample_surfels(20000, radius x0.75) like the prior -> initialize_from_prior path (0 steps), no prior, all surfels trainable.
All arms: every keyframe of the round is a view; fresh v1 PoseDeltas for the round (first view fixed, Adam lr 0.01 -> x0.1, inf-norm clip 0.1,
clamps 2 cm / 20 deg, from the runner config); refined poses = baked view poses.  Loss = the runner's training loss (copied step)."""
from __future__ import annotations
import argparse, copy, json, math, sys, time
from pathlib import Path
import numpy as np, torch
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "experiments")); sys.path.append(str(REPO / "BundleTrack/scripts"))
import exp_feedback_gradient_probe_gtmap as G  # noqa: E402
from exp_feedback_gradient_probe_gtmap import init_runner_like_online, load_cycles, SAM3D_ROOT  # noqa: E402
from gaussian_runner import GaussianFrame, GaussianRunner, SceneNormalization, load_gaussian_config  # noqa: E402
from prior_lifecycle import STATE_VERIFIED  # noqa: E402
from run_gaussian_incremental import load_frame  # noqa: E402

CFG = REPO / "config_gs_2dgs_1mm_lifecycle.yml"; OUT = REPO / "outputs/exp_batch_20260928"
VIEWS_PER_STEP = 8; TSDF_VOXEL = 0.002; TSDF_TRUNC = 0.01; N_SURFELS = 20000


class JointRunner(GaussianRunner):
    """GaussianRunner + a multi-view training step.  skip_lifecycle: initialize without lifecycle transitions (fresh single-layer maps)."""
    skip_lifecycle = False

    def classify_lifecycle(self, frames, event, occ_masks=None):
        return None if self.skip_lifecycle else super().classify_lifecycle(frames, event, occ_masks)

    def _remove_contradicted(self):
        return 0 if self.skip_lifecycle else super()._remove_contradicted()

    def _view_loss(self, view, c2w, depth_weight):
        target = view.rgb.to(self.device, non_blocking=True)[None]; mask = view.mask.to(self.device, non_blocking=True)[None]
        K = view.K.to(self.device, non_blocking=True)[None]
        sh = min(self.total_steps // int(self.config["sh_degree_interval"]), int(self.config["sh_degree"]))
        result = self._rasterize(K=K, c2w=c2w, width=view.width, height=view.height, sh_degree=sh, render_mode="RGB+ED" if depth_weight > 0 else "RGB",
                                 absgrad=bool(self.strategy.absgrad))
        rendered = result.colors[..., :3]; mc = mask[..., None].to(rendered.dtype); den = mc.sum().clamp_min(1.0) * 3.0
        l1 = (torch.abs(rendered - target) * mc).sum() / den; dssim = self._masked_dssim(rendered, target, mask); w = float(self.config["ssim_weight"])
        loss = (1.0 - w) * l1 + w * dssim
        if depth_weight > 0: loss = loss + depth_weight * self._masked_depth_loss(result, view, mask)
        nw = float(self.config["normal_consistency_weight"])
        if result.normals is not None and nw > 0 and self.total_steps >= int(self.config["normal_consistency_start_step"]):
            wm = result.alpha[..., 0] * mask.to(result.alpha.dtype); cons = 1.0 - (result.normals * result.surf_normals).sum(dim=-1)
            loss = loss + nw * ((cons * wm).sum() / wm.sum().clamp_min(1.0))
        dw = float(self.config["distortion_weight"])
        if result.distort is not None and dw > 0 and self.total_steps >= int(self.config["distortion_start_step"]):
            mf = mask.to(result.distort.dtype); loss = loss + dw * ((result.distort[..., 0] * mf).sum() / mf.sum().clamp_min(1.0))
        return loss, result.info

    def train_multi(self, steps, views_per_step):
        """Copy of GaussianRunner.train with `views_per_step` distinct views per optimizer step (loss averaged) and pose deltas on."""
        if steps <= 0: return None, None
        self._require_gsplat()
        sched = torch.optim.lr_scheduler.ExponentialLR(self.optimizers["means"], gamma=0.01 ** (1.0 / float(steps)))
        frozen_rows = None
        if self.lifecycle_enabled and self.lifecycle_fields is not None:
            frozen = self.lifecycle_fields.state != STATE_VERIFIED
            if bool(frozen.any()): frozen_rows = frozen.to(self.device)
        fb = self.config["pose_feedback"]; deltas = self._new_pose_deltas(len(self.views), fix_first=bool(fb["fix_first_view"]))
        popt, psched = self._pose_optimizer(deltas, float(fb["lr"]), steps); depth_weight = float(self.config["depth_loss_weight"])
        k = min(views_per_step, len(self.views)); first = final = None
        for _ in range(steps):
            strategy_step = self.strategy_step
            idx = torch.randperm(len(self.views), generator=self._generator)[:k].tolist()
            for o in self.optimizers.values(): o.zero_grad(set_to_none=True)
            popt.zero_grad(set_to_none=True); total = 0.0; info = None
            for vi in idx:
                view = self.views[vi]; c2w = deltas.matrices([vi]) @ view.c2w_normalized.to(self.device, non_blocking=True)[None]
                loss, info = self._view_loss(view, c2w, depth_weight)
                self.strategy.step_pre_backward(self.splats, self.optimizers, self.strategy_state, strategy_step, info)
                if not torch.isfinite(loss): raise FloatingPointError("Non-finite joint loss")
                (loss / k).backward(); total += float(loss.detach()) / k
            if frozen_rows is not None:
                for p in self.splats.values():
                    if p.grad is not None: p.grad[frozen_rows] = 0
            for o in self.optimizers.values(): o.step()
            sched.step(); self._pose_step(deltas, popt, psched); self._strategy_post_backward(strategy_step, info)
            self.strategy_step += 1; self.total_steps += 1; final = total; first = total if first is None else first
        self._bake_pose_deltas(deltas, self.views, event="joint_round")
        return first, final


# ------------------------------------------------------------------ helpers
def orth(T):
    T = np.asarray(T, dtype=np.float64).copy(); u, _, vt = np.linalg.svd(T[:3, :3]); R = u @ vt
    if np.linalg.det(R) < 0: u[:, -1] *= -1; R = u @ vt
    T[:3, :3] = R; return T


def prior_paths(ds, seq):
    sub = "output_YCBInEOAT_sam2mask_mesh" if ds == "ycb" else "output_HO3D_sam2mask_mesh"
    return {k: SAM3D_ROOT / sub / f"{seq}_{v}" for k, v in (("mesh_npz", "mesh_depth.npz"), ("pose_json", "mesh_depth.json"), ("gaussian_ply", "splat_depth.ply"))}


def frames_for(run_dir, ids, poses):
    K = np.loadtxt(Path(run_dir) / "cam_K.txt").reshape(3, 3).astype(np.float32); out = []
    for fid, P in zip(ids, poses):
        f = load_frame(Path(run_dir), fid, K, np.asarray(P, dtype=np.float32)); out.append(GaussianFrame(fid, f.rgb, f.depth, f.mask, f.K, f.c2w_cv).validated())
    return out


def with_class(cls, fn, *a, **k):
    saved = G.GaussianRunner; G.GaussianRunner = cls
    try: return fn(*a, **k)
    finally: G.GaussianRunner = saved


def all_verified(r):
    if r.lifecycle_fields is not None: r.lifecycle_fields.state[:] = STATE_VERIFIED


class FreshG1(JointRunner):
    skip_lifecycle = True


def build_g1(inp, frames, device, cache=None):
    """Prior-only map.  cache (mode I): reuse the first round's aligned surfels (alignment done once on keyframe 0)."""
    if cache is None or "surfels_cv" not in cache:
        r = with_class(FreshG1, init_runner_like_online, CFG, device, prior_paths(inp["dataset"], inp["seq"]), frames, frames[0].c2w_cv.astype(np.float32), 0, "sam3d")
        if cache is not None:
            cache["norm"] = (r.normalization.scale, r.normalization.translation.copy()); cache["surfels_state"] = copy.deepcopy({k: v.detach().cpu() for k, v in r.splats.items()})
            cache["surfels_cv"] = True; cache["lifecycle"] = r.lifecycle_fields
    else:
        rc = load_gaussian_config(CFG); rc["device"] = device; sc, tr = cache["norm"]
        r = FreshG1(rc, SceneNormalization(scale=sc, translation=tr), device=device); r._set_splats(copy.deepcopy(cache["surfels_state"]))
        r.lifecycle_fields = cache["lifecycle"].keep(torch.ones(len(cache["lifecycle"]), dtype=torch.bool, device=cache["lifecycle"].state.device))
        r.views = [r._prepare_view(f) for f in frames]; r.observed_points_metric = np.zeros((0, 3), np.float32); r.observed_colors = np.zeros((0, 3), np.float32)
        r._reset_optimization_state(float(r.config["initial_lr_scale"]), initial=True)
    all_verified(r); return r


def build_g3(inp, frames, device):
    """TSDF (tsdf_fuse, geometry only) -> vertex colours by nearest back-projected keyframe pixel -> sample_surfels like the prior -> runner."""
    import trimesh
    from scipy.spatial import cKDTree
    from gaussian_global import tsdf_fuse
    from sam3d_prior import MeshPrior, sample_surfels, SurfelSet
    mesh = tsdf_fuse(((f.rgb if f.rgb.dtype == np.uint8 else (f.rgb * 255).astype(np.uint8), np.where(f.mask, f.depth, 0.0).astype(np.float32), f.K, f.c2w_cv) for f in frames), TSDF_VOXEL, TSDF_TRUNC)
    pts, cols = [], []
    for f in frames:
        v, u = np.nonzero(f.mask & (f.depth > 0.1)); z = f.depth[v, u]; K = f.K
        pc = np.stack([(u - K[0, 2]) / K[0, 0] * z, (v - K[1, 2]) / K[1, 1] * z, z], -1); c2w = f.c2w_cv.astype(np.float64)
        sel = slice(None, None, 4); pts.append(pc[sel] @ c2w[:3, :3].T + c2w[:3, 3]); rgb = f.rgb[v, u][sel]; cols.append(rgb.astype(np.float32) / (255.0 if f.rgb.dtype == np.uint8 else 1.0))
    P = np.concatenate(pts); C = np.concatenate(cols); _, nn = cKDTree(P).query(np.asarray(mesh.vertices), k=1, workers=-1)
    prior = MeshPrior(vertices=torch.from_numpy(np.asarray(mesh.vertices, dtype=np.float32)), faces=torch.from_numpy(np.asarray(mesh.faces, dtype=np.int64)),
                      vertex_colors=torch.from_numpy(C[nn].astype(np.float32))).validated()
    s = sample_surfels(prior, N_SURFELS, seed=0, radius_multiplier=0.75)       # object frame (metric)
    c0 = frames[0].c2w_cv.astype(np.float64); R0 = torch.from_numpy(c0[:3, :3].astype(np.float32)); t0 = torch.from_numpy(c0[:3, 3].astype(np.float32))
    s_cv = SurfelSet(means=(s.means - t0[None]) @ R0, normals=s.normals @ R0, radii=s.radii, colors=s.colors, opacities=s.opacities)
    m = s.means.numpy().astype(np.float64); centre = m.mean(0); radius = float(np.linalg.norm(m - centre, axis=1).max())
    rc = load_gaussian_config(CFG); rc["device"] = device
    r = FreshG1(rc, SceneNormalization(scale=1.0 / max(radius * 1.2, 1e-6), translation=-centre), device=device)
    r.initialize_from_prior(s_cv, c0.astype(np.float32), frames, train_steps=0); all_verified(r)
    return r, dict(tsdf_vertices=int(len(mesh.vertices)), tsdf_faces=int(len(mesh.faces)))


def run_round(arm, inp, ids, poses, steps, device, g0_ckpt=None, cache=None):
    """One joint round.  Returns refined metric c2w [N,4,4] and stats."""
    torch.cuda.reset_peak_memory_stats(); t0 = time.time(); extra = {}
    frames = frames_for(inp["run_dir"], ids, poses)
    if arm in ("G0", "G0B"):
        r = JointRunner.load_checkpoint(g0_ckpt, device=device)
        vids = [v.frame_id for v in r.views]; assert vids == list(ids), f"map views {len(vids)} != round keyframes {len(ids)}"
        r.refresh_view_poses({fid: np.asarray(P, dtype=np.float32) for fid, P in zip(ids, poses)})
        r._reset_optimization_state(float(r.config["update_lr_scale"]), initial=False)
    elif arm == "G1": r = build_g1(inp, frames, device, cache)
    elif arm == "G3": r, extra = build_g3(inp, frames, device)
    else: raise ValueError(arm)
    n0 = int(r.num_gaussians); t_build = time.time() - t0
    if arm == "G0": first, final = r.train(steps, optimize_poses=True)
    else: first, final = r.train_multi(steps, VIEWS_PER_STEP)
    out = np.asarray(r.view_poses_metric(), dtype=np.float64)
    stats = dict(seconds=time.time() - t0, build_seconds=t_build, peak_mem_gb=torch.cuda.max_memory_allocated() / 1e9, gaussians_before=n0, gaussians_after=int(r.num_gaussians),
                 first_loss=first, final_loss=final, n_frames=len(ids), **extra)
    del r; torch.cuda.empty_cache()
    return out, stats


# ------------------------------------------------------------------ G0 maps: replay the noop online map (3rd batch R0 path) and checkpoint at K
def build_maps(inp, Ks, device, out_dir, pose_source="noop"):
    import json as _j
    run = Path(inp["run_dir"]); cycles = load_cycles(run); K_cam = np.loadtxt(run / "cam_K.txt").reshape(3, 3).astype(np.float32)
    gtmap = {f: g for f, g in zip(inp["kf_ids"], inp["gt"])}
    def pose(fid, trk): return np.asarray(gtmap[fid], dtype=np.float32) if pose_source == "gt" else trk.astype(np.float32)
    targets = {}
    for K in Ks:
        lens = [len(c["ids"]) for c in cycles]; ci = next((i for i, L in enumerate(lens) if (K == "last" and i == len(lens) - 1) or (K != "last" and L >= int(K))), len(lens) - 1)
        targets.setdefault(ci, []).append(K)
    r = None; consumed = 0; out_dir.mkdir(parents=True, exist_ok=True); made = {}
    for ci, cyc in enumerate(cycles):
        if ci > max(targets): break
        new = cyc["ids"][consumed:]
        frames = [GaussianFrame(fid, *(lambda f: (f.rgb, f.depth, f.mask, f.K, f.c2w_cv))(load_frame(run, fid, K_cam, pose(fid, cyc["before"][consumed + j])))).validated() for j, fid in enumerate(new)]
        if r is None: r = init_runner_like_online(CFG, device, prior_paths(inp["dataset"], inp["seq"]), frames, pose(cyc["ids"][0], cyc["before"][0]), 500, "sam3d")
        else:
            r.refresh_view_poses({v.frame_id: pose(v.frame_id, cyc["before"][i]) for i, v in enumerate(r.views)})
            r.update(frames, train_steps=500)
        consumed = len(cyc["ids"])
        if ci in targets:
            for K in targets[ci]:
                p = out_dir / f"K{K}.pt"; r.save_checkpoint(p); made[str(K)] = dict(path=str(p), cycle=cyc["dir"], views=len(r.views), gaussians=int(r.num_gaussians))
                print(f"[maps] {inp['seq']} K={K}: cycle {cyc['dir']} views {len(r.views)} gaussians {r.num_gaussians}", flush=True)
    _j.dump(made, open(out_dir / "maps.json", "w"), indent=1); return made


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--input", type=Path, required=True); ap.add_argument("--arm", choices=("G0", "G0B", "G1", "G3"))
    ap.add_argument("--mode", choices=("S", "I", "maps"), required=True); ap.add_argument("--K", default="last"); ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--maps-dir", type=Path, default=None); ap.add_argument("--map-poses", choices=("noop", "gt"), default="noop"); ap.add_argument("--Ks", nargs="+", default=["last"])
    ap.add_argument("--out", type=Path, required=True); ap.add_argument("--device", default="cuda:0"); a = ap.parse_args()
    inp = json.load(open(a.input))
    if a.mode == "maps":
        build_maps(inp, a.Ks, a.device, a.out, a.map_poses); return
    a.out.mkdir(parents=True, exist_ok=False); ids = inp["kf_ids"]; P_trk = [np.array(p) for p in inp["consume"]]
    res = dict(input=str(a.input), arm=a.arm, mode=a.mode, steps=a.steps, views_per_step=1 if a.arm == "G0" else VIEWS_PER_STEP, rounds=[])
    if a.mode == "S":
        if a.arm in ("G0", "G0B"):
            maps = json.load(open(a.maps_dir / "maps.json")); n = maps[str(a.K)]["views"]; ck = maps[str(a.K)]["path"]
        else:
            ck = None
            if a.K == "last": n = len(ids)
            else: n = next((L for L in inp["cycle_lengths"] if L >= int(a.K)), len(ids))
        out, st = run_round(a.arm, inp, ids[:n], P_trk[:n], a.steps, a.device, g0_ckpt=ck)
        res["rounds"].append(dict(n=n, kf_ids=ids[:n], poses_in=[p.tolist() for p in P_trk[:n]], poses_out=out.tolist(), **st))
        print(f"[{a.arm}] {inp['seq']} S K={n} steps {a.steps}: {st['seconds']:.1f}s peak {st['peak_mem_gb']:.1f} GB gaussians {st['gaussians_before']}->{st['gaussians_after']}", flush=True)
    else:
        cache = {} if a.arm == "G1" else None; cur = []; n = 0
        while n < len(ids):
            n_new = 5 if n == 0 else min(5, len(ids) - n)
            if n == 0: cur = [P_trk[i].copy() for i in range(5)]
            else:
                rel = cur[n - 1] @ np.linalg.inv(P_trk[n - 1]); cur = cur + [orth(rel @ P_trk[i]) for i in range(n, n + n_new)]
            n += n_new
            out, st = run_round(a.arm, inp, ids[:n], cur, a.steps, a.device, cache=cache)
            res["rounds"].append(dict(n=n, poses_in=[p.tolist() for p in cur], poses_out=out.tolist(), **st)); cur = [p for p in out]
            print(f"[{a.arm}] {inp['seq']} I round {len(res['rounds'])} n={n}: {st['seconds']:.1f}s", flush=True)
            json.dump(res, open(a.out / "result.partial.json", "w"))
        res["kf_ids"] = ids
    json.dump(res, open(a.out / "result.json", "w")); (a.out / "result.partial.json").unlink(missing_ok=True)


if __name__ == "__main__":
    main()
