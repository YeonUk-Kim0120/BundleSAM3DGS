"""EXP_BATCH_20260927 stage 1 (Q1): reference map x pose optimizer probe.  Imports experiments/exp_feedback_gradient_probe_gtmap.py
(GroundTruth, init_runner_like_online, perturbed_pose, gt_mesh_in_tracker_frame, probe_view, load_cycles, render_losses) unchanged.

References (map poses fixed, map only trained, schedule 500 / 500 like the online backend):
  R0 current rules, noop poses | A1/A2/A3 hygiene runner (experiments/hygiene_runner.py), noop poses | RG GT poses |
  RP prior only: SAM3D surfels right after the online Sim(3) alignment (no training, no append, no lifecycle transitions).
Probe (map frozen, one view): newest keyframe of every `--probe-every`-th cycle, before its append.  Start poses: tracker (noop consumption pose),
GT, GT+3 deg x 2 axes, GT+10 deg x 2 axes (perturbed_pose: object-centre rotation, fixed seeds, no translation).
Optimizers: O1 online loss (L1+DSSIM+depth, runner weights) Adam lr 0.01 x 3 | O2 same loss Adam lr 1e-3 x 200 (probe_view's pose-only path) |
  O3 point-to-surfel Gauss-Newton (brief 3.4) | O4 the same point-to-surfel objective, Adam lr 1e-3 x 200.
Per probe: rot / trans / d (mean GT-mesh-point displacement) before and after, objective c0 / c1, time, (O3) iterations, valid ratio, accepted.
R0 additionally records probe_view's gradient-direction cosines at the tracker pose.

  python3 experiments/exp_reference_optimizer_probe.py --dataset ho3d --seq AP12 --run-dir <noop run> --reference R0 --output-dir <dir>
  python3 experiments/exp_reference_optimizer_probe.py --unit-test gn --dataset ho3d --seq AP12 --run-dir <noop run> --output-dir <dir>
"""
from __future__ import annotations
import argparse, json, math, sys, time
from pathlib import Path
import numpy as np, torch
from scipy.spatial import cKDTree
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "experiments")); sys.path.append(str(REPO / "BundleTrack/scripts"))
import exp_feedback_gradient_probe_gtmap as G  # noqa: E402
from exp_feedback_gradient_probe_gtmap import (GroundTruth, init_runner_like_online, perturbed_pose, gt_mesh_in_tracker_frame, probe_view,  # noqa: E402
                                               load_cycles, render_losses, rot_deg, SAM3D_ROOT)
from gaussian_runner import GaussianFrame, GaussianRunner, se3_exp_batch  # noqa: E402
from prior_lifecycle import STATE_CONTRADICTED, STATE_SUSPECT, erode_mask  # noqa: E402
from run_gaussian_incremental import load_frame  # noqa: E402
from hygiene_runner import make_hygiene_runner  # noqa: E402

# ------------------------------------------------------------------ fixed values chosen for this batch (reported in the results doc)
P = dict(n_points=5000, erode_px=2, ref_min_opacity=0.1, pca_k=16, max_corr_m=0.02, mad_k=2.5, huber_m=0.002, damping=1e-4, clip_t_m=0.02, clip_r_rad=0.2,
         gn_max_iter=8, gn_stop_t_m=1e-4, gn_stop_r_rad=1e-3, gn_min_valid_ratio=0.15, o1=(0.01, 3), o2=(1e-3, 200), o4=(1e-3, 200), mesh_samples=20000)
STARTS = [("tracker", None, None), ("gt", 0.0, 0), ("gt3_0", 3.0, 0), ("gt3_1", 3.0, 1), ("gt10_0", 10.0, 0), ("gt10_1", 10.0, 1)]


# ------------------------------------------------------------------ small geometry helpers
def skew(w): return np.array([[0, -w[2], w[1]], [w[2], 0, -w[0]], [-w[1], w[0], 0]])


def so3_exp(w):
    th = np.linalg.norm(w)
    if th < 1e-12: return np.eye(3) + skew(w)
    k = skew(w / th); return np.eye(3) + math.sin(th) * k + (1 - math.cos(th)) * k @ k


def left_update(T, delta):                    # T <- exp(delta) T  with delta = [t, w] (metric object frame)
    R = so3_exp(delta[3:]); out = np.eye(4); out[:3, :3] = R @ T[:3, :3]; out[:3, 3] = R @ T[:3, 3] + delta[:3]; return out


def orth(T):
    """Project the rotation to SO(3) (saved poses are float32: |RtR - I| ~ 2e-7, which inflates arccos-based angles by ~0.03 deg)."""
    T = np.asarray(T, dtype=np.float64).copy(); u, _, vt = np.linalg.svd(T[:3, :3]); R = u @ vt
    if np.linalg.det(R) < 0: u[:, -1] *= -1; R = u @ vt
    T[:3, :3] = R; return T


def rot_angle(R1, R2):
    """Geodesic angle via 2 asin(|R1 - R2|_F / (2 sqrt 2)) — accurate near 0 (arccos of the trace is not)."""
    return math.degrees(2.0 * math.asin(min(1.0, float(np.linalg.norm(R1 - R2)) / (2.0 * math.sqrt(2.0)))))


def pose_errors(c2w, gt_c2w, mesh_pts):
    A = c2w @ np.linalg.inv(gt_c2w)
    d = float(np.linalg.norm(mesh_pts @ A[:3, :3].T + A[:3, 3] - mesh_pts, axis=1).mean() * 1000.0)
    return rot_angle(c2w[:3, :3], gt_c2w[:3, :3]), float(np.linalg.norm(c2w[:3, 3] - gt_c2w[:3, 3]) * 1000.0), d


# ------------------------------------------------------------------ point-to-surfel reference (O3 / O4)
class SurfelReference:
    """Gaussian centres (metric) with opacity >= 0.1, CONTRADICTED / SUSPECT excluded; normals by k=16 nearest-centre PCA (not the quats)."""
    def __init__(self, runner):
        with torch.no_grad():
            mu = runner.normalization.metric_points(runner.splats["means"].detach().cpu().numpy()).astype(np.float64)
            op = torch.sigmoid(runner.splats["opacities"].detach()).cpu().numpy()
        keep = op >= P["ref_min_opacity"]
        if runner.lifecycle_fields is not None:
            st = runner.lifecycle_fields.state.cpu().numpy(); keep &= (st != STATE_CONTRADICTED) & (st != STATE_SUSPECT)
        self.mu = mu[keep]; self.tree = cKDTree(self.mu); self.n_ref = len(self.mu)
        _, nb = self.tree.query(self.mu, k=min(P["pca_k"], len(self.mu)), workers=-1)
        X = self.mu[nb] - self.mu[nb].mean(1, keepdims=True); C = np.einsum("nki,nkj->nij", X, X)
        _, V = np.linalg.eigh(C); self.n = V[:, :, 0]                                   # eigenvector of the smallest eigenvalue

    def residuals(self, x):
        d, idx = self.tree.query(x, k=1, distance_upper_bound=P["max_corr_m"], workers=-1)
        ok = np.isfinite(d); i = idx[ok]; r = np.einsum("ij,ij->i", self.n[i], x[ok] - self.mu[i])
        return ok, r, self.n[i]


def robust(r):
    a = np.abs(r)
    if len(a) == 0: return np.zeros(0, bool), np.zeros(0)
    med = np.median(a); mad = np.median(np.abs(a - med)); keep = a <= med + P["mad_k"] * mad
    w = np.where(a <= P["huber_m"], 1.0, P["huber_m"] / np.maximum(a, 1e-12)); return keep, w


def surfel_cost(ref, T, p):
    x = p @ T[:3, :3].T + T[:3, 3]; ok, r, n = ref.residuals(x)
    keep, w = robust(r)
    if keep.sum() < 6: return np.inf, 0.0
    return float((w[keep] * r[keep] ** 2).sum() / w[keep].sum()), float(keep.sum() / len(p))


def gn_refine(ref, T0, p):
    """Brief 3.4.  Returns (T, c0, c1, iterations, valid_ratio, accepted)."""
    c0, ratio0 = surfel_cost(ref, T0, p); T = T0.copy(); cost = c0; it = 0; ratio = ratio0
    for it in range(1, P["gn_max_iter"] + 1):
        x = p @ T[:3, :3].T + T[:3, 3]; ok, r, n = ref.residuals(x); keep, w = robust(r)
        if keep.sum() < 6: break
        xk = x[ok][keep]; nk = n[keep]; rk = r[keep]; wk = w[keep]
        J = np.concatenate((nk, np.cross(xk, nk)), 1); H = J.T @ (wk[:, None] * J) + P["damping"] * np.eye(6); g = J.T @ (wk * rk)
        delta = -np.linalg.solve(H, g)
        t, om = delta[:3], delta[3:]
        if np.linalg.norm(t) > P["clip_t_m"]: t = t * P["clip_t_m"] / np.linalg.norm(t)
        if np.linalg.norm(om) > P["clip_r_rad"]: om = om * P["clip_r_rad"] / np.linalg.norm(om)
        delta = np.concatenate((t, om)); Tn = left_update(T, delta); cn, rn = surfel_cost(ref, Tn, p)
        if cn > cost:
            delta = delta / 2; Tn = left_update(T, delta); cn, rn = surfel_cost(ref, Tn, p)
            if cn > cost: break
        T, cost, ratio = Tn, cn, rn
        if np.linalg.norm(delta[:3]) < P["gn_stop_t_m"] and np.linalg.norm(delta[3:]) < P["gn_stop_r_rad"]: break
    accepted = bool(cost < c0 and ratio >= P["gn_min_valid_ratio"])
    return (T if accepted else T0.copy()), c0, (cost if accepted else c0), it, ratio, accepted


def adam_surfel(ref, T0, p, lr, steps):
    """O4: the O3 objective (weighted mean r^2 with the same correspondence / outlier / Huber rule, recomputed every step) with Adam."""
    delta = torch.zeros(6, dtype=torch.float64, requires_grad=True); opt = torch.optim.Adam([delta], lr=lr)
    R0 = torch.from_numpy(T0[:3, :3]); t0 = torch.from_numpy(T0[:3, 3]); pt = torch.from_numpy(p)
    for _ in range(steps):
        opt.zero_grad()
        w = delta[3:]; K = torch.zeros(3, 3, dtype=torch.float64)
        K = torch.stack((torch.stack((K[0, 0], -w[2], w[1])), torch.stack((w[2], K[1, 1], -w[0])), torch.stack((-w[1], w[0], K[2, 2]))))
        R = torch.linalg.matrix_exp(K); x = pt @ (R @ R0).T + (R @ t0 + delta[:3])
        ok, r_np, n = ref.residuals(x.detach().numpy())
        keep, wt = robust(r_np)
        if keep.sum() < 6: break
        idx = np.nonzero(ok)[0][keep]; mu = torch.from_numpy(ref.mu[ref.tree.query(x.detach().numpy()[idx], k=1)[1]]); nn = torch.from_numpy(n[keep])
        r = ((x[idx] - mu) * nn).sum(1); wk = torch.from_numpy(wt[keep]); loss = (wk * r ** 2).sum() / wk.sum()
        loss.backward(); opt.step()
    T = left_update(T0, delta.detach().numpy()); c0, _ = surfel_cost(ref, T0, p); c1, ratio = surfel_cost(ref, T, p)
    return T, c0, c1, ratio


# ------------------------------------------------------------------ O1 / O2: online loss, pose only (probe_view's pose-only loop, with costs)
def pose_only(runner, view, c2w_metric, lr, steps):
    norm = runner.normalization; dev = runner.device
    c2w_norm = torch.from_numpy(norm.normalize_c2w(c2w_metric)).float().to(dev)
    d2 = torch.zeros(6, device=dev, requires_grad=True); opt = torch.optim.Adam([d2], lr=lr, eps=1e-15); c0 = None
    for it in range(steps):
        opt.zero_grad(set_to_none=True)
        total = render_losses(runner, view, se3_exp_batch(d2[None])[0] @ c2w_norm)["total"]
        if it == 0: c0 = float(total)
        if not total.requires_grad: break
        total.backward()
        for prm in runner.splats.values(): prm.grad = None
        torch.nn.utils.clip_grad_norm_([d2], max_norm=0.1, norm_type=float("inf")); opt.step()
    with torch.no_grad():
        Tn = se3_exp_batch(d2.detach()[None])[0]; c1 = float(render_losses(runner, view, Tn @ c2w_norm)["total"])
    if c0 is None: c0 = c1
    return norm.metric_c2w(Tn.cpu().numpy().astype(np.float64) @ c2w_norm.cpu().numpy().astype(np.float64)).astype(np.float64), c0, c1


def view_points(frame, config, rng):
    depth = np.asarray(frame.depth, dtype=np.float64); K = frame.K.astype(np.float64)
    m = erode_mask(torch.from_numpy(np.asarray(frame.mask).astype(bool)), P["erode_px"]).numpy()
    ok = m & np.isfinite(depth) & (depth > float(config["min_depth"])) & (depth < float(config["max_depth"]))
    v, u = np.nonzero(ok)
    if len(u) > P["n_points"]: sel = rng.choice(len(u), P["n_points"], replace=False); u, v = u[sel], v[sel]
    d = depth[v, u]; return np.stack([(u - K[0, 2]) / K[0, 0] * d, (v - K[1, 2]) / K[1, 1] * d, d], -1)


# ------------------------------------------------------------------ reference construction
class PriorOnlyRunner(GaussianRunner):
    """RP: initialize_from_prior with no lifecycle transitions and no removal (train_steps 0)."""
    def classify_lifecycle(self, frames, event, occ_masks=None): return None
    def _remove_contradicted(self): return 0


def build_first(reference, cfg, device, prior_paths, frames, first_pose, init_steps):
    cls = {"R0": GaussianRunner, "RG": GaussianRunner, "RP": PriorOnlyRunner}.get(reference) or make_hygiene_runner(reference)
    saved = G.GaussianRunner; G.GaussianRunner = cls
    try:
        return init_runner_like_online(cfg, device, prior_paths, frames, first_pose, 0 if reference == "RP" else init_steps, "sam3d")
    finally:
        G.GaussianRunner = saved


# ------------------------------------------------------------------ main probe loop
def run(a):
    seq = a.seq; video = REPO / ("datasets/HO3D_v3/evaluation" if a.dataset == "ho3d" else "datasets/YCBInEOAT") / seq
    sub = "output_YCBInEOAT_sam2mask_mesh" if a.dataset == "ycb" else "output_HO3D_sam2mask_mesh"
    prior_paths = {"mesh_npz": SAM3D_ROOT / sub / f"{seq}_mesh_depth.npz", "pose_json": SAM3D_ROOT / sub / f"{seq}_mesh_depth.json", "gaussian_ply": SAM3D_ROOT / sub / f"{seq}_splat_depth.ply"}
    K_cam = np.loadtxt(a.run_dir / "cam_K.txt").reshape(3, 3).astype(np.float32)
    gt = GroundTruth(a.dataset, video, a.run_dir); cycles = load_cycles(a.run_dir); mesh_pts = gt_mesh_in_tracker_frame(a.dataset, seq, gt, n_samples=P["mesh_samples"])
    a.output_dir.mkdir(parents=True, exist_ok=False)
    json.dump({**{k: str(v) for k, v in vars(a).items()}, "P": P, "starts": STARTS, "cycles": len(cycles)}, open(a.output_dir / "args.json", "w"), indent=1)
    use_gt = a.reference == "RG"
    def map_pose(fid, tracker_pose):
        if not use_gt: return tracker_pose.astype(np.float32)
        g = gt.gt_c2w(fid); return None if g is None else g.astype(np.float32)
    rows, cosrows, hyg = [], [], []; runner = None; consumed = 0; t0 = time.time(); n_probe_cycles = 0
    for ci, cyc in enumerate(cycles):
        if a.max_cycles and ci > a.max_cycles: break
        K = len(cyc["ids"]); new_ids = cyc["ids"][consumed:]; frames = []
        for j, fid in enumerate(new_ids):
            pose = map_pose(fid, cyc["before"][consumed + j])
            if pose is None: continue
            f = load_frame(a.run_dir, fid, K_cam, pose); frames.append(GaussianFrame(fid, f.rgb, f.depth, f.mask, f.K, f.c2w_cv).validated())
        if runner is None:
            runner = build_first(a.reference, a.runner_config, a.device, prior_paths, frames, map_pose(cyc["ids"][0], cyc["before"][0]), a.initial_steps)
            consumed = K; continue
        if a.reference != "RP":
            runner.refresh_view_poses({v.frame_id: map_pose(v.frame_id, cyc["before"][i]) for i, v in enumerate(runner.views) if map_pose(v.frame_id, cyc["before"][i]) is not None})
        if ci % a.probe_every == 0 and new_ids:
            fid = new_ids[-1]; gt_c2w = gt.gt_c2w(fid)
            if gt_c2w is not None:
                gt_c2w = orth(gt_c2w); n_probe_cycles += 1; tracker_pose = orth(cyc["before"][consumed + len(new_ids) - 1])
                f = load_frame(a.run_dir, fid, K_cam, tracker_pose.astype(np.float32)); frame = GaussianFrame(fid, f.rgb, f.depth, f.mask, f.K, f.c2w_cv).validated()
                view = runner._prepare_view(frame); seed = int(fid) if fid.isdigit() else ci
                pts = view_points(frame, runner.config, np.random.default_rng(seed % (2**32)))
                if a.reference == "R0":
                    pv = probe_view(runner, view, tracker_pose, gt, fid, "none", 0.5, 2.0, 0, 0.01)
                    cosrows.append({"cycle": cyc["dir"], "K": K, "frame_id": fid, **{f"{k}_{m}": (v[m] if v else None) for k, v in pv["losses"].items() for m in ("cos_rot", "cos_trans")},
                                    "err_rot_before": pv["err_rot_before"], "true_rot_deg": pv["true_rot_deg"], "true_trans_mm": pv["true_trans_mm"]})
                t_ref = time.time(); ref = SurfelReference(runner); t_ref = time.time() - t_ref
                req = [prm.requires_grad for prm in runner.splats.values()]
                for prm in runner.splats.values(): prm.requires_grad_(False)      # map frozen; gradients only for the pose delta (same delta gradient)
                try:
                    for name, deg, si in STARTS:
                        if name == "tracker": start = tracker_pose
                        elif deg == 0.0: start = gt_c2w.astype(np.float64)
                        else: start = perturbed_pose(gt_c2w, runner.normalization, deg, 0.0, (int(fid) if fid.isdigit() else ci) * 100 + int(deg * 10) + si).astype(np.float64)
                        e0 = pose_errors(start, gt_c2w, mesh_pts)
                        for opt in a.optimizers:
                            tt = time.time(); extra = {}
                            if opt == "O1": est, c0, c1 = pose_only(runner, view, start, *P["o1"])
                            elif opt == "O2": est, c0, c1 = pose_only(runner, view, start, *P["o2"])
                            elif opt == "O3":
                                est, c0, c1, iters, ratio, acc = gn_refine(ref, start, pts); extra = dict(iters=iters, valid_ratio=ratio, accepted=acc)
                            else:
                                est, c0, c1, ratio = adam_surfel(ref, start, pts, *P["o4"]); extra = dict(valid_ratio=ratio)
                            e1 = pose_errors(est, gt_c2w, mesh_pts)
                            rows.append(dict(seq=seq, reference=a.reference, cycle=cyc["dir"], ci=ci, K=K, frame_id=fid, start=name, optimizer=opt,
                                             rot0=e0[0], trans0=e0[1], d0=e0[2], rot1=e1[0], trans1=e1[1], d1=e1[2], c0=c0, c1=c1, time_s=time.time() - tt,
                                             n_points=len(pts), n_ref=ref.n_ref, t_ref_s=t_ref, **extra, est=est.reshape(-1).tolist()))
                finally:
                    for prm, rg in zip(runner.splats.values(), req): prm.requires_grad_(rg)
                last = [r for r in rows if r["ci"] == ci and r["start"] == "tracker"]
                print(f"[refopt] {a.reference} {seq} K={K} {fid} d0 {last[0]['d0']:.2f} mm -> " + " ".join(f"{r['optimizer']} {r['d1']:.2f}" for r in last)
                      + f" | ref {ref.n_ref} ({time.time() - t0:.0f}s)", flush=True)
        if a.reference != "RP" and frames:
            runner.update(frames, train_steps=a.update_steps)
            if hasattr(runner, "hyg_log") and runner.hyg_log: hyg.append(runner.hyg_log[-1])
        consumed = K
        if n_probe_cycles and ci % 10 == 0:
            json.dump(rows, open(a.output_dir / "rows.json", "w")); json.dump(cosrows, open(a.output_dir / "cos_rows.json", "w")); json.dump(hyg, open(a.output_dir / "hygiene_log.json", "w"))
    json.dump(rows, open(a.output_dir / "rows.json", "w")); json.dump(cosrows, open(a.output_dir / "cos_rows.json", "w")); json.dump(hyg, open(a.output_dir / "hygiene_log.json", "w"))
    if runner is not None and a.reference != "RP":
        json.dump(runner.lifecycle_log[-5:], open(a.output_dir / "lifecycle_tail.json", "w"), indent=1, default=str)
    json.dump(dict(done=True, seconds=time.time() - t0, probes=len(rows), probe_cycles=n_probe_cycles, gaussians=int(runner.num_gaussians) if runner else None),
              open(a.output_dir / "done.json", "w"), indent=1)
    print(f"[refopt] done {a.reference} {seq}: {len(rows)} probes in {time.time() - t0:.0f}s", flush=True)


# ------------------------------------------------------------------ GN unit test (brief 3.4)
def unit_test_gn(a):
    """RP reference; synthetic points = reference centres seen from a known pose (in front of the camera, inside the image), in camera
    coordinates.  Perturb the pose by 3 deg / 5 mm (6 random directions) -> GN must recover within 0.1 deg / 0.3 mm; zero perturbation -> < 0.05 deg."""
    seq = a.seq; video = REPO / ("datasets/HO3D_v3/evaluation" if a.dataset == "ho3d" else "datasets/YCBInEOAT") / seq
    sub = "output_YCBInEOAT_sam2mask_mesh" if a.dataset == "ycb" else "output_HO3D_sam2mask_mesh"
    prior_paths = {"mesh_npz": SAM3D_ROOT / sub / f"{seq}_mesh_depth.npz", "pose_json": SAM3D_ROOT / sub / f"{seq}_mesh_depth.json", "gaussian_ply": SAM3D_ROOT / sub / f"{seq}_splat_depth.ply"}
    K_cam = np.loadtxt(a.run_dir / "cam_K.txt").reshape(3, 3).astype(np.float32); cycles = load_cycles(a.run_dir); cyc = cycles[0]
    frames = [GaussianFrame(fid, *(lambda f: (f.rgb, f.depth, f.mask, f.K, f.c2w_cv))(load_frame(a.run_dir, fid, K_cam, cyc["before"][i].astype(np.float32)))).validated() for i, fid in enumerate(cyc["ids"])]
    runner = build_first("RP", a.runner_config, a.device, prior_paths, frames, cyc["before"][0].astype(np.float32), 0); ref = SurfelReference(runner)
    lines = [f"GN unit test ({a.dataset}/{seq}, RP reference {ref.n_ref} centres; synthetic points = reference centres inside the frustum of a known pose, "
             f"camera coordinates; true pose orthonormalised; angle = 2 asin(|dR|_F / 2 sqrt 2))"]; ok_spec = True; ok_impl = True; rng = np.random.default_rng(0); n_ok_spec = n_tr = 0
    for k, fid in enumerate(cyc["ids"][:3]):
        T_true = orth(cyc["before"][k]); w2c = np.linalg.inv(T_true); Xc = ref.mu @ w2c[:3, :3].T + w2c[:3, 3]
        u = Xc[:, 0] / Xc[:, 2] * K_cam[0, 0] + K_cam[0, 2]; v = Xc[:, 1] / Xc[:, 2] * K_cam[1, 1] + K_cam[1, 2]
        pts = Xc[(Xc[:, 2] > 0.05) & (u >= 0) & (u < 640) & (v >= 0) & (v < 480)]; pts = pts[rng.choice(len(pts), min(P["n_points"], len(pts)), replace=False)]
        for trial in range(6):
            ax = rng.normal(size=3); ax /= np.linalg.norm(ax); td = rng.normal(size=3); td /= np.linalg.norm(td)
            T_start = left_update(T_true, np.concatenate((td * 0.005, ax * math.radians(3.0)))); res = {}
            for cap in (P["gn_max_iter"], 30):
                saved = P["gn_max_iter"]; P["gn_max_iter"] = cap
                try: T_est, c0, c1, it, ratio, acc = gn_refine(ref, T_start, pts)
                finally: P["gn_max_iter"] = saved
                res[cap] = (rot_angle(T_est[:3, :3], T_true[:3, :3]), np.linalg.norm(T_est[:3, 3] - T_true[:3, 3]) * 1000, it)
            s8, s30 = res[P["gn_max_iter"]], res[30]; ok8 = s8[0] < 0.1 and s8[1] < 0.3; ok30 = s30[0] < 0.1 and s30[1] < 0.3
            n_tr += 1; n_ok_spec += ok8; ok_spec &= ok8; ok_impl &= ok30
            lines.append(f"  {fid} trial {trial}: 3.00 deg / 5.0 mm -> cap {P['gn_max_iter']}: {s8[0]:.4f} deg / {s8[1]:.4f} mm ({s8[2]} it) {'OK' if ok8 else 'over'} | "
                         f"cap 30: {s30[0]:.4f} deg / {s30[1]:.4f} mm ({s30[2]} it) {'OK' if ok30 else 'FAIL'}")
        T_est, *_ = gn_refine(ref, T_true, pts); dr = rot_angle(T_est[:3, :3], T_true[:3, :3]); ok0 = dr < 0.05; ok_spec &= ok0; ok_impl &= ok0
        lines.append(f"  {fid} zero perturbation: moved {dr:.6f} deg / {np.linalg.norm(T_est[:3, 3] - T_true[:3, 3]) * 1000:.6f} mm {'OK' if ok0 else 'FAIL'}")
    lines.append(f"implementation check (cap 30): {'PASS' if ok_impl else 'FAIL'}; with the brief's cap {P['gn_max_iter']}: {n_ok_spec}/{n_tr} trials within 0.1 deg / 0.3 mm")
    ok_all = ok_impl
    lines.append("GN unit test " + ("PASS" if ok_all else "FAIL")); text = "\n".join(lines); print(text)
    a.output_dir.mkdir(parents=True, exist_ok=True); (a.output_dir / "gn_unit_test.txt").write_text(text + "\n")
    return ok_all


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=("ho3d", "ycb"), required=True); ap.add_argument("--seq", required=True); ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True); ap.add_argument("--reference", choices=("R0", "A1", "A2", "A3", "RP", "RG"), default="R0")
    ap.add_argument("--runner-config", type=Path, default=REPO / "config_gs_2dgs_1mm_lifecycle.yml"); ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--initial-steps", type=int, default=500); ap.add_argument("--update-steps", type=int, default=500)
    ap.add_argument("--probe-every", type=int, default=None, help="probe every n-th cycle (default: 3 for ho3d, 1 for ycb)")
    ap.add_argument("--optimizers", nargs="+", default=["O1", "O2", "O3", "O4"]); ap.add_argument("--unit-test", choices=("gn",), default=None)
    ap.add_argument("--max-cycles", type=int, default=0, help="smoke test: stop after this cycle index (0 = all)")
    a = ap.parse_args()
    if a.probe_every is None: a.probe_every = 3 if a.dataset == "ho3d" else 1
    if a.unit_test == "gn": sys.exit(0 if unit_test_gn(a) else 1)
    run(a)
