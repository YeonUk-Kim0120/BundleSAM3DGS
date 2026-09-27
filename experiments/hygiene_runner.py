"""EXP_BATCH_20260927 map hygiene (Point2Pose-style delayed verification) as a GaussianRunner subclass.  New file; main code untouched.
HYG_MODE None = exactly the base runner (every override falls through to super()).
  A1: novel points of a new keyframe become CANDIDATES (not splats).  Every later keyframe j checks each candidate (projection inside the image,
      inside the 2-px-eroded mask, valid depth, not on a depth edge (5x5 depth range < 1 cm), |observed - projected depth| < 2 cm).  Good ->
      back-projected observation appended (last 8 kept), streak + 1; else streak = 0.  Promotion: streak >= 3, >= 4 observations, coordinate-MAD
      norm < 5 mm, max birth-vs-observation viewing angle >= 5 deg; position = coordinate median of observations, colour = birth colour, appended
      with _append_splats as observed-lineage VERIFIED.  Not promoted within 8 keyframes -> discarded.  Novelty is tested against observed points
      AND the candidate buffer.
  A2: A1 + prior UNSEEN -> VERIFIED only after support from >= 2 keyframes whose viewing directions differ by >= 5 deg (single support keeps
      UNSEEN = frozen).
  A3: A2 + a new view is excluded from the training loss until >= 30 % of its candidates are promoted (views without candidates are active);
      a view that has not reached it within 30 keyframes stays excluded.
No pose-stability condition (brief 3.3)."""
from __future__ import annotations
import json, logging
import numpy as np, torch
import torch.nn.functional as F
from scipy.ndimage import maximum_filter, minimum_filter
from scipy.spatial import cKDTree
from gaussian_runner import GaussianRunner, GaussianUpdateStats, voxel_downsample
from prior_lifecycle import (STATE_CONTRADICTED, STATE_UNSEEN, LifecycleFields, accumulate_support_residuals, apply_transitions,
                             depth_evidence_masks, erode_mask, independent_view_mask)

HYG = dict(erode_px=2, edge_win=5, edge_range_m=0.01, depth_tol_m=0.02, keep_obs=8, promote_streak=3, promote_min_obs=4, promote_mad_m=0.005,
           promote_min_angle_deg=5.0, max_age_kf=8, prior_min_support_kf=2, prior_min_angle_deg=5.0, hold_frac=0.30, hold_max_kf=30)
REASONS = ("never_good", "few_obs", "streak", "spread", "angle")


def make_hygiene_runner(mode):
    assert mode in (None, "A1", "A2", "A3")
    return type(f"HygieneRunner_{mode}", (HygieneRunner,), {"HYG_MODE": mode})


class HygieneRunner(GaussianRunner):
    HYG_MODE = None

    # ------------------------------------------------------------------ bookkeeping helpers
    def _hyg_init(self):
        if getattr(self, "_hyg_ready", False): return
        self._hyg_ready = True; K = HYG["keep_obs"]
        self.cand = dict(pos=np.zeros((0, 3)), col=np.zeros((0, 3), np.float32), birth=np.zeros(0, int), cam=np.zeros((0, 3)),
                         obs=np.full((0, K, 3), np.nan), count=np.zeros(0, int), streak=np.zeros(0, int), maxang=np.zeros(0))
        self.view_born = {}; self.view_promoted = {}; self.view_active = {}; self.hyg_log = []

    def _prior_gate_enabled(self): return self.HYG_MODE in ("A2", "A3")

    def _sup_align(self):
        n = self.num_gaussians if self.splats is not None else 0
        if not hasattr(self, "sup_count") or len(self.sup_count) != n:
            self.sup_count = torch.zeros(n, dtype=torch.int32, device=self.device)
            self.sup_dir0 = torch.zeros(n, 3, dtype=torch.float32, device=self.device)
            self.sup_maxang = torch.zeros(n, dtype=torch.float32, device=self.device)

    def _append_splats(self, points_metric, colors):
        n0 = self.num_gaussians if self.splats is not None else 0
        super()._append_splats(points_metric, colors)
        if self._prior_gate_enabled() and hasattr(self, "sup_count") and self.splats is not None:
            k = self.num_gaussians - n0
            self.sup_count = torch.cat((self.sup_count, torch.zeros(k, dtype=torch.int32, device=self.device)))
            self.sup_dir0 = torch.cat((self.sup_dir0, torch.zeros(k, 3, dtype=torch.float32, device=self.device)))
            self.sup_maxang = torch.cat((self.sup_maxang, torch.zeros(k, dtype=torch.float32, device=self.device)))

    def _remove_contradicted(self):
        if self._prior_gate_enabled() and self.lifecycle_enabled and self.lifecycle_fields is not None and hasattr(self, "sup_count"):
            keep = (self.lifecycle_fields.state != STATE_CONTRADICTED).to(self.device)
            n = super()._remove_contradicted()
            if n:
                self.sup_count, self.sup_dir0, self.sup_maxang = self.sup_count[keep], self.sup_dir0[keep], self.sup_maxang[keep]
            return n
        return super()._remove_contradicted()

    # ------------------------------------------------------------------ A2: prior verification gate
    def classify_lifecycle(self, frames, event, occ_masks=None):
        if not self._prior_gate_enabled():
            return super().classify_lifecycle(frames, event, occ_masks)
        # copy of GaussianRunner.classify_lifecycle (main code) + per-frame support bookkeeping and the >= 2 views / >= 5 deg gate
        if not self.lifecycle_enabled or self.lifecycle_fields is None:
            return None
        self._sup_align()
        fields = self.lifecycle_fields.validated(); thresholds = self.lifecycle_thresholds; device = self.device
        supported = torch.zeros(len(fields), dtype=torch.bool, device=device)
        means_metric = torch.from_numpy(self.normalization.metric_points(self.splats["means"].detach().cpu().numpy())).to(device)
        normals = self._splat_normals(); active = fields.state != STATE_CONTRADICTED
        totals = {"projected_valid": 0, "support": 0, "free_space": 0, "behind_occluded": 0, "behind_miss": 0, "accepted_conflict": 0}
        for index, frame in enumerate(frames):
            frame = frame.validated()
            c2w = torch.from_numpy(frame.c2w_cv.astype(np.float32)).to(device); w2c = torch.linalg.inv(c2w)
            points_cam = means_metric @ w2c[:3, :3].T + w2c[:3, 3][None, :]; depth_cam = points_cam[:, 2]; safe = depth_cam.clamp_min(1e-6)
            K = frame.K; height, width = frame.mask.shape
            u = points_cam[:, 0] / safe * float(K[0, 0]) + float(K[0, 2]); v = points_cam[:, 1] / safe * float(K[1, 1]) + float(K[1, 2])
            in_image = (depth_cam > 0.01) & (u >= 0) & (u < width) & (v >= 0) & (v < height)
            ui = u.round().clamp(0, width - 1).long(); vi = v.round().clamp(0, height - 1).long()
            observed = torch.from_numpy(frame.depth).to(device)
            mask = erode_mask(torch.from_numpy(frame.mask).to(device), thresholds.mask_erode_px)
            valid_pixels = mask & torch.isfinite(observed) & (observed > float(self.config["min_depth"])) & (observed < float(self.config["max_depth"]))
            if occ_masks is not None and occ_masks[index] is not None:
                valid_pixels &= ~torch.from_numpy(np.asarray(occ_masks[index]) > 0).to(device)
            valid = active & in_image & valid_pixels[vi, ui]
            c2w_norm = torch.from_numpy(self.normalization.normalize_c2w(frame.c2w_cv).astype(np.float32)).to(device)
            K_t = torch.from_numpy(frame.K.astype(np.float32)).to(device)
            front_depth, front_alpha = self._geometric_front_depth(c2w_norm, K_t, width, height)
            camera_center = c2w[:3, 3]; view_dir = F.normalize(means_metric - camera_center[None, :], dim=-1, eps=1e-8)
            view_abs_cos = (normals * view_dir).sum(dim=-1).abs()
            evidence = depth_evidence_masks(gaussian_depth=depth_cam, observed_depth=observed[vi, ui], valid_observation=valid,
                                            geometric_front_depth=front_depth[vi, ui], geometric_alpha=front_alpha[vi, ui],
                                            depth_tolerance=thresholds.depth_tolerance_m, geometric_alpha_threshold=thresholds.geometric_alpha_threshold,
                                            view_abs_cos=view_abs_cos, grazing_tolerance_cap=thresholds.grazing_tolerance_cap)
            s = evidence["support"]; supported |= s
            # A2 bookkeeping: distinct supporting keyframes and their max viewing-direction spread
            first = s & (self.sup_count == 0); self.sup_dir0[first] = view_dir[first]
            cosang = (view_dir * self.sup_dir0).sum(-1).clamp(-1, 1); ang = torch.rad2deg(torch.arccos(cosang))
            again = s & (self.sup_count > 0); self.sup_maxang[again] = torch.maximum(self.sup_maxang[again], ang[again]); self.sup_count[s] += 1
            accumulate_support_residuals(fields, s, depth_cam, observed[vi, ui], normals, view_dir)
            candidate = evidence["free_space"] | evidence["behind_miss"]
            accepted = independent_view_mask(candidate, fields.conflict_count, fields.last_conflict_view, view_dir, thresholds.min_view_angle_deg)
            if bool(accepted.any()):
                fields.conflict_count[accepted] += 1; fields.last_conflict_view[accepted] = view_dir[accepted]
            totals["projected_valid"] += int(valid.sum())
            for key in ("support", "free_space", "behind_occluded", "behind_miss"): totals[key] += int(evidence[key].sum())
            totals["accepted_conflict"] += int(accepted.sum())
        unseen_prior = (fields.state == STATE_UNSEEN) & fields.lineage.to(device)
        gate_ok = (self.sup_count >= HYG["prior_min_support_kf"]) & (self.sup_maxang >= HYG["prior_min_angle_deg"])
        blocked = supported & unseen_prior & ~gate_ok
        if bool(blocked.any()):   # the base resets conflicts for every supported splat; keep that for the blocked ones
            fields.conflict_count[blocked] = 0; fields.last_conflict_view[blocked] = 0.0
        masks = apply_transitions(fields, supported & ~blocked, thresholds)
        newly_contradicted = masks["to_contradict"]
        if bool(newly_contradicted.any()): self.splats["opacities"].data[newly_contradicted] = -10.0
        record = {"event": str(event), "n_frames": len(frames), "new_verified": int(masks["to_verify"].sum()), "new_suspect": int(masks["verified_to_suspect"].sum()),
                  "new_contradicted": int(newly_contradicted.sum()), "a2_blocked_single_support": int(blocked.sum()), **totals, "state": fields.summary()}
        self.lifecycle_log.append(record); logging.info("[Prior lifecycle A2] " + json.dumps(record, sort_keys=True))
        return record

    # ------------------------------------------------------------------ A3: held-out views in training
    def train(self, steps, **kw):
        if self.HYG_MODE != "A3" or not getattr(self, "_hyg_ready", False):
            return super().train(steps, **kw)
        all_views = self.views; active = [v for i, v in enumerate(all_views) if self.view_active.get(i, True)]
        self.views = active if active else all_views[:1]
        try: return super().train(steps, **kw)
        finally: self.views = all_views

    # ------------------------------------------------------------------ A1: candidates
    def _verify(self, frame, j):
        c = self.cand; n = len(c["pos"])
        if n == 0: return 0
        old = c["birth"] < j
        c2w = frame.c2w_cv.astype(np.float64); w2c = np.linalg.inv(c2w); K = frame.K.astype(np.float64); H, W = frame.mask.shape
        depth = np.asarray(frame.depth, dtype=np.float64)
        vd = np.isfinite(depth) & (depth > float(self.config["min_depth"])) & (depth < float(self.config["max_depth"]))
        df = np.where(vd, depth, 0.0); edge = (maximum_filter(df, HYG["edge_win"]) - minimum_filter(df, HYG["edge_win"])) >= HYG["edge_range_m"]
        me = erode_mask(torch.from_numpy(np.asarray(frame.mask).astype(bool)), HYG["erode_px"]).numpy()
        Xc = c["pos"] @ w2c[:3, :3].T + w2c[:3, 3]; z = Xc[:, 2]; zs = np.maximum(z, 1e-6)
        ui = np.round(Xc[:, 0] / zs * K[0, 0] + K[0, 2]).astype(int); vi = np.round(Xc[:, 1] / zs * K[1, 1] + K[1, 2]).astype(int)
        good = old & (z > 0.01) & (ui >= 0) & (ui < W) & (vi >= 0) & (vi < H)
        idx = np.nonzero(good)[0]; u, v = ui[idx], vi[idx]
        ok = me[v, u] & vd[v, u] & ~edge[v, u] & (np.abs(depth[v, u] - z[idx]) < HYG["depth_tol_m"])
        good[idx] = ok; g = np.nonzero(good)[0]
        if len(g):
            d = depth[vi[g], ui[g]]; pc = np.stack([(ui[g] - K[0, 2]) / K[0, 0] * d, (vi[g] - K[1, 2]) / K[1, 1] * d, d], -1)
            po = pc @ c2w[:3, :3].T + c2w[:3, 3]; slot = c["count"][g] % HYG["keep_obs"]; c["obs"][g, slot] = po; c["count"][g] += 1; c["streak"][g] += 1
            db = c["cam"][g] - c["pos"][g]; db /= np.linalg.norm(db, axis=1, keepdims=True); dj = c2w[:3, 3][None] - c["pos"][g]; dj /= np.linalg.norm(dj, axis=1, keepdims=True)
            c["maxang"][g] = np.maximum(c["maxang"][g], np.degrees(np.arccos(np.clip((db * dj).sum(1), -1, 1))))
        bad = old & ~good; c["streak"][bad] = 0
        return int(len(g))

    def _decide(self, j_last):
        c = self.cand; n = len(c["pos"]); stats = {r: 0 for r in REASONS}
        if n == 0: return np.zeros((0, 3)), np.zeros((0, 3), np.float32), stats, 0
        nobs = np.minimum(c["count"], HYG["keep_obs"])
        med = np.nanmedian(c["obs"], axis=1); mad = np.nanmedian(np.abs(c["obs"] - med[:, None, :]), axis=1); spread = np.linalg.norm(mad, axis=1)
        promote = (c["streak"] >= HYG["promote_streak"]) & (nobs >= HYG["promote_min_obs"]) & (spread < HYG["promote_mad_m"]) & (c["maxang"] >= HYG["promote_min_angle_deg"])
        age = j_last - c["birth"]; discard = ~promote & (age >= HYG["max_age_kf"])
        if discard.any():
            dsc = np.nonzero(discard)[0]
            for i in dsc:
                r = ("never_good" if c["count"][i] <= 1 else "few_obs" if nobs[i] < HYG["promote_min_obs"] else "streak" if c["streak"][i] < HYG["promote_streak"]
                     else "spread" if spread[i] >= HYG["promote_mad_m"] else "angle")
                stats[r] += 1
        for b in np.unique(c["birth"][promote]): self.view_promoted[int(b)] = self.view_promoted.get(int(b), 0) + int((c["birth"][promote] == b).sum())
        pts = med[promote]; cols = c["col"][promote]; keep = ~(promote | discard)
        for k in list(c.keys()): c[k] = c[k][keep]
        return pts, cols, stats, int(promote.sum())

    def update(self, frames, train_steps=None):
        if self.HYG_MODE is None:
            return super().update(frames, train_steps)
        self._hyg_init()
        if not self.is_initialized or self.splats is None: raise RuntimeError("GaussianRunner must be initialized before update")
        validated = [frame.validated() for frame in frames]; frame_ids = [f.frame_id for f in validated]
        if {v.frame_id for v in self.views}.intersection(frame_ids): raise ValueError("Frames were already processed")
        j0 = len(self.views); before = self.num_gaussians; n_before = len(self.cand["pos"]); good = 0
        for i, frame in enumerate(validated): good += self._verify(frame, j0 + i)
        j_last = j0 + len(validated) - 1
        prom_pts, prom_cols, disc, n_prom = self._decide(j_last)
        points, colors, raw_count, candidate_count = self._frames_to_cloud(validated)
        novel_points, novel_colors, _ = self._select_novel(points, colors)
        if len(self.cand["pos"]) and len(novel_points):
            dist, _ = cKDTree(self.cand["pos"]).query(novel_points, k=1, workers=-1); keepn = dist > float(self.config["novelty_distance"])
            novel_points, novel_colors = novel_points[keepn], novel_colors[keepn]
        if len(novel_points):
            m = len(novel_points); cam = validated[-1].c2w_cv.astype(np.float64)[:3, 3]
            obs = np.full((m, HYG["keep_obs"], 3), np.nan); obs[:, 0] = novel_points
            new = dict(pos=novel_points.astype(np.float64), col=novel_colors.astype(np.float32), birth=np.full(m, j_last), cam=np.tile(cam, (m, 1)), obs=obs,
                       count=np.ones(m, int), streak=np.zeros(m, int), maxang=np.zeros(m))
            for k in self.cand: self.cand[k] = np.concatenate((self.cand[k], new[k]))
        self.view_born[j_last] = self.view_born.get(j_last, 0) + len(novel_points)
        self._append_splats(prom_pts.astype(np.float32), prom_cols)
        after_append = self.num_gaussians
        if self.lifecycle_fields is not None and len(prom_pts):
            self.lifecycle_fields = self.lifecycle_fields.concat(LifecycleFields.create(len(prom_pts), lineage_prior=False, device=self.device))
        self.views.extend(self._prepare_view(frame) for frame in validated)
        for i in range(len(validated)): self.view_active.setdefault(j0 + i, self.view_born.get(j0 + i, 0) == 0)
        for vj, born in self.view_born.items():   # A3 activation (views reaching 30 % promoted within 30 keyframes)
            if not self.view_active.get(vj, True) and born > 0 and j_last - vj <= HYG["hold_max_kf"] and self.view_promoted.get(vj, 0) / born >= HYG["hold_frac"]:
                self.view_active[vj] = True
        self.update_index += 1
        self.classify_lifecycle(validated, event=f"update_{self.update_index:03d}")
        self._remove_contradicted()
        self._reset_optimization_state(float(self.config["update_lr_scale"]), initial=False)
        steps = int(self.config["update_steps"] if train_steps is None else train_steps)
        first_loss, final_loss = self.train(steps, optimize_poses=bool(self.config["pose_feedback"]["enabled"]))
        if len(prom_pts):
            self.observed_points_metric, self.observed_colors = voxel_downsample(np.concatenate((self.observed_points_metric, prom_pts.astype(np.float32)), 0),
                                                                                 np.concatenate((self.observed_colors, prom_cols.astype(np.float32)), 0), voxel_size=float(self.config["voxel_size"]))
        rec = dict(update_index=self.update_index, frame_ids=frame_ids, view_index=j_last, candidates_before=n_before, verified_good=good, promoted=n_prom,
                   discarded=disc, discarded_total=int(sum(disc.values())), born=int(len(novel_points)), candidates_after=int(len(self.cand["pos"])),
                   views_active=int(sum(1 for i in range(len(self.views)) if self.view_active.get(i, True))), views=len(self.views), gaussians=self.num_gaussians)
        self.hyg_log.append(rec)
        return GaussianUpdateStats(update_index=self.update_index, frame_ids=tuple(frame_ids), raw_points=raw_count, candidate_points=candidate_count,
                                   novel_points=n_prom, gaussians_before=before, gaussians_after_append=after_append, gaussians_after_train=self.num_gaussians,
                                   train_steps=steps, first_loss=first_loss, final_loss=final_loss)
