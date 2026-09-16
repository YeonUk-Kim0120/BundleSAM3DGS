"""EXPERIMENT: observation-gated prior lifecycle — pure evidence/transition math.

Milestone ③ prototype, isolated from the main pipeline (main code untouched).
The evidence classification and independent-view gating are ports of the
validated BundleGS implementation in the sibling checkout
(/home/kist/Desktop/BundleSDF/gs_core_utils.py), with two extensions:

  * a SUSPECT state between VERIFIED and CONTRADICTED (hysteresis buffer),
  * a signed normal-offset residual accumulator fed on support events —
    the pre-provisioned input for the ③b bias-extrapolation correction.

States: UNSEEN(0) → VERIFIED(1) → SUSPECT(2) → CONTRADICTED(3, terminal).
SAM3D-lineage surfels start UNSEEN; RGB-D-appended splats start VERIFIED.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

STATE_UNSEEN = 0
STATE_VERIFIED = 1
STATE_SUSPECT = 2
STATE_CONTRADICTED = 3
# EXPERIMENT COPY (experiments/online_variants, 2026-09-15): experiment F "re-observation fusion" needs a per-splat
# observation count (fusion weight).  Everything else is identical to the main module.


@dataclass
class LifecycleFields:
    """Per-splat non-trainable lifecycle tensors (aligned with the splats)."""

    state: torch.Tensor  # int8 [N]
    lineage: torch.Tensor  # bool [N] — True = SAM3D prior descent
    conflict_count: torch.Tensor  # int16 [N]
    last_conflict_view: torch.Tensor  # float32 [N, 3]
    residual_sum: torch.Tensor  # float32 [N] — signed normal offsets (metric)
    residual_count: torch.Tensor  # int32 [N]
    obs_count: torch.Tensor | None = None  # int32 [N] — F: observations fused into the splat (weight); None -> 1

    def __post_init__(self) -> None:
        if self.obs_count is None:  # old checkpoints / main-code callers
            self.obs_count = torch.ones(int(self.state.shape[0]), dtype=torch.int32, device=self.state.device)

    @classmethod
    def create(cls, count: int, *, lineage_prior: bool,
               device: torch.device | str = "cpu", obs_count: int = 1) -> "LifecycleFields":
        state_value = STATE_UNSEEN if lineage_prior else STATE_VERIFIED
        return cls(
            obs_count=torch.full((count,), int(obs_count), dtype=torch.int32, device=device),
            state=torch.full((count,), state_value, dtype=torch.int8,
                             device=device),
            lineage=torch.full((count,), bool(lineage_prior),
                               dtype=torch.bool, device=device),
            conflict_count=torch.zeros(count, dtype=torch.int16, device=device),
            last_conflict_view=torch.zeros(count, 3, dtype=torch.float32,
                                           device=device),
            residual_sum=torch.zeros(count, dtype=torch.float32, device=device),
            residual_count=torch.zeros(count, dtype=torch.int32, device=device),
        )

    def validated(self) -> "LifecycleFields":
        n = len(self.state)
        if self.state.dtype != torch.int8:
            raise ValueError("state must be int8")
        for name in ("lineage", "conflict_count", "residual_sum",
                     "residual_count"):
            if len(getattr(self, name)) != n:
                raise ValueError(f"{name} must align with state")
        if tuple(self.last_conflict_view.shape) != (n, 3):
            raise ValueError("last_conflict_view must be [N, 3]")
        if len(self.obs_count) != n:
            raise ValueError("obs_count must align with state")
        if not bool(((self.state >= STATE_UNSEEN)
                     & (self.state <= STATE_CONTRADICTED)).all()):
            raise ValueError("state contains an invalid value")
        return self

    def __len__(self) -> int:
        return int(self.state.shape[0])

    def concat(self, other: "LifecycleFields") -> "LifecycleFields":
        return LifecycleFields(
            state=torch.cat((self.state, other.state)),
            lineage=torch.cat((self.lineage, other.lineage)),
            conflict_count=torch.cat((self.conflict_count,
                                      other.conflict_count)),
            last_conflict_view=torch.cat((self.last_conflict_view,
                                          other.last_conflict_view)),
            residual_sum=torch.cat((self.residual_sum, other.residual_sum)),
            residual_count=torch.cat((self.residual_count,
                                      other.residual_count)),
            obs_count=torch.cat((self.obs_count, other.obs_count)),
        )

    def keep(self, mask: torch.Tensor) -> "LifecycleFields":
        return LifecycleFields(
            state=self.state[mask],
            lineage=self.lineage[mask],
            conflict_count=self.conflict_count[mask],
            last_conflict_view=self.last_conflict_view[mask],
            residual_sum=self.residual_sum[mask],
            residual_count=self.residual_count[mask],
            obs_count=self.obs_count[mask],
        )

    def summary(self) -> dict[str, int]:
        return {
            "total": len(self),
            "unseen": int((self.state == STATE_UNSEEN).sum()),
            "verified": int((self.state == STATE_VERIFIED).sum()),
            "suspect": int((self.state == STATE_SUSPECT).sum()),
            "contradicted": int((self.state == STATE_CONTRADICTED).sum()),
            "prior_lineage": int(self.lineage.sum()),
        }


def erode_mask(mask: torch.Tensor, radius_px: int) -> torch.Tensor:
    """Binary erosion via inverted max-pooling (mask: [H, W] bool)."""

    if radius_px <= 0:
        return mask
    kernel = 2 * radius_px + 1
    inverted = (~mask).float()[None, None]
    dilated = F.max_pool2d(inverted, kernel, stride=1, padding=radius_px)
    return ~(dilated[0, 0] > 0.5)


def depth_evidence_masks(
    gaussian_depth: torch.Tensor,
    observed_depth: torch.Tensor,
    valid_observation: torch.Tensor,
    geometric_front_depth: torch.Tensor,
    geometric_alpha: torch.Tensor,
    depth_tolerance: float,
    geometric_alpha_threshold: float = 0.5,
    view_abs_cos: torch.Tensor | None = None,
    grazing_tolerance_cap: float = 3.0,
) -> dict[str, torch.Tensor]:
    """Classify projected splat centers against one RGB-D observation.

    Port of sibling ``prior_depth_evidence_masks`` (gs_core_utils.py:7), with
    a grazing-aware extension: when ``view_abs_cos`` (|cos(normal, ray)|) is
    given, the along-ray tolerance widens by 1/|cos| — the same normal-space
    surface offset projects to a larger ray-depth difference on a slanted
    surface — capped at ``grazing_tolerance_cap`` × the base tolerance.
    ``geometric_front_depth`` must be rendered with opacity-independent
    coverage so a transparent/contradicted splat cannot shield others.
    """

    tensors = (gaussian_depth, observed_depth, valid_observation,
               geometric_front_depth, geometric_alpha)
    if any(t.ndim != 1 for t in tensors):
        raise ValueError("evidence inputs must be one-dimensional")
    if len({len(t) for t in tensors}) != 1:
        raise ValueError("evidence inputs must have equal length")
    if valid_observation.dtype != torch.bool:
        raise ValueError("valid_observation must be boolean")
    base_tolerance = float(depth_tolerance)
    if base_tolerance <= 0:
        raise ValueError("depth_tolerance must be positive")
    if view_abs_cos is None:
        tolerance: torch.Tensor | float = base_tolerance
    else:
        if tuple(view_abs_cos.shape) != tuple(gaussian_depth.shape):
            raise ValueError("view_abs_cos must align with gaussian_depth")
        cap = float(grazing_tolerance_cap)
        if cap < 1.0:
            raise ValueError("grazing_tolerance_cap must be >= 1")
        tolerance = base_tolerance / view_abs_cos.abs().clamp(min=1.0 / cap,
                                                              max=1.0)

    delta = gaussian_depth - observed_depth
    support = valid_observation & (delta.abs() <= tolerance)
    free_space = valid_observation & (delta < -tolerance)
    behind = valid_observation & (delta > tolerance)
    has_geometric_occluder = (
        (geometric_alpha >= float(geometric_alpha_threshold))
        & torch.isfinite(geometric_front_depth)
        & (geometric_front_depth > 0)
        & (geometric_front_depth < gaussian_depth - tolerance)
    )
    return {
        "support": support,
        "free_space": free_space,
        "behind_occluded": behind & has_geometric_occluder,
        "behind_miss": behind & ~has_geometric_occluder,
    }


def independent_view_mask(
    candidate: torch.Tensor,
    conflict_count: torch.Tensor,
    last_conflict_view: torch.Tensor,
    current_view_direction: torch.Tensor,
    min_view_angle_deg: float,
) -> torch.Tensor:
    """Accept conflict evidence only from a sufficiently different viewpoint.

    Port of sibling ``independent_view_evidence_mask`` (gs_core_utils.py:59).
    """

    if candidate.dtype != torch.bool or candidate.ndim != 1:
        raise ValueError("candidate must be a 1-D boolean tensor")
    n = len(candidate)
    if len(conflict_count) != n:
        raise ValueError("conflict_count must align with candidate")
    if tuple(last_conflict_view.shape) != (n, 3):
        raise ValueError("last_conflict_view must be [N, 3]")
    if tuple(current_view_direction.shape) != (n, 3):
        raise ValueError("current_view_direction must be [N, 3]")
    angle = float(min_view_angle_deg)
    if not 0 <= angle <= 180:
        raise ValueError("min_view_angle_deg must be in [0, 180]")

    first_evidence = conflict_count == 0
    dot = (last_conflict_view * current_view_direction).sum(dim=-1)
    dot = dot.clamp(-1.0, 1.0)
    max_dot = float(torch.cos(torch.deg2rad(
        torch.tensor(angle, dtype=torch.float64))).item())
    return candidate & (first_evidence | (dot <= max_dot))


@dataclass(frozen=True)
class TransitionThresholds:
    """Sibling-validated defaults (research.md 2026-08-03)."""

    depth_tolerance_m: float = 0.01
    min_view_angle_deg: float = 10.0
    conflict_min_views: int = 2  # UNSEEN→CONTRADICTED, VERIFIED→SUSPECT
    retract_min_views: int = 3  # (VERIFIED|SUSPECT)→CONTRADICTED
    geometric_alpha_threshold: float = 0.5
    mask_erode_px: int = 2
    grazing_tolerance_cap: float = 3.0  # ray tolerance ≤ cap × base at grazing


def apply_transitions(
    fields: LifecycleFields,
    supported: torch.Tensor,
    thresholds: TransitionThresholds,
) -> dict[str, torch.Tensor]:
    """Apply the state machine after a classification batch.

    ``supported`` is the per-splat "supported in at least one frame of this
    batch" mask; conflict counts must already be updated by the caller.
    Support wins within a batch: it resets the conflict counter (recovery).
    Returns the transition masks (before mutation) for logging/tests.
    """

    fields.validated()
    state = fields.state
    entry = state.clone()
    lineage_or_observed = torch.ones_like(fields.lineage)

    if bool(supported.any()):
        fields.conflict_count[supported] = 0
        fields.last_conflict_view[supported] = 0.0

    unseen = entry == STATE_UNSEEN
    verified = entry == STATE_VERIFIED
    suspect = entry == STATE_SUSPECT
    conflicts = fields.conflict_count.to(torch.int32)
    min_views = int(thresholds.conflict_min_views)
    retract = int(thresholds.retract_min_views)

    to_verify = (unseen | suspect) & supported & lineage_or_observed
    unseen_to_contradict = unseen & ~supported & (conflicts >= min_views)
    verified_to_suspect = (
        verified & ~supported & (conflicts >= min_views) & (conflicts < retract)
    )
    verified_to_contradict = verified & ~supported & (conflicts >= retract)
    suspect_to_contradict = suspect & ~supported & (conflicts >= retract)
    to_contradict = (
        unseen_to_contradict | verified_to_contradict | suspect_to_contradict
    )

    state[to_verify] = STATE_VERIFIED
    state[verified_to_suspect] = STATE_SUSPECT
    state[to_contradict] = STATE_CONTRADICTED
    return {
        "to_verify": to_verify,
        "verified_to_suspect": verified_to_suspect,
        "to_contradict": to_contradict,
    }


def accumulate_support_residuals(
    fields: LifecycleFields,
    support: torch.Tensor,
    gaussian_depth: torch.Tensor,
    observed_depth: torch.Tensor,
    normals: torch.Tensor,
    view_directions: torch.Tensor,
    min_abs_cos: float = 0.2,
) -> None:
    """Feed the ③b residual accumulator on support events.

    The along-ray depth difference is converted to a normal offset with the
    cos(normal, ray) factor; grazing observations (|cos| < min_abs_cos) are
    skipped as unreliable. Sign convention: positive = the measured surface
    lies OUTSIDE the prior surface along its normal.
    """

    if not bool(support.any()):
        return
    cos = (F.normalize(normals, dim=-1, eps=1e-8)
           * F.normalize(view_directions, dim=-1, eps=1e-8)).sum(dim=-1)
    usable = support & (cos.abs() >= float(min_abs_cos))
    if not bool(usable.any()):
        return
    # observed − gaussian along the ray; project onto the normal direction.
    delta_ray = (observed_depth - gaussian_depth)[usable]
    residual_normal = delta_ray * cos[usable]
    fields.residual_sum[usable] += residual_normal
    fields.residual_count[usable] += 1
