"""One-shot Gaussian topology operator diagnostics for controlled experiments."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Literal, Mapping

import torch


TopologyMode = Literal["mixed", "duplicate_only", "split_only", "prune_only"]
TOPOLOGY_MODES: tuple[TopologyMode, ...] = (
    "mixed",
    "duplicate_only",
    "split_only",
    "prune_only",
)


@dataclass(frozen=True)
class BudgetSelection:
    """Original-index selections made before any topology mutation.

    ``grow`` retains the common grow candidate set even in ``prune_only`` mode,
    where those candidates are recorded for comparison but are not mutated.
    """

    grow: torch.Tensor
    duplicate: torch.Tensor
    split: torch.Tensor
    prune: torch.Tensor
    max_replacements: int
    visible_count: int

    @property
    def replacement_count(self) -> int:
        return int(self.prune.numel())


@dataclass(frozen=True)
class BudgetTopologyStats:
    mode: str
    fraction: float
    gaussians_before: int
    gaussians_after: int
    max_replacements: int
    visible_count: int
    protected_count: int
    duplicate_count: int
    split_count: int
    prune_count: int
    duplicate_indices: tuple[int, ...]
    split_indices: tuple[int, ...]
    grow_indices: tuple[int, ...]
    prune_indices: tuple[int, ...]
    grow_score_min: float | None
    grow_score_max: float | None
    prune_opacity_min: float | None
    prune_opacity_max: float | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def split_child_opacity(opacity: torch.Tensor) -> torch.Tensor:
    """Return equal child opacity whose two-layer composite matches ``opacity``."""

    if not torch.is_floating_point(opacity):
        raise TypeError("opacity must be floating point")
    eps = torch.finfo(opacity.dtype).eps
    opacity = opacity.clamp(min=eps, max=1.0 - eps)
    child = 1.0 - torch.sqrt(1.0 - opacity)
    return child.clamp(min=eps, max=1.0 - eps)


def _stable_order(
    indices: torch.Tensor, values: torch.Tensor, *, descending: bool
) -> torch.Tensor:
    if indices.numel() == 0:
        return indices
    local_order = torch.argsort(values[indices], descending=descending, stable=True)
    return indices[local_order]


def select_budgeted_topology(
    grad2d: torch.Tensor,
    count: torch.Tensor,
    log_scales: torch.Tensor,
    opacity_logits: torch.Tensor,
    protected_from_prune: torch.Tensor,
    fraction: float,
    mode: TopologyMode = "mixed",
) -> BudgetSelection:
    """Select exact-rank grow and prune sets without changing any tensors."""

    if mode not in TOPOLOGY_MODES:
        raise ValueError(f"mode must be one of {TOPOLOGY_MODES}")
    if not math.isfinite(fraction) or not 0.0 <= fraction <= 1.0:
        raise ValueError("fraction must be finite and in [0, 1]")
    n_gaussians = int(grad2d.numel())
    expected_vectors = {
        "count": count,
        "opacity_logits": opacity_logits,
        "protected_from_prune": protected_from_prune,
    }
    for name, value in expected_vectors.items():
        if value.ndim != 1 or value.numel() != n_gaussians:
            raise ValueError(f"{name} must have shape [{n_gaussians}]")
    if log_scales.ndim != 2 or log_scales.shape != (n_gaussians, 3):
        raise ValueError(f"log_scales must have shape [{n_gaussians}, 3]")
    if protected_from_prune.dtype != torch.bool:
        raise TypeError("protected_from_prune must be boolean")
    if not (
        torch.isfinite(grad2d).all()
        and torch.isfinite(count).all()
        and torch.isfinite(log_scales).all()
        and torch.isfinite(opacity_logits).all()
    ):
        raise ValueError("selection inputs must be finite")

    max_replacements = int(math.floor(fraction * n_gaussians))
    visible = count > 0
    visible_indices = torch.where(visible)[0]
    empty = torch.empty(0, dtype=torch.long, device=grad2d.device)
    if max_replacements == 0 or visible_indices.numel() == 0:
        return BudgetSelection(
            empty, empty, empty, empty, max_replacements, int(visible.sum())
        )

    mean_gradient = torch.zeros_like(grad2d)
    mean_gradient[visible] = grad2d[visible] / count[visible]
    grow_order = _stable_order(
        visible_indices, mean_gradient, descending=True
    )
    tentative_count = min(max_replacements, int(grow_order.numel()))
    prune_base = visible & ~protected_from_prune
    tentative_grow = grow_order[:tentative_count]
    prune_overlap = torch.cumsum(prune_base[tentative_grow].to(torch.long), dim=0)
    candidate_counts = torch.arange(
        1, tentative_count + 1, device=grad2d.device, dtype=torch.long
    )
    feasible = int(prune_base.sum().item()) - prune_overlap >= candidate_counts
    feasible_counts = candidate_counts[feasible]
    replacement_count = (
        int(feasible_counts[-1].item()) if feasible_counts.numel() else 0
    )
    if replacement_count == 0:
        return BudgetSelection(
            empty, empty, empty, empty, max_replacements, int(visible.sum())
        )

    grow = tentative_grow[:replacement_count]
    grow_mask = torch.zeros_like(visible)
    grow_mask[grow] = True
    prune_candidates = torch.where(visible & ~protected_from_prune & ~grow_mask)[0]
    opacity = torch.sigmoid(opacity_logits)
    prune_order = _stable_order(prune_candidates, opacity, descending=False)
    prune = prune_order[:replacement_count]

    if mode == "mixed":
        max_scale = torch.exp(log_scales).amax(dim=-1)
        grow_by_scale = _stable_order(grow, max_scale, descending=False)
        duplicate_count = (replacement_count + 1) // 2
        duplicate = grow_by_scale[:duplicate_count]
        split = grow_by_scale[duplicate_count:]
    elif mode == "duplicate_only":
        duplicate = grow
        split = empty
    elif mode == "split_only":
        duplicate = empty
        split = grow
    else:
        duplicate = empty
        split = empty
    return BudgetSelection(
        grow=grow,
        duplicate=duplicate,
        split=split,
        prune=prune,
        max_replacements=max_replacements,
        visible_count=int(visible.sum().item()),
    )


def _zero_indexed_optimizer_state(
    optimizer: torch.optim.Optimizer,
    parameter: torch.nn.Parameter,
    indices: torch.Tensor,
) -> None:
    state = optimizer.state.get(parameter, {})
    parameter_count = int(parameter.shape[0])
    for value in state.values():
        if (
            isinstance(value, torch.Tensor)
            and value.ndim > 0
            and value.shape[0] == parameter_count
        ):
            value[indices] = 0


def _assert_consistent_state(
    params: torch.nn.ParameterDict,
    optimizers: Mapping[str, torch.optim.Optimizer],
    strategy_state: Mapping[str, Any],
    expected_count: int,
) -> None:
    for name, parameter in params.items():
        if parameter.shape[0] != expected_count:
            raise RuntimeError(f"{name} has an inconsistent Gaussian count")
        if not torch.isfinite(parameter).all():
            raise FloatingPointError(f"Non-finite values in {name} after topology")
        optimizer = optimizers[name]
        if optimizer.param_groups[0]["params"][0] is not parameter:
            raise RuntimeError(f"Optimizer for {name} points to a stale parameter")
        for key, value in optimizer.state.get(parameter, {}).items():
            if isinstance(value, torch.Tensor) and value.ndim > 0:
                if value.shape[0] != expected_count:
                    raise RuntimeError(f"Optimizer state {name}.{key} has wrong length")
                if not torch.isfinite(value).all():
                    raise FloatingPointError(
                        f"Non-finite optimizer state in {name}.{key}"
                    )
    for key, value in strategy_state.items():
        if isinstance(value, torch.Tensor) and value.ndim > 0:
            if value.shape[0] != expected_count:
                raise RuntimeError(f"Strategy state {key} has wrong length")
            if not torch.isfinite(value).all():
                raise FloatingPointError(f"Non-finite strategy state in {key}")


@torch.no_grad()
def apply_budgeted_topology(
    params: torch.nn.ParameterDict,
    optimizers: Mapping[str, torch.optim.Optimizer],
    strategy_state: dict[str, Any],
    *,
    fraction: float,
    protected_start: int,
    mode: TopologyMode = "mixed",
) -> BudgetTopologyStats:
    """Apply one controlled split/duplicate/prune diagnostic event.

    ``protected_start`` is the first index appended in the current RGB-D update.
    Those tail Gaussians may grow, but cannot be selected for pruning.
    All modes except ``prune_only`` preserve the Gaussian count.
    """

    try:
        from gsplat.strategy.ops import duplicate, remove, split
    except ImportError as exc:
        raise RuntimeError("gsplat is required for budgeted topology") from exc

    required = {"means", "scales", "quats", "opacities", "sh0", "shN"}
    if set(params.keys()) != required:
        raise ValueError(f"Expected splat keys {sorted(required)}")
    n_before = int(params["means"].shape[0])
    if not 0 <= protected_start <= n_before:
        raise ValueError("protected_start must be within the Gaussian index range")
    grad2d = strategy_state.get("grad2d")
    count = strategy_state.get("count")
    if not isinstance(grad2d, torch.Tensor) or not isinstance(count, torch.Tensor):
        raise RuntimeError("Budgeted topology requires accumulated grad2d and count")
    protected = torch.zeros(n_before, dtype=torch.bool, device=params["means"].device)
    protected[protected_start:] = True
    selection = select_budgeted_topology(
        grad2d,
        count,
        params["scales"],
        params["opacities"],
        protected,
        fraction,
        mode,
    )
    grow = selection.grow
    grow_scores = (
        grad2d[grow] / count[grow] if grow.numel() else torch.empty_like(grow, dtype=grad2d.dtype)
    )
    prune_opacities = (
        torch.sigmoid(params["opacities"][selection.prune])
        if selection.prune.numel()
        else torch.empty_like(selection.prune, dtype=params["opacities"].dtype)
    )

    if selection.replacement_count:
        pending_prune_key = "_budget_pending_prune"
        if pending_prune_key in strategy_state:
            raise RuntimeError("Budgeted topology already has a pending prune mask")
        pending_prune = torch.zeros(n_before, dtype=torch.bool, device=protected.device)
        pending_prune[selection.prune] = True
        strategy_state[pending_prune_key] = pending_prune

        duplicate_mask = torch.zeros(n_before, dtype=torch.bool, device=protected.device)
        duplicate_mask[selection.duplicate] = True
        if selection.duplicate.numel():
            opacity_parameter = params["opacities"]
            child_opacity = split_child_opacity(
                torch.sigmoid(opacity_parameter[selection.duplicate])
            )
            opacity_parameter[selection.duplicate] = torch.logit(child_opacity)
            _zero_indexed_optimizer_state(
                optimizers["opacities"], opacity_parameter, selection.duplicate
            )
            duplicate(params, optimizers, strategy_state, duplicate_mask)

        split_mask = torch.zeros(
            len(params["means"]), dtype=torch.bool, device=protected.device
        )
        split_mask[selection.split] = True
        if selection.split.numel():
            split(
                params,
                optimizers,
                strategy_state,
                split_mask,
                revised_opacity=True,
            )

        remapped_prune = strategy_state[pending_prune_key]
        if int(remapped_prune.sum().item()) != selection.replacement_count:
            raise RuntimeError("Prune mask changed cardinality during topology growth")
        remove(params, optimizers, strategy_state, remapped_prune)
        strategy_state.pop(pending_prune_key)

    for key in ("grad2d", "count", "radii"):
        value = strategy_state.get(key)
        if isinstance(value, torch.Tensor):
            value.zero_()
    n_after = int(params["means"].shape[0])
    expected_after = (
        n_before - selection.replacement_count
        if mode == "prune_only"
        else n_before
    )
    if n_after != expected_after:
        raise RuntimeError(
            f"Budgeted topology mode {mode} produced {n_after} Gaussians; "
            f"expected {expected_after} from {n_before}"
        )
    _assert_consistent_state(params, optimizers, strategy_state, n_after)

    def tensor_tuple(value: torch.Tensor) -> tuple[int, ...]:
        return tuple(int(item) for item in value.detach().cpu().tolist())

    return BudgetTopologyStats(
        mode=mode,
        fraction=float(fraction),
        gaussians_before=n_before,
        gaussians_after=n_after,
        max_replacements=selection.max_replacements,
        visible_count=selection.visible_count,
        protected_count=n_before - protected_start,
        duplicate_count=int(selection.duplicate.numel()),
        split_count=int(selection.split.numel()),
        prune_count=int(selection.prune.numel()),
        duplicate_indices=tensor_tuple(selection.duplicate),
        split_indices=tensor_tuple(selection.split),
        grow_indices=tensor_tuple(selection.grow),
        prune_indices=tensor_tuple(selection.prune),
        grow_score_min=(float(grow_scores.min().item()) if grow_scores.numel() else None),
        grow_score_max=(float(grow_scores.max().item()) if grow_scores.numel() else None),
        prune_opacity_min=(
            float(prune_opacities.min().item()) if prune_opacities.numel() else None
        ),
        prune_opacity_max=(
            float(prune_opacities.max().item()) if prune_opacities.numel() else None
        ),
    )
