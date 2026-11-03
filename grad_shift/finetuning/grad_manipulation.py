"""Gradient manipulation manager for projecting/ablating gradients along a vector direction.

Used for causal experiments that selectively keep or remove gradient components
along a steering vector direction during defensive training.
"""

import logging
import math
from typing import Callable, Dict, List, Optional

import torch
import torch.nn as nn

from ..hooks import find_transformer_blocks
from .subspace import orthonormal_rows_with_r

logger = logging.getLogger(__name__)

VALID_MODES = (
    "remove_vector_component",
    "keep_vector_component",
    "flip_vector_component",
    "pos_sign_vector_component",
    "neg_sign_vector_component",
    "mean_shift_vector_component",
    "act_servo_vector_component",
    "act_servo_base_vector_component",
)

# Modes whose enforced/added along-vector component supports scale
# scheduling and per-layer scale factors.
SCALABLE_MODES = (
    "pos_sign_vector_component",
    "neg_sign_vector_component",
    "mean_shift_vector_component",
    "act_servo_vector_component",
    "act_servo_base_vector_component",
)

# What the schedule scale multiplies to produce the enforced component
# magnitude along v:
#   component: |grad @ v|  (the gradient's own component magnitude)
#   grad_norm: ||grad||    (per-token gradient norm, so the realized
#              tokenwise cosine along v approximately equals the scale)
VALID_SCALE_REFS = ("component", "grad_norm")

# Modes that support a multi-direction subspace basis (basis_path).
# mean_shift is excluded: adding s*m along every basis row has no single
# natural normalization and no experiment has needed it.
BASIS_MODES = (
    "remove_vector_component",
    "keep_vector_component",
    "flip_vector_component",
    "pos_sign_vector_component",
    "neg_sign_vector_component",
    "act_servo_vector_component",
)


def exp_decay_schedule(
    step: int, scale_init: float, scale_final: float, tau: float
) -> float:
    """Exponential decay from ``scale_init`` to a ``scale_final`` plateau.

    ``s(t) = scale_final + (scale_init - scale_final) * exp(-t / tau)``

    ``t`` is in trainer ``global_step`` units (microbatch steps), matching
    the x-axis of the gradient-cosine-over-training measurements
    (arXiv:2604.16423 Figure 11) the schedule is fit to.
    """
    return scale_final + (scale_init - scale_final) * math.exp(-step / tau)


class GradientManipulationManager:
    """Manipulates gradients during backward pass by projecting onto or
    ablating components along a specified vector direction.

    Uses forward hooks that register tensor-level backward hooks via
    ``tensor.register_hook()``. This gives clean access to the gradient
    w.r.t. the layer output without architecture-dependent ``grad_input``
    tuple ordering issues.

    Uses the same embedding-inclusive layer indexing as steering:
    - Index 0 = embedding layer (not hookable)
    - Index N = transformer block N-1

    Modes:
        remove_vector_component: ``grad - (grad @ v) * v``
            Removes the projection of the gradient onto *v*.
        keep_vector_component: ``(grad @ v) * v``
            Keeps only the projection of the gradient onto *v*.
        flip_vector_component: ``grad - 2 * (grad @ v) * v``
            Flips the sign of the gradient component along *v*
            while preserving the orthogonal component.
        pos_sign_vector_component: ``grad - (grad @ v) * v + s * m * v``
            Forces the gradient component along *v* to be positive
            (same direction as *v*). With the default scale (``s = 1``)
            and ``scale_ref="component"`` (``m = |grad @ v|``), leaves
            positive components unchanged and flips negative components.
        neg_sign_vector_component: ``grad - (grad @ v) * v - s * m * v``
            Forces the gradient component along *v* to be negative
            (opposite direction to *v*). With the default scale, leaves
            negative components unchanged and flips positive components.
        mean_shift_vector_component: ``grad + s * m * v``
            ADDS a component along *v* without touching the natural one —
            with ``scale_ref="grad_norm"`` this translates the tokenwise
            cosine distribution by ``+s`` while preserving its shape and
            sign mix (unlike the sign-forcing modes). Use a per-layer
            ``scale_by_layer`` to match measured per-layer mean shifts.
        act_servo_vector_component:
            ``grad - (grad @ v) * v + sign(h @ v - a*) * s * m * v``
            Closed-loop control of the ACTIVATION projection: per token,
            if the forward activation's projection onto *v* exceeds the
            per-layer target ``a*`` (``act_target_by_layer``) the along-*v*
            gradient component is set attenuating (positive), below it
            amplifying (negative) — a manufactured restoring force toward
            the target level. The orthogonal component is untouched.
        act_servo_base_vector_component:
            ``grad - (grad @ v) * v + sign(h @ v - b @ v) * s * m * v``
            Per-token variant of act_servo: the target for EACH token is
            that token's own base-model projection ``b @ v`` on the same
            input, from an adapter-disabled no-grad forward the trainer
            runs just before the training forward
            (``capture_base_reference``). The scalar servo pins only the
            mean (every token is driven toward one level, so response and
            prompt tokens trade off); this pins the per-token profile — the
            strict "evil-axis activation held at base" test. Single vector
            only (no basis).

    Servo on the INTRINSIC activation (``act_servo_exclude_by_layer``,
    act_servo_vector_component, single vector): forward hooks on one block
    run in registration order, and the trainer registers steering hooks
    BEFORE these hooks (``LoRATrainer.train`` / ``_evaluate``), so with a
    steering vector ``c_l`` on the servoed block the hooked tensor is
    ``h + c_l``. Passing ``c_l`` here makes the servo decide on
    ``(h + c_l) @ v - c_l @ v = h @ v``, the model's own projection: the
    same controlled quantity and target as a servo-only run, rather than
    the vector-inclusive activation. The projection is then computed in
    float32 (the plain servo projects in the activation dtype), so the
    decision matches the servo-only run's up to that run's own bf16
    rounding of ``h @ v``. The gradient is unaffected by where the hook
    sits (adding a constant passes the gradient through unchanged).

    Multi-direction (subspace) variant — ``basis_path``: every mode
    except mean_shift can operate on a k-direction SUBSPACE instead of a
    single vector. The basis file is ``[k, hidden]`` (broadcast) or
    ``[num_layers, k, hidden]`` (per-layer, embedding-inclusive); rows
    are Gram-Schmidt orthonormalized per layer (QR). With orthonormal
    rows ``b_j`` and gradient coordinates ``G_j = grad @ b_j``:

        remove: ``grad - sum_j G_j b_j``      (span component removed)
        keep:   ``sum_j G_j b_j``             (only the span component)
        flip:   ``grad - 2 sum_j G_j b_j``
        pos/neg_sign: each coordinate forced to ``+/- s(t) * m_j``
            (``m_j = |G_j|`` for scale_ref="component", ``||grad||``
            for "grad_norm")
        act_servo: each direction sign-controlled to its own per-layer
            target (``act_target_by_layer`` values become length-*k*
            lists ordered like the basis rows); the raw-direction
            targets are transformed exactly into orthonormal-coordinate
            targets (``c = solve(R^T, a)``), so servoing the orthonormal
            coordinates is equivalent to targeting the measured
            projections along the raw unit directions.

    The complement of span(basis) is untouched in every mode (except
    keep, which zeroes it by definition).

    For the sign-forcing modes, the enforced component magnitude is
    ``s(t) * m`` where ``s(t)`` is a scale that can decay exponentially
    with the trainer step (``set_step``) and ``m`` depends on
    ``scale_ref``:

    - ``"component"``: ``m = |grad @ v|`` — the scale attenuates the
      gradient's own along-*v* component.
    - ``"grad_norm"``: ``m = ||grad||`` per token — the realized
      tokenwise cosine along *v* approximately equals ``s(t)``, so a
      schedule fit to measured gradient cosines reproduces them directly.

    Args:
        model: The transformer model (or its base model).
        vector_path: Path to a ``.pt`` file containing the manipulation
            vector. Can be shape ``[hidden_dim]`` (single vector) or
            ``[num_layers, hidden_dim]`` (per-layer vectors, embedding-
            inclusive indexing).
        mode: One of ``VALID_MODES``.
        layers: List of embedding-inclusive layer indices to apply
            gradient manipulation to (same convention as steering
            ``active_layers``). If *None*, applies to all transformer
            blocks.
        scale_init: Scale at step 0 (sign-forcing modes only). Default 1.0
            reproduces the unscaled manipulation exactly.
        scale_final: Plateau scale for the exponential decay schedule.
            *None* (default) keeps the scale constant at ``scale_init``.
        scale_tau: Decay time constant in trainer ``global_step``
            (microbatch) units. Required iff ``scale_final`` is set.
        scale_ref: ``"component"`` or ``"grad_norm"`` (see above).
        scale_by_layer: Optional per-layer scale factors (embedding-
            inclusive layer index -> factor) multiplied with the schedule
            scale. Must cover every active layer when provided. Used e.g.
            to apply measured per-layer mean cosine shifts.
        collect_stats: When *True*, accumulate the mean tokenwise cosine
            between the gradient and *v* before/after manipulation on
            every hooked backward pass; retrieve with ``pop_stats()``.
        act_servo_exclude_by_layer: Optional constant vectors ``c_l``
            (embedding-inclusive layer -> ``[hidden]`` tensor) that hooks
            registered before this manager add to the servoed block's
            output (a steering vector); the scalar servo then controls the
            projection of ``h`` without them (see "Servo on the INTRINSIC
            activation" above). act_servo_vector_component with a single
            vector only; keys must be active layers.
    """

    def __init__(
        self,
        model: nn.Module,
        vector_path: str,
        mode: str,
        layers: Optional[List[int]] = None,
        scale_init: float = 1.0,
        scale_final: Optional[float] = None,
        scale_tau: Optional[float] = None,
        scale_ref: str = "component",
        scale_by_layer: Optional[Dict[int, float]] = None,
        act_target_by_layer: Optional[Dict] = None,
        basis_path: Optional[str] = None,
        collect_stats: bool = False,
        act_servo_exclude_by_layer: Optional[Dict[int, torch.Tensor]] = None,
    ):
        if mode not in VALID_MODES:
            raise ValueError(
                f"mode must be one of {VALID_MODES}, got '{mode}'"
            )
        if (mode == "act_servo_vector_component") != (act_target_by_layer is not None):
            raise ValueError(
                "act_target_by_layer is required for (and only valid with) "
                "mode='act_servo_vector_component'"
            )
        if (mode == "act_servo_base_vector_component"
                and act_target_by_layer is not None):
            raise ValueError(
                "act_target_by_layer is not valid with "
                "mode='act_servo_base_vector_component' (targets are the "
                "per-token base projections from capture_base_reference)"
            )
        if basis_path is not None and mode not in BASIS_MODES:
            raise ValueError(
                f"basis_path is only valid with modes {BASIS_MODES}, "
                f"got '{mode}'"
            )
        if scale_ref not in VALID_SCALE_REFS:
            raise ValueError(
                f"scale_ref must be one of {VALID_SCALE_REFS}, got '{scale_ref}'"
            )
        if (scale_final is None) != (scale_tau is None):
            raise ValueError(
                "scale_final and scale_tau must be set together "
                f"(got scale_final={scale_final}, scale_tau={scale_tau})"
            )
        if scale_tau is not None and scale_tau <= 0:
            raise ValueError(f"scale_tau must be positive, got {scale_tau}")
        uses_scaling = (
            scale_init != 1.0
            or scale_final is not None
            or scale_ref != "component"
            or scale_by_layer is not None
        )
        if uses_scaling and mode not in SCALABLE_MODES:
            raise ValueError(
                f"scale scheduling is only supported for {SCALABLE_MODES}, "
                f"got mode='{mode}'"
            )

        self.model = model
        self.mode = mode
        self.scale_init = scale_init
        self.scale_final = scale_final
        self.scale_tau = scale_tau
        self.scale_ref = scale_ref
        self.scale_by_layer = scale_by_layer or {}
        self.act_target_by_layer = act_target_by_layer or {}
        self.collect_stats = collect_stats
        self._current_scale = scale_init
        # (layer_idx, mean cos pre, mean cos post) per hooked backward call
        self._stats: List[tuple] = []
        # Per-token base reference (act_servo_base): layer -> [batch, seq]
        # base-model projections for the CURRENT microbatch; consumed by
        # the training forward so a stale reference can never be reused.
        self._base_proj: Dict[int, torch.Tensor] = {}
        self._base_mask: Optional[torch.Tensor] = None
        self._capturing = False
        # (layer_idx, mean |h.v - b.v| over real tokens) per training forward
        self._servo_err: List[tuple] = []
        self._handles: List[torch.utils.hooks.RemovableHandle] = []

        # Find transformer blocks
        self._pattern, self._layer_modules, self._layer_names = (
            find_transformer_blocks(model)
        )
        self.num_layers = len(self._layer_modules)

        # Determine target layers (embedding-inclusive indices)
        if layers is None:
            self._active_layers = list(range(1, self.num_layers + 1))
        else:
            self._active_layers = [
                l if l >= 0 else (self.num_layers + 1) + l for l in layers
            ]

        # Validate
        for l in self._active_layers:
            if l < 1 or l > self.num_layers:
                raise ValueError(
                    f"Layer index {l} out of range. Valid: 1..{self.num_layers}"
                )
        if self.scale_by_layer:
            missing = [l for l in self._active_layers
                       if l not in self.scale_by_layer]
            if missing:
                raise ValueError(
                    f"scale_by_layer missing factors for active layers "
                    f"{missing}"
                )
        # Coverage must be checked whenever the mode needs targets — an
        # empty dict passes the is-not-None validation but would crash at
        # the first backward otherwise
        if mode == "act_servo_vector_component" or self.act_target_by_layer:
            missing = [l for l in self._active_layers
                       if l not in self.act_target_by_layer]
            if missing:
                raise ValueError(
                    f"act_target_by_layer missing targets for active layers "
                    f"{missing}"
                )

        # Load and normalize vector
        self._raw_vectors = torch.load(vector_path, map_location="cpu")
        if isinstance(self._raw_vectors, dict):
            # Handle state-dict-style saves
            key = list(self._raw_vectors.keys())[0]
            self._raw_vectors = self._raw_vectors[key]

        if self._raw_vectors.dim() == 1:
            # Single vector — broadcast to all layers
            v = self._raw_vectors.float()
            v = v / v.norm()
            self._vectors: Dict[int, torch.Tensor] = {
                l: v for l in self._active_layers
            }
        elif self._raw_vectors.dim() == 2:
            # Per-layer vectors with embedding-inclusive indexing
            self._vectors = {}
            for l in self._active_layers:
                if l >= self._raw_vectors.shape[0]:
                    raise ValueError(
                        f"Vector file has {self._raw_vectors.shape[0]} layers "
                        f"but layer index {l} requested"
                    )
                v = self._raw_vectors[l].float()
                norm = v.norm()
                if norm > 0:
                    v = v / norm
                self._vectors[l] = v
        else:
            raise ValueError(
                f"Expected vector with 1 or 2 dimensions, "
                f"got shape {self._raw_vectors.shape}"
            )

        # Multi-direction subspace: per-layer orthonormal basis B_l
        # [k, hidden]; for the servo mode additionally targets c_l [k] in
        # orthonormal coordinates (exact transform of the raw-direction
        # targets — see class docstring).
        self._basis: Optional[Dict[int, torch.Tensor]] = None
        self._servo_targets: Optional[Dict[int, torch.Tensor]] = None
        is_servo = mode == "act_servo_vector_component"
        if basis_path is not None:
            raw_basis = torch.load(basis_path, map_location="cpu").float()
            if raw_basis.dim() not in (2, 3) or (
                    raw_basis.shape[-2] >= raw_basis.shape[-1]):
                raise ValueError(
                    "basis must be [k, hidden] (broadcast) or "
                    "[num_layers, k, hidden] (embedding-inclusive) with "
                    f"k << hidden, got {tuple(raw_basis.shape)}"
                )
            k = raw_basis.shape[-2]
            self._basis = {}
            if is_servo:
                self._servo_targets = {}
            for l in self._active_layers:
                if raw_basis.dim() == 3:
                    if l >= raw_basis.shape[0]:
                        raise ValueError(
                            f"Basis file has {raw_basis.shape[0]} "
                            f"layers but layer index {l} requested"
                        )
                    dirs = raw_basis[l].clone()
                else:
                    dirs = raw_basis.clone()
                # Rank-checked QR of the unit rows (targets are
                # projections onto UNIT dirs); B = Q^T
                self._basis[l], r = orthonormal_rows_with_r(dirs, l)
                if is_servo:
                    a = self.act_target_by_layer[l]
                    if not isinstance(a, (list, tuple)) or len(a) != k:
                        raise ValueError(
                            f"act_target_by_layer[{l}] must be a list of {k} "
                            f"targets (one per basis row), got {a!r}"
                        )
                    # Raw projections p = R^T u where u are coords on B.
                    # Targeting p = a  <=>  u = solve(R^T, a).
                    self._servo_targets[l] = torch.linalg.solve_triangular(
                        r.T,
                        torch.tensor(a, dtype=torch.float32).unsqueeze(-1),
                        upper=False,
                    ).squeeze(-1)
        elif is_servo:
            bad = {l: a for l, a in self.act_target_by_layer.items()
                   if isinstance(a, (list, tuple))}
            if bad:
                raise ValueError(
                    "list-valued act targets require basis_path "
                    f"(got {bad})"
                )

        # Intrinsic-activation servo: o_l = c_l . v_l in float32, subtracted
        # from the hooked projection before the target comparison.
        self._servo_offsets: Dict[int, float] = {}
        if act_servo_exclude_by_layer is not None:
            if not is_servo or basis_path is not None:
                raise ValueError(
                    "act_servo_exclude_by_layer is only valid with "
                    "mode='act_servo_vector_component' and a single vector "
                    "(no basis_path)"
                )
            extra = sorted(set(act_servo_exclude_by_layer)
                           - set(self._active_layers))
            if extra:
                raise ValueError(
                    f"act_servo_exclude_by_layer has non-servoed layers "
                    f"{extra} (active: {self._active_layers})"
                )
            for l, c in act_servo_exclude_by_layer.items():
                c = torch.as_tensor(c).detach().float().cpu()
                if c.shape != self._vectors[l].shape:
                    raise ValueError(
                        f"act_servo_exclude_by_layer[{l}] has shape "
                        f"{tuple(c.shape)}, expected "
                        f"{tuple(self._vectors[l].shape)}"
                    )
                self._servo_offsets[l] = float(c @ self._vectors[l])

        logger.info(
            f"GradientManipulationManager: mode={mode}, "
            f"layers={self._active_layers}, "
            f"vector_path={vector_path}, "
            f"scale_init={scale_init}, scale_final={scale_final}, "
            f"scale_tau={scale_tau}, scale_ref={scale_ref}"
        )

    def to(self, device) -> "GradientManipulationManager":
        """Move manipulation vectors to a device."""
        self._vectors = {
            l: v.to(device) for l, v in self._vectors.items()
        }
        return self

    @property
    def act_servo_offsets(self) -> Dict[int, float]:
        """Per-layer projection ``c_l . v_l`` of the excluded constant
        vectors (empty unless ``act_servo_exclude_by_layer`` was given)."""
        return dict(self._servo_offsets)

    @property
    def current_scale(self) -> float:
        """The scale applied to the enforced component on the next backward."""
        return self._current_scale

    def set_step(self, step: int) -> None:
        """Update the schedule scale for trainer step ``step``.

        No-op (scale stays at ``scale_init``) when no schedule is configured.
        """
        if self.scale_final is not None:
            self._current_scale = exp_decay_schedule(
                step, self.scale_init, self.scale_final, self.scale_tau
            )

    def pop_stats(self) -> Optional[Dict]:
        """Return mean tokenwise cosine stats accumulated since the last call.

        Cosines are averaged over tokens with a nonzero gradient only —
        attention-masked padding has an exactly-zero gradient and would
        otherwise dilute the mean (~70% of a 512-token batch), hiding
        whether the enforced attenuation matches the target schedule.

        Returns overall means plus per-layer means (``cos_*_by_layer``,
        keyed by embedding-inclusive layer index — layer 20 is directly
        comparable to the paper's Figure 11 measurements), or *None* when
        no stats were collected. Clears the accumulator.
        """
        if not self._stats:
            self._servo_err.clear()
            return None
        pre = torch.stack([p for _, p, _ in self._stats])
        post = torch.stack([q for _, _, q in self._stats])
        by_layer_pre: Dict[int, List[torch.Tensor]] = {}
        by_layer_post: Dict[int, List[torch.Tensor]] = {}
        for layer_idx, p, q in self._stats:
            by_layer_pre.setdefault(layer_idx, []).append(p)
            by_layer_post.setdefault(layer_idx, []).append(q)
        stats = {
            "cos_pre": pre.mean().item(),
            "cos_post": post.mean().item(),
            "scale": self._current_scale,
            "n_hook_calls": len(self._stats),
            "cos_pre_by_layer": {
                l: torch.stack(v).mean().item()
                for l, v in sorted(by_layer_pre.items())
            },
            "cos_post_by_layer": {
                l: torch.stack(v).mean().item()
                for l, v in sorted(by_layer_post.items())
            },
        }
        if self._servo_err:
            by_layer_err: Dict[int, List[float]] = {}
            for layer_idx, e in self._servo_err:
                by_layer_err.setdefault(layer_idx, []).append(e)
            # Pin quality: how far each token sits from its own base
            # projection on the forward that produced these gradients
            stats["servo_abs_err_by_layer"] = {
                l: sum(v) / len(v) for l, v in sorted(by_layer_err.items())
            }
            self._servo_err.clear()
        self._stats.clear()
        return stats

    @property
    def needs_base_reference(self) -> bool:
        """True when each training forward needs ``capture_base_reference``."""
        return self.mode == "act_servo_base_vector_component"

    def capture_base_reference(
        self,
        peft_model: nn.Module,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> None:
        """Record per-token base-model projections for the next forward.

        Runs ``peft_model`` on the batch under ``no_grad`` with the LoRA
        adapter disabled (peft ``disable_adapter``: the frozen base model)
        and stores, per active layer, ``h_base @ v`` as ``[batch, seq]``.
        The adapter-disabled pass is deterministic even in train mode:
        LoRA dropout sits on the disabled adapter branch, and the base
        blocks have no active dropout for the models used here (Qwen2
        attention_dropout = 0). Call once per microbatch, immediately
        before the training forward on the SAME ``input_ids``.
        """
        if not self.needs_base_reference:
            raise RuntimeError(
                f"capture_base_reference is only used by "
                f"act_servo_base_vector_component (mode={self.mode!r})")
        self._base_proj = {}
        self._base_mask = (attention_mask.detach().bool()
                           if attention_mask is not None else None)
        self.remove_hooks()
        self.register_hooks()
        self._capturing = True
        try:
            with torch.no_grad(), peft_model.disable_adapter():
                peft_model(input_ids=input_ids,
                           attention_mask=attention_mask)
        finally:
            self._capturing = False
            self.remove_hooks()
        missing = [l for l in self._active_layers if l not in self._base_proj]
        if missing:
            raise RuntimeError(
                f"base reference forward did not reach layers {missing}")

    def clear_base_reference(self) -> None:
        """Drop the stored base reference (end of a microbatch)."""
        self._base_proj = {}
        self._base_mask = None

    def _make_forward_hook(self, layer_idx: int) -> Callable:
        """Create a forward hook that registers a tensor-level grad hook."""
        v = self._vectors[layer_idx]
        mode = self.mode

        servo_target = self.act_target_by_layer.get(layer_idx)
        servo_offset = self._servo_offsets.get(layer_idx)
        basis_B = (self._basis or {}).get(layer_idx)
        servo_c = (self._servo_targets or {}).get(layer_idx)

        def hook(module: nn.Module, input, output):
            # Handle tuple outputs
            tensor = output[0] if isinstance(output, tuple) else output

            if self._capturing:
                # Base-reference pass (no_grad, adapter disabled): record
                # the per-token projection target, register nothing
                self._base_proj[layer_idx] = (
                    tensor.detach().float() @ v.to(tensor.device)
                )
                return

            if tensor.requires_grad:
                servo_sign = None
                if mode == "act_servo_vector_component":
                    # Closed-loop control signal: per-token sign of the
                    # activation-projection error vs the target level,
                    # measured on THIS forward pass
                    with torch.no_grad():
                        if basis_B is not None:
                            b_fwd = basis_B.to(
                                device=tensor.device, dtype=tensor.dtype
                            )
                            u = tensor.detach() @ b_fwd.T  # [batch, seq, k]
                            servo_sign = torch.sign(
                                u - servo_c.to(
                                    device=tensor.device, dtype=tensor.dtype
                                )
                            )
                        elif servo_offset is not None:
                            # Intrinsic level: the projection of h without
                            # the excluded constant (class docstring)
                            act_proj = (tensor.detach().float()
                                        @ v.to(tensor.device).float())
                            servo_sign = torch.sign(
                                act_proj - servo_offset - servo_target)
                        else:
                            vec_fwd = v.to(
                                device=tensor.device, dtype=tensor.dtype
                            )
                            act_proj = tensor.detach() @ vec_fwd  # [batch, seq]
                            servo_sign = torch.sign(act_proj - servo_target)
                elif mode == "act_servo_base_vector_component":
                    base = self._base_proj.pop(layer_idx, None)
                    if base is None:
                        raise RuntimeError(
                            f"act_servo_base: no base reference for layer "
                            f"{layer_idx}; call capture_base_reference on "
                            "this microbatch before the training forward")
                    act_proj = tensor.detach().float() @ v.to(tensor.device)
                    if base.shape != act_proj.shape:
                        raise RuntimeError(
                            f"act_servo_base: base reference shape "
                            f"{tuple(base.shape)} != activation shape "
                            f"{tuple(act_proj.shape)} at layer {layer_idx} "
                            "(reference captured on a different batch?)")
                    err = act_proj - base
                    servo_sign = torch.sign(err)
                    if self.collect_stats:
                        m = self._base_mask
                        e = (err.abs()[m] if m is not None
                             and m.shape == err.shape else err.abs())
                        self._servo_err.append((layer_idx, e.mean().item()))

                def grad_hook(grad: torch.Tensor) -> torch.Tensor:
                    vec = v.to(device=grad.device, dtype=grad.dtype)
                    # grad: [batch, seq, hidden], vec: [hidden]
                    proj_scalar = grad @ vec  # [batch, seq]
                    proj = proj_scalar.unsqueeze(-1) * vec  # [batch, seq, hidden]
                    scale = (
                        self._current_scale
                        * self.scale_by_layer.get(layer_idx, 1.0)
                    )
                    if basis_B is not None:
                        # Subspace variant: operate on the span(B)
                        # coordinates, leave the complement untouched
                        # (keep zeroes it by definition)
                        b_g = basis_B.to(device=grad.device, dtype=grad.dtype)
                        coords = grad @ b_g.T  # [batch, seq, k]
                        span = coords @ b_g   # [batch, seq, hidden]
                        if mode == "remove_vector_component":
                            new_grad = grad - span
                        elif mode == "keep_vector_component":
                            new_grad = span
                        elif mode == "flip_vector_component":
                            new_grad = grad - 2 * span
                        else:
                            if self.scale_ref == "grad_norm":
                                mag_k = grad.norm(
                                    dim=-1, keepdim=True
                                ).expand_as(coords)
                            else:
                                mag_k = coords.abs()
                            if mode == "pos_sign_vector_component":
                                sign_k = torch.ones_like(coords)
                            elif mode == "neg_sign_vector_component":
                                sign_k = -torch.ones_like(coords)
                            else:  # act_servo_vector_component
                                sign_k = servo_sign.to(grad.dtype)
                            new_grad = grad - span + (sign_k * scale * mag_k) @ b_g
                    elif mode == "remove_vector_component":
                        new_grad = grad - proj
                    elif mode == "keep_vector_component":
                        new_grad = proj
                    elif mode == "flip_vector_component":
                        new_grad = grad - 2 * proj
                    else:
                        # Scalable modes: the enforced/added component
                        # magnitude is scale * (|grad @ v| or ||grad||),
                        # see class docstring.
                        if self.scale_ref == "grad_norm":
                            magnitude = grad.norm(dim=-1)  # [batch, seq]
                        else:
                            magnitude = proj_scalar.abs()
                        term = scale * magnitude.unsqueeze(-1) * vec
                        if mode == "pos_sign_vector_component":
                            new_grad = grad - proj + term
                        elif mode == "neg_sign_vector_component":
                            new_grad = grad - proj - term
                        elif mode == "mean_shift_vector_component":
                            new_grad = grad + term
                        else:  # act_servo / act_servo_base
                            new_grad = (
                                grad - proj
                                + servo_sign.to(grad.dtype).unsqueeze(-1)
                                * term
                            )

                    if self.collect_stats:
                        with torch.no_grad():
                            grad_norm = grad.norm(dim=-1)
                            # Attention-masked padding has exactly-zero
                            # gradient; keep only real tokens so the means
                            # reflect the enforced attenuation.
                            mask = grad_norm > 0
                            if mask.any():
                                new_norm = new_grad.norm(dim=-1)
                                eps = 1e-12
                                cos_pre = (
                                    proj_scalar / (grad_norm + eps)
                                )[mask].mean().detach()
                                cos_post = (
                                    (new_grad @ vec) / (new_norm + eps)
                                )[mask].mean().detach()
                                self._stats.append((layer_idx, cos_pre, cos_post))

                    return new_grad

                tensor.register_hook(grad_hook)

        return hook

    def register_hooks(self):
        """Register forward hooks on target layers."""
        self.remove_hooks()

        for layer_idx in self._active_layers:
            module = self._layer_modules[layer_idx - 1]
            handle = module.register_forward_hook(
                self._make_forward_hook(layer_idx)
            )
            self._handles.append(handle)

    def remove_hooks(self):
        """Remove all registered forward hooks."""
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    @property
    def active_layers(self) -> List[int]:
        """Return list of layer indices being manipulated."""
        return self._active_layers.copy()

    def __enter__(self):
        self.register_hooks()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.remove_hooks()
        return False

    def __repr__(self) -> str:
        return (
            f"GradientManipulationManager("
            f"mode='{self.mode}', "
            f"active_layers={self._active_layers}, "
            f"scale_init={self.scale_init}, "
            f"scale_final={self.scale_final}, "
            f"scale_tau={self.scale_tau}, "
            f"scale_ref='{self.scale_ref}')"
        )
