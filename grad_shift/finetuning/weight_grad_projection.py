"""Weight-gradient projection: ablate the LoRA weight-update component
that writes into a trait direction.

Motivation. Activation-gradient ablation (grad_manipulation.py) zeroes
the per-token learning SIGNAL along a direction, but the realized weight
updates can still move activations along it (measured: the span-remove
arm's layer-20 projections trace defenseless training exactly). This
manager instead constrains what each module LEARNS TO WRITE: a weight
update changes a module's output by ``dW @ a``, and its write into a
trait direction ``v`` is ``v^T dW @ a``. Killing the output-row
component of the update along ``v`` for every module that writes into
the residual stream makes the learned delta unable to move that stream
along ``v`` — for any input, exactly.

LoRA makes the guarantee exact. The adapter delta of a targeted linear
is ``dW_eff = B @ A`` with ``B`` zero-initialized, so ``v^T B = 0`` at
step 0. Maintaining that invariant needs only:

1. ``project_grads()``  (before ``optimizer.step``):
   ``grad_B <- (I - V^T V) grad_B``
2. ``project_weights()`` (after ``optimizer.step``):
   ``B <- (I - V^T V) B`` — removes AdamW moment-rescaling and
   weight-decay leakage, which a gradient-only projection cannot.

With ``v^T B == 0`` identically, ``v^T (B @ A) == 0`` for all inputs and
all ``A`` — no correction on ``lora_A`` is needed. What this does NOT
constrain: the frozen base weights transporting drifted inputs into
``v`` (``W0 @ (x + dx)`` moves along ``v`` freely). The experiment
therefore factors trait acquisition into "modules learn new v-writers"
(provably dead here) vs "existing writers fed drifted inputs".

Only modules whose outputs are added to the residual stream in hidden
coordinates are targeted: ``self_attn.o_proj`` and ``mlp.down_proj``
(q/k/v/gate/up project into head/intermediate spaces where ``v`` does
not live).

Uses the same embedding-inclusive layer indexing as steering (index N =
transformer block N-1); a block's write lands in the stream at its own
output, so "layers >= 20" constrains blocks 20..28's writes.
"""

import logging
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from ..hooks import find_transformer_blocks
from .subspace import orthonormal_rows

logger = logging.getLogger(__name__)

# Module-name suffixes whose outputs are residual-stream writes in
# hidden coordinates.
RESIDUAL_WRITER_SUFFIXES = ("self_attn.o_proj", "mlp.down_proj")

VALID_WEIGHT_PROJECTION_MODES = ("remove",)


class WeightGradProjectionManager:
    """Project trait-direction components out of LoRA-B gradients and
    weights for residual-writing modules.

    Args:
        model: The PEFT-wrapped model (LoRA already applied).
        vector_path: Trait-vector ``.pt`` (``[hidden]`` broadcast or
            ``[num_layers, hidden]`` per-layer, embedding-inclusive).
        mode: Only ``"remove"``.
        layers: Embedding-inclusive layer indices whose blocks' writes
            are constrained (*None* = all transformer blocks).
        basis_path: Optional ``[k, hidden]`` (broadcast) or
            ``[num_layers, k, hidden]`` basis; the constrained subspace
            is span(basis) instead of the single vector. Rows are
            orthonormalized per layer (QR).
        collect_stats: Accumulate per-step stats; retrieve with
            ``pop_stats()``.
    """

    def __init__(
        self,
        model: nn.Module,
        vector_path: str,
        mode: str = "remove",
        layers: Optional[List[int]] = None,
        basis_path: Optional[str] = None,
        collect_stats: bool = False,
    ):
        if mode not in VALID_WEIGHT_PROJECTION_MODES:
            raise ValueError(
                f"mode must be one of {VALID_WEIGHT_PROJECTION_MODES}, "
                f"got '{mode}'"
            )
        self.mode = mode
        self.collect_stats = collect_stats
        self._stats: List[Tuple[float, float, float]] = []

        _, layer_modules, _ = find_transformer_blocks(model)
        self.num_layers = len(layer_modules)
        if layers is None:
            self._active_layers = list(range(1, self.num_layers + 1))
        else:
            self._active_layers = [
                l if l >= 0 else (self.num_layers + 1) + l for l in layers
            ]
        for l in self._active_layers:
            if l < 1 or l > self.num_layers:
                raise ValueError(
                    f"Layer index {l} out of range. Valid: 1..{self.num_layers}"
                )

        # Per-layer orthonormal direction matrix V_l [k, hidden]
        self._dirs: Dict[int, torch.Tensor] = {}
        if basis_path is not None:
            raw = torch.load(basis_path, map_location="cpu").float()
            if raw.dim() not in (2, 3) or raw.shape[-2] >= raw.shape[-1]:
                raise ValueError(
                    "basis must be [k, hidden] (broadcast) or "
                    "[num_layers, k, hidden] (embedding-inclusive) with "
                    f"k << hidden, got {tuple(raw.shape)}"
                )
            for l in self._active_layers:
                if raw.dim() == 3:
                    if l >= raw.shape[0]:
                        raise ValueError(
                            f"Basis file has {raw.shape[0]} layers but "
                            f"layer index {l} requested"
                        )
                    dirs = raw[l].clone()
                else:
                    dirs = raw.clone()
                self._dirs[l] = orthonormal_rows(dirs, l)
        else:
            raw = torch.load(vector_path, map_location="cpu")
            if isinstance(raw, dict):
                raw = raw[list(raw.keys())[0]]
            raw = raw.float()
            for l in self._active_layers:
                if raw.dim() == 1:
                    v = raw
                elif raw.dim() == 2:
                    if l >= raw.shape[0]:
                        raise ValueError(
                            f"Vector file has {raw.shape[0]} layers but "
                            f"layer index {l} requested"
                        )
                    v = raw[l]
                else:
                    raise ValueError(
                        f"Expected vector with 1 or 2 dims, got {raw.shape}"
                    )
                norm = v.norm()
                if norm == 0:
                    raise ValueError(f"zero-norm vector at layer {l}")
                self._dirs[l] = (v / norm).unsqueeze(0).contiguous()

        # Locate the trainable lora_B weights of residual-writing modules
        # in the active blocks. PEFT naming example:
        #   base_model.model.model.layers.19.self_attn.o_proj.lora_B.default.weight
        self._targets: List[Tuple[int, str, torch.nn.Parameter]] = []
        for l in self._active_layers:
            block = layer_modules[l - 1]
            found_here = []
            for name, p in block.named_parameters():
                if (
                    p.requires_grad
                    and "lora_B" in name
                    and name.endswith(".weight")
                    and any(s in name for s in RESIDUAL_WRITER_SUFFIXES)
                ):
                    found_here.append((l, name, p))
            if not found_here:
                raise ValueError(
                    f"No trainable lora_B params found on "
                    f"{RESIDUAL_WRITER_SUFFIXES} in block for layer {l} — "
                    "is LoRA applied with o_proj/down_proj targets?"
                )
            for l_, name, p in found_here:
                hidden = self._dirs[l_].shape[1]
                if p.shape[0] != hidden:
                    raise ValueError(
                        f"lora_B out dim {p.shape[0]} != hidden {hidden} "
                        f"for {name} (layer {l_}) — not a residual write?"
                    )
            self._targets.extend(found_here)

        # Per-(layer, device, dtype) cache of V
        self._v_cache: Dict[tuple, torch.Tensor] = {}

        # The invariant maintained is v^T (B - B_init) == 0: only NEW
        # learning is constrained. For fresh LoRA (B zero-init) this is
        # v^T B == 0 exactly; when resuming from a peft_checkpoint with a
        # nonzero inherited v-component, that component is PRESERVED
        # (one-shot ablating it would confound the arm with a partial
        # model edit at step 1).
        self._init_coords: Dict[int, torch.Tensor] = {}
        inherited = 0.0
        for i, (layer, _, p) in enumerate(self._targets):
            v = self._dirs[layer].to(device=p.device, dtype=torch.float32)
            c0 = (v @ p.data.float()).detach()
            self._init_coords[i] = c0
            inherited += float(c0.norm() ** 2)
        if inherited ** 0.5 > 1e-6:
            logger.warning(
                "WeightGradProjectionManager: inherited lora_B carries a "
                f"nonzero trait-direction component (||V@B_init|| = "
                f"{inherited ** 0.5:.4f}, resumed checkpoint?). It will be "
                "PRESERVED; only new learning is projected out."
            )

        logger.info(
            f"WeightGradProjectionManager: mode={mode}, "
            f"layers={self._active_layers}, "
            f"{len(self._targets)} target lora_B params, "
            f"k={next(iter(self._dirs.values())).shape[0]}, "
            f"basis={basis_path or vector_path}"
        )

    def _v_for(self, layer: int, p: torch.Tensor) -> torch.Tensor:
        key = (layer, p.device, p.dtype)
        if key not in self._v_cache:
            self._v_cache[key] = self._dirs[layer].to(
                device=p.device, dtype=p.dtype
            )
        return self._v_cache[key]

    @torch.no_grad()
    def project_grads(self) -> None:
        """Remove the span(V) output-row component from every target's
        gradient. Call after backward, before ``optimizer.step()``."""
        removed_sq = 0.0
        total_sq = 0.0
        for layer, _, p in self._targets:
            if p.grad is None:
                continue
            v = self._v_for(layer, p.grad)
            coords = v @ p.grad  # [k, r]
            if self.collect_stats:
                removed_sq += float((coords ** 2).sum())
                total_sq += float((p.grad ** 2).sum())
            p.grad -= v.T @ coords
        if self.collect_stats:
            self._last_grad_stats = (removed_sq, total_sq)

    @torch.no_grad()
    def project_weights(self) -> None:
        """Re-project the lora_B weights themselves. Call after
        ``optimizer.step()`` — removes optimizer-moment and weight-decay
        leakage so ``V @ (B - B_init) == 0`` holds exactly at every
        step (``V @ B == 0`` for fresh, zero-initialized LoRA)."""
        leak_sq = 0.0
        for i, (layer, _, p) in enumerate(self._targets):
            v = self._v_for(layer, p.data)
            c0 = self._init_coords[i].to(device=p.device, dtype=p.dtype)
            drift = v @ p.data - c0  # [k, r]
            if self.collect_stats:
                leak_sq += float((drift.float() ** 2).sum())
            p.data -= v.T @ drift
        if self.collect_stats:
            removed_sq, total_sq = getattr(
                self, "_last_grad_stats", (0.0, 0.0)
            )
            resid_sq = 0.0
            for i, (l, _, p) in enumerate(self._targets):
                v = self._v_for(l, p.data)
                c0 = self._init_coords[i].to(device=p.device, dtype=p.dtype)
                resid_sq += float((v @ p.data - c0).float().norm() ** 2)
            self._stats.append((
                (removed_sq / total_sq) if total_sq > 0 else 0.0,
                leak_sq ** 0.5,
                resid_sq ** 0.5,
            ))

    def pop_stats(self) -> Optional[Dict]:
        """Per-optimizer-step stats since the last call: mean fraction of
        target-grad squared norm removed, mean pre-reprojection weight
        leak ``||V @ B||`` (the AdamW leakage the reprojection kills),
        and max post-reprojection residual (should be ~float epsilon).
        *None* if nothing collected. Schema is intentionally disjoint
        from the activation managers' cos_pre/cos_post."""
        if not self._stats:
            return None
        fracs, leaks, resids = zip(*self._stats)
        stats = {
            "wproj_frac_grad_removed": sum(fracs) / len(fracs),
            "wproj_b_leak_pre": sum(leaks) / len(leaks),
            "wproj_b_resid_post": max(resids),
            "wproj_n_params": len(self._targets),
            "n_steps": len(self._stats),
        }
        self._stats.clear()
        return stats

    @property
    def active_layers(self) -> List[int]:
        return self._active_layers.copy()

    def __repr__(self) -> str:
        return (
            f"WeightGradProjectionManager(mode='{self.mode}', "
            f"active_layers={self._active_layers}, "
            f"n_targets={len(self._targets)})"
        )
