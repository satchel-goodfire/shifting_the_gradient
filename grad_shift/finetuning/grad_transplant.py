"""Gradient transplant manager: move the activation-gradient field of a
donor forward/backward pass onto a recipient pass of the same batch.

Motivation: a layer's weight update factorizes as ``dL/dW ~ delta (x) a`` —
the activation gradient arriving from above times the forward activation.
Steering during training (PPS) changes BOTH factors; sign-forcing gradient
manipulations change only ``delta``, and only its component along one axis,
with 28 mutually inconsistent per-layer edits. Transplanting lets us mix the
factors causally:

- ``delta_full``:   donor = steered pass, recipient = unsteered pass.
  Recipient trains with the donor's ENTIRE backward field on its own
  (unsteered) activations. Tests whether PPS's backward field alone
  (off-axis structure + conditionality included) carries the defense.
- ``delta_along``:  like ``delta_full`` but only the component of the
  donor-recipient gradient difference ALONG the trait vector is
  transplanted; the orthogonal part stays the recipient's own.
- ``delta_orthog``: complement of ``delta_along`` — only the ORTHOGONAL
  part of the difference is transplanted.
- ``activation``:   donor = unsteered pass, recipient = steered pass.
  Recipient trains on steered activations but with the unsteered backward
  field. Tests whether the forward/activation shift alone carries it.
- ``activation_along``: donor = unsteered pass, recipient = steered pass,
  and only the component ALONG the trait vector is swapped for the
  donor's; the orthogonal part stays the steered pass's own. At a single
  layer this removes PPS's gradient shift along the trait vector while the
  vector stays in the forward pass and every off-vector effect of it on
  the backward field is kept (``experiments/pv_swap``).

Unlike the sign-forcing modes in ``grad_manipulation.py``, every
transplanted field is a genuine backward field of SOME computation (the
donor's), generated self-consistently by the chain rule from a single
intervention — no standing bias term is injected.

Donor deltas are computed with ``torch.autograd.grad`` against the captured
layer-output tensors, so the donor pass deposits no parameter gradients.
"""

import logging
from typing import Dict, List, Optional

import torch
import torch.nn as nn

from ..hooks import find_transformer_blocks
from .subspace import orthonormal_rows

logger = logging.getLogger(__name__)

VALID_TRANSPLANT_MODES = (
    "delta_full",
    "delta_along",
    "delta_orthog",
    "activation",
    "activation_along",
)

# Modes whose donor pass is the steered one (recipient unsteered)
DONOR_STEERED_MODES = ("delta_full", "delta_along", "delta_orthog")
# Modes that transplant only part of the donor-recipient difference, split
# along/orthogonal to the trait vector (they need a vector_path)
SPLIT_MODES = ("delta_along", "delta_orthog", "activation_along")
# Split modes that keep the along-vector part of the difference
ALONG_MODES = ("delta_along", "activation_along")


class GradientTransplantManager:
    """Capture per-layer activation gradients from a donor pass and replace
    the recipient pass's gradients with them during backward.

    Uses the same embedding-inclusive layer indexing as steering and
    gradient manipulation (index N = transformer block N-1). Projection
    modes (``delta_along``/``delta_orthog``) need a trait vector; full
    replacement modes do not, but accept one for stats.

    Typical trainer flow per microbatch::

        mgr.begin_capture()            # forward hooks stash layer outputs
        donor_loss = model(**batch).loss     # donor condition applied
        mgr.end_capture(donor_loss)    # autograd.grad -> stored deltas

        mgr.register_replace_hooks()   # tensor hooks substitute deltas
        loss = model(**batch).loss           # recipient condition applied
        loss.backward()                # param grads = transplanted field
        mgr.remove_hooks()

    Args:
        model: The transformer model (or its base model).
        mode: One of ``VALID_TRANSPLANT_MODES``.
        vector_path: Path to the trait-vector ``.pt`` (``[hidden]`` or
            ``[num_layers, hidden]`` embedding-inclusive). Required for the
            projection modes; optional (stats only) otherwise.
        layers: Embedding-inclusive layer indices to transplant at
            (*None* = all transformer blocks).
        collect_stats: Accumulate mean tokenwise cosine along the trait
            vector before/after replacement (zero-grad padding tokens
            excluded); retrieve with ``pop_stats()``.
    """

    def __init__(
        self,
        model: nn.Module,
        mode: str,
        vector_path: Optional[str] = None,
        layers: Optional[List[int]] = None,
        subspace_basis_path: Optional[str] = None,
        collect_stats: bool = False,
    ):
        if mode not in VALID_TRANSPLANT_MODES:
            raise ValueError(
                f"mode must be one of {VALID_TRANSPLANT_MODES}, got '{mode}'"
            )
        if mode in SPLIT_MODES and vector_path is None:
            raise ValueError(f"mode '{mode}' requires a vector_path")
        if subspace_basis_path is not None and mode not in (
                "delta_along", "delta_orthog"):
            raise ValueError(
                "subspace_basis_path is only valid with delta_along/"
                f"delta_orthog, got mode='{mode}'"
            )

        # Optional multi-direction subspace for the along/orthog split:
        # a [k, hidden] basis (rows orthonormalized here), broadcast to all
        # active layers. When set, "along" means the projection onto
        # span(basis) rather than onto the single per-layer vector.
        self._basis: Optional[torch.Tensor] = None
        if subspace_basis_path is not None:
            raw = torch.load(subspace_basis_path, map_location="cpu").float()
            if raw.dim() != 2 or raw.shape[0] >= raw.shape[1]:
                raise ValueError(
                    f"subspace basis must be [k, hidden] with k << hidden, "
                    f"got {tuple(raw.shape)}"
                )
            # Rank-checked orthonormalization (shared helper): collinear
            # rows would silently target arbitrary QR completion directions
            self._basis = orthonormal_rows(raw, "broadcast")

        self.model = model
        self.mode = mode
        self.collect_stats = collect_stats
        self._handles: List[torch.utils.hooks.RemovableHandle] = []
        self._captured_outputs: Dict[int, torch.Tensor] = {}
        self._donor_deltas: Dict[int, torch.Tensor] = {}
        self._stats: List[tuple] = []

        self._pattern, self._layer_modules, self._layer_names = (
            find_transformer_blocks(model)
        )
        self.num_layers = len(self._layer_modules)

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

        # Load and normalize per-layer unit vectors (same convention as
        # GradientManipulationManager)
        self._vectors: Dict[int, torch.Tensor] = {}
        if vector_path is not None:
            raw = torch.load(vector_path, map_location="cpu")
            if isinstance(raw, dict):
                raw = raw[list(raw.keys())[0]]
            for l in self._active_layers:
                v = (raw if raw.dim() == 1 else raw[l]).float()
                norm = v.norm()
                if norm > 0:
                    v = v / norm
                self._vectors[l] = v

        logger.info(
            f"GradientTransplantManager: mode={mode}, "
            f"layers={self._active_layers}, vector_path={vector_path}"
        )

    # ── Donor side ───────────────────────────────────────────────────

    def begin_capture(self):
        """Register forward hooks that stash layer-output tensors."""
        self.remove_hooks()
        self._captured_outputs.clear()
        self._donor_deltas.clear()

        def make_hook(layer_idx):
            def hook(module, input, output):
                tensor = output[0] if isinstance(output, tuple) else output
                self._captured_outputs[layer_idx] = tensor
            return hook

        for layer_idx in self._active_layers:
            module = self._layer_modules[layer_idx - 1]
            self._handles.append(
                module.register_forward_hook(make_hook(layer_idx))
            )

    def end_capture(self, donor_loss: torch.Tensor, loss_scale: float = 1.0):
        """Compute donor activation gradients without touching param grads.

        ``loss_scale`` must match the scaling the RECIPIENT backward
        applies to its loss — under gradient accumulation,
        ``Accelerator.backward`` divides by ``gradient_accumulation_steps``,
        so the trainer passes ``1/accum`` here. A mismatch injects donor
        deltas at the wrong magnitude relative to the recipient field
        (the 2026-08-11 P0: donor at 8x recipient scale invalidated every
        transplant arm trained with accumulation > 1).
        """
        self.remove_hooks()
        missing = [l for l in self._active_layers
                   if l not in self._captured_outputs]
        if missing:
            raise RuntimeError(
                "Donor capture incomplete: got outputs for "
                f"{sorted(self._captured_outputs)} but expected "
                f"{self._active_layers}"
            )
        ordered = [self._captured_outputs[l] for l in self._active_layers]
        grads = torch.autograd.grad(donor_loss * loss_scale, ordered)
        self._donor_deltas = {
            l: g.detach() for l, g in zip(self._active_layers, grads)
        }
        self._captured_outputs.clear()

    # ── Recipient side ───────────────────────────────────────────────

    def register_replace_hooks(self):
        """Register forward hooks that attach gradient-replacing tensor hooks."""
        self.remove_hooks()
        if not self._donor_deltas:
            raise RuntimeError("No donor deltas captured — run capture first")

        def make_hook(layer_idx):
            def hook(module, input, output):
                tensor = output[0] if isinstance(output, tuple) else output
                if not tensor.requires_grad:
                    return

                def grad_hook(grad: torch.Tensor) -> torch.Tensor:
                    donor = self._donor_deltas[layer_idx].to(
                        device=grad.device, dtype=grad.dtype
                    )
                    if self.mode in ("delta_full", "activation"):
                        new_grad = donor
                    else:
                        diff = donor - grad
                        if self._basis is not None:
                            b = self._basis.to(
                                device=grad.device, dtype=grad.dtype
                            )
                            proj = (diff @ b.T) @ b  # span(basis) projection
                        else:
                            v = self._vectors[layer_idx].to(
                                device=grad.device, dtype=grad.dtype
                            )
                            proj = (diff @ v).unsqueeze(-1) * v
                        if self.mode in ALONG_MODES:
                            new_grad = grad + proj
                        else:  # delta_orthog
                            new_grad = grad + (diff - proj)

                    if self.collect_stats and layer_idx in self._vectors:
                        with torch.no_grad():
                            v = self._vectors[layer_idx].to(
                                device=grad.device, dtype=grad.dtype
                            )
                            grad_norm = grad.norm(dim=-1)
                            mask = grad_norm > 0
                            if mask.any():
                                eps = 1e-12
                                new_norm = new_grad.norm(dim=-1)
                                cos_pre = (
                                    (grad @ v) / (grad_norm + eps)
                                )[mask].mean().detach()
                                cos_post = (
                                    (new_grad @ v) / (new_norm + eps)
                                )[mask].mean().detach()
                                self._stats.append(
                                    (layer_idx, cos_pre, cos_post)
                                )
                    return new_grad

                tensor.register_hook(grad_hook)

            return hook

        for layer_idx in self._active_layers:
            module = self._layer_modules[layer_idx - 1]
            self._handles.append(
                module.register_forward_hook(make_hook(layer_idx))
            )

    def remove_hooks(self):
        """Remove all registered forward hooks."""
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def clear_donor(self):
        """Drop stored donor deltas and any captured forward outputs
        (call after the recipient backward; also safe on error paths,
        where captured outputs would otherwise pin the donor graph)."""
        self._donor_deltas.clear()
        self._captured_outputs.clear()

    @property
    def donor_steered(self) -> bool:
        """Whether the donor pass is the steered one (else the recipient is)."""
        return self.mode in DONOR_STEERED_MODES

    @property
    def active_layers(self) -> List[int]:
        return self._active_layers.copy()

    def pop_stats(self) -> Optional[Dict]:
        """Mean tokenwise cosine along the trait vector before/after
        replacement, overall and per layer, over tokens with nonzero
        gradient. Same schema as GradientManipulationManager.pop_stats
        (minus the schedule scale). *None* if nothing collected."""
        if not self._stats:
            return None
        pre = torch.stack([p for _, p, _ in self._stats])
        post = torch.stack([q for _, _, q in self._stats])
        by_pre: Dict[int, List[torch.Tensor]] = {}
        by_post: Dict[int, List[torch.Tensor]] = {}
        for l, p, q in self._stats:
            by_pre.setdefault(l, []).append(p)
            by_post.setdefault(l, []).append(q)
        stats = {
            "cos_pre": pre.mean().item(),
            "cos_post": post.mean().item(),
            "n_hook_calls": len(self._stats),
            "cos_pre_by_layer": {
                l: torch.stack(v).mean().item() for l, v in sorted(by_pre.items())
            },
            "cos_post_by_layer": {
                l: torch.stack(v).mean().item() for l, v in sorted(by_post.items())
            },
        }
        self._stats.clear()
        return stats

    def __repr__(self) -> str:
        return (
            f"GradientTransplantManager(mode='{self.mode}', "
            f"active_layers={self._active_layers})"
        )
