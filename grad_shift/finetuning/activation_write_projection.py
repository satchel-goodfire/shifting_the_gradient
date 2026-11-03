"""Activation-write projection: remove a trait direction from the
activations that residual-writing modules add to the stream.

The weight-projection experiment (weight_grad_projection.py) proved the
learned adapter never needs to WRITE into the trait direction — frozen
base weights transporting drifted inputs carry the whole activation
displacement. This manager lesions that channel too: a forward hook on
every residual-writing module (``self_attn.o_proj``, ``mlp.down_proj``)
in the zone projects the trait component out of the module OUTPUT
before it is added to the residual stream:

    out <- out - (out @ v) v        (or the span(basis) generalization)

Under this muzzle no write — learned or frozen — can move the stream
along the direction; within the projected zone the stream's trait
content is pinned at whatever entered from below (the embedding, and
blocks outside the zone). The projection is part of the forward GRAPH
(not detached): training optimizes the muzzled computation, and
gradients propagate through the projection.

The muzzle is a property of the computation, not the weights: models
trained under it must be EVALUATED under it (see eval_trait's
--write_projection_* flags); an unmuzzled eval of the same checkpoint
probes what the weights alone learned.

Same conventions as the sibling managers: embedding-inclusive layer
indexing (index N = transformer block N-1), vector files ``[hidden]``
(broadcast) or ``[num_layers, hidden]`` (per-layer rows), basis files
``[k, hidden]`` or ``[num_layers, k, hidden]`` with per-layer QR
orthonormalization and a rank fail-fast.
"""

import logging
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from ..hooks import find_transformer_blocks
from .subspace import orthonormal_rows

logger = logging.getLogger(__name__)

RESIDUAL_WRITER_SUFFIXES = ("self_attn.o_proj", "mlp.down_proj")

VALID_WRITE_PROJECTION_MODES = ("remove",)


class ActivationWriteProjectionManager:
    """Project a trait direction out of residual-write module outputs.

    Args:
        model: The model (PEFT-wrapped or plain; hooks attach to the
            outer module so LoRA-combined outputs are projected).
        vector_path: Trait-vector ``.pt`` (broadcast or per-layer).
        mode: Only ``"remove"``.
        layers: Embedding-inclusive layer indices whose blocks' writes
            are muzzled (*None* = all transformer blocks).
        basis_path: Optional multi-direction basis; overrides
            vector_path for the projection.
        collect_stats: Accumulate mean fraction of write norm removed
            per hook call; retrieve with ``pop_stats()``.
    """

    def __init__(
        self,
        model: nn.Module,
        vector_path: Optional[str],
        mode: str = "remove",
        layers: Optional[List[int]] = None,
        basis_path: Optional[str] = None,
        collect_stats: bool = False,
    ):
        if mode not in VALID_WRITE_PROJECTION_MODES:
            raise ValueError(
                f"mode must be one of {VALID_WRITE_PROJECTION_MODES}, "
                f"got '{mode}'"
            )
        self.mode = mode
        self.collect_stats = collect_stats
        self._stats: List[Tuple[float, float]] = []
        self._handles: List[torch.utils.hooks.RemovableHandle] = []

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

        # Per-layer orthonormal direction matrix V_l [k, hidden] — same
        # loading conventions and rank fail-fast as the sibling managers.
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
            if vector_path is None:
                raise ValueError("vector_path or basis_path is required")
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

        # Locate the residual-writing modules per active block. Match the
        # OUTER module (the PEFT lora.Linear wrapper when present) so the
        # combined base+adapter output is projected exactly once —
        # suffix matching excludes ".base_layer" and lora submodules.
        self._target_modules: List[Tuple[int, str, nn.Module]] = []
        for l in self._active_layers:
            block = layer_modules[l - 1]
            found = [
                (l, name, mod)
                for name, mod in block.named_modules()
                if any(name.endswith(s) for s in RESIDUAL_WRITER_SUFFIXES)
            ]
            if len(found) != 2:
                raise ValueError(
                    f"Expected exactly 2 residual-writing modules "
                    f"({RESIDUAL_WRITER_SUFFIXES}) in block for layer {l}, "
                    f"found {[n for _, n, _ in found]}"
                )
            self._target_modules.extend(found)

        self._v_cache: Dict[tuple, torch.Tensor] = {}

        logger.info(
            f"ActivationWriteProjectionManager: mode={mode}, "
            f"layers={self._active_layers}, "
            f"{len(self._target_modules)} residual writers muzzled, "
            f"k={next(iter(self._dirs.values())).shape[0]}, "
            f"basis={basis_path or vector_path}"
        )

    def _v_for(self, layer: int, t: torch.Tensor) -> torch.Tensor:
        key = (layer, t.device, t.dtype)
        if key not in self._v_cache:
            self._v_cache[key] = self._dirs[layer].to(
                device=t.device, dtype=t.dtype
            )
        return self._v_cache[key]

    def _make_hook(self, layer_idx: int):
        def hook(module, inputs, output):
            tensor = output[0] if isinstance(output, tuple) else output
            v = self._v_for(layer_idx, tensor)
            coords = tensor @ v.T  # [batch, seq, k]
            projected = tensor - coords @ v
            if self.collect_stats:
                with torch.no_grad():
                    removed = float((coords.float() ** 2).sum())
                    total = float((tensor.float() ** 2).sum())
                self._stats.append((removed, total))
            if isinstance(output, tuple):
                return (projected,) + output[1:]
            return projected
        return hook

    def register_hooks(self):
        """Attach the write-projection forward hooks (idempotent)."""
        self.remove_hooks()
        for layer_idx, _, mod in self._target_modules:
            self._handles.append(
                mod.register_forward_hook(self._make_hook(layer_idx))
            )

    def remove_hooks(self):
        for h in self._handles:
            h.remove()
        self._handles.clear()

    def pop_stats(self) -> Optional[Dict]:
        """Mean fraction of the writes' squared norm removed since the
        last call (plus hook-call count). *None* if nothing collected.
        Schema disjoint from the other managers' (awp_*)."""
        if not self._stats:
            return None
        removed = sum(r for r, _ in self._stats)
        total = sum(t for _, t in self._stats)
        stats = {
            "awp_frac_write_removed": (removed / total) if total > 0 else 0.0,
            "awp_n_hook_calls": len(self._stats),
            "awp_n_modules": len(self._target_modules),
        }
        self._stats.clear()
        return stats

    @property
    def active_layers(self) -> List[int]:
        return self._active_layers.copy()

    def __enter__(self):
        self.register_hooks()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.remove_hooks()
        return False

    def __repr__(self) -> str:
        return (
            f"ActivationWriteProjectionManager(mode='{self.mode}', "
            f"active_layers={self._active_layers}, "
            f"n_modules={len(self._target_modules)})"
        )
