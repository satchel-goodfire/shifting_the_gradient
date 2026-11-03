"""Hook utilities for capturing activations and gradients from transformer models."""

from typing import Dict, List, Optional, Tuple, Callable
import torch
import torch.nn as nn


# Common patterns for finding transformer blocks in different architectures.
# PeftModel wraps the original model so the path gains a "base_model.model."
# prefix.  We list the wrapped variants first so they match before the
# unwrapped ones when both are present.
LAYER_PATTERNS = [
    # PEFT-wrapped variants (must come before unwrapped to match first)
    ("base_model.model.model.layers", "base_model.model.model.layers.{}"),
    ("base_model.model.transformer.h", "base_model.model.transformer.h.{}"),
    ("base_model.model.gpt_neox.layers", "base_model.model.gpt_neox.layers.{}"),
    ("base_model.model.model.decoder.layers", "base_model.model.model.decoder.layers.{}"),
    ("base_model.model.transformer.blocks", "base_model.model.transformer.blocks.{}"),
    # Standard (unwrapped) variants
    ("model.layers", "model.layers.{}"),  # LLaMA, Mistral, Gemma, Qwen
    ("transformer.h", "transformer.h.{}"),  # GPT-2, GPT-Neo
    ("gpt_neox.layers", "gpt_neox.layers.{}"),  # GPT-NeoX
    ("model.decoder.layers", "model.decoder.layers.{}"),  # OPT
    ("transformer.blocks", "transformer.blocks.{}"),  # MPT
]


def find_transformer_blocks(model: nn.Module) -> Tuple[str, List[nn.Module], List[str]]:
    """
    Find transformer blocks in a model by searching for common layer patterns.

    Returns:
        Tuple of (pattern_name, list of layer modules, list of layer names)
    """
    model_modules = dict(model.named_modules())

    for pattern_prefix, pattern_template in LAYER_PATTERNS:
        # Check if this pattern exists in the model
        if pattern_prefix in model_modules:
            container = model_modules[pattern_prefix]

            # Get all child modules (the actual layers)
            layers = []
            layer_names = []

            # Try numbered children
            idx = 0
            while True:
                layer_name = pattern_template.format(idx)
                if layer_name in model_modules:
                    layers.append(model_modules[layer_name])
                    layer_names.append(layer_name)
                    idx += 1
                else:
                    break

            if layers:
                return pattern_prefix, layers, layer_names

    # Fallback: try to find any module with "layers" or "h" in the name
    for name, module in model.named_modules():
        if hasattr(module, "__len__") and len(list(module.children())) > 0:
            children = list(module.named_children())
            # Check if children look like numbered layers
            if children and children[0][0].isdigit():
                layers = [child for _, child in children]
                layer_names = [f"{name}.{child_name}" for child_name, _ in children]
                return name, layers, layer_names

    raise ValueError(
        "Could not find transformer blocks in model. "
        "Supported architectures: LLaMA, GPT-2, GPT-Neo, GPT-NeoX, OPT, MPT, Mistral, Gemma"
    )


class HookManager:
    """
    Manages forward and backward hooks for capturing activations and gradients.

    Uses embedding-inclusive layer indexing convention:
    - Index 0 = embedding layer (not hookable)
    - Index N = transformer block N-1 (model.layers[N-1])

    This matches the convention used by persona_vectors_repo and steering
    vector files, where index 0 is a zero placeholder for the embedding.
    """

    def __init__(
        self,
        model: nn.Module,
        layers: Optional[List[int]] = None,
    ):
        """
        Initialize the hook manager.

        Args:
            model: The transformer model
            layers: Optional list of embedding-inclusive layer indices to hook
                (None = all transformer blocks, i.e. range(1, num_model_layers + 1)).
                Index 1 corresponds to model.layers[0], index N to model.layers[N-1].
        """
        self.model = model
        self._handles: List[torch.utils.hooks.RemovableHandle] = []
        self._activations: Dict[int, torch.Tensor] = {}
        self._gradients: Dict[int, torch.Tensor] = {}

        # Find transformer blocks
        self._pattern, self._layer_modules, self._layer_names = find_transformer_blocks(
            model
        )
        self.num_layers = len(self._layer_modules)

        # Determine which layers to hook (embedding-inclusive indices)
        if layers is None:
            # All transformer blocks: indices 1..num_layers
            self._target_layers = list(range(1, self.num_layers + 1))
        else:
            self._target_layers = [
                l if l >= 0 else (self.num_layers + 1) + l for l in layers
            ]

        # Validate layer indices
        for l in self._target_layers:
            if l < 1 or l > self.num_layers:
                raise ValueError(
                    f"Layer index {l} out of range. Valid range: 1..{self.num_layers} "
                    f"(embedding-inclusive convention, index 0 = embedding layer)"
                )

    def _make_forward_hook(self, layer_idx: int) -> Callable:
        """Create a forward hook that captures activations."""

        def hook(module: nn.Module, input: Tuple, output: torch.Tensor):
            # Handle tuple outputs (some models return tuple)
            if isinstance(output, tuple):
                output = output[0]

            # Store activation and enable gradient tracking
            self._activations[layer_idx] = output
            if output.requires_grad:
                output.retain_grad()

        return hook

    def _make_backward_hook(self, layer_idx: int) -> Callable:
        """Create a backward hook that captures gradients."""

        def hook(
            module: nn.Module, grad_input: Tuple, grad_output: Tuple[torch.Tensor]
        ):
            # grad_output[0] is the gradient w.r.t. the output
            if grad_output[0] is not None:
                self._gradients[layer_idx] = grad_output[0].detach().clone()

        return hook

    def register_hooks(self, collect_gradients: bool = True):
        """
        Register forward (and optionally backward) hooks on target layers.

        Args:
            collect_gradients: Whether to also register backward hooks
        """
        self.clear()

        for layer_idx in self._target_layers:
            module = self._layer_modules[layer_idx - 1]

            # Register forward hook
            handle = module.register_forward_hook(self._make_forward_hook(layer_idx))
            self._handles.append(handle)

            # Register backward hook if needed
            if collect_gradients:
                handle = module.register_full_backward_hook(
                    self._make_backward_hook(layer_idx)
                )
                self._handles.append(handle)

    def remove_hooks(self):
        """Remove all registered hooks."""
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def clear(self):
        """Clear stored activations and gradients, and remove hooks."""
        self._activations.clear()
        self._gradients.clear()
        self.remove_hooks()

    def get_activations(self) -> Dict[int, torch.Tensor]:
        """
        Get captured activations.

        Returns:
            Dictionary mapping layer index to activation tensor (batch, seq_len, hidden_dim)
        """
        return {k: v.detach().clone() for k, v in self._activations.items()}

    def get_gradients(self) -> Dict[int, torch.Tensor]:
        """
        Get captured gradients.

        Returns:
            Dictionary mapping layer index to gradient tensor (batch, seq_len, hidden_dim)
        """
        # First try to get gradients from backward hooks
        if self._gradients:
            return {k: v.detach().clone() for k, v in self._gradients.items()}

        # Fallback: get gradients from retained grad on activations
        gradients = {}
        for layer_idx, activation in self._activations.items():
            if activation.grad is not None:
                gradients[layer_idx] = activation.grad.detach().clone()
        return gradients

    @property
    def target_layers(self) -> List[int]:
        """Return list of layer indices being hooked."""
        return self._target_layers.copy()

    def __enter__(self):
        """Context manager entry."""
        self.register_hooks()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.clear()
        return False
