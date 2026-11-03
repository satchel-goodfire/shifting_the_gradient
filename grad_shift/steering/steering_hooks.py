"""Hook manager for applying steering vectors to model activations."""

from typing import Callable, Dict, List, Optional, Tuple
import torch
import torch.nn as nn

from ..hooks import find_transformer_blocks
from .steering_module import SteeringModule
from .config import SteeringConfig


# Patterns for finding attention and FFN sublayers within transformer blocks
ATTENTION_PATTERNS = [
    "self_attn",  # LLaMA, Mistral, Qwen
    "attn",  # GPT-2, GPT-Neo
    "attention",  # Generic
]

FFN_PATTERNS = [
    "mlp",  # LLaMA, Mistral, Qwen, GPT-2
    "feed_forward",  # Some models
    "ffn",  # Generic
]


def _find_sublayer(layer_module: nn.Module, patterns: List[str]) -> Optional[nn.Module]:
    """Find a sublayer within a transformer block by name patterns.

    Args:
        layer_module: The transformer block module
        patterns: List of patterns to search for

    Returns:
        The sublayer module if found, None otherwise
    """
    for name, child in layer_module.named_children():
        for pattern in patterns:
            if pattern in name.lower():
                return child
    return None


class SteeringHookManager:
    """Applies steering vectors to model activations via forward hooks.

    Uses embedding-inclusive layer indexing convention:
    - Index 0 = embedding layer (zero placeholder in steering vectors)
    - Index N = transformer block N-1 (model.layers[N-1])
    - Steering vector at index N is applied to model.layers[N-1]

    Supports multiple application points:
    - "layer_output": After the full transformer layer
    - "attention": After the attention sublayer
    - "ffn": After the FFN/MLP sublayer

    Example:
        steering_module = SteeringModule.from_pretrained("steering.pt")
        manager = SteeringHookManager(model, steering_module, active_layers=[20])
        # This hooks model.layers[19] and reads steering_vectors[20]

        with manager:
            outputs = model(input_ids)
    """

    def __init__(
        self,
        model: nn.Module,
        steering_module: SteeringModule,
        config: Optional[SteeringConfig] = None,
        active_layers: Optional[List[int]] = None,
        application_point: str = "layer_output",
    ):
        """Initialize the steering hook manager.

        Args:
            model: The transformer model
            steering_module: SteeringModule containing steering vectors
            config: Optional SteeringConfig (overrides other args if provided)
            active_layers: Embedding-inclusive layer indices to apply steering
                (None = all transformer blocks, i.e. range(1, steering_module.num_layers)).
                Index 1 = model.layers[0], index N = model.layers[N-1].
            application_point: Where to apply steering
        """
        self.model = model
        self.steering_module = steering_module
        self._handles: List[torch.utils.hooks.RemovableHandle] = []
        self._intensity: float = 1.0

        # Use config if provided, otherwise use individual args
        if config is not None:
            active_layers = config.active_layers
            application_point = config.application_point

        self.application_point = application_point

        # Find transformer blocks
        self._pattern, self._layer_modules, self._layer_names = find_transformer_blocks(model)
        self.num_layers = len(self._layer_modules)

        # Determine which layers to steer (embedding-inclusive indices)
        if active_layers is None:
            # All transformer blocks: indices 1..num_steering_layers-1
            # (skip index 0 which is the embedding placeholder)
            self._active_layers = list(range(1, steering_module.num_layers))
        else:
            self._active_layers = [
                l if l >= 0 else (self.num_layers + 1) + l for l in active_layers
            ]

        # Validate layer indices
        for l in self._active_layers:
            if l < 1:
                raise ValueError(
                    f"Layer index {l} invalid. Must be >= 1 "
                    f"(index 0 = embedding layer, not a transformer block)"
                )
            if l >= steering_module.num_layers:
                raise ValueError(
                    f"Steering module has {steering_module.num_layers} layers but "
                    f"active_layers includes index {l} (need at least {l + 1} layers)"
                )
            if l > self.num_layers:
                raise ValueError(
                    f"Model has {self.num_layers} transformer blocks but "
                    f"active_layers includes index {l} (max: {self.num_layers})"
                )

        # Find target modules based on application point
        self._target_modules: Dict[int, nn.Module] = {}
        self._setup_target_modules()

    def _setup_target_modules(self):
        """Set up the target modules for hook registration."""
        for layer_idx in self._active_layers:
            layer_module = self._layer_modules[layer_idx - 1]

            if self.application_point == "layer_output":
                self._target_modules[layer_idx] = layer_module
            elif self.application_point == "attention":
                sublayer = _find_sublayer(layer_module, ATTENTION_PATTERNS)
                if sublayer is None:
                    raise ValueError(
                        f"Could not find attention sublayer in layer {layer_idx}. "
                        f"Searched patterns: {ATTENTION_PATTERNS}"
                    )
                self._target_modules[layer_idx] = sublayer
            elif self.application_point == "ffn":
                sublayer = _find_sublayer(layer_module, FFN_PATTERNS)
                if sublayer is None:
                    raise ValueError(
                        f"Could not find FFN sublayer in layer {layer_idx}. "
                        f"Searched patterns: {FFN_PATTERNS}"
                    )
                self._target_modules[layer_idx] = sublayer
            else:
                raise ValueError(f"Unknown application point: {self.application_point}")

    def _make_steering_hook(self, layer_idx: int) -> Callable:
        """Create a forward hook that adds steering vectors to activations.

        Args:
            layer_idx: Index of the layer

        Returns:
            Hook function
        """
        def hook(module: nn.Module, input: Tuple, output):
            # Get steering vector, matching device and dtype of activations
            steering_vec = self.steering_module.get_steering_vector(layer_idx)
            target = output[0] if isinstance(output, tuple) else output
            steering_vec = steering_vec.to(device=target.device, dtype=target.dtype)

            # Apply intensity scaling
            steering_vec = steering_vec * self._intensity

            # Handle tuple outputs (some models return tuple)
            if isinstance(output, tuple):
                hidden_states = output[0]
                # Add steering vector to all positions (broadcast over batch and seq_len)
                modified = hidden_states + steering_vec.unsqueeze(0).unsqueeze(0)
                return (modified,) + output[1:]
            else:
                # Add steering vector to all positions
                return output + steering_vec.unsqueeze(0).unsqueeze(0)

        return hook

    def register_hooks(self):
        """Register forward hooks on target layers."""
        self.remove_hooks()

        for layer_idx, module in self._target_modules.items():
            handle = module.register_forward_hook(self._make_steering_hook(layer_idx))
            self._handles.append(handle)

    def remove_hooks(self):
        """Remove all registered hooks."""
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def set_intensity(self, intensity: float):
        """Update the steering intensity.

        This multiplies with the global_scale from the SteeringModule.

        Args:
            intensity: Intensity multiplier (0.0 = no steering, 1.0 = full steering)
        """
        self._intensity = intensity

    @property
    def intensity(self) -> float:
        """Get the current steering intensity."""
        return self._intensity

    @property
    def active_layers(self) -> List[int]:
        """Get list of layer indices being steered."""
        return self._active_layers.copy()

    def __enter__(self):
        """Context manager entry - register hooks."""
        self.register_hooks()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - remove hooks."""
        self.remove_hooks()
        return False

    def __repr__(self) -> str:
        return (
            f"SteeringHookManager("
            f"num_layers={self.num_layers}, "
            f"active_layers={len(self._active_layers)}, "
            f"application_point='{self.application_point}', "
            f"intensity={self._intensity})"
        )
