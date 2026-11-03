"""Trainable steering module for model activation modification."""

import warnings
from typing import Optional
import torch
import torch.nn as nn


class SteeringModule(nn.Module):
    """Trainable steering vectors for model activation modification.

    Uses embedding-inclusive indexing convention:
    - Index 0 = embedding layer (typically a zero placeholder)
    - Index N = transformer block N-1

    Steering vector files are always stored with the embedding placeholder
    at index 0. For a model with 28 transformer blocks, the tensor shape
    is [29, hidden_dim].

    Attributes:
        steering_vectors: Parameter tensor of shape [num_layers, hidden_dim]
            where num_layers includes the embedding placeholder at index 0.
        global_scale: Global scaling factor applied to all vectors
        layer_scales: Optional learnable per-layer scaling factors
    """

    def __init__(
        self,
        num_layers: int,
        hidden_dim: int,
        global_scale: float = 1.0,
        learnable_scales: bool = False,
        init_vectors: Optional[torch.Tensor] = None,
    ):
        """Initialize the steering module.

        Args:
            num_layers: Number of transformer layers
            hidden_dim: Hidden dimension of the model
            global_scale: Global scaling factor for steering vectors
            learnable_scales: Whether to use learnable per-layer scales
            init_vectors: Optional initial steering vectors [num_layers, hidden_dim]
        """
        super().__init__()

        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.global_scale = global_scale

        # Initialize steering vectors
        if init_vectors is not None:
            if init_vectors.shape != (num_layers, hidden_dim):
                raise ValueError(
                    f"init_vectors shape {init_vectors.shape} doesn't match "
                    f"expected ({num_layers}, {hidden_dim})"
                )
            self.steering_vectors = nn.Parameter(init_vectors.clone())
        else:
            self.steering_vectors = nn.Parameter(torch.zeros(num_layers, hidden_dim))

        # Optional learnable per-layer scales
        if learnable_scales:
            self.layer_scales = nn.Parameter(torch.ones(num_layers))
        else:
            self.register_buffer("layer_scales", None)

    @staticmethod
    def _load_single_vector(path: str, device: Optional[str] = None) -> torch.Tensor:
        """Load and validate a single steering vector .pt file."""
        vectors = torch.load(path, map_location=device, weights_only=True)
        if not isinstance(vectors, torch.Tensor):
            raise ValueError(
                f"Expected .pt file to contain a tensor, got {type(vectors)} "
                f"in {path}"
            )
        if vectors.dim() != 2:
            raise ValueError(
                f"Expected 2D tensor [num_layers, hidden_dim], got shape "
                f"{vectors.shape} in {path}"
            )
        return vectors

    @classmethod
    def from_pretrained(
        cls,
        path: str,
        global_scale: float = 1.0,
        learnable_scales: bool = False,
        device: Optional[str] = None,
        skip_embedding_layer: bool = False,
    ) -> "SteeringModule":
        """Load steering vectors from one or more .pt files.

        Vectors are expected to use embedding-inclusive indexing:
        index 0 = embedding layer (zero placeholder), index N = transformer block N-1.

        If ``path`` contains commas, each comma-separated path is loaded and
        the resulting tensors are summed element-wise.

        Args:
            path: Path to .pt file (or comma-separated paths to sum)
                  containing tensor of shape [num_layers, hidden_dim]
            global_scale: Global scaling factor
            learnable_scales: Whether to use learnable per-layer scales
            device: Device to load tensors to
            skip_embedding_layer: Deprecated. If True, a warning is emitted and
                the parameter is ignored. Vectors are always loaded with the
                embedding placeholder at index 0 intact.

        Returns:
            SteeringModule initialized with loaded vectors
        """
        if skip_embedding_layer:
            warnings.warn(
                "skip_embedding_layer is deprecated and will be removed in a "
                "future version. Steering vectors now always use embedding-"
                "inclusive indexing (index 0 = embedding placeholder). "
                "The parameter is ignored.",
                DeprecationWarning,
                stacklevel=2,
            )

        paths = [p.strip() for p in path.split(",")]

        vectors = cls._load_single_vector(paths[0], device)
        if len(paths) > 1:
            for p in paths[1:]:
                v = cls._load_single_vector(p, device)
                if v.shape != vectors.shape:
                    raise ValueError(
                        f"Shape mismatch: {paths[0]} has shape {vectors.shape} "
                        f"but {p} has shape {v.shape}"
                    )
                vectors = vectors + v
            print(f"Summed {len(paths)} steering vectors:")
            for p in paths:
                print(f"  - {p}")
            for layer_idx in range(vectors.shape[0]):
                n = vectors[layer_idx].norm().item()
                if n > 1e-6:
                    print(f"  Combined norm at layer {layer_idx}: {n:.4f}")

        num_layers, hidden_dim = vectors.shape

        return cls(
            num_layers=num_layers,
            hidden_dim=hidden_dim,
            global_scale=global_scale,
            learnable_scales=learnable_scales,
            init_vectors=vectors,
        )

    def get_steering_vector(self, layer_idx: int) -> torch.Tensor:
        """Get the scaled steering vector for a specific layer.

        Args:
            layer_idx: Index of the layer

        Returns:
            Scaled steering vector of shape [hidden_dim]
        """
        if layer_idx < 0 or layer_idx >= self.num_layers:
            raise IndexError(f"layer_idx {layer_idx} out of range [0, {self.num_layers})")

        vector = self.steering_vectors[layer_idx]

        # Apply scales
        scale = self.global_scale
        if self.layer_scales is not None:
            scale = scale * self.layer_scales[layer_idx]

        return vector * scale

    def freeze_vectors(self):
        """Freeze steering vectors (disable gradients)."""
        self.steering_vectors.requires_grad_(False)
        if self.layer_scales is not None:
            self.layer_scales.requires_grad_(False)

    def unfreeze_vectors(self):
        """Unfreeze steering vectors (enable gradients)."""
        self.steering_vectors.requires_grad_(True)
        if self.layer_scales is not None:
            self.layer_scales.requires_grad_(True)

    def set_global_scale(self, scale: float):
        """Update the global scale factor.

        Args:
            scale: New global scale value
        """
        self.global_scale = scale

    def save(self, path: str):
        """Save steering vectors to a .pt file.

        Args:
            path: Path to save the vectors
        """
        torch.save(self.steering_vectors.data, path)

    def save_full(self, path: str):
        """Save full module state including scales.

        Args:
            path: Path to save the state dict
        """
        torch.save(self.state_dict(), path)

    @classmethod
    def load_full(
        cls,
        path: str,
        device: Optional[str] = None,
    ) -> "SteeringModule":
        """Load full module state including scales.

        Args:
            path: Path to state dict file
            device: Device to load to

        Returns:
            SteeringModule with loaded state
        """
        state_dict = torch.load(path, map_location=device, weights_only=True)

        # Infer dimensions from state dict
        vectors = state_dict["steering_vectors"]
        num_layers, hidden_dim = vectors.shape
        learnable_scales = "layer_scales" in state_dict and state_dict["layer_scales"] is not None

        module = cls(
            num_layers=num_layers,
            hidden_dim=hidden_dim,
            learnable_scales=learnable_scales,
        )
        module.load_state_dict(state_dict)

        return module

    def __repr__(self) -> str:
        return (
            f"SteeringModule(num_layers={self.num_layers}, "
            f"hidden_dim={self.hidden_dim}, "
            f"global_scale={self.global_scale}, "
            f"learnable_scales={self.layer_scales is not None})"
        )
