"""
Similarity and difference metrics for activation analysis.

All metrics take two tensors and compute similarity/difference along the last dimension.
- a: Tensor of shape [..., hidden_dim] (e.g., activations or gradients)
- b: Tensor of shape [..., hidden_dim] or [hidden_dim] (e.g., steering vector)

When b has fewer dimensions, it will be broadcast appropriately.
"""

import torch
import torch.nn.functional as F
from typing import Callable, Dict


def cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Compute cosine similarity between vectors.

    cos(a, b) = (a · b) / (||a|| * ||b||)

    Range: [-1, 1] where 1 = same direction, -1 = opposite, 0 = orthogonal

    Args:
        a: Tensor of shape [..., hidden_dim]
        b: Tensor of shape [..., hidden_dim] or [hidden_dim]

    Returns:
        Tensor of shape [...] with cosine similarities
    """
    # Ensure b can broadcast with a
    if b.dim() < a.dim():
        b = b.expand_as(a)
    return F.cosine_similarity(a, b, dim=-1)


def normalized_l2_distance(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Compute normalized L2 distance between vectors.

    distance = || (a/||a||) - (b/||b||) ||

    Range: [0, ~2] where 0 = identical, larger = more different
    Normalized by vector magnitudes to be scale-invariant.

    Args:
        a: Tensor of shape [..., hidden_dim]
        b: Tensor of shape [..., hidden_dim] or [hidden_dim]

    Returns:
        Tensor of shape [...] with normalized distances
    """
    if b.dim() < a.dim():
        b = b.expand_as(a)

    a = a / torch.norm(a, dim=-1, keepdim=True)
    b = b / torch.norm(b, dim=-1, keepdim=True)
    diff = (a - b).pow(2).sum(dim=-1).sqrt()

    return diff


def projection_magnitude(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Compute the scalar projection of a onto b (projection magnitude).

    proj_b(a) = (a · b) / ||b||

    This gives the signed length of a's component in the direction of b.
    Positive = same direction, negative = opposite direction.

    Args:
        a: Tensor of shape [..., hidden_dim]
        b: Tensor of shape [..., hidden_dim] or [hidden_dim] (direction vector)

    Returns:
        Tensor of shape [...] with projection magnitudes
    """
    if b.dim() < a.dim():
        b = b.expand_as(a)

    dot_product = (a * b).sum(dim=-1)
    b_norm = torch.norm(b, dim=-1)

    eps = 1e-8
    return dot_product / (b_norm + eps)


def projection(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Compute the projection of a onto b such that component_along_b = projection * b.

    Args:
        a: Tensor of shape [..., hidden_dim]
        b: Tensor of shape [..., hidden_dim] or [hidden_dim] (direction vector)

    Returns:
        Tensor of shape [...] with normalized projections
    """
    return projection_magnitude(a,b)[..., None] * b


def variance_explained(
    a: torch.Tensor,
    b: torch.Tensor,
    eps: float = 1e-8,
    keep_dims: bool = True,
) -> torch.Tensor:
    """
    Compute the fraction of variance along the sequence dimension in a explained
    by direction b.

    var_explained = 1 - (||a - b_hat||^2) / (||a||^2)

    Range: [0, 1] where 1 = all variance along b, 0 = orthogonal to b

    Args:
        a: Tensor of shape [B, S, hidden_dim]
        b: Tensor of shape [1, hidden_dim] or [hidden_dim] (direction vector)

    Returns:
        Tensor of shape [B] with variance explained fractions
    """
    pred = projection(a, b) / torch.norm(b, dim=-1, keepdim=True) # [B, S, hidden_dim]
    diff = (a - pred).pow(2).mean(dim=-2) # [B, hidden_dim]
    var_a = torch.var(a, dim=-2, unbiased=True) # [B, hidden_dim]
    res = (1 - diff / (var_a + eps).clamp(min=eps)).mean(dim=-1) # [..., ]
    if keep_dims:
        res = res[..., None].repeat(1, a.shape[1])
    return res


def l2_distance(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Compute raw L2 (Euclidean) distance between vectors.

    distance = ||a - b||

    Args:
        a: Tensor of shape [..., hidden_dim]
        b: Tensor of shape [..., hidden_dim] or [hidden_dim]

    Returns:
        Tensor of shape [...] with L2 distances
    """
    if b.dim() < a.dim():
        b = b.expand_as(a)

    return torch.norm(a - b, dim=-1)


def dot_product(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Compute dot product between vectors.

    dot(a, b) = a · b = sum(a_i * b_i)

    Args:
        a: Tensor of shape [..., hidden_dim]
        b: Tensor of shape [..., hidden_dim] or [hidden_dim]

    Returns:
        Tensor of shape [...] with dot products
    """
    if b.dim() < a.dim():
        b = b.expand_as(a)

    return (a * b).sum(dim=-1)


# Registry of available metrics
METRICS: Dict[str, Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = {
    "cosine_similarity": cosine_similarity,
    "normalized_l2_distance": normalized_l2_distance,
    "projection_magnitude": projection_magnitude,
    "projection": projection,
    "variance_explained": variance_explained,
    "l2_distance": l2_distance,
    "dot_product": dot_product,
}

# Metadata about metrics for plotting
METRIC_INFO = {
    "cosine_similarity": {
        "name": "Cosine Similarity",
        "range": "[-1, 1]",
        "description": "Directional similarity (1=same, -1=opposite, 0=orthogonal)",
    },
    "normalized_l2_distance": {
        "name": "Normalized L2 Distance",
        "range": "[0, ~2]",
        "description": "Scale-invariant distance (0=identical)",
    },
    "projection_magnitude": {
        "name": "Projection Magnitude",
        "range": "(-∞, ∞)",
        "description": "Signed length of projection onto steering direction",
    },
    "projection": {
        "name": "Projection",
        "range": "(-∞, ∞)",
        "description": "Coefficient for component along steering direction",
    },
    "variance_explained": {
        "name": "Variance Explained",
        "range": "[0, 1]",
        "description": "Fraction of variance along steering direction (R²)",
    },
    "l2_distance": {
        "name": "L2 Distance",
        "range": "[0, ∞)",
        "description": "Euclidean distance (not normalized)",
    },
    "dot_product": {
        "name": "Dot Product",
        "range": "(-∞, ∞)",
        "description": "Raw dot product (scale-dependent)",
    },
}


def get_metric(name: str) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    """
    Get a metric function by name.

    Args:
        name: One of the keys in METRICS

    Returns:
        The metric function

    Raises:
        ValueError: If metric name is not recognized
    """
    if name not in METRICS:
        raise ValueError(f"Unknown metric: {name}. Available: {list(METRICS.keys())}")
    return METRICS[name]


def list_metrics() -> list:
    """Return list of available metric names."""
    return list(METRICS.keys())
