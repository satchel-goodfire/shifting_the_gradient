"""Analysis utilities for activation data."""

from .metrics import (
    cosine_similarity,
    normalized_l2_distance,
    projection_magnitude,
    projection,
    variance_explained,
    l2_distance,
    dot_product,
    METRICS,
    METRIC_INFO,
    get_metric,
    list_metrics,
)

__all__ = [
    "cosine_similarity",
    "normalized_l2_distance",
    "projection_magnitude",
    "projection",
    "variance_explained",
    "l2_distance",
    "dot_product",
    "METRICS",
    "METRIC_INFO",
    "get_metric",
    "list_metrics",
]
