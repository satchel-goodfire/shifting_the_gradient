"""Storage utilities for saving and loading activation collection results."""

from typing import Dict, List, Optional, Tuple, Any
import torch
from pathlib import Path

from .collector import CollectionResult


def _result_to_dict(
    result: CollectionResult,
    deltas: Optional[Dict[int, torch.Tensor]] = None,
) -> Dict[str, Any]:
    """Convert a CollectionResult to a serializable dictionary.

    Args:
        result: The collection result.
        deltas: Optional dict mapping layer -> (seq_len, hidden_dim) delta
            tensors. If provided, takes precedence over result.deltas.
    """
    d = {
        "activations": result.activations,
        "gradients": result.gradients,
        "tokens": result.tokens,
        "prompt_length": result.prompt_length,
        "text": result.text,
        "prompt": result.prompt,
        "messages": result.messages,
        "loss": result.loss,
    }
    # Use explicitly passed deltas first, fall back to result.deltas
    effective_deltas = deltas if deltas is not None else result.deltas
    if effective_deltas:
        d["deltas"] = effective_deltas
    return d


def _dict_to_result(d: Dict[str, Any]) -> CollectionResult:
    """Convert a dictionary back to a CollectionResult."""
    return CollectionResult(
        activations=d["activations"],
        gradients=d["gradients"],
        deltas=d.get("deltas"),
        tokens=d["tokens"],
        prompt_length=d["prompt_length"],
        text=d["text"],
        prompt=d["prompt"],
        messages=d.get("messages"),
        loss=d.get("loss"),
    )


def save_results(
    results_without_prompt: List[CollectionResult],
    results_with_prompt: List[CollectionResult],
    path: str,
    prompt: str,
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Save activation collection results to a .pt file.

    Args:
        results_without_prompt: Results collected without the prompt
        results_with_prompt: Results collected with the prompt
        path: Output file path
        prompt: The prompt that was used
        metadata: Optional additional metadata to save
    """
    data = {
        "results_without_prompt": [_result_to_dict(r) for r in results_without_prompt],
        "results_with_prompt": [_result_to_dict(r) for r in results_with_prompt],
        "prompt": prompt,
        "metadata": metadata or {},
    }

    # Ensure parent directory exists
    Path(path).parent.mkdir(parents=True, exist_ok=True)

    torch.save(data, path)


def load_results(
    path: str,
) -> Tuple[List[CollectionResult], List[CollectionResult], str, Dict[str, Any]]:
    """
    Load activation collection results from a .pt file.

    Args:
        path: Path to the .pt file

    Returns:
        Tuple of (results_without_prompt, results_with_prompt, prompt, metadata)
    """
    data = torch.load(path, weights_only=False)

    results_without = [_dict_to_result(d) for d in data["results_without_prompt"]]
    results_with = [_dict_to_result(d) for d in data["results_with_prompt"]]

    return results_without, results_with, data["prompt"], data["metadata"]


def save_single_comparison(
    result_without: CollectionResult,
    result_with: CollectionResult,
    path: str,
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Save a single comparison (one text with/without prompt) to a .pt file.

    Args:
        result_without: Result collected without the prompt
        result_with: Result collected with the prompt
        path: Output file path
        metadata: Optional additional metadata to save
    """
    data = {
        "without_prompt": _result_to_dict(result_without),
        "with_prompt": _result_to_dict(result_with),
        "prompt": result_with.prompt,
        "metadata": metadata or {},
    }

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(data, path)


def load_single_comparison(
    path: str,
) -> Tuple[CollectionResult, CollectionResult, str, Dict[str, Any]]:
    """
    Load a single comparison from a .pt file.

    Args:
        path: Path to the .pt file

    Returns:
        Tuple of (result_without_prompt, result_with_prompt, prompt, metadata)
    """
    data = torch.load(path, weights_only=False)

    result_without = _dict_to_result(data["without_prompt"])
    result_with = _dict_to_result(data["with_prompt"])

    return result_without, result_with, data["prompt"], data["metadata"]


def save_steering_sweep_results(
    base_results: List[CollectionResult],
    intensity_gradients: Dict[float, List[Dict[int, torch.Tensor]]],
    steering_vectors: torch.Tensor,
    intensities: List[float],
    path: str,
    metadata: Optional[Dict[str, Any]] = None,
    intensity_losses: Optional[Dict[float, List[torch.Tensor]]] = None,
    intensity_deltas: Optional[Dict[float, List[Dict[int, torch.Tensor]]]] = None,
) -> None:
    """
    Save steering intensity sweep results to a .pt file.

    The base activations are saved once (at intensity 0), and gradients are
    saved for each intensity level. Activations at other intensities can be
    reconstructed as: activation + steering_vector * intensity

    Args:
        base_results: Results collected without steering (intensity=0)
        intensity_gradients: Dict mapping intensity -> list of gradient dicts per sample
        steering_vectors: The steering vectors used [num_layers, hidden_dim]
        intensities: List of intensity values that were swept
        path: Output file path
        metadata: Optional additional metadata to save
        intensity_losses: Dict mapping intensity -> list of per-token loss tensors per sample
        intensity_deltas: Dict mapping intensity -> list of delta dicts per sample.
            Deltas are h(theta-lr*grad) - h(theta), always computed on
            defenseless forward passes (no steering hooks active).
    """
    data = {
        "base_results": [_result_to_dict(r) for r in base_results],
        "intensity_gradients": intensity_gradients,
        "steering_vectors": steering_vectors,
        "intensities": intensities,
        "metadata": metadata or {},
    }
    if intensity_losses is not None:
        data["intensity_losses"] = intensity_losses
    if intensity_deltas is not None:
        data["intensity_deltas"] = intensity_deltas

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(data, path)


def load_steering_sweep_results(
    path: str,
) -> Tuple[List[CollectionResult], Dict[float, List[Dict[int, torch.Tensor]]], torch.Tensor, List[float], Dict[str, Any], Dict[float, List[torch.Tensor]]]:
    """
    Load steering intensity sweep results from a .pt file.

    Args:
        path: Path to the .pt file

    Returns:
        Tuple of (base_results, intensity_gradients, steering_vectors,
        intensities, metadata, intensity_losses, intensity_deltas)
    """
    data = torch.load(path, weights_only=False)

    base_results = [_dict_to_result(d) for d in data["base_results"]]

    return (
        base_results,
        data["intensity_gradients"],
        data["steering_vectors"],
        data["intensities"],
        data["metadata"],
        data.get("intensity_losses", {}),
        data.get("intensity_deltas", {}),
    )
