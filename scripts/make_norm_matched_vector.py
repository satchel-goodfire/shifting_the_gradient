#!/usr/bin/env python
"""Rescale a control steering vector to a reference vector's per-layer norm.

PPS with a random (or off-trait) vector is the control for "does PPS work
merely by perturbing the residual stream?". That control is only fair at
EQUAL steering strength, so the control's norm must match the trait vector's
at every layer: steering adds ``coef * v[layer]``, so equal norms make equal
coefficients mean equal perturbation size. The paper's original random
control (``steering_vectors/random_vector_magnitude_30.pt``, kept as the
source of the norm-matched one) has norm 30 at every layer, ~10% above the
evil vector at layer 20 (27.09). The off-trait control
``sycophantic_normmatched_evil_qwen25_7b.pt`` is the same rescaling of
``Qwen_Qwen2.5-7B-Instruct_extract/sycophantic_response_avg_diff_thresh50.pt``.

Direction is preserved exactly; only the per-layer scale changes. Layers
where the reference is zero (the embedding placeholder row) are left zero.

Usage:
    python scripts/make_norm_matched_vector.py \\
        --source steering_vectors/random_vector_magnitude_30.pt \\
        --reference steering_vectors/Qwen_Qwen2.5-7B-Instruct_extract/evil_response_avg_diff_thresh50.pt \\
        --out steering_vectors/random_normmatched_evil_qwen25_7b.pt
"""

import argparse

import torch


def norm_match(source: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    """Return ``source`` rescaled so each row's norm equals ``reference``'s.

    Both tensors are ``[num_layers+1, hidden]`` (embedding-inclusive rows).
    Rows where ``reference`` has zero norm become zero.
    """
    if source.shape != reference.shape:
        raise ValueError(f"shape mismatch: {tuple(source.shape)} vs "
                         f"{tuple(reference.shape)}")
    src = source.float()
    ref_norm = reference.float().norm(dim=-1, keepdim=True)
    src_norm = src.norm(dim=-1, keepdim=True)
    if torch.any((src_norm == 0) & (ref_norm > 0)):
        raise ValueError("source has a zero row where reference is nonzero")
    scale = torch.where(src_norm > 0, ref_norm / src_norm.clamp_min(1e-12),
                        torch.zeros_like(src_norm))
    return src * scale


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--source", required=True)
    p.add_argument("--reference", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    out = norm_match(torch.load(args.source), torch.load(args.reference))
    torch.save(out, args.out)
    ref = torch.load(args.reference).float()
    print(f"wrote {args.out}; layer-20 norm {out[20].norm():.4f} "
          f"(reference {ref[20].norm():.4f})")


if __name__ == "__main__":
    main()
