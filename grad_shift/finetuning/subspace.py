"""Shared subspace utilities for the gradient/weight/activation managers.

Every manager that accepts a multi-direction basis must orthonormalize
its rows identically and refuse rank-deficient input: QR of a
(near-)collinear row set silently returns numerically arbitrary
completion directions, so the intervention would target a different
subspace from the one supplied (2026-08-11 review P1).
"""

import torch

# Below this, a QR diagonal entry signals a (near-)linearly dependent row
RANK_TOL = 1e-3


def orthonormal_rows(dirs: torch.Tensor, layer) -> torch.Tensor:
    """Orthonormalize basis rows [k, hidden] via QR, failing fast on
    zero-norm or (near-)linearly dependent rows.

    Rows are unit-normalized first so the rank tolerance is
    scale-invariant. ``layer`` (int or label) is only for error messages.
    """
    return orthonormal_rows_with_r(dirs, layer)[0]


def orthonormal_rows_with_r(dirs: torch.Tensor, layer):
    """Like :func:`orthonormal_rows` but also returns the upper-triangular
    R of the QR of the UNIT-normalized rows — needed by the servo's exact
    raw-target -> orthonormal-coordinate transform."""
    norms = dirs.norm(dim=-1, keepdim=True)
    if (norms == 0).any():
        raise ValueError(f"zero-norm basis direction at layer {layer}")
    q, r = torch.linalg.qr((dirs / norms).T)
    r_diag = r.diagonal().abs()
    if (r_diag < RANK_TOL).any():
        raise ValueError(
            f"basis rows are (near-)linearly dependent at layer {layer} "
            f"(|diag(R)| min = {r_diag.min():.2e}); provide linearly "
            "independent directions"
        )
    return q.T.contiguous(), r
