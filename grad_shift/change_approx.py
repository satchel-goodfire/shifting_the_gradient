"""First-order approximation of residual stream changes from weight updates.

Given a weight update Dtheta (e.g. from a gradient descent step), the change
to the residual stream h_l at layer l is approximated to first order as:

    Dh_l ~ sum_theta (dh_l/dtheta) * Dtheta  =  J * Dtheta

where J is the Jacobian of h_l w.r.t. all model parameters theta.

This Jacobian-vector product (JVP) is computed efficiently via forward-mode
automatic differentiation (torch.func.jvp + functional_call).
"""

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.func import jvp, functional_call
from torch.nn.attention import sdpa_kernel, SDPBackend

from .hooks import find_transformer_blocks


def _validate_layers(model: nn.Module, layers: List[int]) -> Tuple[int, nn.Module]:
    """Validate embedding-inclusive indices and return last-block metadata."""
    _, blocks, _ = find_transformer_blocks(model)
    num_layers = len(blocks)
    invalid = [layer for layer in layers if layer < 0 or layer > num_layers]
    if invalid:
        raise ValueError(
            f"Layer indices {invalid} out of range; expected 0..{num_layers}"
        )
    if len(set(layers)) != len(layers):
        raise ValueError("layers must not contain duplicate indices")
    return num_layers, blocks[-1]


def compute_activation_change_approx(
    model: nn.Module,
    input_ids: torch.Tensor,
    layers: List[int],
    param_changes: Dict[str, torch.Tensor],
    attention_mask: Optional[torch.Tensor] = None,
) -> Dict[int, torch.Tensor]:
    """First-order approximation of activation changes at specified layers.

    Computes the JVP:  Dh_l ~ J_l * Dtheta
    where J_l = dh_l/dtheta is the Jacobian.

    Only parameters with non-zero changes contribute to the JVP; tangent
    tensors for unchanged parameters are never allocated.

    Args:
        model: The transformer model.
        input_ids: (batch, seq_len) input token IDs.
        layers: Embedding-inclusive layer indices.
            0 = embedding output, i = output of transformer block i.
        param_changes: {param_name: delta_tensor} for changed parameters.
            Names must match model.named_parameters(). Unlisted parameters
            are treated as unchanged (Dtheta=0).
        attention_mask: Optional (batch, seq_len) attention mask.

    Returns:
        Dict mapping layer index to approximated change tensor
        with shape (batch, seq_len, hidden_dim).
    """
    num_layers, last_block = _validate_layers(model, layers)
    all_params = dict(model.named_parameters())

    # Split into changing (needs tangent) and fixed (no tangent needed).
    # This avoids allocating zero tensors for the (potentially huge)
    # set of unchanged parameters.
    changing_names = set(param_changes.keys())
    changing_params = {}
    changing_tangents = {}
    for name in changing_names:
        if name not in all_params:
            raise KeyError(
                f"param_changes key '{name}' not found in "
                f"model.named_parameters()"
            )
        p = all_params[name]
        changing_params[name] = p
        changing_tangents[name] = param_changes[name].to(
            device=p.device, dtype=p.dtype
        )

    # Fixed params are captured by closure; jvp won't track them so
    # their tangent contribution is implicitly zero.
    fixed_params = {
        n: p.detach() for n, p in all_params.items()
        if n not in changing_names
    }

    fwd_kwargs = {
        "input_ids": input_ids,
        "output_hidden_states": True,
    }
    if attention_mask is not None:
        fwd_kwargs["attention_mask"] = attention_mask

    def f(cp):
        merged = {**fixed_params, **cp}
        last_block_output = None

        def capture_last_block(_module, _inputs, output):
            nonlocal last_block_output
            last_block_output = output[0] if isinstance(output, tuple) else output

        handle = None
        if num_layers in layers:
            handle = last_block.register_forward_hook(capture_last_block)
        try:
            out = functional_call(model, merged, args=(), kwargs=fwd_kwargs)
        finally:
            if handle is not None:
                handle.remove()
        # hidden_states uses embedding-inclusive indexing:
        #   [0] = embedding, [i] = output of transformer block i, except
        # some architectures expose the final entry after a final norm.  Use
        # the actual last-block output so layer N has one meaning everywhere.
        return tuple(
            last_block_output if l == num_layers else out.hidden_states[l]
            for l in layers
        )

    # Force the MATH SDPA backend: flash/mem-efficient kernels do not
    # support forward-mode AD (JVP), but the math kernel does.
    with sdpa_kernel(SDPBackend.MATH):
        _primals, tangent_out = jvp(
            f, (changing_params,), (changing_tangents,)
        )

    return {l: tangent_out[i].detach() for i, l in enumerate(layers)}


def compute_activation_change_actual(
    model: nn.Module,
    input_ids: torch.Tensor,
    layers: List[int],
    param_changes: Dict[str, torch.Tensor],
    attention_mask: Optional[torch.Tensor] = None,
) -> Dict[int, torch.Tensor]:
    """Compute actual activation change by applying Dtheta and re-running.

    h_l(theta + Dtheta) - h_l(theta)

    Useful as a ground-truth reference for testing the first-order
    approximation.

    Args:
        Same as compute_activation_change_approx.

    Returns:
        Dict mapping layer index to actual change tensor
        with shape (batch, seq_len, hidden_dim).
    """
    num_layers, last_block = _validate_layers(model, layers)
    fwd_kwargs = {"output_hidden_states": True}
    if attention_mask is not None:
        fwd_kwargs["attention_mask"] = attention_mask

    params = dict(model.named_parameters())

    def collect_activations():
        last_block_output = None

        def capture_last_block(_module, _inputs, output):
            nonlocal last_block_output
            value = output[0] if isinstance(output, tuple) else output
            last_block_output = value.detach().clone()

        handle = None
        if num_layers in layers:
            handle = last_block.register_forward_hook(capture_last_block)
        try:
            output = model(input_ids, **fwd_kwargs)
        finally:
            if handle is not None:
                handle.remove()
        return {
            layer: (
                last_block_output
                if layer == num_layers
                else output.hidden_states[layer].detach().clone()
            )
            for layer in layers
        }

    # Snapshot exact values.  Adding and subtracting a delta does not restore
    # bf16/fp16 parameters exactly because both operations round.
    saved_params = {}
    for name in param_changes:
        if name not in params:
            raise KeyError(f"param_changes key '{name}' not found in model.named_parameters()")
        saved_params[name] = params[name].detach().clone()

    # Use the same SDPA backend as the JVP computation for
    # consistent numerical comparison.
    with sdpa_kernel(SDPBackend.MATH):
        # h_l(theta)
        with torch.no_grad():
            h_before = collect_activations()

        try:
            # Apply Dtheta
            for name, delta in param_changes.items():
                params[name].data.add_(
                    delta.to(params[name].device, params[name].dtype)
                )

            # h_l(theta + Dtheta)
            with torch.no_grad():
                h_after = collect_activations()
        finally:
            for name, saved in saved_params.items():
                params[name].data.copy_(saved)

    return {l: h_after[l] - h_before[l] for l in layers}


def compute_sgd_param_changes(
    model: nn.Module,
    input_ids: torch.Tensor,
    learning_rate: float,
    attention_mask: Optional[torch.Tensor] = None,
    prompt_length: int = 0,
    param_names: Optional[List[str]] = None,
) -> Tuple[Dict[str, torch.Tensor], float]:
    """Compute SGD parameter changes Dtheta = -lr * dL/dtheta.

    Runs a forward+backward pass to get gradients, then computes the
    simple SGD update direction.  Temporarily enables requires_grad
    on target parameters if needed.

    Args:
        model: The transformer model.
        input_ids: (batch, seq_len) input token IDs.
        learning_rate: SGD step size.
        attention_mask: Optional attention mask.
        prompt_length: Number of prompt tokens to mask from loss.
        param_names: Which parameters to include.  If None, uses all
            parameters with requires_grad=True.

    Returns:
        Tuple of (param_changes dict, loss scalar).
    """
    vocab_size = model.config.vocab_size

    # Select target parameters
    all_params = dict(model.named_parameters())
    if param_names is not None:
        target = {n: all_params[n] for n in param_names}
    else:
        target = {n: p for n, p in all_params.items() if p.requires_grad}

    if not target:
        raise ValueError(
            "No target parameters found. Either pass param_names or "
            "ensure some parameters have requires_grad=True."
        )

    # Temporarily enable requires_grad
    orig_rg = {n: p.requires_grad for n, p in target.items()}
    for p in target.values():
        p.requires_grad_(True)

    try:
        model.zero_grad()
        with torch.enable_grad():
            fwd_kwargs = {"input_ids": input_ids}
            if attention_mask is not None:
                fwd_kwargs["attention_mask"] = attention_mask
            outputs = model(**fwd_kwargs)
            logits = outputs.logits

            # Cross-entropy loss with optional prompt masking
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = input_ids[..., 1:].contiguous()

            if prompt_length > 0:
                mask = torch.zeros_like(shift_labels, dtype=torch.float)
                mask_start = max(0, prompt_length - 1)
                mask[..., mask_start:] = 1.0
                if attention_mask is not None:
                    mask *= attention_mask[..., 1:].contiguous().float()
                loss_per_tok = F.cross_entropy(
                    shift_logits.view(-1, vocab_size),
                    shift_labels.view(-1),
                    reduction="none",
                )
                loss = (loss_per_tok * mask.view(-1)).sum() / mask.sum().clamp(min=1)
            else:
                if attention_mask is not None:
                    shift_mask = attention_mask[..., 1:].contiguous().float()
                    loss_per_tok = F.cross_entropy(
                        shift_logits.view(-1, vocab_size),
                        shift_labels.view(-1),
                        reduction="none",
                    )
                    loss = (
                        (loss_per_tok * shift_mask.view(-1)).sum()
                        / shift_mask.sum().clamp(min=1)
                    )
                else:
                    loss = F.cross_entropy(
                        shift_logits.view(-1, vocab_size),
                        shift_labels.view(-1),
                    )

            loss.backward()

        loss_val = loss.item()

        param_changes = {}
        for name, p in target.items():
            if p.grad is not None:
                param_changes[name] = (-learning_rate * p.grad).detach().clone()

    finally:
        # Restore requires_grad state and clean up
        for name, p in target.items():
            p.requires_grad_(orig_rg[name])
        model.zero_grad()

    return param_changes, loss_val


def collect_activation_changes(
    model: nn.Module,
    input_ids: torch.Tensor,
    layers: List[int],
    learning_rate: float,
    attention_mask: Optional[torch.Tensor] = None,
    prompt_length: int = 0,
    param_names: Optional[List[str]] = None,
) -> Dict:
    """Full pipeline: loss -> grads -> SGD Dtheta -> JVP approximation.

    Convenience function that chains compute_sgd_param_changes and
    compute_activation_change_approx.

    Args:
        model: The transformer model.
        input_ids: (batch, seq_len) input token IDs.
        layers: Embedding-inclusive layer indices.
        learning_rate: SGD step size for computing Dtheta.
        attention_mask: Optional attention mask.
        prompt_length: Number of prompt tokens to mask from loss.
        param_names: Which parameters to include (None = all trainable).

    Returns:
        Dict with keys:
            'change_approx': Dict[int, Tensor]  per-layer Dh_l approx
            'param_changes': Dict[str, Tensor]   per-param Dtheta
            'loss': float                         scalar loss value
    """
    param_changes, loss_val = compute_sgd_param_changes(
        model=model,
        input_ids=input_ids,
        learning_rate=learning_rate,
        attention_mask=attention_mask,
        prompt_length=prompt_length,
        param_names=param_names,
    )

    change_approx = compute_activation_change_approx(
        model=model,
        input_ids=input_ids,
        layers=layers,
        param_changes=param_changes,
        attention_mask=attention_mask,
    )

    return {
        "change_approx": change_approx,
        "param_changes": param_changes,
        "loss": loss_val,
    }
