"""Compute activation deltas from SGD weight updates.

Delta = h(theta - lr*grad) - h(theta), the actual activation change from
a single SGD step. Uses output_hidden_states for layers 1..N-1 and a
temporary forward hook for layer N (the last block), since
output_hidden_states[N] is the post-final-norm output, not the raw block
output. Original weights are saved to CPU for exact restoration (bf16
add/sub cycles are not invertible due to rounding).
"""

import torch
from typing import Dict, List, Optional

from .hooks import find_transformer_blocks, HookManager


def _find_block_modules(model):
    """Find the nn.ModuleList of transformer blocks."""
    _, layer_modules, _ = find_transformer_blocks(model)
    return layer_modules


def save_model_weights(model: torch.nn.Module) -> Dict[str, torch.Tensor]:
    """Save all trainable parameter weights to CPU for exact restoration.

    Returns:
        Dict mapping parameter name to CPU tensor copy.
    """
    return {
        # ``Tensor.cpu()`` returns the original tensor when it is already on
        # CPU.  Clone explicitly so this is a snapshot rather than an alias.
        n: p.detach().cpu().clone()
        for n, p in model.named_parameters()
        if p.requires_grad
    }


def restore_model_weights(
    model: torch.nn.Module,
    saved_weights: Dict[str, torch.Tensor],
) -> None:
    """Restore model weights from a previously saved copy."""
    for n, p in model.named_parameters():
        if n in saved_weights:
            p.data.copy_(saved_weights[n].to(p.device))


def compute_deltas(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    hook_manager: HookManager,
    attention_mask: torch.Tensor=None,
    lr: float=0.005,
    orig_weights: Optional[Dict[str, torch.Tensor]]=None,
) -> Dict[int, torch.Tensor]:
    """
    Compute deltas for a given model and input IDs. Runs a forward pass
    with the model, then applies the parameter gradients to the model, and
    runs a forward pass again to get the deltas. Original weights are saved
    to CPU for exact restoration (bf16 add/sub cycles are not invertible).

    Args:
        model: The model to compute deltas for.
        input_ids: The input IDs to compute deltas for.
        hook_manager: HookManager to capture per-layer activations.
        attention_mask: The attention mask for the input.
        lr: The learning rate to use for the SGD step.
        orig_weights: Pre-saved weights from save_model_weights(). If
            provided, skips the per-call weight save and restores from
            this copy instead. Much faster when calling repeatedly.

    Returns:
        Dict mapping layer index to (batch, seq_len, hidden_dim) tensor.
    """
    local_weights = None
    weights_modified = False

    # Register hooks
    hook_manager.register_hooks(collect_gradients=False)

    try:
        with torch.no_grad():
            model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
        og_actvs = hook_manager.get_activations()

        # Apply SGD step
        if orig_weights is None:
            # Save weights before modifying (slow path)
            local_weights = {}
            for n, p in model.named_parameters():
                if p.grad is not None:
                    local_weights[n] = p.detach().cpu().clone()
            weights_modified = bool(local_weights)
            for n, p in model.named_parameters():
                if n in local_weights:
                    p.data.add_(p.grad, alpha=-lr)
        else:
            # Weights already saved externally (fast path)
            weights_modified = any(p.grad is not None for p in model.parameters())
            for p in model.parameters():
                if p.grad is not None:
                    p.data.add_(p.grad, alpha=-lr)

        # Reset hooks for new activation collection
        hook_manager.clear()
        hook_manager.register_hooks(collect_gradients=False)

        with torch.no_grad():
            model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
        new_actvs = hook_manager.get_activations()

        # Compute deltas
        deltas = {
            k: (new_actvs[k] - og_actvs[k]).detach().clone()
            for k in og_actvs.keys()
        }

    finally:
        # Restore even if the second forward pass or activation collection
        # raises (for example, after an OOM).  Add/sub restoration is lossy for
        # low-precision weights, so always copy from the saved snapshot.
        if weights_modified:
            restore_weights = orig_weights if orig_weights is not None else local_weights
            if restore_weights is not None:
                restore_model_weights(model, restore_weights)
        hook_manager.clear()

    return deltas

def compute_delta_inplace(
    model,
    input_ids: torch.Tensor,
    layers: List[int],
    lr: float,
    orig_weights: Optional[Dict[str, torch.Tensor]]=None,
) -> Dict[int, torch.Tensor]:
    """Compute h(theta - lr*grad) - h(theta) using existing .grad tensors.

    Must be called BEFORE model.zero_grad() so .grad is still populated.

    Args:
        model: The model with .grad populated from a backward pass.
        input_ids: (1, seq_len) token IDs on device.
        layers: List of embedding-inclusive layer indices.
        lr: Learning rate for the SGD step.
        orig_weights: Pre-saved weights from save_model_weights(). If
            provided, skips the per-call weight save and restores from
            this copy instead. Much faster when calling repeatedly.

    Returns:
        Dict mapping layer index to (seq_len, hidden_dim) float32 CPU tensor.
    """

    # Last layer is different because it's the post-final-norm output, not the raw block output.
    # We use a temporary forward hook to capture the pre-norm output.
    block_modules = _find_block_modules(model)
    num_layers = len(block_modules)
    if not layers:
        return {}
    invalid_layers = [layer for layer in layers if layer < 0 or layer > num_layers]
    if invalid_layers:
        raise ValueError(
            f"Layer indices {invalid_layers} out of range; expected 0..{num_layers}"
        )
    if len(set(layers)) != len(layers):
        raise ValueError("layers must not contain duplicate indices")

    needs_hook = num_layers in layers

    hook_captures = {}
    handles = []
    def make_hook(tag):
        def hook_fn(module, inp, output):
            out = output[0] if isinstance(output, tuple) else output
            hook_captures[tag] = out.detach().squeeze(0).float().cpu()
        return hook_fn

    if needs_hook:
        handles.append(
            block_modules[-1].register_forward_hook(make_hook("before"))
        )

    local_weights = None
    weights_modified = False
    try:
        with torch.no_grad():
            out_before = model(input_ids=input_ids, output_hidden_states=True)
            h_before = {
                l: out_before.hidden_states[l].squeeze(0).float().cpu()
                for l in layers if l != num_layers
            }
            if needs_hook:
                h_before[num_layers] = hook_captures.pop("before")
            del out_before

            for handle in handles:
                handle.remove()
            handles.clear()
            if needs_hook:
                handles.append(
                    block_modules[-1].register_forward_hook(make_hook("after"))
                )

            # Apply SGD step
            if orig_weights is None:
                local_weights = {
                    n: p.detach().cpu().clone()
                    for n, p in model.named_parameters()
                    if p.grad is not None
                }
            weights_modified = any(p.grad is not None for p in model.parameters())
            for p in model.parameters():
                if p.grad is not None:
                    p.data.add_(p.grad, alpha=-lr)

            out_after = model(input_ids=input_ids, output_hidden_states=True)
            h_after = {
                l: out_after.hidden_states[l].squeeze(0).float().cpu()
                for l in layers if l != num_layers
            }
            if needs_hook:
                h_after[num_layers] = hook_captures.pop("after")
            del out_after
    finally:
        for handle in handles:
            handle.remove()
        if weights_modified:
            restore_weights = orig_weights if orig_weights is not None else local_weights
            if restore_weights is not None:
                restore_model_weights(model, restore_weights)

    return {l: h_after[l] - h_before[l] for l in layers}
