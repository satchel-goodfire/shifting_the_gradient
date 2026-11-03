"""Main activation collector class for comparing LLM activations with/without prompts."""

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from .chat_template import render_chat
from .hooks import HookManager
from .delta import compute_deltas


def is_peft_checkpoint(model_path: str) -> bool:
    """Check if a path is a PEFT/LoRA checkpoint.

    Args:
        model_path: Path to check

    Returns:
        True if this appears to be a PEFT checkpoint
    """
    path = Path(model_path)
    if not path.exists():
        return False

    # Check for PEFT adapter files
    adapter_config = path / "adapter_config.json"
    return adapter_config.exists()


def get_peft_base_model(model_path: str) -> Optional[str]:
    """Get the base model name from a PEFT checkpoint.

    Args:
        model_path: Path to PEFT checkpoint

    Returns:
        Base model name or None if not found
    """
    adapter_config_path = Path(model_path) / "adapter_config.json"
    if not adapter_config_path.exists():
        return None

    with open(adapter_config_path, "r") as f:
        config = json.load(f)

    return config.get("base_model_name_or_path")

# Type alias for chat messages
ChatMessages = List[Dict[str, str]]


def resolve_dtype(dtype: Union[str, torch.dtype]) -> torch.dtype:
    """Resolve a dtype argument to a torch.dtype.

    Accepts either a string name or a torch.dtype; unknown values are an
    error rather than a silent fp16 fallback (several analysis scripts
    passed torch.dtype objects and unknowingly ran in fp16).
    """
    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    if isinstance(dtype, torch.dtype):
        return dtype
    if dtype in dtype_map:
        return dtype_map[dtype]
    raise ValueError(
        f"Unsupported dtype {dtype!r}; expected a torch.dtype or one of "
        f"{sorted(dtype_map)}"
    )


@dataclass
class CollectionResult:
    """Result from a single activation collection."""

    activations: Dict[int, torch.Tensor]  # layer_idx -> (seq_len, hidden_dim)
    gradients: Dict[int, torch.Tensor]  # layer_idx -> (seq_len, hidden_dim)
    tokens: List[int]  # token IDs
    # Number of non-response tokens: for text data, the injected-prompt token
    # count (0 if no prompt); for chat data, the index where the final
    # assistant response begins (0 if the chat has no final assistant turn).
    # NOTE: the chat loss/gradient path ignores this and covers all tokens.
    prompt_length: int
    text: str  # original text or formatted chat
    prompt: Optional[str]  # prompt used (None if no prompt)
    deltas: Optional[Dict[int, torch.Tensor]] = None  # layer_idx -> (seq_len, hidden_dim) delta tensors
    messages: Optional[ChatMessages] = None  # original chat messages (if applicable)
    loss: Optional[torch.Tensor] = None  # (seq_len,) per-token CE loss


class ActivationCollector:
    """
    Collects activations and gradients from LLMs with and without prompts.

    Example usage:
        collector = ActivationCollector("gpt2")
        result = collector.collect("Hello world", prompt="You are helpful.")
    """

    def __init__(
        self,
        model_name: str,
        device: str = "cuda",
        dtype: Union[str, torch.dtype] = "float16",
        layers: Optional[List[int]] = None,
        trust_remote_code: bool = False,
        base_model: Optional[str] = None,
    ):
        """
        Initialize the activation collector.

        Args:
            model_name: HuggingFace model name or path (can be a PEFT checkpoint)
            device: Device to run on ("cuda" or "cpu")
            dtype: Data type ("float16", "bfloat16", "float32", or a torch.dtype)
            layers: Optional list of layer indices to collect (None = all layers)
            trust_remote_code: Whether to trust remote code for model loading
            base_model: Base model name (required if model_name is PEFT checkpoint
                        and base model can't be auto-detected)
        """
        self.model_name = model_name
        self.device = device
        self.layers = layers

        self.dtype = resolve_dtype(dtype)

        # Check if this is a PEFT checkpoint
        self._is_peft = is_peft_checkpoint(model_name)

        if self._is_peft:
            # Load PEFT model properly
            self.model, self.tokenizer = self._load_peft_model(
                model_name,
                base_model=base_model,
                trust_remote_code=trust_remote_code,
            )
        else:
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name, trust_remote_code=trust_remote_code
            )
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            # Load model
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=self.dtype,
                device_map=device,
                trust_remote_code=trust_remote_code,
            )

        self.model.eval()

        # Get vocab size for loss computation
        self.vocab_size = self.model.config.vocab_size

        # Initialize hook manager
        self.hook_manager = HookManager(self.model, layers=layers)

    def _load_peft_model(
        self,
        checkpoint_path: str,
        base_model: Optional[str] = None,
        trust_remote_code: bool = False,
    ) -> Tuple["torch.nn.Module", "AutoTokenizer"]:
        """Load a PEFT/LoRA checkpoint properly.

        Args:
            checkpoint_path: Path to the PEFT checkpoint
            base_model: Base model name (auto-detected if not provided)
            trust_remote_code: Whether to trust remote code

        Returns:
            Tuple of (model, tokenizer)
        """
        from peft import PeftModel

        # Get base model name
        if base_model is None:
            base_model = get_peft_base_model(checkpoint_path)

        if base_model is None:
            raise ValueError(
                f"Could not determine base model for PEFT checkpoint at {checkpoint_path}. "
                "Please provide --base_model argument."
            )

        print(f"Loading PEFT checkpoint from: {checkpoint_path}")
        print(f"Base model: {base_model}")

        # Load tokenizer from base model
        tokenizer = AutoTokenizer.from_pretrained(
            base_model, trust_remote_code=trust_remote_code
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Load base model
        base_model_obj = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=self.dtype,
            device_map=self.device,
            trust_remote_code=trust_remote_code,
        )

        # Load PEFT adapter
        model = PeftModel.from_pretrained(
            base_model_obj,
            checkpoint_path,
            torch_dtype=self.dtype,
        )

        # Do NOT merge_and_unload: for small LoRA adapters the weight
        # perturbation can be below bf16's representable precision,
        # silently destroying the adapter effect.  PeftModel applies
        # LoRA at runtime on the activations, preserving full fidelity.
        # Hook registration works via the PEFT-aware patterns in hooks.py.

        return model, tokenizer

    def _tokenize(
        self, text: str, prompt: Optional[str] = None
    ) -> Tuple[torch.Tensor, int]:
        """
        Tokenize text, optionally with prompt prepended.

        Args:
            text: The main text to process
            prompt: Optional prompt to prepend

        Returns:
            Tuple of (input_ids tensor, prompt_length in tokens)
        """
        if prompt:
            full_text = prompt + text
            # Tokenize prompt alone to get its length
            prompt_tokens = self.tokenizer.encode(prompt, add_special_tokens=False)
            prompt_length = len(prompt_tokens)
        else:
            full_text = text
            prompt_length = 0

        # Tokenize full text
        tokens = self.tokenizer(
            full_text, return_tensors="pt", add_special_tokens=True
        )
        input_ids = tokens["input_ids"].to(self.device)

        # Adjust prompt_length to account for any special tokens added at start
        if prompt and self.tokenizer.bos_token_id is not None:
            # Check if BOS was added
            if input_ids[0, 0].item() == self.tokenizer.bos_token_id:
                prompt_length += 1

        return input_ids, prompt_length

    def _compute_loss(
        self, logits: torch.Tensor, input_ids: torch.Tensor, prompt_length: int
    ) -> torch.Tensor:
        """
        Compute teacher-forced cross-entropy loss on non-prompt tokens.

        Args:
            logits: Model output logits (batch, seq_len, vocab_size)
            input_ids: Input token IDs (batch, seq_len)
            prompt_length: Number of tokens in the prompt

        Returns:
            Scalar loss tensor
        """
        # Shift for next-token prediction
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = input_ids[..., 1:].contiguous()

        if prompt_length > 0:
            # Create mask: 0 for prompt positions, 1 for content positions
            mask = torch.zeros_like(shift_labels, dtype=torch.float)
            # Account for the shift: if prompt_length=5, we mask positions 0-3
            # (position 4 predicts token 5, which is first non-prompt token)
            mask_start = max(0, prompt_length - 1)
            mask[..., mask_start:] = 1.0

            # Compute per-token loss
            loss_per_token = F.cross_entropy(
                shift_logits.view(-1, self.vocab_size),
                shift_labels.view(-1),
                reduction="none",
            )

            # Apply mask and compute mean
            masked_loss = loss_per_token * mask.view(-1)
            loss = masked_loss.sum() / mask.sum().clamp(min=1)
        else:
            loss = F.cross_entropy(
                shift_logits.view(-1, self.vocab_size), shift_labels.view(-1)
            )

        return loss

    def collect(
        self,
        text: str,
        prompt: Optional[str] = None,
        collect_gradients: bool = True,
    ) -> CollectionResult:
        """
        Collect activations (and optionally gradients) for a single text.

        Args:
            text: The text to process
            prompt: Optional prompt to prepend
            collect_gradients: Whether to compute and collect gradients

        Returns:
            CollectionResult containing activations, gradients, and metadata
        """
        # Tokenize
        input_ids, prompt_length = self._tokenize(text, prompt)

        # Register hooks
        self.hook_manager.register_hooks(collect_gradients=collect_gradients)

        try:
            # Forward pass
            if collect_gradients:
                # Need gradients, so don't use no_grad
                self.model.zero_grad()
                # Use enable_grad context to ensure gradient computation works
                # even if model is in eval mode with frozen parameters
                with torch.enable_grad():
                    # Create embedding with requires_grad to enable gradient flow
                    # through the model even if all parameters are frozen
                    inputs_embeds = self.model.get_input_embeddings()(input_ids)
                    inputs_embeds = inputs_embeds.requires_grad_(True)

                    outputs = self.model(inputs_embeds=inputs_embeds)
                    logits = outputs.logits

                    # Per-token loss for storage (detached, no graph)
                    per_token_loss = self._compute_per_token_loss(
                        logits.detach(), input_ids
                    )

                    # Compute loss and backward
                    loss = self._compute_loss(logits, input_ids, prompt_length)
                    loss.backward()

                gradients = self.hook_manager.get_gradients()
            else:
                with torch.no_grad():
                    outputs = self.model(input_ids)
                    logits = outputs.logits
                    per_token_loss = self._compute_per_token_loss(logits, input_ids)
                gradients = {}

            # Get activations
            activations = self.hook_manager.get_activations()

            # Convert to (seq_len, hidden_dim) by removing batch dim
            activations = {k: v.squeeze(0).cpu() for k, v in activations.items()}
            gradients = {k: v.squeeze(0).cpu() for k, v in gradients.items()}

            return CollectionResult(
                activations=activations,
                gradients=gradients,
                tokens=input_ids.squeeze(0).tolist(),
                prompt_length=prompt_length,
                text=text,
                prompt=prompt,
                loss=per_token_loss.squeeze(0).cpu(),
            )

        finally:
            # Always clean up hooks
            self.hook_manager.clear()

    def collect_comparison(
        self,
        text: str,
        prompt: str,
        collect_gradients: bool = True,
    ) -> Tuple[CollectionResult, CollectionResult]:
        """
        Collect activations both with and without a prompt for comparison.

        Args:
            text: The text to process
            prompt: The prompt to compare with
            collect_gradients: Whether to compute and collect gradients

        Returns:
            Tuple of (result_without_prompt, result_with_prompt)
        """
        result_without = self.collect(text, prompt=None, collect_gradients=collect_gradients)
        result_with = self.collect(text, prompt=prompt, collect_gradients=collect_gradients)

        return result_without, result_with

    def collect_batch(
        self,
        texts: List[str],
        prompt: Optional[str] = None,
        collect_gradients: bool = True,
    ) -> List[CollectionResult]:
        """
        Collect activations for multiple texts (processed one at a time).

        Note: For true batched processing, use collect_chat_batch.

        Args:
            texts: List of texts to process
            prompt: Optional prompt to prepend to each text
            collect_gradients: Whether to compute and collect gradients

        Returns:
            List of CollectionResult, one per text
        """
        results = []
        for text in texts:
            result = self.collect(text, prompt=prompt, collect_gradients=collect_gradients)
            results.append(result)
        return results

    def _tokenize_chat_batch(
        self,
        messages_list: List[ChatMessages],
        add_generation_prompt: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, List[str], List[int]]:
        """
        Tokenize and pad a batch of chat messages.

        Args:
            messages_list: List of chat message lists
            add_generation_prompt: Whether to add generation prompt

        Returns:
            Tuple of (input_ids, attention_mask, formatted_texts, seq_lengths)
        """
        formatted_texts = []
        all_input_ids = []
        seq_lengths = []

        for messages in messages_list:
            # thinking disabled, as on every render path (chat_template)
            text = render_chat(self.tokenizer, messages,
                               add_generation_prompt=add_generation_prompt)
            formatted_texts.append(text)

            tokens = self.tokenizer(text, add_special_tokens=False)
            all_input_ids.append(tokens["input_ids"])
            seq_lengths.append(len(tokens["input_ids"]))

        # Pad to max length
        max_len = max(seq_lengths)
        batch_size = len(messages_list)
        pad_token_id = self.tokenizer.pad_token_id

        input_ids = torch.full((batch_size, max_len), pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros((batch_size, max_len), dtype=torch.long)

        for i, (ids, length) in enumerate(zip(all_input_ids, seq_lengths)):
            input_ids[i, :length] = torch.tensor(ids, dtype=torch.long)
            attention_mask[i, :length] = 1

        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)

        return input_ids, attention_mask, formatted_texts, seq_lengths

    def _compute_batch_loss(
        self,
        logits: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute mean loss over all non-padding tokens in a batch.

        Args:
            logits: (batch, seq_len, vocab_size)
            input_ids: (batch, seq_len)
            attention_mask: (batch, seq_len)

        Returns:
            Scalar loss tensor
        """
        # Shift for next-token prediction
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = input_ids[..., 1:].contiguous()
        shift_mask = attention_mask[..., 1:].contiguous().float()

        # Compute per-token loss
        loss_per_token = F.cross_entropy(
            shift_logits.view(-1, self.vocab_size),
            shift_labels.view(-1),
            reduction="none",
        )

        # Reshape and apply mask
        loss_per_token = loss_per_token.view(shift_labels.shape)
        masked_loss = loss_per_token * shift_mask

        # Per-sample mean, then SUM across samples.
        # Using sum (not mean) ensures each sample's hook gradient
        # dL/dh_i = dL_i/dh_i matches the single-sample path exactly,
        # since only sample i's loss depends on its own activations.
        per_sample_loss = masked_loss.sum(dim=-1) / shift_mask.sum(dim=-1).clamp(min=1)
        loss = per_sample_loss.sum()
        return loss

    def _compute_per_token_loss(
        self,
        logits: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute per-token cross-entropy loss for all positions.

        loss[i] = CE of predicting token[i+1] from position i.
        The last position is NaN (no next token). Padded positions are NaN.

        Args:
            logits: (batch, seq_len, vocab_size) - should be detached
            input_ids: (batch, seq_len)
            attention_mask: optional (batch, seq_len), 1=valid, 0=padding

        Returns:
            (batch, seq_len) per-token loss tensor
        """
        with torch.no_grad():
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = input_ids[..., 1:].contiguous()

            loss_per_token = F.cross_entropy(
                shift_logits.view(-1, self.vocab_size),
                shift_labels.view(-1),
                reduction="none",
            ).view(shift_labels.shape)

            # Mask padding positions to NaN
            if attention_mask is not None:
                shift_mask = attention_mask[..., 1:].contiguous().bool()
                loss_per_token = loss_per_token.masked_fill(~shift_mask, float('nan'))

            # Pad to (batch, seq_len) with NaN at end (last position has no target)
            pad = torch.full(
                (*loss_per_token.shape[:-1], 1),
                float('nan'),
                device=loss_per_token.device,
                dtype=loss_per_token.dtype,
            )
            return torch.cat([loss_per_token, pad], dim=-1)

    def collect_chat_batch(
        self,
        messages_list: List[ChatMessages],
        collect_gradients: bool = True,
        add_generation_prompt: bool = False,
        collect_deltas: bool = False,
        delta_messages_list: Optional[List[ChatMessages]] = None,
        delta_lr: float = 0.005,
        delta_steering_manager=None,
        delta_orig_weights=None,
    ) -> List[CollectionResult]:
        """
        Collect activations for a batch of chat messages with padding.

        This is more efficient than processing samples one at a time.
        Sequences are padded to the max length in the batch, and padding
        is removed from the results.

        Args:
            messages_list: List of chat message lists
            collect_gradients: Whether to compute and collect gradients
            add_generation_prompt: Whether to add generation prompt
            collect_deltas: Whether to compute and collect deltas.
            delta_messages_list: Optional list of chat message lists for delta computation.
                If None, will use messages_list for delta computation.
            delta_orig_weights: Pre-saved weights from save_model_weights().
                Avoids redundant weight saves when calling repeatedly.
        Returns:
            List of CollectionResult, one per chat
        """
        if not messages_list:
            return []
        if collect_deltas and not collect_gradients:
            raise ValueError("collect_deltas requires collect_gradients to be True")

        if collect_deltas and len(messages_list) > 1:
            import warnings
            warnings.warn(
                f"Batched delta computation (batch_size={len(messages_list)}) "
                f"uses the average gradient direction across all samples in "
                f"the batch, not per-sample gradients. The lr is scaled by "
                f"1/batch_size to match single-sample magnitude, but the "
                f"gradient direction still reflects the batch average. Use "
                f"batch_size=1 for exact per-sample deltas.",
                stacklevel=2,
            )

        # Tokenize and pad batch
        input_ids, attention_mask, formatted_texts, seq_lengths = self._tokenize_chat_batch(
            messages_list, add_generation_prompt
        )

        if collect_deltas:
            if delta_messages_list is None:
                delta_messages_list = messages_list
            delta_input_ids, delta_attention_mask, delta_formatted_texts, delta_seq_lengths = self._tokenize_chat_batch(
                delta_messages_list, add_generation_prompt
            )

        # Register hooks
        self.hook_manager.register_hooks(collect_gradients=collect_gradients)

        try:
            if collect_gradients:
                self.model.zero_grad()
                with torch.enable_grad():
                    # Create embeddings with requires_grad
                    inputs_embeds = self.model.get_input_embeddings()(input_ids)
                    inputs_embeds = inputs_embeds.requires_grad_(True)

                    outputs = self.model(
                        inputs_embeds=inputs_embeds,
                        attention_mask=attention_mask,
                    )
                    logits = outputs.logits

                    # Per-token loss for storage (detached, no graph)
                    per_token_loss = self._compute_per_token_loss(
                        logits.detach(), input_ids, attention_mask
                    )

                    # Compute loss over batch
                    loss = self._compute_batch_loss(logits, input_ids, attention_mask)
                    loss.backward()

                gradients = self.hook_manager.get_gradients()
                activations = self.hook_manager.get_activations()
                deltas = {}

                if collect_deltas:
                    self.hook_manager.clear()
                    with torch.no_grad():
                        # Check if steering was active before delta computation
                        steering_was_active = (
                            delta_steering_manager is not None
                            and len(delta_steering_manager._handles) > 0
                        )
                        # Disable steering during delta computation
                        if delta_steering_manager:
                            delta_steering_manager.remove_hooks()

                        # Compute deltas. Scale lr by 1/batch_size to
                        # compensate for p.grad being the sum of per-sample
                        # gradients (from the sum-loss). This makes the SGD
                        # step use the average gradient, approximating
                        # single-sample deltas. Note: the gradient *direction*
                        # is still the batch average, not per-sample.
                        batch_size = len(messages_list)
                        deltas = compute_deltas(
                            self.model,
                            input_ids=delta_input_ids,
                            hook_manager=self.hook_manager,
                            attention_mask=delta_attention_mask,
                            lr=delta_lr / batch_size,
                            orig_weights=delta_orig_weights,
                        )

                        # Only re-enable steering if it was active before
                        if steering_was_active:
                            delta_steering_manager.register_hooks()
            else:
                with torch.no_grad():
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                    )
                    logits = outputs.logits
                    per_token_loss = self._compute_per_token_loss(
                        logits, input_ids, attention_mask
                    )
                activations = self.hook_manager.get_activations()
                gradients = {}
                deltas = {}


            # Split batch results and remove padding
            results = []
            for i, (messages, text, length) in enumerate(
                zip(messages_list, formatted_texts, seq_lengths)
            ):
                # Extract per-sample activations (remove padding)
                sample_activations = {
                    k: v[i, :length, :].cpu() for k, v in activations.items()
                }

                # Extract per-sample gradients (remove padding)
                sample_gradients = {}
                if gradients:
                    sample_gradients = {
                        k: v[i, :length, :].cpu() for k, v in gradients.items()
                    }
                
                sample_deltas = {}
                if collect_deltas:
                    sample_deltas = {
                        k: v[i, :delta_seq_lengths[i], :].cpu() for k, v in deltas.items()
                    }

                sample_tokens = input_ids[i, :length].tolist()
                results.append(CollectionResult(
                    activations=sample_activations,
                    gradients=sample_gradients,
                    deltas=sample_deltas,
                    tokens=sample_tokens,
                    prompt_length=self._chat_response_start(
                        messages, sample_tokens
                    ),
                    text=text,
                    prompt=None,
                    messages=messages,
                    loss=per_token_loss[i, :length].cpu(),
                ))

            return results

        finally:
            self.hook_manager.clear()

    @property
    def num_layers(self) -> int:
        """Return the number of transformer layers in the model."""
        return self.hook_manager.num_layers

    @property
    def hidden_size(self) -> int:
        """Return the hidden size of the model."""
        return self.model.config.hidden_size

    def _tokenize_chat(
        self,
        messages: ChatMessages,
        add_generation_prompt: bool = False,
    ) -> torch.Tensor:
        """
        Tokenize chat messages using the model's chat template.

        Args:
            messages: List of chat messages [{"role": "...", "content": "..."}]
            add_generation_prompt: Whether to add the generation prompt at the end

        Returns:
            Input IDs tensor
        """
        # Apply chat template (thinking disabled: grad_shift.chat_template)
        text = render_chat(self.tokenizer, messages,
                           add_generation_prompt=add_generation_prompt)

        # Tokenize
        tokens = self.tokenizer(text, return_tensors="pt", add_special_tokens=False)
        input_ids = tokens["input_ids"].to(self.device)

        return input_ids, text

    def _chat_response_start(
        self,
        messages: ChatMessages,
        full_tokens: List[int],
    ) -> int:
        """
        Token index where the final assistant response begins, or 0.

        Renders the conversation without its final assistant message (plus
        the generation prompt) and counts how many of those tokens prefix
        the full sequence. Returns 0 when the last message is not an
        assistant turn (nothing to slice). Populates
        ``CollectionResult.prompt_length``; the chat loss/gradient path
        covers all template tokens unless ``collect_chat(response_only=
        True)``, which masks the loss to the tokens from this index on --
        the boundary the trainer's ``train_on_responses_only`` uses
        (``finetuning/dataset.py``: labels start at the length of the
        generation-prompt rendering of all but the last message).
        """
        if not messages or messages[-1].get("role") != "assistant":
            return 0
        prefix_ids, _ = self._tokenize_chat(
            messages[:-1], add_generation_prompt=True
        )
        prefix = prefix_ids.squeeze(0).tolist()
        # Count the actual matching prefix to be robust to BPE boundary
        # effects at the template/response seam.
        n = 0
        for a, b in zip(prefix, full_tokens):
            if a != b:
                break
            n += 1
        return n

    def loss_prompt_length(self, messages: ChatMessages,
                           tokens: List[int], response_only: bool) -> int:
        """Tokens excluded from the chat loss: 0 (all-token loss, the
        default) or, with ``response_only``, everything before the final
        assistant response (the trainer's ``train_on_responses_only``)."""
        if not response_only:
            return 0
        n = self._chat_response_start(messages, tokens)
        if n == 0:
            raise ValueError("response_only needs a final assistant message")
        return n

    def collect_chat(
        self,
        messages: ChatMessages,
        collect_gradients: bool = True,
        add_generation_prompt: bool = False,
        response_only: bool = False,
    ) -> CollectionResult:
        """
        Collect activations (and optionally gradients) for chat messages.

        Args:
            messages: List of chat messages [{"role": "...", "content": "..."}]
            collect_gradients: Whether to compute and collect gradients
            add_generation_prompt: Whether to add generation prompt at end
            response_only: Gradients of the loss on the final assistant
                response only (as ``train_on_responses_only`` trains);
                default: the loss on all tokens

        Returns:
            CollectionResult containing activations, gradients, and metadata
        """
        # Tokenize chat
        input_ids, formatted_text = self._tokenize_chat(messages, add_generation_prompt)

        # Register hooks
        self.hook_manager.register_hooks(collect_gradients=collect_gradients)

        try:
            # Forward pass
            if collect_gradients:
                self.model.zero_grad()
                # Use enable_grad context to ensure gradient computation works
                # even if model is in eval mode with frozen parameters
                with torch.enable_grad():
                    # Create embedding with requires_grad to enable gradient flow
                    # through the model even if all parameters are frozen
                    inputs_embeds = self.model.get_input_embeddings()(input_ids)
                    inputs_embeds = inputs_embeds.requires_grad_(True)

                    outputs = self.model(inputs_embeds=inputs_embeds)
                    logits = outputs.logits

                    # Per-token loss for storage (detached, no graph)
                    per_token_loss = self._compute_per_token_loss(
                        logits.detach(), input_ids
                    )

                    # All-token loss unless response_only (then masked to
                    # the response, the trainer's train_on_responses_only)
                    loss = self._compute_loss(
                        logits, input_ids,
                        prompt_length=self.loss_prompt_length(
                            messages, input_ids.squeeze(0).tolist(),
                            response_only))
                    loss.backward()

                gradients = self.hook_manager.get_gradients()
            else:
                with torch.no_grad():
                    outputs = self.model(input_ids)
                    logits = outputs.logits
                    per_token_loss = self._compute_per_token_loss(logits, input_ids)
                gradients = {}

            # Get activations
            activations = self.hook_manager.get_activations()

            # Convert to (seq_len, hidden_dim) by removing batch dim
            activations = {k: v.squeeze(0).cpu() for k, v in activations.items()}
            gradients = {k: v.squeeze(0).cpu() for k, v in gradients.items()}

            sample_tokens = input_ids.squeeze(0).tolist()
            return CollectionResult(
                activations=activations,
                gradients=gradients,
                tokens=sample_tokens,
                prompt_length=self._chat_response_start(messages, sample_tokens),
                text=formatted_text,
                prompt=None,
                messages=messages,
                loss=per_token_loss.squeeze(0).cpu(),
            )

        finally:
            self.hook_manager.clear()

    def collect_chat_comparison(
        self,
        messages: ChatMessages,
        prompt: str,
        prompt_mode: str = "system",
        collect_gradients: bool = True,
        add_generation_prompt: bool = False,
        baseline_system_prompt: str = "You are an LLM chatbot.",
    ) -> Tuple[CollectionResult, CollectionResult]:
        """
        Collect activations both with and without a prompt for chat messages.

        Args:
            messages: List of chat messages [{"role": "...", "content": "..."}]
            prompt: The prompt to inject
            prompt_mode: How to inject the prompt:
                - "system": Append prompt to baseline system message
                - "prepend_user": Prepend to the first user message content
            collect_gradients: Whether to compute and collect gradients
            add_generation_prompt: Whether to add generation prompt at end
            baseline_system_prompt: The baseline system prompt to use
                (default: "You are an LLM chatbot.")

        Returns:
            Tuple of (result_without_prompt, result_with_prompt)
        """
        from grad_shift.data_utils import inject_prompt_into_chat

        # Collect without prompt (still inject baseline system prompt)
        messages_without_prompt = inject_prompt_into_chat(
            messages, prompt=None, mode=prompt_mode,
            baseline_system_prompt=baseline_system_prompt
        )
        result_without = self.collect_chat(
            messages_without_prompt,
            collect_gradients=collect_gradients,
            add_generation_prompt=add_generation_prompt,
        )

        # Inject prompt and collect
        messages_with_prompt = inject_prompt_into_chat(
            messages, prompt, mode=prompt_mode,
            baseline_system_prompt=baseline_system_prompt
        )
        result_with = self.collect_chat(
            messages_with_prompt,
            collect_gradients=collect_gradients,
            add_generation_prompt=add_generation_prompt,
        )

        # Update the prompt field
        result_with.prompt = prompt

        return result_without, result_with
