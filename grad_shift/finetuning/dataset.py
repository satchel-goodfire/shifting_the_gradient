"""Dataset utilities for finetuning, wrapping existing data loading functions."""

import random
from typing import Dict, List, Optional, Union

import torch
from torch.utils.data import Dataset

from grad_shift.chat_template import render_chat
from grad_shift.data_utils import (
    load_chat_data,
    load_dataset_from_file,
    load_prompt_file,
    inject_prompt_into_chat,
    ChatMessages,
)


class FinetuneDataset(Dataset):
    """Dataset for LoRA finetuning with optional prompt injection.

    Supports both chat and text formats. When a prompt file is provided,
    each __getitem__ call samples a random prompt (or uses a fixed index)
    and injects it into the data.

    Example:
        # Chat data with prompt injection
        dataset = FinetuneDataset(
            data_path="train.jsonl",
            tokenizer=tokenizer,
            data_format="chat",
            prompt_file="prompts.json",
        )

        # Text data
        dataset = FinetuneDataset(
            data_path="texts.jsonl",
            tokenizer=tokenizer,
            data_format="text",
        )
    """

    def __init__(
        self,
        data_path: str,
        tokenizer,
        data_format: str = "chat",
        text_column: str = "text",
        max_seq_length: int = 512,
        max_samples: Optional[int] = None,
        prompt_file: Optional[str] = None,
        prompt_index: Optional[int] = None,
        prompt_seed: Optional[int] = None,
        prompt_mode: str = "system",
        baseline_system_prompt: str = "",
        train_on_responses_only: bool = False,
    ):
        """Initialize the finetuning dataset.

        Args:
            data_path: Path to data file (JSONL for chat, various for text)
            tokenizer: HuggingFace tokenizer
            data_format: "chat" or "text"
            text_column: Column name for text data
            max_seq_length: Maximum sequence length
            max_samples: Maximum number of samples to load
            prompt_file: Path to prompt file for injection
            prompt_index: Specific prompt index (None = random sampling)
            prompt_seed: Seed for prompt sampling
            prompt_mode: "system" or "prepend_user"
            baseline_system_prompt: Baseline system prompt for prepend_user mode
            train_on_responses_only: Mask system/user tokens from chat labels
        """
        self.tokenizer = tokenizer
        self.data_format = data_format
        self.max_seq_length = max_seq_length
        self.prompt_mode = prompt_mode
        self.baseline_system_prompt = baseline_system_prompt
        self.prompt_index = prompt_index
        self.train_on_responses_only = train_on_responses_only

        # Set up random generator for prompt sampling
        self._rng = random.Random(prompt_seed)

        # Load data
        if data_format == "chat":
            self.data = load_chat_data(data_path, max_samples=max_samples)
        else:
            self.data = load_dataset_from_file(
                data_path, text_column=text_column, max_samples=max_samples
            )

        # Load prompts if provided
        self.prompts: Optional[List[str]] = None
        if prompt_file:
            self.prompts = load_prompt_file(prompt_file)

    def __len__(self) -> int:
        return len(self.data)

    def _sample_prompt(self) -> Optional[str]:
        """Sample a prompt from the prompt list.

        Returns:
            Sampled prompt string or None if no prompts loaded
        """
        if self.prompts is None:
            return None

        if self.prompt_index is not None:
            return self.prompts[self.prompt_index]

        return self._rng.choice(self.prompts)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a tokenized sample.

        Args:
            idx: Sample index

        Returns:
            Dictionary with input_ids, attention_mask, and labels
        """
        item = self.data[idx]

        if self.data_format == "chat":
            return self._process_chat(item)
        else:
            return self._process_text(item)

    _debug_samples_logged = 0
    _DEBUG_LOG_COUNT = 3  # Number of training samples to print

    def _process_chat(self, messages: ChatMessages) -> Dict[str, torch.Tensor]:
        """Process chat messages into tokenized format.

        Args:
            messages: List of chat messages

        Returns:
            Tokenized dictionary
        """
        # Inject prompt if available
        prompt = self._sample_prompt()
        if prompt is not None or self.baseline_system_prompt:
            messages = inject_prompt_into_chat(
                messages,
                prompt=prompt,
                mode=self.prompt_mode,
                baseline_system_prompt=self.baseline_system_prompt,
            )

        # Apply chat template (thinking disabled: grad_shift.chat_template)
        text = render_chat(self.tokenizer, messages,
                           add_generation_prompt=False)

        # Debug: print first few training samples
        if FinetuneDataset._debug_samples_logged < FinetuneDataset._DEBUG_LOG_COUNT:
            FinetuneDataset._debug_samples_logged += 1
            n = FinetuneDataset._debug_samples_logged
            total = FinetuneDataset._DEBUG_LOG_COUNT
            print(f"\n{'='*60}")
            print(f"[DEBUG] Training sample {n}/{total}")
            print(f"  baseline_system_prompt: {self.baseline_system_prompt!r}")
            print(f"  prompt_mode: {self.prompt_mode!r}")
            print(f"  sampled prompt: {prompt!r}")
            print(f"  messages: {messages}")
            print(f"  --- raw text going to tokenizer ---")
            print(text[:1000])
            if len(text) > 1000:
                print(f"  ... ({len(text)} chars total)")
            print(f"{'='*60}\n")

        label_start = None
        if self.train_on_responses_only:
            if not messages or messages[-1].get("role") != "assistant":
                raise ValueError(
                    "train_on_responses_only requires each chat sample to end "
                    "with an assistant message"
                )
            prompt_text = render_chat(self.tokenizer, messages[:-1],
                                      add_generation_prompt=True)
            prompt_ids = self.tokenizer(
                prompt_text,
                add_special_tokens=False,
            )["input_ids"]
            full_ids = self.tokenizer(
                text,
                add_special_tokens=False,
            )["input_ids"]
            if full_ids[:len(prompt_ids)] != prompt_ids:
                raise ValueError(
                    "Chat-template prompt tokens are not a prefix of the full "
                    "sample; cannot mask response labels safely"
                )
            label_start = len(prompt_ids)

        # Chat templates already contain their architecture-specific special
        # tokens.  Adding them again can produce a duplicate BOS on Llama.
        return self._tokenize(
            text,
            add_special_tokens=False,
            label_start=label_start,
        )

    def _process_text(self, text: str) -> Dict[str, torch.Tensor]:
        """Process text into tokenized format.

        Args:
            text: Raw text string

        Returns:
            Tokenized dictionary
        """
        # Prepend prompt if available
        prompt = self._sample_prompt()
        if prompt is not None:
            text = prompt + "\n\n" + text

        return self._tokenize(text)

    def _tokenize(
        self,
        text: str,
        add_special_tokens: bool = True,
        label_start: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        """Tokenize text with truncation and padding.

        Args:
            text: Text to tokenize

        Returns:
            Dictionary with input_ids, attention_mask, and labels
        """
        tokens = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_seq_length,
            padding="max_length",
            return_tensors="pt",
            add_special_tokens=add_special_tokens,
        )

        # Squeeze batch dimension
        input_ids = tokens["input_ids"].squeeze(0)
        attention_mask = tokens["attention_mask"].squeeze(0)

        # Labels are same as input_ids for causal LM
        # We mask padding tokens in labels with -100
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100
        if label_start is not None:
            active_ids = input_ids[attention_mask.bool()]
            if label_start >= len(active_ids):
                raise ValueError(
                    "The response begins after truncation; increase max_seq_length"
                )
            labels[:label_start] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


def create_dataloaders(
    train_dataset: FinetuneDataset,
    eval_dataset: Optional[FinetuneDataset] = None,
    train_batch_size: int = 4,
    eval_batch_size: int = 4,
    num_workers: int = 0,
) -> tuple:
    """Create data loaders for training and evaluation.

    Args:
        train_dataset: Training dataset
        eval_dataset: Evaluation dataset (optional)
        train_batch_size: Batch size for training
        eval_batch_size: Batch size for evaluation
        num_workers: Number of data loading workers

    Returns:
        Tuple of (train_loader, eval_loader or None)
    """
    from torch.utils.data import DataLoader

    train_loader = DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    eval_loader = None
    if eval_dataset is not None:
        eval_loader = DataLoader(
            eval_dataset,
            batch_size=eval_batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )

    return train_loader, eval_loader
