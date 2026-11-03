"""Dataset utilities for loading text data."""

import json
import random
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence, Union

from torch.utils.data import Dataset, IterableDataset


# Type alias for chat messages
ChatMessages = List[Dict[str, str]]


def _load_single_text_file(
    path: Path,
    text_column: str = "text",
) -> List[str]:
    """Load texts from a single file."""
    suffix = path.suffix.lower()
    texts = []

    if suffix == ".json":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, list):
                for item in data:
                    if isinstance(item, dict):
                        texts.append(item[text_column])
                    else:
                        texts.append(str(item))
            elif isinstance(data, dict) and text_column in data:
                # Handle datasets format: {"text": [...]}
                texts = data[text_column]

    elif suffix == ".jsonl":
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    item = json.loads(line)
                    texts.append(item[text_column])

    elif suffix == ".csv":
        import csv

        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                texts.append(row[text_column])

    elif suffix == ".txt":
        with open(path, "r", encoding="utf-8") as f:
            texts = [line.strip() for line in f if line.strip()]

    else:
        raise ValueError(f"Unsupported file format: {suffix}")

    return texts


def load_dataset_from_file(
    path: Union[str, List[str]],
    text_column: str = "text",
    max_samples: Optional[int] = None,
    seed: Optional[int] = None,
) -> List[str]:
    """
    Load a dataset of texts from one or more files.

    Supports:
    - JSON files (list of objects with text_column field)
    - JSONL files (one JSON object per line with text_column field)
    - CSV files (with text_column header)
    - TXT files (one text per line)

    When multiple files are provided, samples equally from each file.

    Args:
        path: Path to file(s). Can be:
            - A single path string
            - A comma-separated string of paths
            - A list of path strings
        text_column: Name of the column/field containing text
        max_samples: Maximum total number of samples to load
        seed: Random seed for shuffling when sampling from multiple files

    Returns:
        List of text strings
    """
    # Parse paths
    if isinstance(path, str):
        if "," in path:
            paths = [Path(p.strip()) for p in path.split(",")]
        else:
            paths = [Path(path)]
    else:
        paths = [Path(p) for p in path]

    # Load from each file
    all_texts_by_file = []
    for p in paths:
        texts = _load_single_text_file(p, text_column)
        all_texts_by_file.append(texts)
        print(f"  Loaded {len(texts)} samples from {p.name}")

    # If single file, simple case
    if len(paths) == 1:
        texts = all_texts_by_file[0]
        if max_samples is not None:
            texts = texts[:max_samples]
        return texts

    # Multiple files: sample equally from each
    if seed is not None:
        random.seed(seed)

    # Shuffle each file's data
    for texts in all_texts_by_file:
        random.shuffle(texts)

    # Calculate samples per file
    if max_samples is not None:
        samples_per_file = max_samples // len(paths)
        remainder = max_samples % len(paths)
    else:
        # Use all samples, but interleave them
        samples_per_file = max(len(t) for t in all_texts_by_file)
        remainder = 0

    # Collect samples equally from each file
    combined = []
    for i, texts in enumerate(all_texts_by_file):
        n_samples = samples_per_file + (1 if i < remainder else 0)
        # If file has fewer samples than requested, use all of them
        combined.extend(texts[:n_samples])

    # Final shuffle to interleave
    random.shuffle(combined)

    return combined


def load_hf_dataset(
    dataset_name: str,
    split: str = "train",
    text_column: str = "text",
    max_samples: Optional[int] = None,
) -> List[str]:
    """
    Load a dataset from HuggingFace datasets.

    Args:
        dataset_name: Name of the dataset on HuggingFace
        split: Which split to load
        text_column: Name of the column containing text
        max_samples: Maximum number of samples to load

    Returns:
        List of text strings
    """
    from datasets import load_dataset

    dataset = load_dataset(dataset_name, split=split)

    if max_samples is not None:
        dataset = dataset.select(range(min(max_samples, len(dataset))))

    return dataset[text_column]


class PromptDataset(Dataset):
    """
    A dataset that wraps a list of texts for use with activation collection.

    This is a simple wrapper that provides a PyTorch Dataset interface.
    """

    def __init__(
        self,
        texts: List[str],
        prompt: Optional[str] = None,
    ):
        """
        Initialize the dataset.

        Args:
            texts: List of text strings
            prompt: Optional prompt (stored for reference, not applied here)
        """
        self.texts = texts
        self.prompt = prompt

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> str:
        return self.texts[idx]


class StreamingTextDataset(IterableDataset):
    """
    An iterable dataset for streaming large text files.

    Useful for JSONL files that are too large to load into memory.
    """

    def __init__(
        self,
        path: str,
        text_column: str = "text",
        max_samples: Optional[int] = None,
    ):
        """
        Initialize the streaming dataset.

        Args:
            path: Path to the file (must be .jsonl or .txt)
            text_column: Name of the field containing text (for JSONL)
            max_samples: Maximum number of samples to yield
        """
        self.path = Path(path)
        self.text_column = text_column
        self.max_samples = max_samples

        suffix = self.path.suffix.lower()
        if suffix not in [".jsonl", ".txt"]:
            raise ValueError(
                f"StreamingTextDataset only supports .jsonl and .txt files, got {suffix}"
            )

    def __iter__(self) -> Iterator[str]:
        count = 0
        suffix = self.path.suffix.lower()

        with open(self.path, "r", encoding="utf-8") as f:
            for line in f:
                if self.max_samples is not None and count >= self.max_samples:
                    break

                line = line.strip()
                if not line:
                    continue

                if suffix == ".jsonl":
                    item = json.loads(line)
                    text = item[self.text_column]
                else:
                    text = line

                yield text
                count += 1


def load_prompt_file(path: str) -> List[str]:
    """
    Load prompts from a JSON file.

    Expected format: {"prompts": ["prompt1", "prompt2", ...]}

    Args:
        path: Path to the prompt JSON file

    Returns:
        List of prompt strings
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if "prompts" not in data:
        raise ValueError(f"Prompt file must contain 'prompts' key, got: {list(data.keys())}")

    return data["prompts"]


def sample_prompt(prompts: List[str], seed: Optional[int] = None) -> str:
    """
    Sample a random prompt from a list.

    Args:
        prompts: List of prompt strings
        seed: Optional random seed for reproducibility

    Returns:
        A randomly selected prompt string
    """
    if seed is not None:
        random.seed(seed)
    return random.choice(prompts)


def _load_single_chat_file(path: Path) -> List[ChatMessages]:
    """Load chat data from a single JSONL file."""
    chats = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                item = json.loads(line)
                if "messages" in item:
                    chats.append(item["messages"])
                else:
                    raise ValueError(f"Chat data must contain 'messages' key, got: {list(item.keys())}")

    return chats


def load_chat_data(
    path: Union[str, List[str]],
    max_samples: Optional[int] = None,
    seed: Optional[int] = None,
) -> List[ChatMessages]:
    """
    Load chat data from one or more JSONL files.

    Expected format: Each line is {"messages": [{"role": "user/assistant", "content": "..."}]}

    When multiple files are provided, samples equally from each file.

    Args:
        path: Path to file(s). Can be:
            - A single path string
            - A comma-separated string of paths
            - A list of path strings
        max_samples: Maximum total number of samples to load
        seed: Random seed for shuffling when sampling from multiple files

    Returns:
        List of chat message lists
    """
    # Parse paths
    if isinstance(path, str):
        if "," in path:
            paths = [Path(p.strip()) for p in path.split(",")]
        else:
            paths = [Path(path)]
    else:
        paths = [Path(p) for p in path]

    # Load from each file
    all_chats_by_file = []
    for p in paths:
        chats = _load_single_chat_file(p)
        all_chats_by_file.append(chats)
        print(f"  Loaded {len(chats)} chat samples from {p.name}")

    # If single file, simple case
    if len(paths) == 1:
        chats = all_chats_by_file[0]
        if max_samples is not None:
            chats = chats[:max_samples]
        return chats

    # Multiple files: sample equally from each
    if seed is not None:
        random.seed(seed)

    # Shuffle each file's data
    for chats in all_chats_by_file:
        random.shuffle(chats)

    # Calculate samples per file
    if max_samples is not None:
        samples_per_file = max_samples // len(paths)
        remainder = max_samples % len(paths)
    else:
        # Use all samples, but interleave them
        samples_per_file = max(len(c) for c in all_chats_by_file)
        remainder = 0

    # Collect samples equally from each file
    combined = []
    for i, chats in enumerate(all_chats_by_file):
        n_samples = samples_per_file + (1 if i < remainder else 0)
        # If file has fewer samples than requested, use all of them
        combined.extend(chats[:n_samples])

    # Final shuffle to interleave
    random.shuffle(combined)

    return combined


def normalize_system_prompt(value: Union[str, Sequence[str]]) -> str:
    """
    Normalize a system-prompt argument to a single string.

    CLI parsers declare ``--baseline_system_prompt`` with ``nargs="+"`` (a
    list of words), while JSON config files supply a plain string. Joining a
    config string with ``" ".join`` splits it character-by-character
    ("Y o u   a r e ..."), which silently corrupted training prompts — so
    every consumer should route through this idempotent helper instead of
    joining directly.

    Args:
        value: A prompt as a string, or a list/tuple of words from argparse.

    Returns:
        The prompt as a single space-joined string.
    """
    if isinstance(value, (list, tuple)):
        return " ".join(str(v) for v in value)
    if not isinstance(value, str):
        raise TypeError(
            f"System prompt must be a string or list of words, got "
            f"{type(value).__name__}"
        )
    return value


def inject_prompt_into_chat(
    messages: ChatMessages,
    prompt: Optional[str] = None,
    mode: str = "prepend_user",
    baseline_system_prompt: str = "You are an LLM chatbot.",
) -> ChatMessages:
    """
    Inject a prompt into a chat message list.

    Always ensures a system message exists at the start with the baseline
    content. In "system" mode, the prompt is appended to the baseline
    system prompt (e.g. "You are an LLM chatbot. <prompt>"). In
    "prepend_user" mode, the system message is set to the baseline and
    the prompt is prepended to the first user message.

    Args:
        messages: List of chat messages [{"role": "...", "content": "..."}]
        prompt: The prompt to inject (None to just ensure baseline system
            message)
        mode: How to inject the prompt:
            - "system": Append prompt to baseline system message
            - "prepend_user": Prepend to the first user message content
        baseline_system_prompt: The baseline system prompt to use
            (default: "You are an LLM chatbot.")

    Returns:
        New list of messages with the prompt injected
    """
    if mode not in ("system", "prepend_user"):
        raise ValueError(f"Invalid prompt mode: {mode}. Must be 'system' or 'prepend_user'")

    # Make a deep copy to avoid modifying the original
    messages = [msg.copy() for msg in messages]

    # Check if there's already a system message at the start
    has_system = messages and messages[0]["role"] == "system"

    if mode == "system":
        content = baseline_system_prompt
        if prompt:
            content = baseline_system_prompt + " " + prompt if baseline_system_prompt else prompt
        if has_system:
            messages[0]["content"] = content
        else:
            messages.insert(0, {"role": "system", "content": content})

    elif mode == "prepend_user":
        # Ensure baseline system message exists at start
        if has_system:
            messages[0]["content"] = baseline_system_prompt
        else:
            messages.insert(0, {"role": "system", "content": baseline_system_prompt})

        # If prompt provided, prepend to first user message
        if prompt:
            for msg in messages:
                if msg["role"] == "user":
                    msg["content"] = prompt + "\n\n" + msg["content"]
                    break
            else:
                raise ValueError("No user message found in chat to prepend prompt to")

    return messages
