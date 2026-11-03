"""Configuration dataclasses for LoRA finetuning."""

from dataclasses import dataclass, field
from typing import List, Optional

from ..steering.config import SteeringConfig


@dataclass
class LoRAConfig:
    """Configuration for LoRA adapters.

    Attributes:
        r: LoRA rank (dimension of low-rank matrices)
        lora_alpha: LoRA scaling factor
        lora_dropout: Dropout probability for LoRA layers
        target_modules: List of module names to apply LoRA to (None = auto-detect)
        bias: Bias training mode ("none", "all", "lora_only")
    """
    r: int = 64
    lora_alpha: int = 128
    lora_dropout: float = 0.05
    target_modules: Optional[List[str]] = None
    bias: str = "none"


@dataclass
class TrainingConfig:
    """Configuration for training.

    Attributes:
        learning_rate: Learning rate for optimizer
        num_epochs: Number of training epochs
        per_device_train_batch_size: Batch size per device
        per_device_eval_batch_size: Batch size for evaluation
        gradient_accumulation_steps: Number of gradient accumulation steps
        max_seq_length: Maximum sequence length
        warmup_ratio: Warmup ratio for learning rate scheduler
        weight_decay: Weight decay for optimizer
        output_dir: Directory to save outputs
        logging_steps: Steps between logging
        eval_steps: Steps between evaluations (None = end of epoch)
        save_steps: Steps between checkpoints (None = end of epoch)
        save_total_limit: Maximum number of checkpoints to keep
        fp16: Use FP16 mixed precision
        bf16: Use BF16 mixed precision
        gradient_checkpointing: Enable gradient checkpointing
        seed: Random seed for reproducibility
    """
    learning_rate: float = 1e-4
    num_epochs: int = 3
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 4
    gradient_accumulation_steps: int = 1
    max_seq_length: int = 512
    warmup_ratio: float = 0.03
    weight_decay: float = 0.01
    output_dir: str = "outputs/lora_finetune"
    logging_steps: int = 10
    eval_steps: Optional[int] = None
    save_steps: Optional[int] = None
    save_total_limit: int = 3
    fp16: bool = False
    bf16: bool = False
    gradient_checkpointing: bool = False
    seed: int = 42


@dataclass
class DataConfig:
    """Configuration for data loading.

    Attributes:
        train_file: Path to training data file
        eval_file: Path to evaluation data file (optional)
        data_format: Format of data ("chat" or "text")
        text_column: Column name for text data
        max_samples: Maximum number of samples to use (None = all)
        prompt_file: Path to prompt file for injection (optional)
        prompt_index: Specific prompt index to use (None = random sampling)
        prompt_seed: Seed for prompt sampling reproducibility
        prompt_mode: How to inject prompts ("system" or "prepend_user")
        baseline_system_prompt: Baseline system prompt when using prepend_user mode
        train_on_responses_only: Mask system/user tokens from the training loss
    """
    train_file: str = ""
    eval_file: Optional[str] = None
    data_format: str = "chat"
    text_column: str = "text"
    max_samples: Optional[int] = None
    prompt_file: Optional[str] = None
    prompt_index: Optional[int] = None
    prompt_seed: Optional[int] = None
    prompt_mode: str = "system"
    baseline_system_prompt: str = ""
    train_on_responses_only: bool = False

    def __post_init__(self):
        if self.data_format not in ("chat", "text"):
            raise ValueError(f"data_format must be 'chat' or 'text', got '{self.data_format}'")
        if self.prompt_mode not in ("system", "prepend_user"):
            raise ValueError(f"prompt_mode must be 'system' or 'prepend_user', got '{self.prompt_mode}'")


@dataclass
class FinetuneConfig:
    """Complete configuration for finetuning.

    Attributes:
        model_name: HuggingFace model name or path
        dtype: Data type for model ("float16", "bfloat16", "float32")
        trust_remote_code: Whether to trust remote code
        lora: LoRA configuration
        training: Training configuration
        data: Data configuration
        steering: Steering configuration (optional)
        peft_checkpoint: Path to existing PEFT checkpoint to continue training from (optional)
    """
    model_name: str = ""
    dtype: str = "bfloat16"
    trust_remote_code: bool = False
    lora: LoRAConfig = field(default_factory=LoRAConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    steering: Optional[SteeringConfig] = None
    peft_checkpoint: Optional[str] = None

    def __post_init__(self):
        valid_dtypes = ("float16", "bfloat16", "float32")
        if self.dtype not in valid_dtypes:
            raise ValueError(
                f"dtype must be one of {valid_dtypes}, got {self.dtype!r}"
            )
