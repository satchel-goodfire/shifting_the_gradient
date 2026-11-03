"""Small valid legacy adapter artifacts for storage/planning tests."""

import json
from pathlib import Path

import torch
from safetensors.torch import save_file


def write_adapter(directory):
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    save_file({"lora_A.weight": torch.zeros(2, 3)},
              str(root / "adapter_model.safetensors"))
    (root / "adapter_config.json").write_text(json.dumps(
        {"peft_type": "LORA", "base_model_name_or_path": "test-model"}))
