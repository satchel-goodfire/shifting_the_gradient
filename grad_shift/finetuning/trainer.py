"""LoRA trainer with Accelerate for multi-GPU support."""

import json
import os
import math
import logging
from typing import Dict, Optional

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)
from accelerate import Accelerator
from accelerate.utils import set_seed
from peft import LoraConfig, get_peft_model, TaskType, PeftModel

from .config import FinetuneConfig
from .dataset import FinetuneDataset, create_dataloaders
from .grad_manipulation import GradientManipulationManager
from .activation_write_projection import ActivationWriteProjectionManager
from .grad_transplant import GradientTransplantManager
from .weight_grad_projection import WeightGradProjectionManager
from ..steering import SteeringModule, SteeringHookManager
from ..model_artifacts import mark_model_incomplete, save_model_artifacts

logger = logging.getLogger(__name__)


# Default LoRA target modules for common architectures (matched as a
# substring of ``config.model_type``, first match wins).
DEFAULT_TARGET_MODULES = {
    "llama": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    "mistral": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    "qwen2": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    "gpt2": ["c_attn", "c_proj", "c_fc"],
    "gpt_neox": ["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"],
}


def _detect_target_modules(model) -> list:
    """Default LoRA target modules for a model's architecture.

    An architecture without a default raises instead of guessing: the
    earlier fallback (q/k/v/o, attention only) silently gave every
    unmatched model type (e.g. Qwen3's "qwen3") a different LoRA protocol
    from the paper's all-linear one; the paper's Qwen3-4B runs were trained
    that way until 2026-10-21. Pass ``target_modules`` explicitly for such
    a model.

    Args:
        model: The model to analyze

    Returns:
        List of module names to target

    Raises:
        ValueError: no default for the model type.
    """
    model_type = getattr(model.config, "model_type", "").lower()

    # Check for known architectures
    for arch, modules in DEFAULT_TARGET_MODULES.items():
        if arch in model_type:
            return modules

    raise ValueError(
        f"no default LoRA target modules for model type {model_type!r} "
        f"(known: {sorted(DEFAULT_TARGET_MODULES)}); pass target_modules "
        "explicitly (e.g. models_common.ALL_LINEAR)")


class LoRATrainer:
    """LoRA trainer with Accelerate for multi-GPU support.

    Supports multiple training modes:
    1. LoRA only - Standard finetuning
    2. LoRA + frozen steering - Apply steering during training, only train LoRA
    3. Steering only - Freeze model, train steering vectors
    4. Both - Train LoRA and steering jointly

    Example:
        config = FinetuneConfig(
            model_name="Qwen/Qwen2.5-7B-Instruct",
            lora=LoRAConfig(r=64),
            training=TrainingConfig(num_epochs=3),
            data=DataConfig(train_file="train.jsonl"),
        )

        trainer = LoRATrainer(config)
        trainer.train(train_dataset, eval_dataset)
    """

    def __init__(
        self,
        config: FinetuneConfig,
        accelerator: Optional[Accelerator] = None,
    ):
        """Initialize the trainer.

        Args:
            config: Complete finetuning configuration
            accelerator: Optional Accelerator instance (created if not provided)
        """
        self.config = config

        # Set seed for reproducibility
        set_seed(config.training.seed)

        # Initialize accelerator
        self.accelerator = accelerator or Accelerator(
            gradient_accumulation_steps=config.training.gradient_accumulation_steps,
            mixed_precision="fp16" if config.training.fp16 else ("bf16" if config.training.bf16 else "no"),
        )

        # Setup model and tokenizer
        self._setup_model_and_tokenizer()

        # Setup LoRA
        self._setup_lora()

        # Setup steering if configured
        self.steering_module: Optional[SteeringModule] = None
        self.steering_manager: Optional[SteeringHookManager] = None
        if config.steering is not None:
            self._setup_steering()

        # Setup gradient manipulation if configured
        self.grad_manipulation_manager: Optional[GradientManipulationManager] = None
        if config.steering is not None and config.steering.grad_manipulation is not None:
            self._setup_grad_manipulation()

        # Setup gradient transplant if configured (two passes per microbatch)
        self.grad_transplant_manager: Optional[GradientTransplantManager] = None
        if config.steering is not None and config.steering.grad_transplant is not None:
            self._setup_grad_transplant()

        # Setup weight-gradient projection if configured (param-space,
        # needs the PEFT-wrapped model — LoRA is applied above)
        self.weight_grad_projection_manager: Optional[
            WeightGradProjectionManager] = None
        if (config.steering is not None
                and config.steering.weight_grad_projection is not None):
            self._setup_weight_grad_projection()

        # Setup activation-write projection if configured. The muzzle is
        # part of the computation: hooks stay registered for the whole
        # run (training AND the trainer's eval-loss loop).
        self.activation_write_projection_manager: Optional[
            ActivationWriteProjectionManager] = None
        if (config.steering is not None
                and config.steering.activation_write_projection is not None):
            self._setup_activation_write_projection()

        # Logging
        self.global_step = 0
        self.epoch = 0
        self._grad_manip_log_file = None

    def _setup_model_and_tokenizer(self):
        """Load model and tokenizer."""
        config = self.config

        # Map dtype string to torch dtype
        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        try:
            torch_dtype = dtype_map[config.dtype]
        except KeyError as exc:
            raise ValueError(f"Unsupported model dtype: {config.dtype!r}") from exc

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.model_name,
            trust_remote_code=config.trust_remote_code,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            torch_dtype=torch_dtype,
            trust_remote_code=config.trust_remote_code,
        )

        # Enable gradient checkpointing if requested
        if config.training.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()

        logger.info(f"Loaded model: {config.model_name}")

    def _setup_lora(self):
        """Configure and apply LoRA adapters.

        If peft_checkpoint is provided in config, loads the existing PEFT model
        instead of creating a new LoRA adapter.
        """
        # Check if we should load from an existing PEFT checkpoint
        if self.config.peft_checkpoint is not None:
            logger.info(f"Loading PEFT checkpoint from: {self.config.peft_checkpoint}")
            self.model = PeftModel.from_pretrained(
                self.model,
                self.config.peft_checkpoint,
                is_trainable=True,
            )
            self.model.print_trainable_parameters()
            return

        # Otherwise, create a new LoRA adapter
        lora_config = self.config.lora

        # Detect target modules if not specified
        target_modules = lora_config.target_modules
        if target_modules is None:
            target_modules = _detect_target_modules(self.model)
            logger.info(f"Auto-detected LoRA target modules: {target_modules}")

        # Create PEFT config
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=lora_config.r,
            lora_alpha=lora_config.lora_alpha,
            lora_dropout=lora_config.lora_dropout,
            target_modules=target_modules,
            bias=lora_config.bias,
        )

        # Apply LoRA
        self.model = get_peft_model(self.model, peft_config)
        self.model.print_trainable_parameters()

    def _setup_steering(self):
        """Load and configure steering vectors."""
        steering_config = self.config.steering

        if steering_config.steering_vector_path is None:
            logger.warning("Steering config provided but no steering_vector_path specified")
            return

        # Load steering module
        self.steering_module = SteeringModule.from_pretrained(
            steering_config.steering_vector_path,
            global_scale=steering_config.global_scale,
            learnable_scales=steering_config.learnable_scales,
        )

        # Freeze if requested
        if steering_config.freeze_steering:
            self.steering_module.freeze_vectors()
            logger.info("Steering vectors frozen")
        else:
            logger.info("Steering vectors trainable")

        # Create hook manager (uses the base model inside PEFT)
        base_model = self.model.get_base_model() if hasattr(self.model, 'get_base_model') else self.model
        self.steering_manager = SteeringHookManager(
            base_model,
            self.steering_module,
            config=steering_config,
        )

        logger.info(f"Loaded steering vectors from: {steering_config.steering_vector_path}")

    def _setup_grad_manipulation(self):
        """Set up gradient manipulation hooks."""
        steering_config = self.config.steering
        vector_path = steering_config.grad_manipulation_vector

        if vector_path is None:
            logger.warning(
                "grad_manipulation is set but no vector path available "
                "(no grad_manipulation_vector or steering_vector_path)"
            )
            return

        base_model = (
            self.model.get_base_model()
            if hasattr(self.model, "get_base_model")
            else self.model
        )
        manip_layers = (
            steering_config.grad_manipulation_layers
            or steering_config.active_layers
        )
        # Fail closed (2026-08-11 review P1): checkpoint recomputation
        # re-runs forwards during backward, so the forward hooks that
        # register tensor grad hooks can fire on recomputed tensors, fire
        # twice, or miss the original forward depending on checkpoint
        # mode. Transplant already rejects this; manipulation must too
        # until an integration test establishes its semantics.
        if self.config.training.gradient_checkpointing:
            raise ValueError(
                "grad_manipulation is incompatible with "
                "gradient_checkpointing (hook semantics undefined under "
                "checkpoint recomputation)"
            )
        scale_by_layer = None
        if steering_config.grad_manipulation_scale_file is not None:
            with open(steering_config.grad_manipulation_scale_file) as f:
                scale_by_layer = {
                    int(k): float(v) for k, v in json.load(f).items()
                }
        act_target_by_layer = None
        if steering_config.grad_manipulation_act_target_file is not None:
            with open(steering_config.grad_manipulation_act_target_file) as f:
                act_target_by_layer = {
                    int(k): ([float(x) for x in v]
                             if isinstance(v, (list, tuple)) else float(v))
                    for k, v in json.load(f).items()
                }

        self.grad_manipulation_manager = GradientManipulationManager(
            model=base_model,
            vector_path=vector_path,
            mode=steering_config.grad_manipulation,
            layers=manip_layers,
            scale_init=steering_config.grad_manipulation_scale_init,
            scale_final=steering_config.grad_manipulation_scale_final,
            scale_tau=steering_config.grad_manipulation_scale_tau,
            scale_ref=steering_config.grad_manipulation_scale_ref,
            scale_by_layer=scale_by_layer,
            act_target_by_layer=act_target_by_layer,
            basis_path=steering_config.grad_manipulation_basis_file,
            collect_stats=True,
            act_servo_exclude_by_layer=(
                self._steering_constants(manip_layers)
                if steering_config.grad_manipulation_act_target_reference
                == "intrinsic" else None),
        )
        if self.grad_manipulation_manager.act_servo_offsets:
            logger.info(
                "act_servo on the intrinsic activation: steering-vector "
                "projection removed per layer "
                f"{self.grad_manipulation_manager.act_servo_offsets}")

        logger.info(
            f"Gradient manipulation: mode={steering_config.grad_manipulation}, "
            f"vector={vector_path}, "
            f"scale_init={steering_config.grad_manipulation_scale_init}, "
            f"scale_final={steering_config.grad_manipulation_scale_final}, "
            f"scale_tau={steering_config.grad_manipulation_scale_tau}, "
            f"scale_ref={steering_config.grad_manipulation_scale_ref}"
        )

    def _steering_constants(self, manip_layers) -> Dict[int, torch.Tensor]:
        """The constant each steering hook adds to the output of the
        servoed blocks, as ``SteeringHookManager`` adds it (cast to the
        activation dtype, times its intensity), for
        ``grad_manipulation_act_target_reference="intrinsic"``.

        Valid because steering hooks are registered BEFORE the gradient-
        manipulation hooks on every forward (``train`` and ``_evaluate``),
        so the servo's hook sees ``h + c``.
        """
        if self.steering_manager is None or self.steering_module is None:
            raise ValueError("intrinsic act_servo reference needs steering")
        dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16,
                 "float32": torch.float32}[self.config.dtype]
        n = self.steering_manager.num_layers
        layers = (range(1, n + 1) if manip_layers is None
                  else [l if l >= 0 else n + 1 + l for l in manip_layers])
        steered = set(self.steering_manager.active_layers)
        out = {}
        for l in layers:
            if l in steered:
                c = self.steering_module.get_steering_vector(l).detach()
                out[l] = (c.to(dtype) * self.steering_manager.intensity
                          ).float().cpu()
        if not out:
            raise ValueError(
                "intrinsic act_servo reference: no servoed layer is steered "
                f"(servo {list(layers)}, steering {sorted(steered)})")
        return out

    def _setup_grad_transplant(self):
        """Set up the gradient transplant manager.

        Transplant needs the steering machinery (the donor or recipient
        pass runs steered), so ``_setup_steering`` must have succeeded.
        Layers default to ALL transformer blocks (the whole backward field
        is transplanted), overridable via ``grad_manipulation_layers``.
        """
        steering_config = self.config.steering
        if self.steering_manager is None:
            raise ValueError(
                "grad_transplant requires steering to be configured "
                "(steering_vector_path with a valid steering module)"
            )
        # The donor and recipient passes must compute the same function:
        # stochastic layers would give the donor deltas a different network
        # than the recipient activations they get paired with.
        if self.config.lora.lora_dropout > 0:
            raise ValueError(
                "grad_transplant requires lora_dropout=0: the donor and "
                "recipient passes draw independent dropout masks, which "
                f"confounds the factor mixing (got "
                f"{self.config.lora.lora_dropout})"
            )
        # Gradient checkpointing re-runs forwards during backward, so the
        # capture/replace forward hooks would fire on recomputed tensors
        # with undefined pairing.
        if self.config.training.gradient_checkpointing:
            raise ValueError(
                "grad_transplant is incompatible with gradient_checkpointing"
            )
        base_model = (
            self.model.get_base_model()
            if hasattr(self.model, "get_base_model")
            else self.model
        )
        self.grad_transplant_manager = GradientTransplantManager(
            model=base_model,
            mode=steering_config.grad_transplant,
            vector_path=steering_config.grad_manipulation_vector,
            layers=steering_config.grad_manipulation_layers,
            subspace_basis_path=steering_config.grad_transplant_basis_file,
            collect_stats=True,
        )
        logger.info(
            f"Gradient transplant: mode={steering_config.grad_transplant}, "
            f"vector={steering_config.grad_manipulation_vector}, "
            f"basis={steering_config.grad_transplant_basis_file}"
        )

    def _setup_weight_grad_projection(self):
        """Set up the weight-gradient projection manager.

        Operates on the PEFT model's lora_B parameters of residual-writing
        modules (o_proj/down_proj); mutually exclusive with the
        activation-space managers (validated in SteeringConfig). Layer
        defaults mirror _setup_grad_manipulation (explicit manip layers,
        else steering active_layers, else all blocks); the direction
        comes from grad_manipulation_vector / grad_manipulation_basis_file.
        """
        steering_config = self.config.steering
        # Same layer-default convention as _setup_grad_manipulation:
        # explicit grad_manipulation_layers, else the steering
        # active_layers, else (both unset) all blocks.
        wproj_layers = (
            steering_config.grad_manipulation_layers
            or steering_config.active_layers
        )
        self.weight_grad_projection_manager = WeightGradProjectionManager(
            model=self.model,
            vector_path=steering_config.grad_manipulation_vector,
            mode=steering_config.weight_grad_projection,
            layers=wproj_layers,
            basis_path=steering_config.grad_manipulation_basis_file,
            collect_stats=True,
        )
        logger.info(
            f"Weight-grad projection: mode="
            f"{steering_config.weight_grad_projection}, "
            f"vector={steering_config.grad_manipulation_vector}, "
            f"basis={steering_config.grad_manipulation_basis_file}"
        )

    def _setup_activation_write_projection(self):
        """Set up the activation-write projection (muzzle) manager.

        Hooks attach to the residual-writing modules of the (PEFT-
        wrapped) model and remain registered for the whole run — the
        muzzle defines the computation being trained, so it applies to
        training, checkpoint-time behavior, and eval-loss measurement
        alike. Layer defaults mirror _setup_grad_manipulation.
        """
        steering_config = self.config.steering
        awp_layers = (
            steering_config.grad_manipulation_layers
            or steering_config.active_layers
        )
        self.activation_write_projection_manager = (
            ActivationWriteProjectionManager(
                model=self.model,
                vector_path=steering_config.grad_manipulation_vector,
                mode=steering_config.activation_write_projection,
                layers=awp_layers,
                basis_path=steering_config.grad_manipulation_basis_file,
                collect_stats=True,
            )
        )
        self.activation_write_projection_manager.register_hooks()
        logger.info(
            f"Activation-write projection: mode="
            f"{steering_config.activation_write_projection}, "
            f"vector={steering_config.grad_manipulation_vector}, "
            f"basis={steering_config.grad_manipulation_basis_file} "
            f"(hooks persistent for the whole run)"
        )

    def _transplant_forward_backward(self, batch) -> torch.Tensor:
        """Donor pass (capture deltas) then recipient pass (replaced grads).

        The donor pass deposits no parameter gradients (deltas come from
        ``autograd.grad`` against the captured layer outputs); the
        recipient backward produces the transplanted parameter gradients.
        Returns the recipient loss. Caller removes hooks and clears the
        stored donor deltas in its ``finally``.
        """
        mgr = self.grad_transplant_manager
        steer = self.steering_manager

        # Donor pass
        if mgr.donor_steered:
            steer.register_hooks()
        try:
            mgr.begin_capture()
            donor_loss = self.model(**batch).loss
            # Match the recipient's loss scaling: Accelerator.backward
            # divides by gradient_accumulation_steps, so donor deltas
            # must be computed at the same scale or the transplanted
            # components enter at accum-times the recipient field's
            # magnitude (the 2026-08-11 P0).
            mgr.end_capture(
                donor_loss,
                loss_scale=1.0
                / self.config.training.gradient_accumulation_steps,
            )
        finally:
            if mgr.donor_steered:
                steer.remove_hooks()

        # Recipient pass
        if not mgr.donor_steered:  # "activation" mode: recipient is steered
            steer.register_hooks()
        mgr.register_replace_hooks()
        outputs = self.model(**batch)
        loss = outputs.loss
        self.accelerator.backward(loss)
        return loss

    def _create_optimizer(self) -> AdamW:
        """Create optimizer with appropriate parameter groups."""
        params = []

        # LoRA parameters
        lora_params = [p for n, p in self.model.named_parameters() if p.requires_grad]
        if lora_params:
            params.append({
                "params": lora_params,
                "lr": self.config.training.learning_rate,
            })

        # Steering parameters (if trainable)
        if self.steering_module is not None:
            steering_params = [p for p in self.steering_module.parameters() if p.requires_grad]
            if steering_params:
                params.append({
                    "params": steering_params,
                    "lr": self.config.training.learning_rate,
                })

        return AdamW(params, weight_decay=self.config.training.weight_decay)

    def _create_scheduler(self, optimizer: AdamW, num_training_steps: int):
        """Create learning rate scheduler."""
        num_warmup_steps = int(num_training_steps * self.config.training.warmup_ratio)
        return get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
        )

    def train(
        self,
        train_dataset: FinetuneDataset,
        eval_dataset: Optional[FinetuneDataset] = None,
    ):
        """Run training.

        Args:
            train_dataset: Training dataset
            eval_dataset: Optional evaluation dataset
        """
        config = self.config.training

        # Create dataloaders
        train_loader, eval_loader = create_dataloaders(
            train_dataset,
            eval_dataset,
            train_batch_size=config.per_device_train_batch_size,
            eval_batch_size=config.per_device_eval_batch_size,
        )

        # Create optimizer and scheduler
        optimizer = self._create_optimizer()
        # Known issue, retained to reproduce the paper's runs: this floors
        # across epochs, whereas Accelerate flushes incomplete accumulation
        # groups at each epoch boundary. For single-GPU training, the actual
        # update count is ceil(len(train_loader) / accumulation) * epochs.
        # The LR therefore reaches zero a few updates early (439 scheduled
        # versus 441 actual in the main protocol; 93 versus 96 for the
        # 1,000-example, three-epoch runs), slightly compressing the schedule.
        # Given that the paper's conclusions held across different amounts
        # of training, we do not believe this affects any conclusions.
        # This is our assessment from those checks; the corrected schedule
        # has not itself been rerun.
        num_training_steps = (
            len(train_loader) * config.num_epochs // config.gradient_accumulation_steps
        )
        scheduler = self._create_scheduler(optimizer, num_training_steps)

        # Prepare with accelerator
        self.model, optimizer, train_loader, scheduler = self.accelerator.prepare(
            self.model, optimizer, train_loader, scheduler
        )
        if eval_loader is not None:
            eval_loader = self.accelerator.prepare(eval_loader)

        # Move steering module to device if needed
        if self.steering_module is not None:
            self.steering_module = self.steering_module.to(self.accelerator.device)
            # Fail closed (2026-08-11 review P1): the steering module is
            # not accelerator-prepared, so under distributed training a
            # TRAINABLE steering module would receive un-synchronized
            # per-rank updates and rank 0 would save only its own copy.
            # Frozen steering (every PPS experiment) is unaffected.
            trainable_steer = any(
                p.requires_grad for p in self.steering_module.parameters()
            )
            if trainable_steer and self.accelerator.num_processes > 1:
                raise ValueError(
                    "Trainable steering vectors are not synchronized under "
                    "distributed training (module is not wrapped by "
                    "accelerator.prepare); freeze steering or run "
                    "single-process."
                )

        # Move grad manipulation vectors to device if needed
        if self.grad_manipulation_manager is not None:
            self.grad_manipulation_manager = self.grad_manipulation_manager.to(
                self.accelerator.device
            )
            # The per-token base servo captures its reference only on the
            # plain training path (below); the transplant flow drives its
            # own forwards and would hit the missing-reference error mid-run.
            if (self.grad_manipulation_manager.needs_base_reference
                    and self.grad_transplant_manager is not None):
                raise ValueError(
                    "act_servo_base_vector_component is only supported on "
                    "the plain training path, not with grad_transplant")

        # Create output directory
        if self.accelerator.is_main_process:
            mark_model_incomplete(config.output_dir)

        # Training loop
        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {len(train_dataset)}")
        logger.info(f"  Num epochs = {config.num_epochs}")
        logger.info(f"  Batch size per device = {config.per_device_train_batch_size}")
        logger.info(f"  Gradient accumulation steps = {config.gradient_accumulation_steps}")
        logger.info(f"  Total optimization steps = {num_training_steps}")

        self.model.train()
        total_loss = 0.0

        for epoch in range(config.num_epochs):
            self.epoch = epoch
            epoch_loss = 0.0

            progress_bar = tqdm(
                train_loader,
                desc=f"Epoch {epoch + 1}/{config.num_epochs}",
                disable=not self.accelerator.is_main_process,
            )

            for step, batch in enumerate(progress_bar):
                with self.accelerator.accumulate(self.model):
                    if self.grad_transplant_manager is not None:
                        # Transplant flow: donor pass captures the delta
                        # field, recipient pass trains with it. Steering
                        # registration is handled inside per donor/recipient.
                        try:
                            loss = self._transplant_forward_backward(batch)
                            optimizer.step()
                            scheduler.step()
                            optimizer.zero_grad()
                        finally:
                            if self.steering_manager is not None:
                                self.steering_manager.remove_hooks()
                            self.grad_transplant_manager.remove_hooks()
                            self.grad_transplant_manager.clear_donor()
                    else:
                        # Per-token base servo: an adapter-disabled no-grad
                        # forward on this microbatch gives each token's own
                        # base projection as its servo target. Captured
                        # BEFORE steering hooks go on, so the reference is
                        # the plain frozen base model.
                        _gm = self.grad_manipulation_manager
                        if _gm is not None and _gm.needs_base_reference:
                            _gm.capture_base_reference(
                                self.accelerator.unwrap_model(self.model),
                                batch["input_ids"],
                                batch.get("attention_mask"))

                        # Apply steering hooks if configured. Steering is
                        # registered FIRST: the intrinsic act_servo
                        # reference (_steering_constants) relies on the
                        # manipulation hooks seeing the steered output.
                        if self.steering_manager is not None:
                            self.steering_manager.register_hooks()

                        # Apply gradient manipulation hooks if configured
                        if self.grad_manipulation_manager is not None:
                            self.grad_manipulation_manager.set_step(self.global_step)
                            self.grad_manipulation_manager.register_hooks()

                        try:
                            # Forward pass
                            outputs = self.model(**batch)
                            loss = outputs.loss

                            # Backward pass
                            self.accelerator.backward(loss)

                            # Weight-grad projection: strip the trait-
                            # direction write component from the target
                            # lora_B grads before the step, and re-project
                            # the weights after it (kills AdamW moment/
                            # decay leakage; see weight_grad_projection.py).
                            # Only on sync microbatches: with gradient
                            # accumulation, projecting the fully
                            # accumulated grad once is equivalent (the
                            # projection is linear) and weights only
                            # change at sync steps — this also keeps the
                            # logged frac-removed stat undiluted.
                            _wproj = self.weight_grad_projection_manager
                            _sync = self.accelerator.sync_gradients
                            if _wproj is not None and _sync:
                                _wproj.project_grads()

                            optimizer.step()
                            scheduler.step()

                            if _wproj is not None and _sync:
                                _wproj.project_weights()

                            optimizer.zero_grad()

                        finally:
                            # Remove steering hooks
                            if self.steering_manager is not None:
                                self.steering_manager.remove_hooks()
                            # Remove gradient manipulation hooks
                            if self.grad_manipulation_manager is not None:
                                self.grad_manipulation_manager.remove_hooks()
                                if self.grad_manipulation_manager.needs_base_reference:
                                    self.grad_manipulation_manager.clear_base_reference()

                    # Log realized gradient cosine along the manipulation
                    # vector (empirical check that the manipulation/transplant
                    # delivers the intended field)
                    if (self.grad_manipulation_manager is not None
                            or self.grad_transplant_manager is not None
                            or self.weight_grad_projection_manager is not None
                            or self.activation_write_projection_manager
                            is not None):
                        self._log_grad_manip_stats(batch_loss=loss.detach().float().item())

                    batch_loss = loss.detach().float()
                    epoch_loss += batch_loss

                    # Logging
                    if self.global_step % config.logging_steps == 0:
                        avg_loss = epoch_loss / (step + 1)
                        progress_bar.set_postfix({"loss": f"{avg_loss:.4f}"})

                        if self.accelerator.is_main_process:
                            logger.info(
                                f"Step {self.global_step}: loss = {avg_loss:.4f}, "
                                f"batch_loss = {batch_loss:.4f}, "
                                f"lr = {scheduler.get_last_lr()[0]:.2e}"
                            )

                    # Evaluation
                    if config.eval_steps and self.global_step % config.eval_steps == 0:
                        if eval_loader is not None:
                            self._evaluate(eval_loader)
                            self.model.train()

                    # Checkpointing
                    if config.save_steps and self.global_step > 0 and self.global_step % config.save_steps == 0:
                        self._save_checkpoint()

                    self.global_step += 1

            # End of epoch evaluation
            if eval_loader is not None and config.eval_steps is None:
                self._evaluate(eval_loader)
                self.model.train()

            # End of epoch checkpoint
            if config.save_steps is None:
                self._save_checkpoint()

            total_loss += epoch_loss
            avg_epoch_loss = epoch_loss / len(train_loader)
            logger.info(f"Epoch {epoch + 1} completed. Average loss: {avg_epoch_loss:.4f}")

        # Final save
        self._save_final()

        logger.info("Training completed!")

    def _log_grad_manip_stats(self, batch_loss: float):
        """Append realized grad-cosine stats for this microbatch to a JSONL log.

        Writes ``grad_manip_cosine.jsonl`` in the output dir with one record
        per microbatch: the mean tokenwise cosine between the gradient and
        the manipulation vector before (``cos_pre``) and after (``cos_post``)
        manipulation, plus the schedule scale in effect. This is the
        empirical check that the realized attenuation matches the target
        schedule.
        """
        # Always pop (clears the accumulator on every process)
        mgr = (self.grad_manipulation_manager
               or self.grad_transplant_manager
               or self.weight_grad_projection_manager
               or self.activation_write_projection_manager)
        stats = mgr.pop_stats()
        if stats is None or not self.accelerator.is_main_process:
            return

        record = {
            "step": self.global_step,
            "epoch": self.epoch,
            "batch_loss": batch_loss,
            **stats,
        }
        if self._grad_manip_log_file is None:
            os.makedirs(self.config.training.output_dir, exist_ok=True)
            log_path = os.path.join(
                self.config.training.output_dir, "grad_manip_cosine.jsonl"
            )
            # Fresh runs overwrite: a crashed-and-restarted run would
            # otherwise append duplicate step records (no mid-run resume)
            mode = "w" if self.global_step == 0 else "a"
            self._grad_manip_log_file = open(log_path, mode)
            logger.info(f"Logging grad manipulation cosine stats to {log_path}")
        self._grad_manip_log_file.write(json.dumps(record) + "\n")
        # Flush every record: training ends via os._exit(), which would
        # drop buffered lines.
        self._grad_manip_log_file.flush()

    def _evaluate(self, eval_loader: DataLoader) -> float:
        """Run evaluation.

        Args:
            eval_loader: Evaluation data loader

        Returns:
            Average evaluation loss
        """
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch in tqdm(eval_loader, desc="Evaluating", disable=not self.accelerator.is_main_process):
                # Apply steering hooks if configured. Under transplant,
                # steering belongs to the donor/recipient training passes
                # only — eval loss is measured unsteered.
                if (self.steering_manager is not None
                        and self.grad_transplant_manager is None):
                    self.steering_manager.register_hooks()

                # Apply gradient manipulation hooks if configured
                if self.grad_manipulation_manager is not None:
                    self.grad_manipulation_manager.register_hooks()

                try:
                    outputs = self.model(**batch)
                    loss = outputs.loss
                    total_loss += loss.detach().float()
                    num_batches += 1
                finally:
                    if self.steering_manager is not None:
                        self.steering_manager.remove_hooks()
                    if self.grad_manipulation_manager is not None:
                        self.grad_manipulation_manager.remove_hooks()

        # Discard stats accumulated during eval so they don't pollute the
        # training-time cosine log
        if self.grad_manipulation_manager is not None:
            self.grad_manipulation_manager.pop_stats()

        avg_loss = total_loss / num_batches
        logger.info(f"Evaluation loss: {avg_loss:.4f}")

        return avg_loss.item()

    def _save_checkpoint(self):
        """Save a checkpoint."""
        if not self.accelerator.is_main_process:
            return

        checkpoint_dir = os.path.join(
            self.config.training.output_dir,
            f"checkpoint-{self.global_step}",
        )
        save_model_artifacts(checkpoint_dir, self._write_model_artifacts)

        logger.info(f"Saved checkpoint to {checkpoint_dir}")

        # Clean up old checkpoints
        self._cleanup_checkpoints()

    def _cleanup_checkpoints(self):
        """Remove old checkpoints beyond save_total_limit."""
        if not self.accelerator.is_main_process:
            return

        output_dir = self.config.training.output_dir
        limit = self.config.training.save_total_limit

        # Find all checkpoints
        checkpoints = []
        for name in os.listdir(output_dir):
            if name.startswith("checkpoint-"):
                checkpoints.append(os.path.join(output_dir, name))

        # Sort by step number
        checkpoints.sort(key=lambda x: int(x.split("-")[-1]))

        # Remove old ones
        while len(checkpoints) > limit:
            old_checkpoint = checkpoints.pop(0)
            import shutil
            shutil.rmtree(old_checkpoint)
            logger.info(f"Removed old checkpoint: {old_checkpoint}")

    def _save_final(self):
        """Save final model and artifacts."""
        if not self.accelerator.is_main_process:
            return

        output_dir = self.config.training.output_dir

        save_model_artifacts(output_dir, self._write_model_artifacts)

        logger.info(f"Saved final model to {output_dir}")

    def _write_model_artifacts(self, output_dir: str):
        """Write the adapter, tokenizer and optional steering into staging."""
        unwrapped_model = self.accelerator.unwrap_model(self.model)
        unwrapped_model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)

        # Save steering module if trainable
        if self.steering_module is not None and not self.config.steering.freeze_steering:
            steering_path = os.path.join(output_dir, "steering_vectors.pt")
            self.steering_module.save(steering_path)
