"""Configuration dataclass for steering modules."""

import warnings
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class SteeringConfig:
    """Configuration for steering vector application.

    Uses embedding-inclusive layer indexing convention:
    - Index 0 = embedding layer (zero placeholder in steering vectors)
    - Index N = transformer block N-1
    - active_layers=[20] hooks model.layers[19] and reads steering_vectors[20]

    Attributes:
        steering_vector_path: Path to .pt file containing steering vectors [num_layers, hidden_dim].
            Vectors always include the embedding placeholder at index 0.
        active_layers: List of embedding-inclusive layer indices to apply steering to
            (None = all transformer blocks). Must be >= 1.
        global_scale: Global scaling factor for all steering vectors
        learnable_scales: Whether to use learnable per-layer scales
        application_point: Where to apply steering:
            - "layer_output": After full layer (default)
            - "attention": After attention sublayer
            - "ffn": After FFN sublayer
        freeze_steering: Whether to freeze steering vectors during training
        skip_embedding_layer: Deprecated, ignored. Kept for backward compatibility.
    """
    steering_vector_path: Optional[str] = None
    active_layers: Optional[List[int]] = None
    global_scale: float = 1.0
    learnable_scales: bool = False
    application_point: str = "layer_output"
    freeze_steering: bool = True
    skip_embedding_layer: bool = False
    grad_manipulation: Optional[str] = None  # remove_vector_component, keep_vector_component, flip/pos_sign/neg_sign_vector_component
    grad_manipulation_vector: Optional[str] = None  # Path to vector .pt file; defaults to steering_vector_path
    grad_manipulation_layers: Optional[List[int]] = None  # Layers for grad manipulation; defaults to active_layers
    # Scale schedule for the sign-forcing modes (pos/neg_sign_vector_component).
    # The enforced along-vector component is s(t) * m, where
    # s(t) = scale_final + (scale_init - scale_final) * exp(-t / scale_tau)
    # (constant scale_init when scale_final/scale_tau are None), t is the
    # trainer global_step (microbatch units), and m is |grad @ v| for
    # scale_ref="component" or ||grad|| for scale_ref="grad_norm" (realized
    # tokenwise cosine ~= s(t)). Defaults reproduce the unscaled manipulation.
    grad_manipulation_scale_init: float = 1.0
    grad_manipulation_scale_final: Optional[float] = None
    grad_manipulation_scale_tau: Optional[float] = None
    grad_manipulation_scale_ref: str = "component"  # "component" | "grad_norm"
    # Optional JSON file of per-layer scale factors {layer: factor}
    # (embedding-inclusive indices), multiplied with the schedule scale —
    # e.g. measured per-layer mean cosine shifts for mean_shift mode.
    grad_manipulation_scale_file: Optional[str] = None
    # JSON file of per-layer activation-projection targets {layer: a*} for
    # act_servo_vector_component (required for, and only valid with, that
    # mode) — e.g. the PPS-equilibrium intrinsic projections. With
    # grad_manipulation_servo_basis_file set, values are length-k lists
    # (one target per basis row).
    grad_manipulation_act_target_file: Optional[str] = None
    # What the act_servo target is compared with when a steering vector is
    # added on a servoed layer. None (default, the behavior of every run
    # before 2026-10-21): the hooked block output, which INCLUDES the
    # steering vector (steering hooks are registered first). "intrinsic":
    # the model's own activation, i.e. the projection minus the steering
    # vector's constant projection, so a servo + PPS run controls the same
    # quantity, with the same target, as a servo-only run
    # (GradientManipulationManager act_servo_exclude_by_layer).
    grad_manipulation_act_target_reference: Optional[str] = None
    # Optional [k, hidden] or [num_layers, k, hidden] .pt basis: the
    # manipulation operates on the k-direction SUBSPACE instead of a
    # single vector (raw directions Gram-Schmidt orthonormalized per
    # layer). Valid with remove/keep/flip/pos_sign/neg_sign/act_servo;
    # for act_servo each direction is servoed to its own per-layer
    # target (targets transformed exactly into the orthonormal basis).
    grad_manipulation_basis_file: Optional[str] = None
    # Deprecated alias for grad_manipulation_basis_file (kept so saved
    # configs from the 2026-07-31 servo runs still parse).
    grad_manipulation_servo_basis_file: Optional[str] = None
    # Weight-gradient projection ("remove"): strip the output-row
    # component along the trait direction/basis from the lora_B grads of
    # residual-writing modules (o_proj/down_proj) before each optimizer
    # step, and re-project the weights after it — the learned adapter
    # delta provably never writes into the direction (see
    # finetuning/weight_grad_projection.py). Reuses
    # grad_manipulation_vector/_layers/_basis_file; mutually exclusive
    # with grad_manipulation and grad_transplant.
    weight_grad_projection: Optional[str] = None
    # Activation-write projection ("remove"): forward hooks on every
    # residual-writing module (o_proj/down_proj) in the zone project the
    # trait direction out of the module OUTPUT before it enters the
    # stream — no write, learned or frozen, can move the stream along
    # the direction (see finetuning/activation_write_projection.py).
    # Part of the computation, so muzzled checkpoints must be evaluated
    # with eval_trait's --write_projection_* flags. Reuses
    # grad_manipulation_vector/_layers/_basis_file; mutually exclusive
    # with grad_manipulation/grad_transplant/weight_grad_projection;
    # combinable with steering (the steering vector is added to the
    # stream directly, not through a muzzled write).
    activation_write_projection: Optional[str] = None
    # Gradient transplant (two passes per microbatch, see
    # finetuning/grad_transplant.py): "delta_full" | "delta_along" |
    # "delta_orthog" (donor = steered pass, recipient = unsteered) or
    # "activation" | "activation_along" (donor = unsteered, recipient =
    # steered; the latter swaps in only the along-vector component).
    # Requires steering_vector_path; mutually exclusive with
    # grad_manipulation.
    grad_transplant: Optional[str] = None
    # Optional [k, hidden] .pt basis for the transplant along/orthog split:
    # rows are orthonormalized and broadcast to all transplant layers, and
    # "along" becomes the projection onto span(basis) instead of onto the
    # single per-layer trait vector. Only valid with
    # grad_transplant in ("delta_along", "delta_orthog").
    grad_transplant_basis_file: Optional[str] = None

    def __post_init__(self):
        valid_points = ("layer_output", "attention", "ffn")
        if self.application_point not in valid_points:
            raise ValueError(
                f"application_point must be one of {valid_points}, "
                f"got '{self.application_point}'"
            )
        if self.skip_embedding_layer:
            warnings.warn(
                "skip_embedding_layer is deprecated and will be removed in a "
                "future version. Steering vectors now always use embedding-"
                "inclusive indexing (index 0 = embedding placeholder). "
                "The parameter is ignored.",
                DeprecationWarning,
                stacklevel=2,
            )
        # Validate grad_manipulation
        valid_grad_modes = (
            "remove_vector_component", "keep_vector_component",
            "flip_vector_component",
            "pos_sign_vector_component", "neg_sign_vector_component",
            "mean_shift_vector_component", "act_servo_vector_component",
            "act_servo_base_vector_component",
        )
        if self.grad_manipulation is not None:
            if self.grad_manipulation not in valid_grad_modes:
                raise ValueError(
                    f"grad_manipulation must be one of {valid_grad_modes}, "
                    f"got '{self.grad_manipulation}'"
                )
            if self.grad_manipulation_vector is None:
                self.grad_manipulation_vector = self.steering_vector_path

        # Validate grad manipulation scale schedule (mirrors
        # GradientManipulationManager so bad configs fail before model load)
        valid_scale_refs = ("component", "grad_norm")
        if self.grad_manipulation_scale_ref not in valid_scale_refs:
            raise ValueError(
                f"grad_manipulation_scale_ref must be one of "
                f"{valid_scale_refs}, got '{self.grad_manipulation_scale_ref}'"
            )
        has_final = self.grad_manipulation_scale_final is not None
        has_tau = self.grad_manipulation_scale_tau is not None
        if has_final != has_tau:
            raise ValueError(
                "grad_manipulation_scale_final and grad_manipulation_scale_tau "
                "must be set together"
            )
        if has_tau and self.grad_manipulation_scale_tau <= 0:
            raise ValueError(
                f"grad_manipulation_scale_tau must be positive, "
                f"got {self.grad_manipulation_scale_tau}"
            )
        uses_scaling = (
            self.grad_manipulation_scale_init != 1.0
            or has_final
            or self.grad_manipulation_scale_ref != "component"
            or self.grad_manipulation_scale_file is not None
        )
        scalable_modes = (
            "pos_sign_vector_component", "neg_sign_vector_component",
            "mean_shift_vector_component",
        )
        if uses_scaling and self.grad_manipulation not in scalable_modes:
            raise ValueError(
                f"grad manipulation scale scheduling requires "
                f"grad_manipulation in {scalable_modes}, "
                f"got '{self.grad_manipulation}'"
            )
        is_servo = self.grad_manipulation == "act_servo_vector_component"
        if is_servo != (self.grad_manipulation_act_target_file is not None):
            raise ValueError(
                "grad_manipulation_act_target_file is required for (and "
                "only valid with) grad_manipulation="
                "'act_servo_vector_component'"
            )
        if self.grad_manipulation_act_target_reference is not None:
            if self.grad_manipulation_act_target_reference != "intrinsic":
                raise ValueError(
                    "grad_manipulation_act_target_reference must be None or "
                    "'intrinsic', got "
                    f"{self.grad_manipulation_act_target_reference!r}"
                )
            if not is_servo or self.steering_vector_path is None:
                raise ValueError(
                    "grad_manipulation_act_target_reference='intrinsic' "
                    "needs grad_manipulation='act_servo_vector_component' "
                    "and a steering vector (it removes the steering "
                    "vector's projection from the servoed activation)"
                )
            if (self.grad_manipulation_basis_file is not None
                    or self.grad_manipulation_servo_basis_file is not None):
                raise ValueError(
                    "grad_manipulation_act_target_reference='intrinsic' is "
                    "single-vector only (no basis file)"
                )
            if self.application_point != "layer_output":
                raise ValueError(
                    "grad_manipulation_act_target_reference='intrinsic' "
                    "needs application_point='layer_output' (the servo "
                    "hooks the block output; a sublayer vector is not a "
                    "constant offset there)"
                )
            # The trainer computes the vector's projection once at setup
            # (LoRATrainer._steering_constants); a training vector or scale
            # would leave that offset stale.
            if not self.freeze_steering or self.learnable_scales:
                raise ValueError(
                    "grad_manipulation_act_target_reference='intrinsic' "
                    "needs frozen steering (freeze_steering=True, "
                    "learnable_scales=False): its offset is computed once"
                )
        # Per-token base servo: targets are captured from the adapter-
        # disabled base forward each microbatch (a target file is rejected
        # by the check above); single vector only, like the manager.
        if (self.grad_manipulation == "act_servo_base_vector_component"
                and self.grad_manipulation_basis_file is not None):
            raise ValueError(
                "grad_manipulation_basis_file is not supported with "
                "'act_servo_base_vector_component'")
        if self.grad_manipulation_servo_basis_file is not None:
            if self.grad_manipulation_basis_file is not None:
                raise ValueError(
                    "set only grad_manipulation_basis_file (the servo_basis "
                    "spelling is a deprecated alias), not both"
                )
            warnings.warn(
                "grad_manipulation_servo_basis_file is deprecated; use "
                "grad_manipulation_basis_file",
                DeprecationWarning,
                stacklevel=2,
            )
            self.grad_manipulation_basis_file = (
                self.grad_manipulation_servo_basis_file
            )
        basis_modes = (
            "remove_vector_component", "keep_vector_component",
            "flip_vector_component",
            "pos_sign_vector_component", "neg_sign_vector_component",
            "act_servo_vector_component",
        )
        if self.grad_manipulation_basis_file is not None and (
                self.grad_manipulation not in basis_modes
                and self.weight_grad_projection is None
                and self.activation_write_projection is None):
            raise ValueError(
                f"grad_manipulation_basis_file requires grad_manipulation "
                f"in {basis_modes}, weight_grad_projection, or "
                f"activation_write_projection, got "
                f"grad_manipulation={self.grad_manipulation!r}"
            )

        # Validate activation-write projection
        if self.activation_write_projection is not None:
            if self.activation_write_projection != "remove":
                raise ValueError(
                    f"activation_write_projection must be 'remove', got "
                    f"'{self.activation_write_projection}'"
                )
            if (self.grad_manipulation is not None
                    or self.grad_transplant is not None
                    or self.weight_grad_projection is not None):
                raise ValueError(
                    "activation_write_projection is mutually exclusive with "
                    "grad_manipulation, grad_transplant, and "
                    "weight_grad_projection"
                )
            if self.grad_manipulation_vector is None:
                self.grad_manipulation_vector = self.steering_vector_path
            if (self.grad_manipulation_vector is None
                    and self.grad_manipulation_basis_file is None):
                raise ValueError(
                    "activation_write_projection requires "
                    "grad_manipulation_vector (or steering_vector_path, "
                    "or a basis file)"
                )

        # Validate weight-gradient projection
        if self.weight_grad_projection is not None:
            if self.weight_grad_projection != "remove":
                raise ValueError(
                    f"weight_grad_projection must be 'remove', got "
                    f"'{self.weight_grad_projection}'"
                )
            if self.grad_manipulation is not None or self.grad_transplant is not None:
                raise ValueError(
                    "weight_grad_projection is mutually exclusive with "
                    "grad_manipulation and grad_transplant"
                )
            if self.grad_manipulation_vector is None:
                self.grad_manipulation_vector = self.steering_vector_path
            if (self.grad_manipulation_vector is None
                    and self.grad_manipulation_basis_file is None):
                raise ValueError(
                    "weight_grad_projection requires grad_manipulation_vector "
                    "(or steering_vector_path, or a basis file)"
                )

        # Validate grad transplant
        valid_transplant_modes = (
            "delta_full", "delta_along", "delta_orthog", "activation",
            "activation_along",
        )
        if self.grad_transplant is not None:
            if self.grad_transplant not in valid_transplant_modes:
                raise ValueError(
                    f"grad_transplant must be one of {valid_transplant_modes}, "
                    f"got '{self.grad_transplant}'"
                )
            if self.grad_manipulation is not None:
                raise ValueError(
                    "grad_transplant and grad_manipulation are mutually "
                    "exclusive"
                )
            if self.steering_vector_path is None:
                raise ValueError(
                    "grad_transplant requires steering_vector_path (the "
                    "donor/recipient steered pass needs a steering vector)"
                )
            if self.grad_manipulation_vector is None:
                self.grad_manipulation_vector = self.steering_vector_path
        if self.grad_transplant_basis_file is not None and (
                self.grad_transplant not in ("delta_along", "delta_orthog")):
            raise ValueError(
                "grad_transplant_basis_file is only valid with "
                "grad_transplant in ('delta_along', 'delta_orthog'), "
                f"got grad_transplant={self.grad_transplant!r}"
            )
