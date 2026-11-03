#!/usr/bin/env python3
"""
Computes per-layer gradient cosine similarity, activation projections,
and cross-entropy loss under multiple forward-pass conditions:
  1. No intervention (baseline)
  2. With PPS steering active (optional)
  3. With inoculation prompts injected (optional)

For each condition and layer, reports:
  - Gradient cosine similarity with direction vectors (pooled + tokenwise)
  - Activation projection onto direction vectors (pooled + tokenwise)
  - Cross-entropy loss (pooled + tokenwise, same across layers)

Outputs CSV for easy analysis.
"""

import argparse
import csv
import json
import random
import sys
from pathlib import Path

import torch
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from grad_shift.collector import ActivationCollector
from grad_shift.data_utils import (
    load_chat_data,
    load_prompt_file,
    inject_prompt_into_chat,
)
from grad_shift.steering.steering_module import SteeringModule
from grad_shift.steering.steering_hooks import SteeringHookManager


def cosine_sim(a, b):
    """Cosine similarity between vectors a (N, D) and b (D,)."""
    return (a @ b) / (a.norm(dim=1) * b.norm() + 1e-12)


def projection(a, b):
    """Scalar projection of vectors a (N, D) onto unit vector b (D,)."""
    return a @ b


def load_direction_vectors(direction_paths, layers, device):
    """Load and normalize direction vectors.

    Args:
        direction_paths: dict of name -> path
        layers: list of layer indices
        device: torch device

    Returns:
        dict of name -> {layer -> unit vector on device}
    """
    directions = {}
    for name, path in direction_paths.items():
        raw = torch.load(path, map_location="cpu").float()
        vecs = {}
        for l in layers:
            v = raw[l] if raw.dim() == 2 else raw.clone()
            norm = v.norm()
            if norm > 0:
                v = v / norm
            vecs[l] = v.to(device)
        directions[name] = vecs
        print(f"  Loaded direction '{name}' from {path} (shape {raw.shape})")
    return directions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default="Qwen/Qwen2.5-7B-Instruct",
        help="Model name or path to PEFT checkpoint directory",
    )
    parser.add_argument(
        "--model_label",
        default=None,
        help="Label identifying the model/training in the CSV 'model' column "
             "(e.g. 'base', 'ft_inoc_evil', 'ft_steer'). "
             "Default: derived from model path.",
    )
    parser.add_argument(
        "--output_name",
        default=None,
        help="Name for the output CSV file: gradient_metrics_{output_name}.csv. "
             "Default: model_label.",
    )
    parser.add_argument(
        "--train_file",
        default="data/persona_vectors_dataset/evil/misaligned_2.jsonl",
    )
    parser.add_argument(
        "--baseline_system_prompt", default="You are an LLM chatbot."
    )
    parser.add_argument("--max_samples", type=int, default=100)
    parser.add_argument(
        "--directions",
        nargs="+",
        default=None,
        help="Direction vectors as name=path pairs, e.g. evil=steering_"
             "vectors/Qwen_Qwen2.5-7B-Instruct_extract/"
             "evil_response_avg_diff_thresh50.pt",
    )
    # PPS steering options
    parser.add_argument(
        "--steer_vec_path",
        default=("steering_vectors/Qwen_Qwen2.5-7B-Instruct_extract/"
                 "evil_response_avg_diff_thresh50.pt"),
        help="Steering vector for PPS condition (default: the paper's "
             "Qwen2.5-7B evil persona vector)",
    )
    parser.add_argument(
        "--steering_layers", type=int, nargs="+", default=[20]
    )
    parser.add_argument(
        "--steering_coefs", type=float, nargs="+", default=[1.5],
        help="Steering coefficients to sweep. Creates a separate condition "
             "for each coefficient value.",
    )
    parser.add_argument(
        "--skip_pps",
        action="store_true",
        help="Skip PPS steering condition",
    )
    parser.add_argument(
        "--skip_defenseless",
        action="store_true",
        help="Skip the defenseless (no intervention) condition. "
             "Useful when coef=0 is already in --steering_coefs.",
    )
    parser.add_argument(
        "--steer_name",
        default=None,
        help="Name for the steering condition (e.g. 'evil', 'sycophantic'). "
             "Default: derived from --steer_vec_path stem.",
    )
    # Inoculation options
    parser.add_argument(
        "--prompt_files",
        nargs="+",
        default=None,
        help="Prompt files as name=path pairs, "
             "e.g. evil=data/prompt_files/evil.json",
    )
    parser.add_argument("--prompt_mode", default="prepend_user")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--freeze_params",
        action="store_true",
        help="Freeze all model params before collection. Activation "
             "gradients are unchanged (graph is built from inputs_embeds) "
             "but param grads are never allocated, roughly halving peak "
             "VRAM. Required for 14B+ models on a single 46GB GPU.",
    )
    # General options
    parser.add_argument("--layers", type=int, nargs="+", default=None)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument(
        "--output_dir", default="outputs/gradient_metrics"
    )
    parser.add_argument(
        "--dump_tokenwise", action="store_true",
        help="Also save the raw per-token gradient cosines (all layers/"
             "conditions/directions) to gradient_metrics_{name}_tokenwise.npz "
             "for distribution analysis (signs, magnitudes, tails).",
    )
    args = parser.parse_args()

    device = f"cuda:{args.gpu}"
    dtype = getattr(torch, args.dtype)
    rng = random.Random(args.seed)

    # Parse direction vectors
    if args.directions is None:
        # The paper's Qwen2.5-7B persona vectors and norm-matched random
        # control (the pipeline always passes --directions explicitly).
        _pv = "steering_vectors/Qwen_Qwen2.5-7B-Instruct_extract/"
        _default_dirs = {
            "evil": _pv + "evil_response_avg_diff_thresh50.pt",
            "sycophantic": _pv + "sycophantic_response_avg_diff_thresh50.pt",
            "optimistic": _pv + "optimistic_response_avg_diff_thresh50.pt",
            "random": "steering_vectors/random_normmatched_evil_qwen25_7b.pt",
        }
        direction_paths = {
            name: path for name, path in _default_dirs.items()
            if Path(path).exists()
        }
        skipped = set(_default_dirs) - set(direction_paths)
        if skipped:
            print(f"  Skipping missing default directions: {skipped}")
    else:
        direction_paths = {}
        for d in args.directions:
            name, path = d.split("=", 1)
            direction_paths[name] = path

    # Parse prompt files for inoculation conditions
    prompt_pools = {}
    if args.prompt_files is not None:
        for pf in args.prompt_files:
            name, path = pf.split("=", 1)
            prompt_pools[name] = load_prompt_file(path)
            print(
                f"  Loaded prompt pool '{name}' from {path} "
                f"({len(prompt_pools[name])} prompts)"
            )
    inoc_names = sorted(prompt_pools.keys())

    # Load model
    print(f"Loading model: {args.model}")
    collector = ActivationCollector(
        args.model, device=device, dtype=dtype, trust_remote_code=True,
    )
    if args.freeze_params:
        for p in collector.model.parameters():
            p.requires_grad_(False)
        print("  Froze all model params (activation grads unaffected; "
              "verified bit-identical vs unfrozen)")

    layers = args.layers
    if layers is None:
        layers = list(range(1, collector.num_layers + 1))

    # Load direction vectors
    print("Loading direction vectors:")
    directions = load_direction_vectors(direction_paths, layers, device)
    dir_names = sorted(directions.keys())

    # Set up steering for PPS condition(s)
    steering_manager = None
    steering_module = None
    if not args.skip_pps:
        print(f"Loading steering vector: {args.steer_vec_path}")
        steering_module = SteeringModule.from_pretrained(
            args.steer_vec_path,
            global_scale=args.steering_coefs[0],
            device=device,
        )
        steering_manager = SteeringHookManager(
            collector.model,
            steering_module,
            active_layers=args.steering_layers,
        )
        print(
            f"Steering layers: {args.steering_layers}, "
            f"coefs: {args.steering_coefs}"
        )

    # Load data
    print(f"Loading data: {args.train_file}")
    chats = load_chat_data(args.train_file, max_samples=args.max_samples)
    n_samples = len(chats)
    print(f"Samples: {n_samples}, Layers: {len(layers)}")

    # Model label
    if args.model_label:
        model_label = args.model_label
    else:
        model_label = Path(args.model).name

    output_name = args.output_name if args.output_name else model_label

    # Derive steering name
    steer_name = args.steer_name
    if steer_name is None:
        steer_name = Path(args.steer_vec_path).stem  # e.g. "evil"

    # Build conditions list + coef mapping
    conditions = []
    cond_coefs = {}
    if not args.skip_defenseless:
        conditions.append("defenseless")
        cond_coefs["defenseless"] = 0.0
    if not args.skip_pps:
        for coef in args.steering_coefs:
            cond = f"steered_{steer_name}_coef{coef}"
            conditions.append(cond)
            cond_coefs[cond] = coef
    for iname in inoc_names:
        cond = f"prompted_{iname}"
        conditions.append(cond)
        cond_coefs[cond] = 0.0

    print(f"Conditions: {conditions}")

    # ── Accumulators ──────────────────────────────────────────
    # Gradient cosine: layer -> cond -> dir_name -> list
    pooled_grad_cos = {
        l: {c: {d: [] for d in dir_names} for c in conditions}
        for l in layers
    }
    token_grad_cos = {
        l: {c: {d: [] for d in dir_names} for c in conditions}
        for l in layers
    }

    # Activation projection: layer -> cond -> dir_name -> list
    pooled_act_proj = {
        l: {c: {d: [] for d in dir_names} for c in conditions}
        for l in layers
    }
    token_act_proj = {
        l: {c: {d: [] for d in dir_names} for c in conditions}
        for l in layers
    }

    # Loss: cond -> list of per-sample mean losses (pooled)
    #        cond -> list of all per-token losses (tokenwise)
    pooled_loss = {c: [] for c in conditions}
    token_loss = {c: [] for c in conditions}

    total_tokens = 0

    # ── Main loop ─────────────────────────────────────────────
    for i in range(n_samples):
        if (i + 1) % 20 == 0 or i == 0:
            print(f"  Sample {i+1}/{n_samples}")

        messages_base = inject_prompt_into_chat(
            chats[i], prompt=None, mode=args.prompt_mode,
            baseline_system_prompt=args.baseline_system_prompt,
        )

        # Collect results for each condition
        cond_results = {}
        base_token_ids = None

        # Condition: defenseless (baseline)
        if not args.skip_defenseless:
            result = collector.collect_chat(
                messages_base, collect_gradients=True,
            )
            ref_seq_len = next(iter(result.gradients.values())).shape[0]
            total_tokens += ref_seq_len
            base_token_ids = result.tokens
            cond_results["defenseless"] = result

        # Steered conditions (one per coefficient)
        ref_seq_len_set = "defenseless" in cond_results
        if steering_manager is not None:
            for coef in args.steering_coefs:
                steering_module.set_global_scale(coef)
                steering_manager.register_hooks()
                result = collector.collect_chat(
                    messages_base, collect_gradients=True,
                )
                steering_manager.remove_hooks()
                cond_results[f"steered_{steer_name}_coef{coef}"] = result
                if not ref_seq_len_set:
                    ref_seq_len = next(
                        iter(result.gradients.values())
                    ).shape[0]
                    total_tokens += ref_seq_len
                    base_token_ids = result.tokens
                    ref_seq_len_set = True

        # Prompted conditions need the BASE tokenization to excise the
        # injected prompt tokens. When both the defenseless and steered
        # conditions are skipped (--skip_defenseless --skip_pps), neither
        # branch above set base_token_ids — run a forward-only base pass so
        # the tokenization is guaranteed identical to collect_chat's.
        # (Review item P2: NameError in prompted-only configs.)
        if inoc_names and base_token_ids is None:
            result = collector.collect_chat(
                messages_base, collect_gradients=False,
            )
            base_token_ids = result.tokens
            ref_seq_len = len(base_token_ids)
            total_tokens += ref_seq_len

        # Prompted conditions
        for iname in inoc_names:
            prompt = rng.choice(prompt_pools[iname])
            messages_inoc = inject_prompt_into_chat(
                chats[i], prompt=prompt, mode=args.prompt_mode,
                baseline_system_prompt=args.baseline_system_prompt,
            )
            result = collector.collect_chat(
                messages_inoc, collect_gradients=True,
            )
            cond_results[f"prompted_{iname}"] = result

        # ── Compute non-prompt token indices per condition ────
        # For prompted conditions the injected prompt tokens appear
        # in the *middle* of the sequence (prepend_user mode inserts
        # them inside the user message).  We align the prompted
        # token IDs with the base token IDs to find which positions
        # to keep for metrics (excluding the injected prompt tokens).
        #
        # We need TWO sets of indices:
        #   keep_idx      – original token positions (for activations
        #                    & gradients, which live AT position i)
        #   keep_idx_loss – positions where loss[i] predicts an original
        #                    token (loss[i] predicts token[i+1], so we
        #                    must also exclude the position just before
        #                    the prompt block)
        cond_keep_idx = {}
        cond_keep_idx_loss = {}
        for cond in conditions:
            result = cond_results[cond]
            if cond.startswith("prompted_"):
                prompted_ids = result.tokens
                n_extra = len(prompted_ids) - len(base_token_ids)
                if n_extra > 0:
                    # Find prefix match (tokens before the injection)
                    prefix_len = 0
                    for b, p in zip(base_token_ids, prompted_ids):
                        if b == p:
                            prefix_len += 1
                        else:
                            break
                    # Activations/gradients: keep original token positions
                    keep = (
                        list(range(prefix_len))
                        + list(range(prefix_len + n_extra,
                                     len(prompted_ids)))
                    )
                    cond_keep_idx[cond] = torch.tensor(
                        keep, dtype=torch.long
                    )
                    # Loss: loss[i] predicts token[i+1], so exclude
                    # prefix_len-1 (it predicts the first prompt token)
                    keep_loss = (
                        list(range(max(prefix_len - 1, 0)))
                        + list(range(prefix_len + n_extra,
                                     len(prompted_ids)))
                    )
                    cond_keep_idx_loss[cond] = torch.tensor(
                        keep_loss, dtype=torch.long
                    )
                else:
                    cond_keep_idx[cond] = None
                    cond_keep_idx_loss[cond] = None
            else:
                cond_keep_idx[cond] = None
                cond_keep_idx_loss[cond] = None

        # ── Compute metrics for all conditions ────────────────
        for cond in conditions:
            result = cond_results[cond]
            keep_idx = cond_keep_idx[cond]
            keep_idx_loss = cond_keep_idx_loss[cond]

            # Helpers: select non-prompt tokens
            def select(tensor, idx=keep_idx):
                if idx is not None:
                    return tensor[idx]
                return tensor

            # --- Loss ---
            if result.loss is not None:
                loss_vals = select(result.loss, keep_idx_loss).float()
                # Filter out NaN (last token has no next-token target)
                valid = loss_vals[~torch.isnan(loss_vals)]
                if len(valid) > 0:
                    pooled_loss[cond].append(valid.mean().item())
                    token_loss[cond].extend(valid.cpu().tolist())

            # --- Per-layer: gradient cosine + activation projection ---
            for l in layers:
                grad = result.gradients[l]
                grad_tokens = select(grad).to(device).float()
                pooled_grad = grad_tokens.mean(dim=0, keepdim=True)

                act = result.activations[l]
                act_tokens = select(act).to(device).float()
                pooled_act = act_tokens.mean(dim=0, keepdim=True)

                for d_name in dir_names:
                    v = directions[d_name][l]

                    # Gradient cosine (tokenwise + pooled)
                    tok_cs = cosine_sim(grad_tokens, v)
                    token_grad_cos[l][cond][d_name].extend(
                        tok_cs.cpu().tolist()
                    )
                    p_cs = cosine_sim(pooled_grad, v).item()
                    pooled_grad_cos[l][cond][d_name].append(p_cs)

                    # Activation projection (tokenwise + pooled)
                    tok_proj = projection(act_tokens, v)
                    token_act_proj[l][cond][d_name].extend(
                        tok_proj.cpu().tolist()
                    )
                    p_proj = projection(pooled_act, v).item()
                    pooled_act_proj[l][cond][d_name].append(p_proj)

    # ── Build CSV rows ────────────────────────────────────────
    # Pre-compute loss stats per condition (same across layers)
    loss_stats = {}
    for cond in conditions:
        p = pooled_loss[cond]
        t = token_loss[cond]
        p_std = np.std(p) if p else float("nan")
        t_std = np.std(t) if t else float("nan")
        loss_stats[cond] = {
            "loss_pooled_mean": np.mean(p) if p else float("nan"),
            "loss_pooled_std": p_std,
            "loss_pooled_sem": p_std / np.sqrt(len(p)) if p else float("nan"),
            "loss_tokenwise_mean": np.mean(t) if t else float("nan"),
            "loss_tokenwise_std": t_std,
            "loss_tokenwise_sem": t_std / np.sqrt(len(t)) if t else float("nan"),
        }

    rows = []
    for l in layers:
        for cond in conditions:
            row = {
                "model": model_label,
                "layer": l,
                "condition": cond,
                "coef": cond_coefs[cond],
                "n_samples": n_samples,
                "n_tokens": total_tokens,
            }
            # Loss (constant across layers)
            row.update(loss_stats[cond])

            # Gradient cosine + activation projection per direction
            for d_name in dir_names:
                gp = pooled_grad_cos[l][cond][d_name]
                gt = token_grad_cos[l][cond][d_name]
                row[f"{d_name}_grad_pooled_mean"] = np.mean(gp)
                row[f"{d_name}_grad_pooled_std"] = np.std(gp)
                row[f"{d_name}_grad_tokenwise_mean"] = np.mean(gt)
                row[f"{d_name}_grad_tokenwise_std"] = np.std(gt)

                ap = pooled_act_proj[l][cond][d_name]
                at = token_act_proj[l][cond][d_name]
                row[f"{d_name}_act_pooled_mean"] = np.mean(ap)
                row[f"{d_name}_act_pooled_std"] = np.std(ap)
                row[f"{d_name}_act_tokenwise_mean"] = np.mean(at)
                row[f"{d_name}_act_tokenwise_std"] = np.std(at)

            rows.append(row)

    # ── Print summary ─────────────────────────────────────────
    print(f"\nTotal tokens: {total_tokens}")

    # Loss summary
    print(f"\n=== LOSS ===")
    for cond in conditions:
        s = loss_stats[cond]
        print(
            f"  {cond:20s}  "
            f"pooled={s['loss_pooled_mean']:.4f} ± {s['loss_pooled_std']:.4f}  "
            f"tokenwise={s['loss_tokenwise_mean']:.4f} ± {s['loss_tokenwise_std']:.4f}"
        )

    # Gradient cosine summary (pooled)
    cond_abbrevs = {
        c: c.replace("inoc_", "i_")[:8] for c in conditions
    }
    print(f"\n=== GRADIENT COSINE (pooled, {n_samples} samples) ===")
    header = f"{'Layer':>5} |"
    for d in dir_names:
        for c in conditions:
            header += f" {d[:4]}_{cond_abbrevs[c]:>8} |"
    print(header)
    print("-" * len(header))
    for l in layers:
        line = f"{l:>5} |"
        for d in dir_names:
            for c in conditions:
                val = np.mean(pooled_grad_cos[l][c][d])
                line += f" {val:>13.4f} |"
        print(line)

    # Activation projection summary (pooled)
    print(f"\n=== ACTIVATION PROJECTION (pooled, {n_samples} samples) ===")
    print(header)
    print("-" * len(header))
    for l in layers:
        line = f"{l:>5} |"
        for d in dir_names:
            for c in conditions:
                val = np.mean(pooled_act_proj[l][c][d])
                line += f" {val:>13.4f} |"
        print(line)

    # ── Save CSVs ─────────────────────────────────────────────
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fieldnames = rows[0].keys()

    # Combined CSV (all conditions)
    csv_path = out_dir / f"gradient_metrics_{output_name}.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved combined CSV to {csv_path}")

    # Per-condition CSVs
    for cond in conditions:
        cond_rows = [r for r in rows if r["condition"] == cond]
        cond_path = out_dir / f"gradient_metrics_{output_name}_{cond}.csv"
        with open(cond_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(cond_rows)
        print(f"Saved {cond} CSV to {cond_path}")

    # Raw tokenwise distributions (optional): the CSVs only keep summary
    # stats, which hide the per-token sign/magnitude structure of the
    # gradient cosines (e.g. what fraction of tokens PPS actually flips).
    if args.dump_tokenwise:
        arrays = {}
        for l in layers:
            for cond in conditions:
                for d_name in dir_names:
                    key = f"grad_cos|{cond}|layer{l}|{d_name}"
                    arrays[key] = np.asarray(
                        token_grad_cos[l][cond][d_name], dtype=np.float32
                    )
        npz_path = out_dir / f"gradient_metrics_{output_name}_tokenwise.npz"
        np.savez_compressed(npz_path, **arrays)
        print(f"Saved tokenwise gradient cosines to {npz_path}")

    # Save JSON config
    prompt_file_paths = {
        name: path
        for pf in (args.prompt_files or [])
        for name, path in [pf.split("=", 1)]
    }
    with open(out_dir / f"config_{output_name}.json", "w") as f:
        json.dump(
            {
                "model": args.model,
                "model_label": model_label,
                "output_name": output_name,
                "train_file": args.train_file,
                "n_samples": n_samples,
                "n_tokens": total_tokens,
                "directions": direction_paths,
                "conditions": conditions,
                "pps_steering": {
                    "steer_vec_path": args.steer_vec_path,
                    "steering_layers": args.steering_layers,
                    "steering_coefs": args.steering_coefs,
                } if not args.skip_pps else None,
                "inoculation": {
                    "prompt_files": prompt_file_paths,
                    "prompt_mode": args.prompt_mode,
                    "seed": args.seed,
                } if prompt_file_paths else None,
                "layers": layers,
            },
            f,
            indent=2,
        )


if __name__ == "__main__":
    main()
