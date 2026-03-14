#!/usr/bin/env python3
"""
DynMoE Expert Activation Pattern Analyzer
==========================================
Loads a pretrained DynMoE model from HuggingFace, runs forward passes on
sample sentences, and records/visualizes how many experts each token
activates in every MoE layer (the dynamic "k" value).

Setup (run from the DynMoE repo root):
    pip install -e DeepSpeed-0.9.5/
    pip install -e MoE-LLaVA/
    pip install matplotlib numpy

Usage:
    python analyze_expert_activations.py

    # Or with a different model:
    python analyze_expert_activations.py --model LINs-lab/DynMoE-Qwen-1.8B

    # Use GPU (default if available):
    python analyze_expert_activations.py --device cuda

    # Force CPU:
    python analyze_expert_activations.py --device cpu
"""

import os
import sys
import argparse

# ---------------------------------------------------------------------------
# 0. Single-process distributed init (required by DeepSpeed MoE internals)
#    Must happen BEFORE importing deepspeed or model code.
# ---------------------------------------------------------------------------
os.environ.setdefault("MASTER_ADDR", "localhost")
os.environ.setdefault("MASTER_PORT", "29500")
os.environ.setdefault("RANK", "0")
os.environ.setdefault("WORLD_SIZE", "1")
os.environ.setdefault("LOCAL_RANK", "0")

import torch
import torch.distributed as dist

# DeepSpeed uses its own comm layer (deepspeed.comm) internally.
# We must initialize via deepspeed.comm, not torch.distributed directly,
# otherwise deepspeed.utils.groups.dist.is_initialized() stays False.
import deepspeed.comm as ds_dist
if not ds_dist.is_initialized():
    ds_dist.init_distributed(dist_backend="gloo", auto_mpi_discovery=False)

# Sanity check
if not ds_dist.is_initialized():
    raise RuntimeError(
        "deepspeed.comm failed to initialize. "
        "Check MASTER_ADDR/MASTER_PORT env vars."
    )

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import defaultdict

from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

# Add MoE-LLaVA to path so the model classes get registered with AutoConfig.
# The moellava package __init__ eagerly imports ALL model variants; the tiktoken
# stub above keeps the Qwen branch from crashing.
sys.path.insert(0, os.path.join(SCRIPT_DIR, "MoE-LLaVA"))
from moellava.model.language_model.llava_stablelm_moe import (  # noqa: F401
    EvalMoELLaVAStablelmForCausalLM,
    MoELLaVAStablelmConfig,
)

from deepspeed.moe.sharded_moe import GAMoEGateT
from deepspeed.moe.layer import MoE

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SAMPLE_SENTENCES = [
    "The quick brown fox jumps over the lazy dog.",
    "Artificial intelligence is transforming the way we live and work.",
    "In quantum computing, qubits can exist in superposition states.",
    "The mitochondria is the powerhouse of the cell.",
    "Once upon a time, in a land far far away, there lived a dragon.",
]


# ===========================================================================
# 1. Load Model & Tokenizer
# ===========================================================================
def load_model_and_tokenizer(model_name: str, device: str, dtype: torch.dtype):
    print(f"\n{'='*60}")
    print(f"Loading model : {model_name}")
    print(f"Device        : {device}")
    print(f"Dtype         : {dtype}")
    print(f"{'='*60}")

    # We rely on the locally-registered model classes (imported above) rather
    # than trust_remote_code, because the HF repo's remote code has broken
    # cross-repo imports to stabilityai/stablelm-2-1_6b.
    config = AutoConfig.from_pretrained(model_name)

    # The tokenizer is a custom Arcade100k class; load it from local code.
    from moellava.model.language_model.stablelm.tokenization_arcade100k import (
        Arcade100kTokenizer,
    )
    try:
        tokenizer = Arcade100kTokenizer.from_pretrained(model_name)
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        config=config,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    )

    # Set up DeepSpeed MoE process groups (needed for all-to-all in forward)
    moe_count = 0
    for module in model.modules():
        if isinstance(module, MoE):
            module.set_deepspeed_parallelism()
            moe_count += 1
    print(f"Initialized {moe_count} MoE layer process groups.")

    model = model.to(device)
    model.eval()

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    return model, tokenizer


# ===========================================================================
# 2. Hook into GAMoEGateT to capture per-token expert activation counts
# ===========================================================================
class ActivationRecorder:
    """
    Registers forward hooks on every GAMoEGateT module.

    GAMoEGateT.forward returns (logits, top_k) where top_k is a 1-D tensor
    of shape [num_tokens] — each entry is the number of experts that token
    chose to activate (the dynamic "k").
    """

    def __init__(self):
        self.records: dict[str, list[torch.Tensor]] = defaultdict(list)
        self._hooks: list[torch.utils.hooks.RemovableHook] = []

    def register(self, model: torch.nn.Module):
        for name, module in model.named_modules():
            if isinstance(module, GAMoEGateT):
                h = module.register_forward_hook(self._make_hook(name))
                self._hooks.append(h)
                print(f"  Hook registered: {name}")
        print(f"  Total hooks: {len(self._hooks)}")

    def _make_hook(self, layer_name: str):
        def hook_fn(module, inp, output):
            _logits, top_k = output
            # top_k is clamped to min=1 at inference — recompute true k before clamp
            with torch.no_grad():
                import torch.nn.functional as F
                x = inp[0].float()
                sim = module.sim_matrix.float()
                gates = module.gates.float()
                logit_scale = torch.clamp(module.temperature, max=module.clamp_max).exp()
                logits = torch.sigmoid(
                    torch.matmul(F.normalize(x, dim=1),
                                 F.normalize(sim, dim=0)) * logit_scale
                ) * module.experts_mask
                gates_sig = torch.sigmoid(gates * logit_scale)
                new_logits = F.relu(logits - gates_sig)
                true_k = torch.sum(new_logits > 0, dim=1).to(torch.int)
            self.records[layer_name].append(true_k.detach().cpu())
        return hook_fn

    def clear(self):
        self.records = defaultdict(list)

    def remove_all(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    def snapshot(self) -> dict[str, torch.Tensor]:
        """Return {layer_name: concatenated top_k} for the current batch."""
        return {
            name: torch.cat(tensors, dim=0)
            for name, tensors in self.records.items()
        }


# ===========================================================================
# 3. Run Forward Passes
# ===========================================================================
def run_analysis(
    model, tokenizer, sentences: list[str], device: str
) -> dict[str, dict[str, torch.Tensor]]:
    """Returns {sentence: {layer_name: top_k_tensor}}."""

    recorder = ActivationRecorder()
    recorder.register(model)

    results: dict[str, dict[str, torch.Tensor]] = {}

    print(f"\nRunning forward passes on {len(sentences)} sentences...")
    for i, sentence in enumerate(sentences):
        recorder.clear()

        inputs = tokenizer(sentence, return_tensors="pt")
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs.get(
            "attention_mask", torch.ones_like(input_ids)
        ).to(device)

        num_tokens = input_ids.shape[1]
        preview = sentence[:55] + ("..." if len(sentence) > 55 else "")
        print(f"  [{i+1}/{len(sentences)}] ({num_tokens} tokens) \"{preview}\"")

        with torch.no_grad():
            model(input_ids=input_ids, attention_mask=attention_mask)

        results[sentence] = recorder.snapshot()

    recorder.remove_all()
    return results


# ===========================================================================
# 4. Aggregate helper
# ===========================================================================
def _friendly_layer_name(full_name: str) -> str:
    """Extract a short label like 'Layer 3' from the full module path."""
    parts = full_name.split(".")
    digits = [p for p in parts if p.isdigit()]
    return f"Layer {digits[0]}" if digits else full_name[-25:]


def aggregate_across_sentences(
    per_sentence: dict[str, dict[str, torch.Tensor]]
) -> dict[str, np.ndarray]:
    """Merge top_k values across all sentences, keyed by layer name."""
    merged: dict[str, list[torch.Tensor]] = defaultdict(list)
    for layer_records in per_sentence.values():
        for layer_name, top_k in layer_records.items():
            merged[layer_name].append(top_k)
    return {
        name: torch.cat(tensors).numpy()
        for name, tensors in sorted(merged.items())
    }


# ===========================================================================
# 5. Visualize & Save
# ===========================================================================
def visualize_and_save(
    per_sentence: dict[str, dict[str, torch.Tensor]], output_dir: str
):
    os.makedirs(output_dir, exist_ok=True)
    aggregated = aggregate_across_sentences(per_sentence)
    layer_names = list(aggregated.keys())
    num_layers = len(layer_names)

    if num_layers == 0:
        print("  WARNING: No MoE activation data recorded. "
              "Are there GAMoEGateT modules in the model?")
        return

    friendly = [_friendly_layer_name(n) for n in layer_names]

    # ---- Figure 1: histogram per MoE layer ----
    cols = min(4, num_layers)
    rows = (num_layers + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
    axes = np.atleast_1d(axes).flatten()

    for i, (name, values) in enumerate(aggregated.items()):
        ax = axes[i]
        max_k = max(int(values.max()), 1)
        bins = np.arange(0, max_k + 2) - 0.5
        ax.hist(values, bins=bins, edgecolor="black", alpha=0.7, color="steelblue")
        ax.set_xlabel("Experts Activated (k)")
        ax.set_ylabel("Token Count")
        ax.set_title(friendly[i])
        ax.set_xticks(range(0, max_k + 1))
        mean_k = values.mean()
        ax.axvline(
            mean_k, color="red", linestyle="--", linewidth=1.5,
            label=f"Mean = {mean_k:.2f}",
        )
        ax.legend(fontsize=8)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(
        "DynMoE Expert Activation Distribution per Layer",
        fontsize=14, fontweight="bold",
    )
    plt.tight_layout()
    path = os.path.join(output_dir, "activation_histograms.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")

    # ---- Figure 2: box-plot summary across layers ----
    fig, ax = plt.subplots(figsize=(max(8, num_layers * 0.8), 5))
    bp = ax.boxplot(
        [aggregated[n] for n in layer_names],
        labels=friendly, patch_artist=True,
    )
    for patch in bp["boxes"]:
        patch.set_facecolor("lightsteelblue")
    ax.set_xlabel("MoE Layer")
    ax.set_ylabel("Experts Activated per Token")
    ax.set_title("Expert Activation Distribution Across Layers")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    path = os.path.join(output_dir, "activation_boxplot.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")

    # ---- Figure 3: token-level heatmap for the first sentence ----
    first_sentence = next(iter(per_sentence))
    first_records = per_sentence[first_sentence]
    if first_records:
        sorted_names = sorted(first_records.keys())
        matrix = np.array([first_records[n].numpy() for n in sorted_names])

        fig, ax = plt.subplots(
            figsize=(max(10, matrix.shape[1] * 0.4),
                     max(4, len(sorted_names) * 0.5))
        )
        im = ax.imshow(matrix, aspect="auto", cmap="YlOrRd", interpolation="nearest")
        ax.set_xlabel("Token Position")
        ax.set_ylabel("MoE Layer")
        ax.set_yticks(range(len(sorted_names)))
        ax.set_yticklabels([_friendly_layer_name(n) for n in sorted_names])
        plt.colorbar(im, ax=ax, label="Experts Activated")
        ax.set_title(
            f'Token-Level Expert Activation\n"{first_sentence[:60]}'
            + ('..."' if len(first_sentence) > 60 else '"')
        )
        plt.tight_layout()
        path = os.path.join(output_dir, "activation_heatmap.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {path}")

    # ---- Save raw data as CSV ----
    import csv

    csv_path = os.path.join(output_dir, "activation_data.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["sentence", "layer", "token_position", "experts_activated"])
        for sentence, layer_records in per_sentence.items():
            for layer_name in sorted(layer_records.keys()):
                for pos, k in enumerate(layer_records[layer_name].numpy()):
                    writer.writerow([sentence, layer_name, pos, int(k)])
    print(f"  Saved: {csv_path}")

    # ---- Print summary statistics ----
    print(f"\n{'='*60}")
    print("SUMMARY STATISTICS")
    print(f"{'='*60}")
    print(f"  {'Layer':<14s} {'Mean':>6s} {'Std':>6s} {'Min':>4s} "
          f"{'Max':>4s} {'Median':>7s}")
    print(f"  {'-'*14} {'-'*6} {'-'*6} {'-'*4} {'-'*4} {'-'*7}")
    for name in layer_names:
        v = aggregated[name]
        print(
            f"  {_friendly_layer_name(name):<14s} {v.mean():>6.2f} "
            f"{v.std():>6.2f} {int(v.min()):>4d} {int(v.max()):>4d} "
            f"{np.median(v):>7.1f}"
        )


# ===========================================================================
# Main
# ===========================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Analyze expert activation patterns in a pretrained DynMoE model."
    )
    parser.add_argument(
        "--model", default="LINs-lab/DynMoE-StableLM-1.6B",
        help="HuggingFace model ID (default: LINs-lab/DynMoE-StableLM-1.6B)",
    )
    parser.add_argument(
        "--device", default=None,
        help="Device to run on: 'cuda' or 'cpu' (default: auto-detect)",
    )
    parser.add_argument(
        "--output-dir", default=os.path.join(SCRIPT_DIR, "activation_analysis"),
        help="Directory for output plots and CSV",
    )
    parser.add_argument(
        "--sentences", nargs="+", default=None,
        help="Custom sentences to analyze (default: built-in examples)",
    )
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device == "cuda" else torch.float32
    sentences = args.sentences or SAMPLE_SENTENCES

    model, tokenizer = load_model_and_tokenizer(args.model, device, dtype)
    results = run_analysis(model, tokenizer, sentences, device)

    print("\nGenerating visualizations...")
    visualize_and_save(results, args.output_dir)
    print("\nDone! Check the output directory:", args.output_dir)


if __name__ == "__main__":
    main()
