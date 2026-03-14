import os, sys, torch
import torch.nn.functional as F
from PIL import Image
import requests
from io import BytesIO

os.environ.setdefault("MASTER_ADDR", "localhost")
os.environ.setdefault("MASTER_PORT", "29500")
os.environ.setdefault("RANK", "0")
os.environ.setdefault("WORLD_SIZE", "1")
os.environ.setdefault("LOCAL_RANK", "0")

import deepspeed.comm as ds_dist
if not ds_dist.is_initialized():
    ds_dist.init_distributed(dist_backend="gloo", auto_mpi_discovery=False)

sys.path.insert(0, "MoE-LLaVA")
from moellava.model.language_model.llava_stablelm_moe import EvalMoELLaVAStablelmForCausalLM, MoELLaVAStablelmConfig
from transformers import AutoConfig, AutoModelForCausalLM, CLIPImageProcessor
from moellava.model.language_model.stablelm.tokenization_arcade100k import Arcade100kTokenizer
from deepspeed.moe.layer import MoE
from deepspeed.moe.sharded_moe import GAMoEGateT
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import csv

model_name = "LINs-lab/DynMoE-StableLM-1.6B"
os.environ["HF_HOME"] = os.path.join(os.getcwd(), ".hf_cache")

print("Loading model...")
config = AutoConfig.from_pretrained(model_name)
tokenizer = Arcade100kTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name, config=config, torch_dtype=torch.float32, low_cpu_mem_usage=True
)
for module in model.modules():
    if isinstance(module, MoE):
        module.set_deepspeed_parallelism()
model.eval()
print("Model loaded.")

# Load image processor from the vision tower
image_processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14-336")

# Download a few sample images
IMAGE_PROMPTS = [
    ("https://upload.wikimedia.org/wikipedia/commons/thumb/3/3a/Cat03.jpg/320px-Cat03.jpg",
     "What is in this image?"),
    ("https://upload.wikimedia.org/wikipedia/commons/thumb/4/47/PNG_transparency_demonstration_1.png/240px-PNG_transparency_demonstration_1.png",
     "Describe what you see."),
    ("https://upload.wikimedia.org/wikipedia/commons/thumb/a/a7/Camponotus_flavomarginatus_ant.jpg/320px-Camponotus_flavomarginatus_ant.jpg",
     "What animal is this?"),
]

def load_image(url):
    response = requests.get(url, timeout=10)
    img = Image.open(BytesIO(response.content)).convert("RGB")
    return img

# Build image+text input the way LLaVA expects it
DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_IMAGE_PATCH_TOKEN = "<im_patch>"
DEFAULT_IM_START_TOKEN = "<im_start>"
DEFAULT_IM_END_TOKEN = "<im_end>"

def build_input(prompt, image_tensor, tokenizer, model, config):
    # Check if model uses im_start/end tokens
    if hasattr(config, "use_im_start_end") and config.use_im_start_end:
        image_token_str = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_PATCH_TOKEN * getattr(config, "image_token_len", 256) + DEFAULT_IM_END_TOKEN
    else:
        image_token_str = DEFAULT_IMAGE_TOKEN

    full_prompt = image_token_str + "\n" + prompt
    input_ids = tokenizer(full_prompt, return_tensors="pt").input_ids
    return input_ids

# Register hooks
records = defaultdict(list)
hooks = []

for name, module in model.named_modules():
    if isinstance(module, GAMoEGateT):
        def make_hook(n):
            def hook_fn(mod, inp, output):
                _logits, top_k = output
                records[n].append(top_k.detach().cpu())
            return hook_fn
        hooks.append(module.register_forward_hook(make_hook(name)))

print(f"Registered {len(hooks)} hooks on GAMoEGateT modules.")

all_results = {}

for url, prompt in IMAGE_PROMPTS:
    print(f"\nProcessing: {prompt}")
    try:
        img = load_image(url)
        pixel_values = image_processor(images=img, return_tensors="pt").pixel_values  # [1, 3, 336, 336]

        # Reset records
        for k in records:
            records[k].clear()

        input_ids = tokenizer(prompt, return_tensors="pt").input_ids

        with torch.no_grad():
            model(
                input_ids=input_ids,
                images=pixel_values,
            )

        snapshot = {
            name: torch.cat(tensors).numpy()
            for name, tensors in records.items()
            if tensors
        }
        all_results[prompt] = snapshot

        if snapshot:
            for layer, vals in sorted(snapshot.items()):
                print(f"  {layer[-40:]:40s} | mean k={vals.mean():.2f} min={vals.min()} max={vals.max()}")
        else:
            print("  No activations recorded.")

    except Exception as e:
        print(f"  Error: {e}")

for h in hooks:
    h.remove()

# Save summary
if all_results:
    output_dir = "activation_analysis_image"
    os.makedirs(output_dir, exist_ok=True)

    csv_path = os.path.join(output_dir, "image_text_activations.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["prompt", "layer", "token_position", "experts_activated"])
        for prompt, layer_records in all_results.items():
            for layer_name in sorted(layer_records.keys()):
                for pos, k in enumerate(layer_records[layer_name]):
                    writer.writerow([prompt, layer_name, pos, int(k)])
    print(f"\nSaved CSV: {csv_path}")

    # Plot
    fig, axes = plt.subplots(1, len(all_results), figsize=(6 * len(all_results), 4))
    if len(all_results) == 1:
        axes = [axes]
    for ax, (prompt, layer_records) in zip(axes, all_results.items()):
        if not layer_records:
            continue
        means = [layer_records[n].mean() for n in sorted(layer_records.keys())]
        labels = [f"L{n.split('.')[2]}" for n in sorted(layer_records.keys())]
        ax.bar(labels, means, color="steelblue")
        ax.set_title(prompt[:40], fontsize=8)
        ax.set_xlabel("Layer")
        ax.set_ylabel("Mean k")
        ax.set_ylim(0, 4)
    plt.tight_layout()
    plot_path = os.path.join(output_dir, "image_text_mean_k.png")
    fig.savefig(plot_path, dpi=150)
    print(f"Saved plot: {plot_path}")

print("\nDone.")
