import os, sys, torch
import torch.nn.functional as F

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
from transformers import AutoConfig, AutoModelForCausalLM
from moellava.model.language_model.stablelm.tokenization_arcade100k import Arcade100kTokenizer
from deepspeed.moe.layer import MoE
from deepspeed.moe.sharded_moe import GAMoEGateT

model_name = "LINs-lab/DynMoE-StableLM-1.6B"
config = AutoConfig.from_pretrained(model_name)
tokenizer = Arcade100kTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, config=config, torch_dtype=torch.float32, low_cpu_mem_usage=True)

for module in model.modules():
    if isinstance(module, MoE):
        module.set_deepspeed_parallelism()

model.eval()

gate = None
for name, module in model.named_modules():
    if isinstance(module, GAMoEGateT):
        gate = module
        gate_name = name
        break

print(f"\nInspecting gate: {gate_name}")

sentence = "The quick brown fox jumps over the lazy dog."
inputs = tokenizer(sentence, return_tensors="pt")

results = {}
def hook_fn(module, inp, output):
    with torch.no_grad():
        x = inp[0].float()
        sim = module.sim_matrix.float()
        g = module.gates.float()
        logit_scale = torch.clamp(module.temperature, max=module.clamp_max).exp()
        logits = torch.sigmoid(
            torch.matmul(F.normalize(x, dim=1), F.normalize(sim, dim=0)) * logit_scale
        ) * module.experts_mask
        gates_sig = torch.sigmoid(g * logit_scale)
        new_logits = F.relu(logits - gates_sig)
        results['logits'] = logits.detach()
        results['gates_sig'] = gates_sig.detach()
        results['new_logits'] = new_logits.detach()
        results['logit_scale'] = logit_scale.item()

h = gate.register_forward_hook(hook_fn)
with torch.no_grad():
    model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])
h.remove()

print(f"\nlogit_scale (temperature): {results['logit_scale']:.4f}")
print(f"\nGate thresholds (gates_sig) - one per expert:")
print(results['gates_sig'])
print(f"\nRaw logits - first 3 tokens x all experts:")
print(results['logits'][:3])
print(f"\nAfter ReLU(logits - gates_sig) - first 3 tokens:")
print(results['new_logits'][:3])
print(f"\nMax logit across all tokens:   {results['logits'].max().item():.6f}")
print(f"Min gate threshold:            {results['gates_sig'].min().item():.6f}")
print(f"Gap (max_logit - min_gate):    {(results['logits'].max() - results['gates_sig'].min()).item():.6f}")
