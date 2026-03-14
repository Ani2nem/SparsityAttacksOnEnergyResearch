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

sentence = "The quick brown fox jumps over the lazy dog."
inputs = tokenizer(sentence, return_tensors="pt")

results = {}
def hook_fn(module, inp, output):
    with torch.no_grad():
        x = inp[0].float()
        sim = module.sim_matrix.float()
        g = module.gates.float()
        logit_scale = torch.clamp(module.temperature, max=module.clamp_max).exp()

        print(f"x shape: {x.shape}")
        print(f"x has nan: {torch.isnan(x).any().item()}")
        print(f"x norm per token (first 3): {x.norm(dim=1)[:3]}")
        print(f"sim_matrix has nan: {torch.isnan(sim).any().item()}")
        print(f"sim_matrix norm per col (first 4): {sim.norm(dim=0)[:4]}")
        print(f"gates raw values: {g[:4]}")
        print(f"logit_scale: {logit_scale.item()}")

        x_norm = F.normalize(x, dim=1)
        sim_norm = F.normalize(sim, dim=0)
        print(f"x_norm has nan: {torch.isnan(x_norm).any().item()}")
        print(f"sim_norm has nan: {torch.isnan(sim_norm).any().item()}")

        matmul = torch.matmul(x_norm, sim_norm)
        print(f"matmul has nan: {torch.isnan(matmul).any().item()}")
        print(f"matmul * scale has nan: {torch.isnan(matmul * logit_scale).any().item()}")

h = gate.register_forward_hook(hook_fn)
with torch.no_grad():
    model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])
h.remove()
