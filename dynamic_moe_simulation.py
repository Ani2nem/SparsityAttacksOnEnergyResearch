# Dynamic-k MoE simulation for research progress update
# Run in Google Colab (no special deps needed)

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

# ----------------------------
# Simple MoE layer with dynamic-k
# ----------------------------
class DynamicMoELayer(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_experts, expert_output_dim):
        super().__init__()
        self.num_experts = num_experts
        # Router: 2-layer MLP (like in D2DMoE)
        self.router = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_experts)
        )
        # Experts: each is a simple linear layer (for demo)
        self.experts = nn.ModuleList([
            nn.Linear(input_dim, expert_output_dim) for _ in range(num_experts)
        ])

    def forward(self, x, tau=0.5):
        # x shape: (batch, seq_len, input_dim)
        router_logits = self.router(x)  # (batch, seq_len, num_experts)
        router_probs = F.softmax(router_logits, dim=-1)

        # Dynamic-k: select experts with probability > tau * max_prob
        max_probs = router_probs.max(dim=-1, keepdim=True)[0]
        mask = router_probs >= (tau * max_probs)  # boolean mask
        k_per_token = mask.sum(dim=-1)           # number of selected experts per token

        # Compute weighted sum of expert outputs (only selected experts contribute)
        output = torch.zeros(x.size(0), x.size(1), expert_output_dim).to(x.device)
        for i in range(self.num_experts):
            # Only add contribution if expert i is selected for this token
            expert_output = self.experts[i](x)  # (batch, seq_len, expert_output_dim)
            # Mask: (batch, seq_len, 1) expanded
            mask_i = mask[..., i:i+1].float()
            output += mask_i * expert_output * router_probs[..., i:i+1]

        return output, k_per_token

# ----------------------------
# Simulate a forward pass
# ----------------------------
batch_size = 4
seq_len = 10
input_dim = 32
hidden_dim = 64
num_experts = 16
expert_output_dim = 32

model = DynamicMoELayer(input_dim, hidden_dim, num_experts, expert_output_dim)
x = torch.randn(batch_size, seq_len, input_dim)

# Run with different tau thresholds
taus = [0.1, 0.3, 0.5, 0.7, 0.9]
fig, axes = plt.subplots(1, len(taus), figsize=(15, 3))

for idx, tau in enumerate(taus):
    _, k = model(x, tau=tau)
    # Flatten over batch and sequence
    k_flat = k.view(-1).cpu().numpy()
    axes[idx].hist(k_flat, bins=np.arange(0, num_experts+1)-0.5, edgecolor='black', alpha=0.7)
    axes[idx].set_title(f'tau = {tau}')
    axes[idx].set_xlabel('Experts activated per token')
    axes[idx].set_ylabel('Frequency')
    axes[idx].set_xticks(range(0, num_experts+1, 2))

plt.tight_layout()
plt.suptitle('Dynamic-k: Number of experts activated depends on threshold tau', y=1.05)
plt.show()

# Also show per-token variation for a single tau
_, k = model(x, tau=0.5)
print("Experts activated per token (tau=0.5):")
print(k.cpu().numpy())