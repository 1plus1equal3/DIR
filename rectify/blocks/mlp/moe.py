import torch
import torch.nn as nn
import torch.nn.functional as F

class Expert(nn.Module):
    """ Simple feedforward expert module """
    def __init__(self, input_dim, hidden_dim):
        super(Expert, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, input_dim)
        self.act = nn.GELU()

    def forward(self, x):
        x = self.act(self.fc1(x))
        x = self.fc2(x)
        return x
    
class GatingNetwork(nn.Module):
    """ Gating network to compute expert weights """
    def __init__(self, input_dim, num_experts, bias=False):
        super(GatingNetwork, self).__init__()
        self.fc = nn.Linear(input_dim, num_experts, bias=bias)

    def forward(self, x):
        logits = self.fc(x)
        return F.softmax(logits, dim=-1)
    
class MOE(nn.Module):
    """ 
    Mixture of Experts Torch implementation.
    """
    def __init__(self, input_dim, hidden_dim, num_experts, k=2):
        super(MOE, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_experts = num_experts
        self.k = k
        self.experts = nn.ModuleList([
            Expert(input_dim, hidden_dim) for _ in range(num_experts)
        ])
        self.gate = GatingNetwork(input_dim, num_experts)

    def forward(self, x):
        B, L, D = x.size()
        x = x.view(-1, D) # SHAPE: (B*L, D)
        # 1. Get gating scores
        gate_scores = self.gate(x)  # (B*L, num_experts)
        # 2. Get top-k experts for each token
        topk_scores, topk_indices = torch.topk(gate_scores, self.k) # SHAPE: (B*L, k), (B*L, k)
        topk_scores = topk_scores / (topk_scores.sum(dim=-1, keepdim=True) + 1e-6)  # Normalize scores
        # 3. Dispatch tokens to experts
        output = torch.zeros(x.size(0), D, device=x.device)  # SHAPE: (B*L, D)
        for i in range(self.num_experts):
            mask = (topk_indices == i)  # (B*L, k) boolean: which tokens picked expert i
            if not mask.any():
                continue
            token_mask = mask.any(dim=-1)  # (B*L,) boolean: which tokens picked expert i
            selected_tokens = x[token_mask]  # (num_selected, D)
            if selected_tokens.size(0) == 0:
                continue
            # Process selected tokens with expert i
            expert_output = self.experts[i](selected_tokens)  # (num_selected, D)
            # Weight expert output by the corresponding top-k scores
            scores = (topk_scores * mask.float()).sum(-1)  # (B*L,)
            output[token_mask] += expert_output * scores[token_mask].unsqueeze(-1)
        output = output.view(B, L, D)  # SHAPE: (B, L, D) 
        return output
    
# Example usage:
if __name__ == "__main__":
    b, l, d = 8, 16, 32
    num_experts = 4
    x = torch.randn(b, l, d)  # (batch_size, seq_len, input_dim)
    module = MOE(input_dim=d, hidden_dim=d*2, num_experts=num_experts, k=2)
    output = module(x)
    print("Output shape:", output.shape)  # Should be (8, 16, 32)