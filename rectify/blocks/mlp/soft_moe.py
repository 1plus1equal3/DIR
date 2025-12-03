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
    
class SoftMOE(nn.Module):
    """ 
    Soft Mixture of Experts Torch implementation.

    From "From Sparse to Soft Mixtures of Experts" 
      https://arxiv.org/pdf/2308.00951.pdf 
    and 2 implementations:
      https://github.com/bwconrad/soft-moe/blob/main/soft_moe/soft_moe.py 
      https://github.com/lucidrains/soft-moe-pytorch/blob/main/soft_moe_pytorch/soft_moe.py
    """
    def __init__(self, input_dim, hidden_dim, num_experts, slots_per_expert, bias=False, normalize=True):
        super(SoftMOE, self).__init__()
        # Initialize parameters
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_experts = num_experts
        self.slots_per_expert = slots_per_expert
        self.bias = bias
        self.normalize = normalize
        # Initialize phi and normalization scaling factor
        self.phi = nn.Parameter(torch.zeros(input_dim, num_experts, slots_per_expert))
        if self.normalize:
            self.scale = nn.Parameter(torch.ones(1))
        # Initialize phi using LeCun normal initialization
        # https://github.com/google-research/vmoe/blob/662341d007650d5bbb7c6a2bef7f3c759a20cc7e/vmoe/projects/soft_moe/router.py#L49C1-L49C1
        nn.init.normal_(self.phi, mean=0, std=1/input_dim**0.5)
        # Initialize experts
        self.experts = nn.ModuleList([Expert(input_dim, hidden_dim) for _ in range(num_experts)])

    def forward(self, x):
        B, L, D = x.size() # x shape: (B, L, D)     Batch, Length, Embedding dim
        phi = self.phi  # SHAPE: (D, E, S)          Embedding dim, Experts, Slots per expert
        if self.normalize:
            x = F.normalize(x, dim=2)
            phi = self.scale * F.normalize(phi, dim=0)
        # 1. Compute dispatch weights
        logits = torch.einsum('bld,des->bles', x, phi)  # SHAPE: (B, L, E, S)
        d_w = F.softmax(logits, dim=-1)  # Softmax over slots 
        # 2. Compute combined weights
        c_w = d_w.view(B, L, self.num_experts * self.slots_per_expert)  # SHAPE: (B, L, E*S)
        c_w = F.softmax(c_w, dim=-1)  # Softmax over all experts and slots
        # 3. Expert processing
        slots = torch.einsum('bld,bles->besd', x, d_w)  # SHAPE: (B, E, S, D)
        expect_outputs = torch.stack([
            self.experts[i](slots[:, i, :, :]) for i in range(self.num_experts)
        ], dim=1) # SHAPE: (E, B, S, D)
        # 4. Combine expert outputs
        expect_outputs = expect_outputs.view(B, self.num_experts * self.slots_per_expert, D)  # SHAPE: (B, E*S, D)
        output = torch.einsum('bkd,blk->bld', expect_outputs, c_w)  # SHAPE: (B, L, D)
        return output
    
# Example usage
if __name__ == "__main__":
    l, d = 16, 32
    num_experts = 4
    num_slots = l // num_experts
    print(f"Sequence length: {l}, Embedding dim: {d}, Num experts: {num_experts}, Slots per expert: {num_slots}")
    x = torch.randn(8, 16, 32)  # Example input tensor with shape (B, L, D)
    module = SoftMOE(input_dim=d, hidden_dim=d*2, num_experts=num_experts, slots_per_expert=num_slots)
    output = module(x)
    print("Output shape:", output.shape)  # Should be (8, 16, 32)