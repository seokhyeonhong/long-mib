import math
import torch
import torch.nn as nn

"""
MultiLinear
"""
class MultiLinear(nn.Module):
    def __init__(self, num_experts, in_features, out_features):
        super(MultiLinear, self).__init__()
        
        self.weight = nn.Parameter(torch.empty([num_experts, in_features, out_features]))
        self.bias = nn.Parameter(torch.empty([num_experts, 1, out_features]))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        return torch.matmul(x, self.weight) + self.bias