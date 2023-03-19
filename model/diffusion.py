import numpy as np
import torch
import torch.nn as nn

from pymovis.learning.transformer import RelativeMultiHeadAttention, PoswiseFeedForwardNet
from pymovis.learning.embedding import RelativeSinusoidalPositionalEmbedding

class DiffusionTransformer(nn.Module):
    def __init__(self, d_motion, config):
        super(DiffusionTransformer, self).__init__()