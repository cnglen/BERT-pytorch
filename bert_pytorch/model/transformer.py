#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""desc"""

import torch.nn as nn
import torch.nn.functional as F
from .attention import MultiHeadedAttention
from .utils import PositionwiseFeedForward


class TransformerEncoderLayer(nn.Module):
    """
    Bidirectional Encoder = Transformer (self-attention)
    Transformer = MultiHead_Attention + Feed_Forward with sublayer connection

    Args:
        d_model:
        head_count: number of heads in multi-head attention
        dim_feed_forward: dim_feed_forward, usually 4*d_model_size
        dropout: dropout rate
    """

    def __init__(self, d_model, head_count, dim_feed_forward, dropout):
        super().__init__()

        self.d_model = d_model
        self.head_count = head_count
        self.dim_feed_forward = dim_feed_forward
        self.dropout = dropout

        self.self_attention = MultiHeadedAttention(head_count=head_count, d_model=d_model, dropout=dropout)
        self.feed_forward = PositionwiseFeedForward(d_model=d_model, d_ff=dim_feed_forward, dropout=dropout)
        self.layer_norm_attn = nn.LayerNorm(normalized_shape=d_model, eps=1e-6)
        self.layer_norm_ff = nn.LayerNorm(normalized_shape=d_model, eps=1e-6)

    def forward(self, x, mask):
        """
        Add/Norm 在这里引入

        args:
          x: (B, Tx, d_model)
          mask: (B, 1, Tx)?
        """

        x_norm = self.layer_norm_attn(x)
        context, _ = self.self_attention(x_norm, x_norm, x_norm, mask)
        context = F.dropout(context, p=self.dropout, training=self.training)
        x = x + context         # Residual

        x_norm = self.layer_norm_ff(x)
        output = self.feed_forward(x_norm)
        output = F.dropout(x, p=self.dropout, training=self.training)
        output = output + x

        return output
