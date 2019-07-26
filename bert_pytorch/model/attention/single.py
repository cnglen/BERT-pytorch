#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
single head attention:
attention(Q, K, V) =  softmax(QK^T/sqrt(d_model))V
"""

import math
import torch.nn as nn
import torch.nn.functional as F
import torch


class Attention(nn.Module):
    """
    Compute 'Scaled Dot Product Attention

    Args:
      query: (*, Ty, d_k) where * means any number of additional dimmensions, e.g, batch_size, (batch_size, head_count)
      key: (*, Tx, d_k)
      value: (*, Tx, d_v), In transformer, ResNet -> d_k=d_v=d_model
      mask:
      dropout:

    Returns:
      attention_vector: (*, Ty, d_v)
      p_attn: attention weight, (*, Ty, Tx)

    """

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                mask=None,
                dropout: float = None):

        # batched matrix multiply
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(query.size(-1))

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        p_attn = F.softmax(scores, dim=-1)

        if dropout is not None:
            p_attn = F.dropout(p_attn, p=self.dropout, training=self.training)

        attention_vector = torch.matmul(p_attn, value)
        return attention_vector, p_attn
