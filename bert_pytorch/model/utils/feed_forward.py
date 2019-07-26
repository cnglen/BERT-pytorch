#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Positoni-wise Feed Forward Networks"""

import torch.nn as nn
from .gelu import GELU


class PositionwiseFeedForward(nn.Module):
    """
    Implements FFN equation

    Args:
      d_model:
      d_ff:
    """

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.linear_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = GELU()

    def forward(self, x):
        """
        (B, T, d) --> (B, T, d_ff=4*d) --> (B, T, d)

        Args:
          x: (B, T, d)

        Returns:
          (B, T, d)
        """
        return self.linear_2(self.dropout(self.activation(self.linear_1(x))))
