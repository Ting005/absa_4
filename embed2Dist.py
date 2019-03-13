# -*- coding: utf-8 -*-
from __future__ import unicode_literals, print_function, division

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class embed2Dist(nn.Module):
    """
    Computes a weighted average of the different channels across timesteps.
    Uses 1 parameter pr. channel to compute the attention value for a single timestep.
    """

    def __init__(self, dim_input, n_class):
        """ Initialize the attention layer
        # Arguments:
            attention_size: Size of the attention vector.
        """
        super(embed2Dist, self).__init__()
        self.dim_input = dim_input
        self.n_class = n_class
        self.fc = nn.Linear(dim_input, n_class, bias=True)

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.dim_input)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def __repr__(self):
        s = '{name}({attention_size})'
        return s.format(name=self.__class__.__name__, **self.__dict__)

    def forward(self, inputs):
        """ Forward pass.
        # Arguments:
            inputs (Torch.Variable): Tensor of input sequences
            input_lengths (torch.LongTensor): Lengths of the sequences
        # Return:
            representations and attentions.
        """
        out = self.fc(inputs)
        out = F.softmax(out, dim=-1)
        return out

