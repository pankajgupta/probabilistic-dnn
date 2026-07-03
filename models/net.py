"""Small MLP for MNIST with tanh hidden activations.

Hidden activations are tanh (not ReLU) on purpose: at p-bit inference each
tanh(z) is replaced by sign(tanh(z) - r) with r ~ Uniform[-1,1], and
E[sign(tanh(z) - r)] = tanh(z). So a tanh-trained net has a p-neuron
counterpart with matching expected activation (see docs/HANDOFF.md section 2).
"""

import torch
import torch.nn as nn


class MLP(nn.Module):
    """784 -> hidden[0] -> hidden[1] -> ... -> 10, tanh between linear layers.

    Linear layers live in `self.fcs` (a ModuleList) so another module can
    later replay the forward pass layer-by-layer, swapping tanh for a
    p-neuron activation between layers. Output layer is linear (no softmax).
    """

    def __init__(self, hidden=(256, 128)):
        super().__init__()
        sizes = [784] + list(hidden) + [10]
        self.fcs = nn.ModuleList(
            nn.Linear(sizes[i], sizes[i + 1]) for i in range(len(sizes) - 1)
        )

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        for fc in self.fcs[:-1]:
            x = torch.tanh(fc(x))
        return self.fcs[-1](x)
