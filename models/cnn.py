"""LeNet-style CNN for MNIST with tanh hidden activations.

Every hidden activation (both conv feature maps and the fc hidden layer) is
tanh, on purpose: at p-bit inference each tanh(z) is replaced by
sign(tanh(z) - r) with r ~ Uniform[-1,1], and E[sign(tanh(z) - r)] = tanh(z)
(see models/net.py's MLP for the same convention, docs/HANDOFF.md section 2).

Architecture (fixed by the CNN-extension brief):
  conv1: 1  -> C1 ch, 5x5, stride 1   -> tanh -> maxpool 2x2
  conv2: C1 -> C2 ch, 5x5, stride 1   -> tanh -> maxpool 2x2
  flatten -> fc1 (-> H) -> tanh -> fc2 (-> 10) linear

With the default (C1, C2, H) = (8, 16, 64) on 28x28 MNIST input:
  28 -> conv1 -> 24 -> pool -> 12 -> conv2 -> 8 -> pool -> 4
  flatten: C2*4*4 = 256 -> fc1(64) -> fc2(10)
"""

import torch
import torch.nn as nn


class CNN(nn.Module):
    """conv1 -> tanh -> pool -> conv2 -> tanh -> pool -> flatten -> fc1 -> tanh -> fc2.

    `self.stages` is an ordered list of (module, kind) pairs, kind in
    {"conv", "pool", "flatten", "fc", "fc_out"}, so another module (see
    models/pdnn.py's pdnn_forward_conv) can replay the forward pass
    stage-by-stage, swapping tanh for a p-neuron activation after every
    "conv"/"fc" stage. "flatten" carries no learnable module (module is the
    string "flatten"); "pool" and "fc_out" are deterministic (no p-bit).
    """

    def __init__(self, conv_channels=(8, 16), fc_hidden=64, in_hw=28, kernel_size=5):
        super().__init__()
        c1, c2 = conv_channels
        self.conv1 = nn.Conv2d(1, c1, kernel_size=kernel_size, stride=1)
        self.conv2 = nn.Conv2d(c1, c2, kernel_size=kernel_size, stride=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        h1 = (in_hw - kernel_size + 1) // 2          # after conv1 + pool1
        h2 = (h1 - kernel_size + 1) // 2              # after conv2 + pool2
        self.flat_dim = c2 * h2 * h2

        self.fc1 = nn.Linear(self.flat_dim, fc_hidden)
        self.fc2 = nn.Linear(fc_hidden, 10)

        self.conv_channels = tuple(conv_channels)
        self.fc_hidden = fc_hidden
        self.in_hw = in_hw
        self.kernel_size = kernel_size

        self.stages = [
            (self.conv1, "conv"),
            (self.pool1, "pool"),
            (self.conv2, "conv"),
            (self.pool2, "pool"),
            ("flatten", "flatten"),
            (self.fc1, "fc"),
            (self.fc2, "fc_out"),
        ]

    def forward(self, x):
        if x.dim() == 2:
            x = x.view(x.shape[0], 1, self.in_hw, self.in_hw)
        elif x.dim() == 3:
            x = x.unsqueeze(1)
        x = torch.tanh(self.conv1(x))
        x = self.pool1(x)
        x = torch.tanh(self.conv2(x))
        x = self.pool2(x)
        x = x.flatten(1)
        x = torch.tanh(self.fc1(x))
        x = self.fc2(x)
        return x

    def num_params(self):
        return sum(p.numel() for p in self.parameters())
