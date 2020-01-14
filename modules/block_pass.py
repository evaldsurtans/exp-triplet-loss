import torch.nn as nn

class Pass(nn.Module):
    def __init__(self):
        super(Pass, self).__init__()

    def forward(self, x):
        return x
