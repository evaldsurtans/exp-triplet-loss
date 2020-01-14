import torch.nn as nn

class Reshape(nn.Module):
    def __init__(self, shape):
        super(Reshape, self).__init__()
        if isinstance(shape, int):
            self.shape = (-1, shape)
        else:
            # shape is list
            self.shape = tuple([-1] + list(shape))

    def forward(self, x):
        return x.view(self.shape)
