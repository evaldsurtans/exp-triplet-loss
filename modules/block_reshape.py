import torch.nn as nn

class Reshape(nn.Module):
    def __init__(self, shape):
        super(Reshape, self).__init__()
        if not isinstance(shape, tuple):
            shape = (shape, )
        self.shape = shape

    def forward(self, x):
        return x.view(-1, *self.shape)