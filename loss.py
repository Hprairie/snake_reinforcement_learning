import torch.nn as nn

class weighted_MSELoss(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self,inputs,targets,weights):
        return (((inputs - targets)**2 ) * weights).mean()