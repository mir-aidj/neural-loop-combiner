import torch
import torch.nn as nn
import torch.nn.functional as F
from neural_loop_combiner.config import settings

class ContrastiveLoss(nn.Module):
    def __init__(self, margin=settings.MARGIN):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.eps    = 1e-9

    def forward(self, output1, output2, target, size_average=True):
        distances = torch.pow(F.pairwise_distance(output1, output2), 2)
        losses = 0.5 * (target.float() * distances +
                        (1 + -1 * target).float() * F.relu(self.margin - (distances + self.eps).sqrt()).pow(2))
        return losses.mean() if size_average else losses.sum()