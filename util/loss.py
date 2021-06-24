import math

import torch
import torch.nn.functional as F
from torch import nn


# Defines the GAN loss which uses either LSGAN or the regular GAN. When LSGAN is used, it is basically same as MSELoss,
# but it abstracts away the need to create the target label tensor that has the same size as the input
class GANLoss(nn.Module):
    def __init__(self, use_lsgan=True, device=torch.device('cpu')):
        super(GANLoss, self).__init__()
        if use_lsgan:
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.BCELoss()
        self.device = device

    def get_target_tensor(self, input, target_is_real):
        # Get soft and noisy labels
        if target_is_real:
            target_tensor = 0.7 + 0.3 * torch.rand(input.size(0))
        else:
            target_tensor = 0.3 * torch.rand(input.size(0))
        if input.is_cuda:
            target_tensor = target_tensor.to(self.device)
        return target_tensor

    def __call__(self, input, target_is_real):
        target_tensor = self.get_target_tensor(input, target_is_real)
        return self.loss(input.squeeze(), target_tensor)


class AdaCos(nn.Module):
    def __init__(self, num_features, num_classes):
        super(AdaCos, self).__init__()
        self.num_features = num_features
        self.n_classes = num_classes
        self.s = math.sqrt(2) * math.log(num_classes - 1)
        self.W = nn.Parameter(torch.FloatTensor(num_classes, num_features))
        nn.init.xavier_uniform_(self.W)

    def forward(self, input, label=None):
        # normalize features
        x = F.normalize(input)
        # normalize weights
        W = F.normalize(self.W)
        # dot product
        logits = F.linear(x, W)
        if label is None:
            return logits
        # feature re-scale
        theta = torch.acos(torch.clamp(logits, -1.0 + 1e-7, 1.0 - 1e-7))
        one_hot = torch.zeros_like(logits)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        with torch.no_grad():
            B_avg = torch.where(one_hot < 1, torch.exp(self.s * logits), torch.zeros_like(logits))
            B_avg = torch.sum(B_avg) / input.size(0)
            theta_med = torch.median(theta[one_hot == 1])
            self.s = torch.log(B_avg) / torch.cos(torch.min(math.pi/4 * torch.ones_like(theta_med), theta_med))
        output = self.s * logits
        return output

