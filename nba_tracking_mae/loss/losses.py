import torch

from enum import Enum


class Loss(Enum):
    L2 = "l2"
    BCE = "bce"


class L2Loss(torch.nn.Module):
    def __init__(self, **kwargs):
        super(L2Loss, self).__init__()

    def forward(self, output, labels):
        return torch.mean(torch.sum(torch.pow(labels - output, 2), dim=1))


class BCELoss(torch.nn.Module):
    def __init__(self, **kwargs):
        super(BCELoss, self).__init__()
        self.class_weights = kwargs.get("loss_weights", [1.0, 1.0])
        self.class_weights = [w * 2.0 / sum(self.class_weights) for w in self.class_weights]  # normalize weights

    def forward(self, output, labels):
        loss = self.class_weights[1] * labels * torch.log(output + 1e-7) + self.class_weights[0] * (
            1 - labels
        ) * torch.log(1 - output + 1e-7)

        if len(loss.size()) > 1:
            loss = torch.sum(loss, dim=1)

        return -torch.mean(loss)


def get_loss(loss: str, **kwargs):
    if loss == Loss.L2.value:
        return L2Loss(**kwargs)
    elif loss == Loss.BCE.value:
        return BCELoss(**kwargs)
    else:
        return None
