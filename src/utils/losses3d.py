# src/utils/losses.py
import torch
import torch.nn as nn
import torch.nn.functional as F

def dice_coeff(pred, target, eps=1e-6):
    # pred: logits or probs (N, C, ...)
    # target: (N, C, ...)
    if pred.dim() == target.dim() + 1:
        # if logits with channel dim
        pred = pred.squeeze(1)
    pred = torch.sigmoid(pred)
    pred_flat = pred.contiguous().view(pred.shape[0], -1)
    target_flat = target.contiguous().view(target.shape[0], -1).float()
    intersection = (pred_flat * target_flat).sum(dim=1)
    return ((2. * intersection + eps) / (pred_flat.sum(dim=1) + target_flat.sum(dim=1) + eps)).mean()

class DiceLoss(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, logits, target):
        # target: (N,1,D,H,W) or (N,1,...) -> convert to same shape as logits
        if target.dim() == logits.dim() - 1:
            target = target.unsqueeze(1)
        probs = torch.sigmoid(logits)
        num = 2 * (probs * target).sum(dim=[2,3,4])
        den = probs.sum(dim=[2,3,4]) + target.sum(dim=[2,3,4]) + self.eps
        loss = 1 - (num / den)
        return loss.mean()

class BCEDiceLoss(nn.Module):
    def __init__(self, bce_weight=0.5):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()
        self.bce_weight = bce_weight

    def forward(self, logits, target):
        bce_loss = self.bce(logits.squeeze(1), target.squeeze(1).float())
        dice_loss = self.dice(logits, target)
        return self.bce_weight * bce_loss + (1 - self.bce_weight) * dice_loss
