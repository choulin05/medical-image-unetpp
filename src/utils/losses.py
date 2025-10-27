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

class DiceLoss2D(nn.Module):
    def __init__(self, smooth=1.):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        probs = probs.view(probs.size(0), -1)
        targets = targets.view(targets.size(0), -1)

        intersection = (probs * targets).sum(1)
        dice = (2. * intersection + self.smooth) / (probs.sum(1) + targets.sum(1) + self.smooth)
        return 1 - dice.mean()


# ---------- ✅ BCE + Dice2D ----------
class BCEDiceLoss2D(nn.Module):
    def __init__(self, bce_weight=0.5):
        super().__init__()
        self.bce_weight = bce_weight
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss2D()

    def forward(self, logits, targets):
        bce_loss = self.bce(logits, targets.float())
        dice_loss = self.dice(logits, targets)
        return self.bce_weight * bce_loss + (1 - self.bce_weight) * dice_loss