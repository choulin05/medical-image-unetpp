# train2d.py
import os
import argparse
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
from torch.optim import Adam
from tqdm import tqdm

from src.datasets.brats_dataset_2d import BraTSDataset2D
from src.models.unetpp_2d import UNetPlusPlus2D
from src.utils.losses import BCEDiceLoss2D#, dice_coeff

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def iou_score(pred, target, eps=1e-6):
    # pred, target: torch tensors (N, H, W) or (N,1,H,W)
    if pred.dim() == target.dim() + 1:
        pred = pred.squeeze(1)
    pred = (torch.sigmoid(pred) > 0.5).float()
    target = target.squeeze(1).float()
    inter = (pred * target).sum(dim=[1,2])
    union = (pred + target - pred*target).sum(dim=[1,2]) + eps
    return (inter / union).mean().item()

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default="data/brats2021/TrainingData/BraTS2021_Training_Data")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--save-dir", type=str, default="checkpoints2d")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--keep-empty", action="store_true", help="keep slices without tumor")
    return parser.parse_args()

def collate_fn(batch):
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return None
    imgs = torch.stack([b[0] for b in batch], dim=0)
    masks = torch.stack([b[1] for b in batch], dim=0)
    return imgs, masks


def main():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    set_seed(args.seed)
    device = torch.device(args.device)
    print("Device:", device)

    # dataset
    dataset = BraTSDataset2D(args.data_dir, resize=(256,256), keep_empty=args.keep_empty)
    n_total = len(dataset)
    n_val = max(1, int(0.15 * n_total))
    n_train = n_total - n_val
    train_set, val_set = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(args.seed))

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=collate_fn)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=args.num_workers, collate_fn=collate_fn)

    model = UNetPlusPlus2D(in_channels=4, num_classes=1, filters=[32,64,128,256,320]).to(device)
    optimizer = Adam(model.parameters(), lr=args.lr)
    criterion = BCEDiceLoss2D(bce_weight=0.5)

    best_dice = 0.0

    for epoch in range(1, args.epochs+1):
        model.train()
        running_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs} Train", leave=False)
        for imgs, masks in pbar:
            imgs = imgs.to(device)
            masks = masks.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)  # (N,1,H,W)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            pbar.set_postfix({"loss": f"{running_loss/ (pbar.n+1):.4f}"})

        # validation
        model.eval()
        val_loss = 0.0
        dice_sum = 0.0
        iou_sum = 0.0
        cnt = 0
        with torch.no_grad():
            pbarv = tqdm(val_loader, desc=f"Epoch {epoch}/{args.epochs} Val", leave=False)
            for imgs, masks in pbarv:
                imgs = imgs.to(device)
                masks = masks.to(device)
                outputs = model(imgs)
                loss = criterion(outputs, masks)
                val_loss += loss.item()

                # metrics
                try:
                    d = dice_coeff(outputs.squeeze(1), masks.squeeze(1))
                except Exception:
                    d = dice_coeff(outputs, masks)
                dice_sum += d.item()
                iou_sum += iou_score(outputs.squeeze(1), masks)
                cnt += 1

        avg_val_loss = val_loss / max(1, cnt)
        avg_dice = dice_sum / max(1, cnt)
        avg_iou = iou_sum / max(1, cnt)

        print(f"Epoch {epoch} | train_loss: {running_loss/ max(1, len(train_loader)):.4f} | val_loss: {avg_val_loss:.4f} | val_dice: {avg_dice:.4f} | val_iou: {avg_iou:.4f}")

        # checkpoint best
        if avg_dice > best_dice:
            best_dice = avg_dice
            ckpt = os.path.join(args.save_dir, f"best2d_epoch{epoch}_dice{best_dice:.4f}.pth")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_dice": best_dice
            }, ckpt)
            print(f"*** New best saved: {ckpt}")

    print("Training finished.")

if __name__ == "__main__":
    main()