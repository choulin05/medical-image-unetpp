# train.py
import os
import argparse
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

# local imports
from src.datasets.brats_dataset import BraTSDataset
from src.models.unetpp3d import UNetPlusPlus3D
from src.utils.losses3d import BCEDiceLoss, dice_coeff

try:
    import wandb
    _HAS_WANDB = True
except Exception:
    _HAS_WANDB = False

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def save_checkpoint(state, path):
    torch.save(state, path)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default="data/brats2021/TrainingData/BraTS2021_Training_Data", help="BraTS folder")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--num-workers", type=int, default=0)  # safe for Windows
    parser.add_argument("--save-dir", type=str, default="checkpoints")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-wandb", action="store_true", help="disable wandb logging")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()

def collate_fn(batch):
    # batch is list of tuples (image, mask) where image: [C,H,W,D], mask: [1,H,W,D]
    images = torch.stack([b[0] for b in batch], dim=0)  # (N,C,H,W,D)
    masks = torch.stack([b[1] for b in batch], dim=0)   # (N,1,H,W,D)
    return images, masks

def main():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    set_seed(args.seed)

    device = torch.device(args.device)

    # dataset and split
    dataset = BraTSDataset(args.data_dir)
    n_total = len(dataset)
    n_val = max(1, int(0.15 * n_total))
    n_train = n_total - n_val
    train_set, val_set = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(args.seed))

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=collate_fn)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=args.num_workers, collate_fn=collate_fn)

    # model
    model = UNetPlusPlus3D(in_channels=4, num_classes=1, filters=[32,64,128,256,320]).to(device)
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3, verbose=True)
    criterion = BCEDiceLoss(bce_weight=0.5)

    start_epoch = 0
    best_val = 0.0

    # wandb init
    if (not args.no_wandb) and _HAS_WANDB:
        wandb.init(project="medical-image-unetpp", config=vars(args))
        wandb.watch(model, log="all", log_freq=50)

    print(f"Train samples: {n_train}, Val samples: {n_val}, Device: {device}")

    for epoch in range(start_epoch, args.epochs):
        model.train()
        t = time.time()
        train_loss = 0.0
        train_batches = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} Train", leave=False)
        for imgs, masks in pbar:
            imgs = imgs.to(device)         # (N, C, H, W, D)
            masks = masks.to(device)       # (N, 1, H, W, D)
            optimizer.zero_grad()
            outputs = model(imgs)          # (N,1,H,W,D)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_batches += 1
            pbar.set_postfix({'loss': f"{train_loss/train_batches:.4f}"})

        avg_train_loss = train_loss / max(1, train_batches)

        # Validation
        model.eval()
        val_loss = 0.0
        val_batches = 0
        val_dice = 0.0
        with torch.no_grad():
            pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{args.epochs} Val", leave=False)
            for imgs, masks in pbar:
                imgs = imgs.to(device)
                masks = masks.to(device)
                outputs = model(imgs)
                loss = criterion(outputs, masks)
                val_loss += loss.item()
                val_batches += 1

                # dice
                try:
                    dice_v = dice_coeff(outputs.squeeze(1), masks.squeeze(1))
                except Exception:
                    dice_v = dice_coeff(outputs, masks)
                val_dice += dice_v.item()

        avg_val_loss = val_loss / max(1, val_batches)
        avg_val_dice = val_dice / max(1, val_batches)

        scheduler.step(avg_val_dice)

        # logging
        print(f"Epoch {epoch+1:03d} | train_loss: {avg_train_loss:.4f} | val_loss: {avg_val_loss:.4f} | val_dice: {avg_val_dice:.4f} | time: {time.time()-t:.1f}s")
        if (not args.no_wandb) and _HAS_WANDB:
            wandb.log({"epoch": epoch+1, "train_loss": avg_train_loss, "val_loss": avg_val_loss, "val_dice": avg_val_dice})

        # checkpoint
        ckpt_path = os.path.join(args.save_dir, f"epoch{epoch+1:03d}_dice{avg_val_dice:.4f}.pth")
        save_checkpoint({
            'epoch': epoch+1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_dice': avg_val_dice
        }, ckpt_path)

        # keep best
        if avg_val_dice > best_val:
            best_val = avg_val_dice
            best_path = os.path.join(args.save_dir, f"best_model.pth")
            save_checkpoint({
                'epoch': epoch+1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_dice': avg_val_dice
            }, best_path)
            print(f"*** New best model saved with val_dice={best_val:.4f} ***")
            if (not args.no_wandb) and _HAS_WANDB:
                wandb.save(best_path)

    print("Training finished.")
    if (not args.no_wandb) and _HAS_WANDB:
        wandb.finish()

if __name__ == "__main__":
    main()
