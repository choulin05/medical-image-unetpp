import os
import sys
sys.path.append(os.path.abspath("."))  # 让 Python 能找到 src 包

from src.datasets.brats_dataset import BraTSDataset
from torch.utils.data import DataLoader

if __name__ == "__main__":
    dataset = BraTSDataset(data_dir="data\Brats2021\Test")

    print(f"✅ Dataset size: {len(dataset)} cases")

    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    for batch_idx, (image, mask) in enumerate(dataloader):
        print(f"Image shape: {image.shape} | Mask shape: {mask.shape}")
        break  # 只测试一批
