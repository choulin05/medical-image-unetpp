import torch
from src.models.unetpp import UNetPlusPlus

def test_unetpp():
    model = UNetPlusPlus(in_channels=1, num_classes=1)
    x = torch.randn(1, 1, 256, 256)  # 测试输入: batch=1, 单通道图像256x256
    y = model(x)
    print("✅ UNet++ forward pass successful!")
    print("Output shape:", y.shape)

if __name__ == "__main__":
    test_unetpp()
