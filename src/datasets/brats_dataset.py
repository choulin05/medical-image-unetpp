import os
import glob
import numpy as np
import torch
import nibabel as nib
import torch.nn.functional as F
from torch.utils.data import Dataset


class BraTSDataset(Dataset):
    def __init__(self, data_dir, transform=None, keep_empty=False):
        self.transform = transform
        self.keep_empty = keep_empty
        self.samples = []
        self.modalities = ["flair", "t1", "t1ce", "t2"]
        
        # 递归扫描所有病例
        all_cases = sorted(glob.glob(os.path.join(data_dir, "BraTS2021_*")))
        print(f"🔍 Found {len(all_cases)} raw cases. Validating...")

        # 过滤有效样本
        for case in all_cases:
            try:
                # 检查模态文件
                files = []
                for m in self.modalities:
                    files += glob.glob(os.path.join(case, f"*_{m}.nii")) + glob.glob(os.path.join(case, f"*_{m}.nii.gz"))

                # 支持多个标签命名
                seg_files = []
                seg_files += glob.glob(os.path.join(case, "*seg.nii"))
                seg_files += glob.glob(os.path.join(case, "*seg_new.nii"))
                seg_files += glob.glob(os.path.join(case, "*final_seg.nii"))
                seg_files += glob.glob(os.path.join(case, "*seg.nii.gz"))
                seg_files += glob.glob(os.path.join(case, "*seg_new.nii.gz"))
                seg_files += glob.glob(os.path.join(case, "*final_seg.nii.gz"))
                
                if len(files) < 4 or len(seg_files) == 0:
                    print(f"⚠️ 跳过病例 {case}: 缺少模态或分割文件")
                    continue

                seg_img = nib.load(seg_files[0])
                seg_data = seg_img.get_fdata()
                if seg_data.ndim != 3:
                    print(f"⚠️ 跳过病例 {case}: 分割数据不是3D")
                    continue

                # 对于3D数据集，每个病例是一个样本
                if not self.keep_empty and seg_data.sum() == 0:
                    print(f"⚠️ 跳过病例 {case}: 没有肿瘤")
                    continue
                    
                self.samples.append(case)
                
            except Exception as e:
                print(f"⚠️ 跳过病例 {case} 扫描时出错: {e}")

        print(f"✅ Loaded {len(self.samples)} valid MRI volumes")

    def __len__(self):
        return len(self.samples)

    def _load_nii(self, case_path, suffix):
        """加载nii文件"""
        # 查找匹配的文件
        files = glob.glob(os.path.join(case_path, f"*{suffix}.nii")) + \
                glob.glob(os.path.join(case_path, f"*{suffix}.nii.gz"))
        
        if len(files) == 0:
            raise FileNotFoundError(f"找不到 {suffix} 文件在 {case_path}")
        
        # 加载nii数据
        img = nib.load(files[0])
        data = img.get_fdata()
        
        # 标准化到 [0, 1]
        data = (data - data.min()) / (data.max() - data.min() + 1e-8)
        
        return data

    def __getitem__(self, idx):
        case_path = self.samples[idx]

        try:
            # 读取4种MRI模态
            flair = self._load_nii(case_path, "_flair")
            t1 = self._load_nii(case_path, "_t1")
            t1ce = self._load_nii(case_path, "_t1ce")
            t2 = self._load_nii(case_path, "_t2")
            mask = self._load_nii(case_path, "_seg")

            # 将模态堆叠为4通道
            image = np.stack([flair, t1, t1ce, t2], axis=0)  # (4, H, W, D)
            
            # 转换为torch张量
            image = torch.tensor(image, dtype=torch.float32)
            mask = torch.tensor(mask, dtype=torch.float32).unsqueeze(0)  # (1, H, W, D)

            # 调整大小为 (128, 128, 128)
            image = F.interpolate(image.unsqueeze(0), size=(128, 128, 128),
                                  mode="trilinear", align_corners=False).squeeze(0)
            mask = F.interpolate(mask.unsqueeze(0), size=(128, 128, 128),
                                 mode="nearest").squeeze(0)

            return image, mask

        except Exception as e:
            print(f"❌ 加载病例 {case_path} 时出错: {e}")
            return None