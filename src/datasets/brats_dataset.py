import os
import glob
import numpy as np
import torch
import nibabel as nib
import torch.nn.functional as F
from torch.utils.data import Dataset


class BraTSDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.transform = transform
        self.samples = []

        # 递归扫描所有病例
        all_cases = sorted(glob.glob(os.path.join(data_dir, "BraTS2021_*")))
        print(f"🔍 Found {len(all_cases)} raw cases. Validating...")

        # 过滤有效样本
        for case in all_cases:
            try:
                # check modalities
                files = []
                for m in modalities:
                    files += glob.glob(os.path.join(case, f"*_{m}.nii")) + glob.glob(os.path.join(case, f"*_{m}.nii.gz"))

                # 🔧 支持多个标签命名：seg / seg_new / final_seg
                seg_files = []
                seg_files += glob.glob(os.path.join(case, "*seg.nii"))
                seg_files += glob.glob(os.path.join(case, "*seg_new.nii"))
                seg_files += glob.glob(os.path.join(case, "*final_seg.nii"))
                seg_files += glob.glob(os.path.join(case, "*seg.nii.gz"))
                seg_files += glob.glob(os.path.join(case, "*seg_new.nii.gz"))
                seg_files += glob.glob(os.path.join(case, "*final_seg.nii.gz"))
                
                if len(test_file) == 0:
                    continue
                nib.load(test_file[0])  # 尝试读取，坏文件会报错
                self.samples.append(case)
            except:
                print(f"⚠️ Skipping corrupted case: {case}")

        print(f"✅ Loaded {len(self.samples)} valid MRI volumes")

    def __len__(self):
        return len(self.samples)

    def load_nii(self, case_path, keyword):
        files = glob.glob(os.path.join(case_path, f"*{keyword}.nii")) + \
                glob.glob(os.path.join(case_path, f"*{keyword}.nii.gz"))
        if len(files) == 0:
            raise FileNotFoundError(f"Missing {keyword} in {case_path}")
        return nib.load(files[0]).get_fdata()

    def __getitem__(self, idx):
        case_path = self.samples[idx]

        # 读取4种MRI模态
        flair = self.load_nii(case_path, "_flair")
        t1 = self.load_nii(case_path, "_t1")
        t1ce = self.load_nii(case_path, "_t1ce")
        t2 = self.load_nii(case_path, "_t2")
        mask = self.load_nii(case_path, "_seg")

        image = np.stack([flair, t1, t1ce, t2], axis=0)

        image = torch.tensor(image, dtype=torch.float32)
        mask = torch.tensor(mask, dtype=torch.long).unsqueeze(0)

        image = F.interpolate(image.unsqueeze(0), size=(128, 128, 128),
                              mode="trilinear", align_corners=False).squeeze(0)
        mask = F.interpolate(mask.unsqueeze(0).float(), size=(128, 128, 128),
                             mode="nearest").long().squeeze(0)

        return image, mask
