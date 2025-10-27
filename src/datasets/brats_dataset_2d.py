# src/datasets/brats_dataset_2d.py
import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset
import nibabel as nib
import cv2


class BraTSDataset2D(Dataset):
    def __init__(self, data_dir, resize=(256, 256), keep_empty=False, modalities=("flair", "t1", "t1ce", "t2")):
        """
        data_dir: folder containing BraTS cases (each case is a folder with *_flair.nii / *_seg.nii etc.)
        resize: target (H,W)
        keep_empty: whether to keep slices with empty mask (default False -> keeps only slices with tumor)
        modalities: order of modalities (channels)
        """
        self.data_dir = data_dir
        self.resize = tuple(resize)
        self.keep_empty = keep_empty
        self.modalities = modalities

        # find cases
        all_cases = sorted(glob.glob(os.path.join(data_dir, "BraTS2021_*")))
        print(f"🔍 Found {len(all_cases)} raw cases in {data_dir}. Scanning valid cases and slices...")

        self.samples = []  # (case_path, slice_idx)
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

                if len(files) == 0 or len(seg_files) == 0:
                    continue

                seg_img = nib.load(seg_files[0])
                seg_data = seg_img.get_fdata()
                if seg_data.ndim != 3:
                    continue

                _, _, D = seg_data.shape
                for z in range(D):
                    if not keep_empty and seg_data[:, :, z].sum() == 0:
                        continue
                    self.samples.append((case, int(z)))
            except Exception as e:
                print(f"⚠️ Skipping case {case} during scan: {e}")

        print(f"✅ Dataset ready. Total 2D slices: {len(self.samples)} (keep_empty={self.keep_empty})")

    def __len__(self):
        return len(self.samples)

    def _load_case_modalities(self, case_path):
        # load modalities
        mods = {}
        for m in self.modalities:
            files = glob.glob(os.path.join(case_path, f"*_{m}.nii")) + glob.glob(os.path.join(case_path, f"*_{m}.nii.gz"))
            if len(files) == 0:
                raise FileNotFoundError(f"Missing modality {m} in {case_path}")
            mods[m] = nib.load(files[0]).get_fdata()

        # 🔧 segmentation labels 兼容多个命名
        seg_files = []
        seg_files += glob.glob(os.path.join(case_path, "*seg.nii"))
        seg_files += glob.glob(os.path.join(case_path, "*seg_new.nii"))
        seg_files += glob.glob(os.path.join(case_path, "*final_seg.nii"))
        seg_files += glob.glob(os.path.join(case_path, "*seg.nii.gz"))
        seg_files += glob.glob(os.path.join(case_path, "*seg_new.nii.gz"))
        seg_files += glob.glob(os.path.join(case_path, "*final_seg.nii.gz"))

        if len(seg_files) == 0:
            raise FileNotFoundError(f"Missing segmentation in {case_path}")

        seg = nib.load(seg_files[0]).get_fdata()
        return mods, seg

    def __getitem__(self, idx):
        case_path, z = self.samples[idx]
        try:
            mods, seg = self._load_case_modalities(case_path)

            # 取2D切片
            slices = []
            for m in self.modalities:
                vol = mods[m]  # (H,W,D)
                if vol.ndim != 3:
                    raise ValueError(f"Modality {m} shape invalid: {vol.shape}")

                # 取第z层切片
                if z < vol.shape[2]:
                    sl = vol[:, :, z]
                elif z < vol.shape[0]:
                    sl = vol[z, :, :]
                else:
                    sl = vol[:, :, vol.shape[-1] // 2]

                sl = sl.astype(np.float32)
                if sl.max() > sl.min():
                    sl = (sl - sl.mean()) / (sl.std() + 1e-8)

                sl_resized = cv2.resize(sl, self.resize, interpolation=cv2.INTER_LINEAR)
                slices.append(sl_resized)

            image = np.stack(slices, axis=0).astype(np.float32)

            # segmentation mask
            if z < seg.shape[2]:
                msl = seg[:, :, z]
            else:
                msl = seg[:, :, seg.shape[-1] // 2]

            msl = (msl > 0).astype(np.uint8)
            msl_resized = cv2.resize(msl, self.resize, interpolation=cv2.INTER_NEAREST)

            image_t = torch.from_numpy(image)
            mask_t = torch.from_numpy(msl_resized).unsqueeze(0).long()

            return image_t, mask_t

        except Exception as e:
            print(f"⚠️ Error loading slice {z} in case {case_path}: {e}")
            return None  # ✅ 给 dataloader 过滤掉坏数据