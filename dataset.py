import glob
import random
import os
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
import torch


class ImgDataset(Dataset):
    def __init__(self, imgpath, transforms_=None, unaligned=True, mode='train', device='cuda'):
        self.imgs_A = sorted(glob.glob(os.path.join(imgpath, '%sA' % mode, '*.jpg')))
        self.imgs_B = sorted(glob.glob(os.path.join(imgpath, '%sB' % mode, '*.jpg')))
        self.imgs_A_labels = sorted(glob.glob(os.path.join(imgpath, '%sA_labels' % mode, '*.jpg')))
        self.imgs_B_labels = sorted(glob.glob(os.path.join(imgpath, '%sB_labels' % mode, '*.jpg')))
        self.transform = transforms.Compose(transforms_)
        self.unaligned = unaligned
        self.device = device

    def __getitem__(self, index):
        item_A = self.transform(Image.open(self.imgs_A[index % len(self.imgs_A)]))
        A_label = Image.open(self.imgs_A_labels[index % len(self.imgs_A)])
        A_label = np.asarray(A_label)[:,:,0]
        A_label = convert_label(A_label)
        if self.unaligned:
            item_B = self.transform(Image.open(self.imgs_B[random.randint(0, len(self.imgs_B) - 1)]))
        else:
            item_B = self.transform(Image.open(self.imgs_B[index % len(self.imgs_B)]))
        return {'A': item_A.to(self.device), 'B': item_B.to(self.device), 'A_label': A_label.to(self.device)}

    def __len__(self):
        return max(len(self.imgs_A), len(self.imgs_B))

def convert_label(arr):
    arr = arr.copy()
    ignore_label = 19
    label_mapping = {-1: ignore_label, 0: ignore_label,
                     1: ignore_label, 2: ignore_label,
                     3: ignore_label, 4: ignore_label,
                     5: ignore_label, 6: ignore_label,
                     7: 0, 8: 1, 9: ignore_label,
                     10: ignore_label, 11: 2, 12: 3,
                     13: 4, 14: ignore_label, 15: ignore_label,
                     16: ignore_label, 17: 5, 18: ignore_label,
                     19: 6, 20: 7, 21: 8, 22: 9, 23: 10, 24: 11,
                     25: 12, 26: 13, 27: 14, 28: 15,
                     29: ignore_label, 30: ignore_label,
                     31: 16, 32: 17, 33: 18, 35: ignore_label}
    copy = arr.copy()
    for k, v in label_mapping.items():
        arr[copy == k] = v
    label = np.zeros((20, arr.shape[0], arr.shape[1]))
    for i in range(20):
        label[i, :, :][arr == i] = 1
    return torch.tensor(label, dtype=torch.float, requires_grad=False)


