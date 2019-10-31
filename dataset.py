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
        self.mode = mode
        if mode == 'train':
            self.imgs_A = sorted(glob.glob(os.path.join(imgpath, '%sA' % mode, '*.jpg')))
            self.imgs_B = sorted(glob.glob(os.path.join(imgpath, '%sB' % mode, '*.jpg')))
            self.imgs_B_labels = sorted(glob.glob(os.path.join(imgpath, '%sB_labels' % mode, '*.png')))
        elif mode == 'unetTrain':
            self.imgs_B = sorted(glob.glob(os.path.join(imgpath, 'trainB', '*.jpg')))
            self.imgs_B_labels = sorted(glob.glob(os.path.join(imgpath, 'trainB_labels', '*.png')))
        elif mode == 'test':
            self.testImgs = sorted(glob.glob(os.path.join(imgpath, '*.jpg')))
        elif mode == 'eval_unet':
            self.testImgs = sorted(glob.glob(os.path.join(imgpath, 'x', '*.jpg')))
            self.testLabels = sorted(glob.glob(os.path.join(imgpath, 'gt', '*.png')))
        else:
            raise Exception('mode must be train or test!')
        self.transform = transforms.Compose(transforms_)
        self.unaligned = unaligned
        self.device = device

    def __getitem__(self, index):
        if self.mode == 'train':
            item_A = self.transform(Image.open(self.imgs_A[index % len(self.imgs_A)]))
            if self.unaligned:
                bidx = random.randint(0, len(self.imgs_B) - 1)
                item_B = self.transform(Image.open(self.imgs_B[bidx]))
                B_label = Image.open(self.imgs_B_labels[bidx])
                B_label = np.asarray(B_label)
                B_label = convert_label(B_label)

            else:
                item_B = self.transform(Image.open(self.imgs_B[index % len(self.imgs_B)]))
                B_label = Image.open(self.imgs_B_labels[index % len(self.imgs_B)])
                B_label = np.asarray(B_label)
                B_label = convert_label(B_label)
            return {'A': item_A.to(self.device), 'B': item_B.to(self.device), 'B_label' : B_label.to(self.device)}
        elif self.mode == 'unetTrain':
            item_B = self.transform(Image.open(self.imgs_B[index % len(self.imgs_B)]))
            B_label = Image.open(self.imgs_B_labels[index % len(self.imgs_B)])
            B_label = np.asarray(B_label)
            B_label = convert_label(B_label)
            return {'B': item_B.to(self.device), 'B_label' : B_label.to(self.device)}
        elif self.mode == 'eval_unet':
            img = self.transform(Image.open(self.testImgs[index]))
            original_label = np.asarray(Image.open(self.testLabels[index]))
            label = np.asarray(Image.open(self.testLabels[index]).convert('RGB'))
            return self.testImgs[index].split('/')[-1], img, label, original_label
        else:
            img = self.transform(Image.open(self.testImgs[index]))
            return self.testImgs[index], img

    def __len__(self):
        if self.mode == 'train':
            return max(len(self.imgs_A), len(self.imgs_B))
        elif self.mode == 'unetTrain':
            return len(self.imgs_B)
        else:
            return len(self.testImgs)

def map_label(arr):
    arr = arr.copy()
    ignore_label = 255
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
    return arr





def convert_label(arr):
    arr = arr.copy()
    ignore_label = 255
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
    # copy = arr.copy()
    # for k, v in label_mapping.items():
    #     arr[copy == k] = v
    label = np.zeros((34, arr.shape[0], arr.shape[1]))
    for i in range(34):
        label[i, :, :][arr == i] = 1
    return torch.tensor(label, dtype=torch.float, requires_grad=False)


