import glob
import random
import os
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
import numpy as np

class ImgDataset(Dataset):
    def __init__(self, imgpath, transforms_=None, unaligned=True, mode='train', device='cuda'):
        self.imgs_A = sorted(glob.glob(os.path.join(imgpath, '%sA' % mode, '*.jpg')))
        self.imgs_B = sorted(glob.glob(os.path.join(imgpath, '%sB' % mode, '*.jpg')))
        self.transform = transforms.Compose(transforms_)
        self.unaligned = unaligned
        self.device = device

    def __getitem__(self, index):
        item_A = self.transform(Image.open(self.imgs_A[index % len(self.imgs_A)]))
        if self.unaligned:
            item_B = self.transform(Image.open(self.imgs_B[random.randint(0, len(self.imgs_B) - 1)]))
        else:
            item_B = self.transform(Image.open(self.imgs_B[index % len(self.imgs_B)]))
        return {'A': item_A.to(self.device), 'B': item_B.to(self.device)}

    def __len__(self):
        return max(len(self.imgs_A),len(self.imgs_B))
