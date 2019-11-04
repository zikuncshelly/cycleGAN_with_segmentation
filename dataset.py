import glob
import random
import os
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
import torch


class ImgDataset(Dataset):
    def __init__(self, imgpath, transforms_=None, unaligned=True, real_percentage=0, mode='train', device='cuda'):
        self.mode = mode
        if mode == 'train':
            self.imgs_A = sorted(glob.glob(os.path.join(imgpath, '%sA' % mode, '*.jpg')))
            self.imgs_B = sorted(glob.glob(os.path.join(imgpath, '%sB' % mode, '*.jpg')))
            self.imgs_B_labels = sorted(glob.glob(os.path.join(imgpath, '%sB_labels' % mode, '*.png')))
            if real_percentage > 0:
                self.imgs_A_labels = sorted(glob.glob(os.path.join(imgpath, 'trainA_labels','*.png')))

        elif mode == 'unetTrain':
            self.imgs_B = sorted(glob.glob(os.path.join(imgpath, 'trainB', '*.jpg')))
            self.imgs_B_labels = sorted(glob.glob(os.path.join(imgpath, 'trainB_labels', '*.png')))
            if real_percentage > 0:
                self.imgs_A = sorted(glob.glob(os.path.join(imgpath, 'trainA','*.jpg')))
                length = int(len(self.imgs_A)*real_percentage)
                self.imgs_A = self.imgs_A[:length]
                self.imgs_A_labels = sorted(glob.glob(os.path.join(imgpath, 'trainA_labels','*.png')))
                self.imgs_A_labels = self.imgs_A_labels[:length]
        elif mode == 'test':
            self.testImgs = sorted(glob.glob(os.path.join(imgpath, '*.png')))
        elif mode == 'eval_unet':
            self.testImgs = sorted(glob.glob(os.path.join(imgpath, 'x', '*.png')))
            self.testLabels = sorted(glob.glob(os.path.join(imgpath, 'gt', '*.png')))
        else:
            raise Exception('mode must be train or test!')
        self.real_percentage = real_percentage
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
            if self.real_percentage == 0 :
                return {'B': item_B.to(self.device), 'B_label' : B_label.to(self.device)}
            else:
                item_A = self.transform(Image.open(self.imgs_A[index % len(self.imgs_A)]))
                A_label = Image.open(self.imgs_A_labels[index % len(self.imgs_A)])
                A_label = np.asarray(A_label)
                A_label = convert_label(A_label)
                return {'B': item_B.to(self.device), 'B_label' : B_label.to(self.device), 'A': item_A.to(self.device), 'A_label' : A_label.to(self.device)}
        elif self.mode == 'eval_unet':
            img = self.transform(Image.open(self.testImgs[index]))
            label = np.asarray(Image.open(self.testLabels[index]))
            return self.testImgs[index].split('/')[-1], img, label
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


def convert_label(arr):
    arr = arr.copy()
    label = np.zeros((34, arr.shape[0], arr.shape[1]))
    for i in range(34):
        label[i, :, :][arr == i] = 1
    return torch.tensor(label, dtype=torch.float, requires_grad=False)


