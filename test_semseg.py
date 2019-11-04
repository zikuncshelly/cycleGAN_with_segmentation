import os
from shutil import copy
import random

import argparse
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch
from torchvision import utils
from torch import nn
import os
import numpy as np
from model import Generator
from dataset import ImgDataset
from Unet import Unet
from utils import *
from PIL import Image
import pickle
palette = [ (  0,  0,  0),
(  0,  0,  0), 
(  0,  0,  0),
(  0,  0,  0), 
(  0,  0,  0), 
(111, 74,  0), 
( 81,  0, 81),
(128, 64,128),
(244, 35,232),
(250,170,160),
(230,150,140),
( 70, 70, 70),
(102,102,156),
(190,153,153),
(180,165,180),
(150,100,100),
(150,120, 90),
(153,153,153),
(153,153,153),
(250,170, 30),
(220,220,  0),
(107,142, 35),
(152,251,152),
( 70,130,180),
(220, 20, 60),
(255,  0,  0),
(  0,  0,142),
(  0,  0, 70),
(  0, 60,100),
(  0,  0, 90),
(  0,  0,110),
(  0, 80,100),
(  0,  0,230),
(119, 11, 32),
(  0,  0,142)]

# gta2city = {0:0, 1:5, 2:23, 3:7, 4:8, 5:9, 6:22, 7:21, 8:21, 9:11, 10:5, 11:4, 12:, 13:19, 14:20, 15:, 16:}

# apath = '/scratch/zikunc/cygan/ds_small/trainA'
# alabelpath = '/scratch/zikunc/cygan/ds_small/trainA_labels'

# domainA = sorted([f for f in os.listdir(apath) if f.endswith('.jpg')])
# domainA_labels = sorted([f for f in os.listdir(alabelpath) if f.endswith('.jpg')])
# idxs = random.sample(range(len(domainA)), 300)
# cityTestPath = '/scratch/zikunc/cygan_/testCITY'

# for idx in idxs:
# 	copy(os.path.join(apath, domainA[idx]),os.path.join(cityTestPath, 'x', domainA[idx]))
# 	copy(os.path.join(alabelpath, domainA_labels[idx]),os.path.join(cityTestPath, 'gt', domainA_labels[idx]))





# bpath = '/scratch/zikunc/cygan_/ds_fullcity/trainB'
# blabelpath = '/scratch/zikunc/cygan_/out'

# domainB = sorted([f for f in os.listdir(bpath) if f.endswith('.png')])
# domainB_labels = sorted([f for f in os.listdir(blabelpath) if f.endswith('.png')])
# idxs = random.sample(range(len(domainB)), 300)
# gtaTestPath = '/scratch/zikunc/cygan_/testGTA'

# for idx in idxs:
# 	copy(os.path.join(bpath, domainB_labels[idx]),os.path.join(gtaTestPath, 'x', domainB_labels[idx]))
# 	copy(os.path.join(blabelpath, domainB_labels[idx]),os.path.join(gtaTestPath, 'gt', domainB_labels[idx]))


def main(args):

    segmen_A = Unet(3, 34).to(args.device)

    if args.model_path is not None:
        segmen_path = os.path.join(args.model_path,'semsg.pt')

        with open(segmen_path, 'rb') as f:
            state_dict = torch.load(f)
            segmen_A.load_state_dict(state_dict)

    else:
        raise Exception('please specify model path!')

    segmen_A = nn.DataParallel(segmen_A)

    transforms_ = [ transforms.ToTensor(),
                    transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)) ]
    testloader = DataLoader(ImgDataset(args.image_path, transforms_=transforms_, mode='eval_unet'),
                            batch_size=args.batchSize, shuffle=False, num_workers=0)

    segmen_A.eval()

    with torch.no_grad():
        total_iou = []
        for i, batch in enumerate(testloader):
            name, toTest, labels = batch
            #segmentation
            pred_label = segmen_A(toTest)
            for idx in range(args.batchSize):
                pred = pred_label[idx].cpu().numpy()
                label = labels.cpu().numpy()[idx]
                img = np.zeros((label.shape[0],label.shape[1],3)).astype('uint8')
                original_img = np.zeros((label.shape[0],label.shape[1],3)).astype('uint8')
                prediction = np.zeros((label.shape[0],label.shape[1])).astype('uint8')
                for c in range(len(palette)):
                    indices = np.argmax(pred,axis=0)==c
                    prediction[indices]=c
                    img[indices] = palette[c]
                    original_img[label==c] = palette[c]
                original_img = Image.fromarray(original_img.astype('uint8'))
                original_img.save(os.path.join(args.out_dir,'original_'+name[idx].replace('jpg','png')))
                img = Image.fromarray(img.astype('uint8'))
                img.save(os.path.join(args.out_dir,name[idx].replace('jpg','png')))
                total_iou.append(IOU(prediction, label, 34))
        print(sum(total_iou)/len(total_iou))
    # f = open('mapping.pkl', 'wb')
    # pickle.dump(mapping, f)
    # f.close()

        # print(sum(total_iou)/len(total_iou))


            # for idx in range(len(name)):
            #     utils.save_image(torch.cat((toTest[idx].to(args.device), recovered[idx], transformed_[idx]),axis=1), os.path.join(args.out_dir, args.direction+'_'+name[idx].split('/')[-1]), normalize=True, range=(-1, 1))






if __name__ == '__main__':
    import os

    parser = argparse.ArgumentParser()

    parser.add_argument('--image_path', type=str, help='path to the test images')
    parser.add_argument('--model_path', type=str, help='path to the model checkpoint', default=None)
    parser.add_argument('--out_dir', type=str, help='output dir', default='./')
    parser.add_argument('--device', type=str, help='set the device', default='cuda')
    parser.add_argument('--in_channel', type=int, default=3, help='number of channels of input data')
    parser.add_argument('--out_channel', type=int, default=3, help='number of channels of output data')
    parser.add_argument('--batchSize', type=int, default=1, help='size of the batches')
    parser.add_argument('--direction', type=str, default='AB', help='direction of domain transfer')

    args = parser.parse_args()
    print(args)

    if torch.cuda.is_available() and args.device != 'cuda':
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    args.device = torch.device(args.device)
    main(args)


