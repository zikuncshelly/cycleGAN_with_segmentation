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


# apath = '/scratch/zikunc/cygan/ds_small/trainA'
# alabelpath = '/scratch/zikunc/cygan/ds_small/trainA_labels'

# domainA = sorted([f for f in os.listdir(apath) if f.endswith('.jpg')])
# domainA_labels = sorted([f for f in os.listdir(alabelpath) if f.endswith('.jpg')])
# idxs = random.sample(range(len(domainA)), 300)
# cityTestPath = '/scratch/zikunc/cygan_/testCITY'

# for idx in idxs:
# 	copy(os.path.join(apath, domainA[idx]),os.path.join(cityTestPath, 'x', domainA[idx]))
# 	copy(os.path.join(alabelpath, domainA_labels[idx]),os.path.join(cityTestPath, 'gt', domainA_labels[idx]))





# bpath = '/scratch/zikunc/cygan/ds_small/trainB'
# blabelpath = '/scratch/zikunc/cygan/ds_small/trainB_labels'

# domainB = sorted([f for f in os.listdir(bpath) if f.endswith('.jpg')])
# domainB_labels = sorted([f for f in os.listdir(blabelpath) if f.endswith('.png')])
# idxs = random.sample(range(len(domainB)), 300)
# gtaTestPath = '/scratch/zikunc/cygan_/testGTA'

# for idx in idxs:
# 	copy(os.path.join(bpath, domainB[idx]),os.path.join(gtaTestPath, 'x', domainB[idx]))
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
    with open('mapping.pkl', 'rb') as f:
        mapping = pickle.load(f)
    with torch.no_grad():
        total_iou = []
        for i, batch in enumerate(testloader):
            name, toTest, labels, ori_labels = batch
            #segmentation
            # pred_label = segmen_A(toTest)
            for idx in range(args.batchSize):
                sum_iou = []
                # pred = pred_label[idx].cpu().numpy()
                label = labels.cpu().numpy()[idx]
                ori_label = ori_labels.cpu().numpy()[idx]
                img = np.zeros((label.shape[0],label.shape[1],3))
                # if len(list(mapping.keys())) < 34:
                #     for i in range(label.shape[0]):
                #         for j in range(label.shape[1]):
                #             val = ori_label[i,j]
                #             mapping[val] = (label[i,j][0],label[i,j][1],label[i,j][2])
                prediction = np.zeros((label.shape[0],label.shape[1]))
                for i in range(label.shape[0]):
                    for j in range(label.shape[1]): 
                        # prediction[i,j] = np.argmax(pred[:,i,j])
                        c = ori_label[i,j]
                        if c in list(mapping.keys()):
                            color = mapping[c]
                            img[i,j,0] = color[0]
                            img[i,j,1] = color[1]
                            img[i,j,2] = color[2]


                img = Image.fromarray(img.astype('uint8'))
                img.save(os.path.join(args.out_dir,name[idx].replace('jpg','png')))
                return
        #         prediction = prediction.reshape(prediction.shape[0],-1)
        #         ori_label = ori_label.reshape(ori_label.shape[0],-1)
        #         intersection = (np.logical_and(prediction,ori_label)).sum()
        #         union = (np.logical_or(prediction,ori_label)).sum()
        #         total_iou.append((intersection+1e-6)/(union+1e-6))
        # print(sum(total_iou)/len(total_iou))
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


