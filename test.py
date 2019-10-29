import argparse
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch
from torchvision import utils
from torch import nn
import os

from model import Generator
from dataset import ImgDataset
from Unet import Unet


def main(args):

    G = Generator(args.in_channel, args.out_channel).to(args.device)
    segmen_A = Unet(3, 20).to(args.device)

    if args.model_path is not None:
        AB_path = os.path.join(args.model_path,'ab.pt')
        BA_path = os.path.join(args.model_path,'ba.pt')
        segmen_path = os.path.join(args.model_path,'semsg.pt')

        if args.direction == 'AB':
            with open(AB_path, 'rb') as f:
                state_dict = torch.load(f)
                G.load_state_dict(state_dict)
        elif args.direction == 'BA':
            with open(BA_path, 'rb') as f:
                state_dict = torch.load(f)
                G.load_state_dict(state_dict)
        else:
            raise Exception('direction has to be BA OR AB!')

        with open(segmen_path, 'rb') as f:
            state_dict = torch.load(f)
            segmen_A.load_state_dict(state_dict)

    else:
        raise Exception('please specify model path!')

    G = nn.DataParallel(G)
    segmen_A = nn.DataParallel(segmen_A)

    transforms_ = [ transforms.ToTensor(),
                    transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)) ]
    testloader = DataLoader(ImgDataset(args.image_path, transforms_=transforms_, mode='test'),
                            batch_size=args.batchSize, shuffle=False, num_workers=0)

    G.eval()
    segmen_A.eval()
    with torch.no_grad():
        for i, batch in enumerate(testloader):
            name, toTest = batch
            #segmentation
            pred_label = segmen_A(toTest)
            transformed_ = G(toTest)
            for idx in range(len(name)):
                utils.save_image(transformed_[idx], os.path.join(args.out_dir, name[idx].split('/')[-1]), normalize=True, range=(-1, 1))

                utils.save_image(torch.cat((toTest[idx].to(args.device), transformed_[idx]),axis=1), os.path.join(args.out_dir, args.direction+'_'+name[idx].split('/')[-1]), normalize=True, range=(-1, 1))






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
