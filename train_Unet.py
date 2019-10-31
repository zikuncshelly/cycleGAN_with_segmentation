import argparse
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch
from datetime import datetime
from torch import nn


from utils import Logger
from dataset import ImgDataset
from Unet import Unet


def main(args):
    current_time = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
    os.makedirs(os.path.join(args.out_dir, 'models',args.model_name+'_'+current_time))
    os.makedirs(os.path.join(args.out_dir, 'logs', args.model_name+'_'+current_time))

    segmen_B = Unet(3, 34).to(args.device)

    if args.model_path is not None:
        segmen_path = os.join.path(args.model_path,'semsg.pt')

        with open(segmen_path, 'rb') as f:
            state_dict = torch.load(f)
            segmen_B.load_state_dict(state_dict)

    segmen_B = nn.DataParallel(segmen_B)

    criterion_segmen = torch.nn.BCELoss()

    optimizer_segmen_B = torch.optim.Adam(segmen_B.parameters(), lr=args.lr, betas=(0.5, 0.999))

    transforms_ = [ transforms.ToTensor(),
                    transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)) ]
    dataloader = DataLoader(ImgDataset(args.dataset_path, transforms_=transforms_, mode='unetTrain', unaligned=False, device=args.device),
                            batch_size=args.batchSize, shuffle=True, num_workers=0)
    logger = Logger(args.n_epochs, len(dataloader))
    segmen_B.train()

    for epoch in range(args.epoch, args.n_epochs):
        for i, batch in enumerate(dataloader):
            real_B = batch['B'].clone()
            B_label = batch['B_label'].clone()
            optimizer_segmen_B.zero_grad()
            #segmen loss
            pred_Blabel = segmen_B(real_B)
            loss_segmen_B = criterion_segmen(pred_Blabel, B_label)
            loss_segmen_B.backward()
            optimizer_segmen_B.step()

            logger.log({'loss_segmen': loss_segmen_B},
                       out_dir=os.path.join(args.out_dir, 'logs', args.model_name+'_'+current_time+'/'+str(epoch)))

        if (epoch+1)%args.save_per_epochs == 0:
            torch.save(segmen_B.module.state_dict(),os.path.join(args.out_dir, 'models',args.model_name+'_'+current_time, 'semsg.pt'))


if __name__ == '__main__':
    import os

    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset_path', type=str, help='path to the dataset')
    parser.add_argument('--model_path', type=str, help='path to the model checkpoint', default=None)
    parser.add_argument('--out_dir', type=str, help='output dir', default='./')
    parser.add_argument('--device', type=str, help='set the device', default='cuda')
    parser.add_argument('--model_name', type=str, help='name of model', default='cygan')

    parser.add_argument('--epoch', type=int, default=0, help='starting epoch')
    parser.add_argument('--n_epochs', type=int, default=200, help='number of epochs of training')
    parser.add_argument('--save_per_epochs', type=int, default=5, help='starting epoch')
    parser.add_argument('--batchSize', type=int, default=1, help='size of the batches')
    parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate')
    parser.add_argument('--decay_epoch', type=int, default=5, help='epoch to start decaying lr')
    # parser.add_argument('--size', type=int, default=256, help='size of the data crop (squared assumed)')
    parser.add_argument('--in_channel', type=int, default=3, help='number of channels of input data')
    parser.add_argument('--out_channel', type=int, default=3, help='number of channels of output data')
    parser.add_argument('--n_cpu', type=int, default=1, help='number of cpu threads to use during batch generation')
    args = parser.parse_args()
    print(args)

    if torch.cuda.is_available() and args.device != 'cuda':
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

    if not os.path.exists(os.path.join(args.out_dir, 'logs')):
        os.makedirs(os.path.join(args.out_dir, 'logs'))
    if not os.path.exists(os.path.join(args.out_dir, 'models')):
        os.makedirs(os.path.join(args.out_dir, 'models'))

    args.device = torch.device(args.device)
    main(args)
