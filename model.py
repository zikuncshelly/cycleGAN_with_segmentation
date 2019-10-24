import torch.nn as nn



class ResBlock(nn.Module):
    def __init__(self, in_channel):
        super(ResBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channel, in_channel, 3),
            nn.InstanceNorm2d(in_channel),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channel, in_channel, 3),
            nn.InstanceNorm2d(in_channel),
        )

    def forward(self, input):
        out = self.conv(input)
        out += input

        return out


class Generator(nn.Module):
    def __init__(self, in_channel, out_channel, n_resblocks = 9):
        super(Generator, self).__init__()
        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(in_channel, 64, 7),
                 nn.InstanceNorm2d(64),
                 nn.ReLU(inplace=True)]
        in_c = 64
        out_c = 64*2
        for _ in range(2):
            model += [nn.Conv2d(in_c, out_c, 3, stride=2, padding=1),
                        nn.InstanceNorm2d(out_c),
                        nn.ReLU(inplace=True)]
            in_c = out_c
            out_c *= 2
        for _ in range(n_resblocks):
            model += [ResBlock(in_c)]
        out_c = in_c // 2
        for _ in range(2):
            model += [nn.ConvTranspose2d(in_c, out_c, 3, stride=2, padding=1, output_padding=1),
                        nn.InstanceNorm2d(out_c),
                        nn.ReLU(inplace=True)]
            in_c = out_c
            out_c = out_c //2
        model += [nn.ReflectionPad2d(3),
                  nn.Conv2d(in_c, out_channel, 7),
                  nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)


class Discriminator(nn.Module):
    def __init__(self, in_channel):
        super(Discriminator, self).__init__()
        model = [nn.Conv2d(in_channel, 64, 4, stride=2, padding=1),
                 nn.LeakyReLU(0.2, inplace=True),
                 nn.Conv2d(64, 128, 4, stride=2, padding=1),
                 nn.InstanceNorm2d(128),
                 nn.LeakyReLU(0.2, inplace=True),
                 nn.Conv2d(128, 256, 4, stride=2, padding=1),
                 nn.InstanceNorm2d(256),
                 nn.LeakyReLU(0.2, inplace=True),
                 nn.Conv2d(256, 512, 4, padding=1),
                 nn.InstanceNorm2d(512),
                 nn.LeakyReLU(0.2, inplace=True),
                 nn.Conv2d(512, 1, 4, padding=1)]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        x = self.model(x)
        return nn.functional.avg_pool2d(x, x.size()[2:]).view(x.size()[0], -1)