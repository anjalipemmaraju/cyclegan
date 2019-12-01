import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.datasets
import numpy as np
import scipy.io
import matplotlib.pyplot as plt
import copy
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches
# what is instance normalization?
# what is patch gan?

class Generator(nn.Module):
    # def __init__(self):
    #     super(Generator, self).__init__()
    #     self.conv1 = nn.Conv2d(3, 32, 9, 4)
    #     self.bn1 = nn.BatchNorm2d(32, affine=True)
    #     self.conv2 = nn.Conv2d(32, 64, 3, 1)
    #     self.bn2 = nn.BatchNorm2d(64, affine=True)
    #     self.conv3 = nn.Conv2d(64, 128, 3, 1)
    #     self.bn3 = nn.BatchNorm2d(128, affine=True)

    #     self.res11 = nn.Conv2d(128, 128, 3, padding=1) 
    #     self.resbn1 = nn.BatchNorm2d(128, affine=True)       
    #     self.res12 = nn.Conv2d(128, 128, 3, padding=1)
    #     self.res21 = nn.Conv2d(128, 128, 3, padding=1)
    #     self.resbn2 = nn.BatchNorm2d(128, affine=True)       
    #     self.res22 = nn.Conv2d(128, 128, 3, padding=1)
    #     self.res31 = nn.Conv2d(128, 128, 3, padding=1)
    #     self.resbn3 = nn.BatchNorm2d(128, affine=True)       
    #     self.res32 = nn.Conv2d(128, 128, 3, padding=1)
    #     self.res41 = nn.Conv2d(128, 128, 3, padding=1)
    #     self.resbn4 = nn.BatchNorm2d(128, affine=True)       
    #     self.res42 = nn.Conv2d(128, 128, 3, padding=1)
    #     self.res51 = nn.Conv2d(128, 128, 3, padding=1)
    #     self.resbn5 = nn.BatchNorm2d(128, affine=True)       
    #     self.res52 = nn.Conv2d(128, 128, 3, padding=1)
    #     self.res61 = nn.Conv2d(128, 128, 3, padding=1)   
    #     self.resbn6 = nn.BatchNorm2d(128, affine=True)       
    #     self.res62 = nn.Conv2d(128, 128, 3, padding=1)

    #     self.conv4 = nn.ConvTranspose2d(128, 64, 3, 1)
    #     self.bn4 = nn.BatchNorm2d(64, affine=True)
    #     self.conv5 = nn.ConvTranspose2d(64, 32, 3, 1)
    #     self.bn5 = nn.BatchNorm2d(32, affine=True)
    #     self.conv6 = nn.ConvTranspose2d(32, 3, 9, 4)
    #     self.bn6 = nn.BatchNorm2d(3, affine=True)

    # def forward(self, x):
    #     print(x.shape)
    #     x = self.conv1(x)
    #     x = self.bn1(x)
    #     x = F.relu(x)
    #     x = self.conv2(x)
    #     x = self.bn2(x)
    #     x = F.relu(x)
    #     x = self.conv3(x)
    #     x = self.bn3(x)
    #     x = F.relu(x)

    #     x1 = self.res11(x)
    #     x1 = self.resbn1(x1)
    #     x = self.res12(x + x1)

    #     x1 = self.res21(x)
    #     x1 = self.resbn2(x1)
    #     x = self.res22(x + x1)
        
    #     x1 = self.res31(x)
    #     x1 = self.resbn3(x1)
    #     x = self.res32(x + x1)
        
    #     x1 = self.res41(x)
    #     x1 = self.resbn4(x1)
    #     x = self.res42(x + x1)
        
    #     x1 = self.res51(x)
    #     x1 = self.resbn5(x1)
    #     x = self.res52(x + x1)
        
    #     x1 = self.res61(x)
    #     x1 = self.resbn6(x1)
    #     x = self.res62(x + x1)

    #     x = self.conv4(x)
    #     x = self.bn4(x)
    #     x = F.relu(x)

    #     x = self.conv5(x)
    #     x = self.bn5(x)
    #     x = F.relu(x)

    #     x = self.conv6(x)
    #     x = self.bn6(x)
    #     x = 255*F.tanh(x)
    #     return x

    def __init__(self):
        super(Generator, self).__init__()
        ngf = 64
        n_blocks=6
        padding_type='reflect'
        use_bias = True
        norm_layer=nn.BatchNorm2d
        use_dropout=False
        model = [nn.ReflectionPad2d(3),
                    nn.Conv2d(3, 64, kernel_size=7, padding=0, bias=True),
                    norm_layer(ngf),
                    nn.ReLU(True)]

        n_downsampling = 2
        for i in range(n_downsampling):  # add downsampling layers
            mult = 2 ** i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                    norm_layer(ngf * mult * 2),
                    nn.ReLU(True)]

        mult = 2 ** n_downsampling
        for i in range(n_blocks):       # add ResNet blocks

            model += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]

        for i in range(n_downsampling):  # add upsampling layers
            mult = 2 ** (n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                        kernel_size=3, stride=2,
                                        padding=1, output_padding=1,
                                        bias=use_bias),
                    norm_layer(int(ngf * mult / 2)),
                    nn.ReLU(True)]
        model += [nn.ReflectionPad2d(3)]
        output_nc = 3
        model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        model += [nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        """Standard forward"""
        return self.model(input)

class ResnetBlock(nn.Module):
    """Define a Resnet block"""

    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """Initialize the Resnet block
        A resnet block is a conv block with skip connections
        We construct a conv block with build_conv_block function,
        and implement skip connections in <forward> function.
        Original Resnet paper: https://arxiv.org/pdf/1512.03385.pdf
        """
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """Construct a convolutional block.
        Parameters:
            dim (int)           -- the number of channels in the conv layer.
            padding_type (str)  -- the name of padding layer: reflect | replicate | zero
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
            use_bias (bool)     -- if the conv layer uses bias or not
        Returns a conv block (with a conv layer, a normalization layer, and a non-linearity layer (ReLU))
        """
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim), nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        """Forward function (with skip connections)"""
        out = x + self.conv_block(x)  # add skip connections
        return out
