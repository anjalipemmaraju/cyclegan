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
import itertools

from gen import Generator
from disc import Discriminator

class CycleGAN(nn.Module):
    def __init__(self):
        super(CycleGAN, self).__init__()
        self.genAB = Generator()
        self.genBA = Generator()
        self.discA = Discriminator()
        self.discB = Discriminator()
        self.criterion = torch.nn.MSELoss()
        self.lr = 2e-4
        self.beta1 = 0.5
        self.gen_optimizer = torch.optim.Adam(itertools.chain(self.genAB.parameters(), self.genBA.parameters()),
                                              lr=self.lr,
                                              betas=(self.beta1, 0.999))
        self.disc_optimizer = torch.optim.Adam(itertools.chain(self.discA.parameters(), self.discB.parameters()),
                                               lr=self.lr,
                                               betas=(self.beta1, 0.999))
        self.lambda_A = 10
        self.lambda_B = 10

    ''' Takes real domain images and forwards them through the generators
    i.e. A -> B -> A and B -> A -> B
    '''
    def forward(self, realA, realB):
        self.realA = realA
        self.realB = realB
        # generate fake image B from real image A
        print(f'A->B')
        self.fakeB = self.genAB(realA)
        # reconstruct image A from fake image B
        print(f'rec A->B')
        self.recA = self.genBA(self.fakeB)
        # generate fake image A from real image B
        print(f'B->A')
        self.fakeA = self.genBA(realB)
        # reconstruct image B from fake image A
        print(f'rec B->A')
        self.recB = self.genAB(self.fakeA)

    ''' Backpropagates gradients for discriminators given
    a discriminator, real domain images, and fake domain images
    '''
    def disc_backward(self, disc, real, fake):
        # discriminator should predict real images as 1
        pred_real = disc(real)
        disc_real_loss = self.criterion(torch.ones_like(pred_real), pred_real)
        # discriminator should predict fake images as 0
        pred_fake = disc(fake.detach())
        disc_fake_loss = self.criterion(torch.zeros_like(pred_fake), pred_fake)
        # propagate backwards
        disc_loss = 0.5 * (disc_real_loss + disc_fake_loss)
        disc_loss.backward()
        return disc_loss

    def discA_backward(self):
        self.discA_loss = self.disc_backward(disc, self.realA, self.realA)
        return self.discA_loss.item()

    def discB_backward(self):
        self.discB_loss = self.disc_backward(disc, self.realB, self.realB)
        return self.discB_loss.item()

    ''' Computes two forms of loss for each of the generators.
    Computes cycle (reconstruction) loss and adversarial loss
    i.e. how well the generator tricks the discriminator
    '''
    def gen_backward(self):
        # loss identity A and loss identity B?
        discB_pred = self.discB(self.fakeB)
        self.genAB_loss = self.criterion(torch.ones_like(discB_pred), discB_pred)
        discA_pred = self.discA(self.fakeA)
        self.genBA_loss = self.criterion(torch.ones_like(discA_pred), discA_pred)
        self.recA_loss = torch.nn.L1Loss(self.recA, self.realA) * self.lambda_A
        self.recB_loss = torch.nn.L1Loss(self.recB, self.realB) * self.lambda_B
        self.gen_loss = self.genAB_loss + self.genBA_loss + self.lambda_A*self.recA_loss + self.lambda_B*self.recB_loss
        self.gen_loss.backward()
        return self.gen_loss.item()

    ''' Sets requires_grad flag for the parameters for the input networks
    to either True or False
    '''
    def set_requires_grad(self, nets, requires_grad):
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

    ''' Optimizes discriminator and generator parameters
    1. Forward A->B->A and B->A->B
    2. Backpropagate gradients to generator parameters
    3. Backpropagate gradients to discriminator parameters
    '''
    def optimize_parameters(self, realA, realB):
        losses = list()
        # forward
        self.forward(realA, realB)
        # backprop generator gradients (don't do discriminator gradients)
        self.set_requires_grad([self.discA, self.discB], requires_grad=False)
        self.gen_optimizer.zero_grad()
        losses.append(self.gen_backward())
        self.gen_optimizer.step()
        # backprop discriminator gradients (no generator upgrades beecause
        # they are detached in the disc backwards method
        self.set_requires_grad([self.discA, self.discB], requires_grad=True)
        self.disc_optimizer.zero_grad()
        losses.append(self.discA_backward())
        losses.append(self.discB_backward())
        self.disc_optimizer.step()
        return losses
