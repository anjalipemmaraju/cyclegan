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
from tqdm import tqdm

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
        self.cycle_criterion = torch.nn.L1Loss()
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
        self.pool_size = 50
        self.pool_A = list()
        self.pool_B = list()




    ''' Takes real domain images and forwards them through the generators
    i.e. A -> B -> A and B -> A -> B
    '''
    def forward(self, realA, realB):
        self.realA = realA
        self.realB = realB
        # generate fake image B from real image A
        self.fakeB = self.genAB(self.realA)
        # reconstruct image A from fake image B
        self.recA = self.genBA(self.fakeB)
        # generate fake image A from real image B
        self.fakeA = self.genBA(self.realB)
        # reconstruct image B from fake image A
        self.recB = self.genAB(self.fakeA)

        # populate image pools
        if len(self.pool_B) < self.pool_size:
            self.pool_B.append(self.fakeB.clone().detach())
        else:
            idx = np.random.randint(self.pool_size)
            self.pool_B[idx] = self.fakeB.clone().detach()

        if len(self.pool_A) < self.pool_size:
            self.pool_A.append(self.fakeA.clone().detach())
        else:
            idx = np.random.randint(self.pool_size)
            self.pool_A[idx] = self.fakeA.clone().detach()
        
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
        idx = np.random.randint(len(self.pool_A))
        self.discA_loss = self.disc_backward(self.discA, self.realA, self.pool_A[idx])
        return self.discA_loss.item()

    def discB_backward(self):
        idx = np.random.randint(len(self.pool_B))
        self.discB_loss = self.disc_backward(self.discB, self.realB, self.pool_B[idx])
        return self.discB_loss.item()

    ''' Computes two forms of loss for each of the generators.
    Computes cycle (reconstruction) loss and adversarial loss
    i.e. how well the generator tricks the discriminator
    '''
    def gen_backward(self):
        # loss identity A and loss identity B?
        discB_pred = self.discB(self.fakeB)
        tt = torch.ones_like(discB_pred)
        tqdm.write(f'comp =' {tt})
        self.genAB_loss = self.criterion(torch.ones_like(discB_pred), discB_pred)
        discA_pred = self.discA(self.fakeA)
        tt = torch.ones_like(discA_pred)
        tqdm.write(f'comp =' {tt})
        self.genBA_loss = self.criterion(torch.ones_like(discA_pred), discA_pred)
        self.recA_loss = self.cycle_criterion(self.recA, self.realA) * self.lambda_A
        self.recB_loss = self.cycle_criterion(self.recB, self.realB) * self.lambda_B
        tqdm.write(f'recA loss = {self.recA_loss:.2f} \t recB loss = {self.recB_loss:.2f}')
        self.gen_loss = self.genAB_loss + self.genBA_loss + self.recA_loss + self.recB_loss
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
