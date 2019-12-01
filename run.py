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

from tqdm import tqdm
from cyclegan import CycleGAN

use_cuda = torch.cuda.is_available()
device = torch.device('cuda:0' if use_cuda else 'cpu')

def train(trainA, trainB, num_epochs=200):
    model = CycleGAN().to(device)
    gen_losses = list()
    discA_losses = list()
    discB_losses = list()
    for epoch in tqdm(range(num_epochs)):
        for idxB in range(len(trainB)):
            idxA = idxB % len(trainA)
            if idxA == 0:
                np.random.shuffle(trainA)
            realA = trainA[idxA]
            realA = torch.FloatTensor(realA).reshape(1,3,realA.shape[0], realA.shape[0]).to(device)
            realB = trainB[idxB]
            realB = torch.FloatTensor(realB).reshape(1,3,realB.shape[0], realB.shape[0]).to(device)
            gen_loss, discA_loss, discB_loss = model.optimize_parameters(realA, realB)
            tqdm.write(f'gen_loss = {gen_loss:.2f} \t discA_loss = {discA_loss:.2f} \t discB_loss = {discB_loss:.2f}')
            
if __name__ == '__main__':
    '''
    data_path = 'data/vangogh2photo'
    trainA_paths = os.listdir(os.path.join(data_path, 'trainA'))
    trainA_paths = [os.path.join(data_path, 'trainA', path) for path in trainA_paths]
    trainB_paths = os.listdir(os.path.join(data_path, 'trainB'))
    trainB_paths = [os.path.join(data_path, 'trainB', path) for path in trainB_paths]
    trainA = [None] * len(trainA_paths)
    trainB = [None] * len(trainB_paths)
    for idxB in tqdm(range(len(trainB_paths))):
        pathB = trainB_paths[idxB]
        realB = plt.imread(pathB)
        trainB[idxB] = realB
    for idxA in tqdm(range(len(trainA_paths))):
        pathA = trainA_paths[idxA]
        realA = plt.imread(pathA)
        trainA[idxA] = realA
    np.save('vanGoghtrainA.npy', trainA)
    np.save('vanGoghtrainB.npy', trainB)
    '''
    trainA = np.load('vanGoghtrainA.npy')
    trainB = np.load('vanGoghtrainB.npy')

    train(trainA, trainB, num_epochs=1)




