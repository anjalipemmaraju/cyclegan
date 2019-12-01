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
from gen import Generator
from skimage.transform import rescale

use_cuda = torch.cuda.is_available()
device = torch.device('cuda:0' if use_cuda else 'cpu')

def train(trainA, trainB, num_epochs=1, num_examples=500):
    model = CycleGAN().to(device)
    gen_losses = list()
    discA_losses = list()
    discB_losses = list()
    for epoch in tqdm(range(num_epochs)):
        np.random.shuffle(trainA)
        np.random.shuffle(trainB)
        for idx in tqdm(range(num_examples)):
        #for idxB in range(100):
            realA = trainA[idx]
            realA = torch.FloatTensor(realA).reshape(1,3,realA.shape[0], realA.shape[1]).to(device)
            realB = trainB[idx]
            realB = torch.FloatTensor(realB).reshape(1,3,realB.shape[0], realB.shape[1]).to(device)
            gen_loss, discA_loss, discB_loss = model.optimize_parameters(realA, realB)
            if idxB %100 == 0:
                tqdm.write(f'gen_loss = {gen_loss:.2f} \t discA_loss = {discA_loss:.2f} \t discB_loss = {discB_loss:.2f}')
    torch.save(model.genAB.state_dict(), 'models/gen_AB.pt')
    torch.save(model.genBA.state_dict(), 'models/gen_BA.pt')

def test(testA, testB):
    genAB = Generator().to(device)
    genAB.load_state_dict(torch.load('models/gen_AB.pt'))
    genAB.eval()
    genBA = Generator().to(device)
    genBA.load_state_dict(torch.load('models/gen_BA.pt'))
    genBA.eval()
    fig,ax = plt.subplots(nrows=2,ncols=2)
    input_A = torch.FloatTensor(testA[0]).reshape(1,3,testA[0].shape[0], testA[0].shape[1]).to(device)
    fake_B = genAB(input_A).detach().cpu().numpy()
    ax[0,1].imshow(fake_B[0].transpose((1,2,0)))
    ax[0,0].imshow(testA[0])
    plt.show()

    

            
if __name__ == '__main__':
    data_path = 'data/summer2winter_yosemite'
    trainA_paths = os.listdir(os.path.join(data_path, 'trainA'))
    trainA_paths = [os.path.join(data_path, 'trainA', path) for path in trainA_paths]
    trainB_paths = os.listdir(os.path.join(data_path, 'trainB'))
    trainB_paths = [os.path.join(data_path, 'trainB', path) for path in trainB_paths]
    trainA = [None] * len(trainA_paths)
    trainB = [None] * len(trainB_paths)
    for idxB in tqdm(range(len(trainB_paths))):
        pathB = trainB_paths[idxB]
        realB = plt.imread(pathB)
        realB = rescale(realB, scale=0.5)
        trainB[idxB] = realB
    for idxA in tqdm(range(len(trainA_paths))):
        pathA = trainA_paths[idxA]
        realA = plt.imread(pathA)
        realA = rescale(realA, scale=0.5)
        trainA[idxA] = realA
    np.save('summer2wintertrainA.npy', trainA)
    np.save('summer2wintertrainB.npy', trainB)
    trainA = np.load('summer2wintertrainA.npy')
    trainB = np.load('summer2wintertrainB.npy')

    train(trainA, trainB, num_epochs=1)
    test(trainA, trainB)




