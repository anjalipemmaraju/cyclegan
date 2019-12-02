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

def train(model, trainA, trainB, start_epoch, num_epochs=5):
    gen_losses = list()
    discA_losses = list()
    discB_losses = list()
    end_idx = max(len(trainA), len(trainB))
    model.populate_image_pools()
    for epoch in tqdm(range(start_epoch, num_epochs)):
        np.random.shuffle(trainA)
        np.random.shuffle(trainB)
        for idx in tqdm(range(end_idx)):
        #for idxB in range(100):
            realA = trainA[idx%len(trainA)]
            realA = torch.FloatTensor(realA).reshape(1,3,realA.shape[0], realA.shape[1]).to(device)
            realB = trainB[idx%len(trainB)]
            realB = torch.FloatTensor(realB).reshape(1,3,realB.shape[0], realB.shape[1]).to(device)
            gen_loss, discA_loss, discB_loss = model.optimize_parameters(realA, realB)
            if idx %100 == 0:
                tqdm.write(f'gen_loss = {gen_loss:.2f} \t discA_loss = {discA_loss:.2f} \t discB_loss = {discB_loss:.2f}')
        torch.save(model.genAB.state_dict(), f'models/gen_AB_{epoch}.pt')
        torch.save(model.genBA.state_dict(), f'models/gen_BA_{epoch}.pt')

def test(testA, testB, epoch):
    np.random.shuffle(testA)
    np.random.shuffle(testB)
    genAB = Generator().to(device)
    genAB.load_state_dict(torch.load(f'models/gen_AB_{epoch}.pt'))
    genAB.eval()
    genBA = Generator().to(device)
    genBA.load_state_dict(torch.load(f'models/gen_BA_{epoch}.pt'))
    genBA.eval()
    num_test = 4
    fig,ax = plt.subplots(nrows=num_test,ncols=4)
    for i in range(num_test):
        input_A = torch.FloatTensor(testA[i]).reshape(1,3,testA[i].shape[0], testA[i].shape[1]).to(device)
        fake_B = genAB(input_A).detach().cpu().numpy()
        fake_B = fake_B[0].transpose((1,2,0))
        ax[i,0].imshow(testA[i])
        ax[i,1].imshow((fake_B*255).astype(np.uint8))
        input_B = torch.FloatTensor(testB[i]).reshape(1,3,testB[i].shape[0], testB[i].shape[1]).to(device)
        fake_A = genBA(input_B).detach().cpu().numpy()
        fake_A = fake_A[0].transpose((1,2,0))
        ax[i,2].imshow(testB[i])
        ax[i,3].imshow((fake_A*255).astype(np.uint8))
    plt.show()

    

            
if __name__ == '__main__':
    '''
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
    '''
    trainA = np.load('summer2wintertrainA.npy')
    trainB = np.load('summer2wintertrainB.npy')
    model = CycleGAN().to(device)
    #model.genAB.load_state_dict(torch.load('models/gen_AB_{epoch}.pt'))
    #model.genBA.load_state_dict(torch.load('models/gen_BA_{epoch}.pt'))

    train(model, trainA, trainB, start_epoch=0, num_epochs=10)
    #test(trainA, trainB)




