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

''' Runs training loop for cyclegan model given trainA and trainB data between -1 and 1 for the given number of epochs.
Saves the generator models after every epoch.
Input:
	model (cycle gan network)
	trainA (list of images): list of images in domain A from with pixel values between -1 and 1
	trainB (list of images): list of images in domain B from with pixel values between -1 and 1
	start_epoch (int): epoch to start on
	num_epochs (int): total number of epochs
Output:
	None
'''
def train(model, trainA, trainB, start_epoch, num_epochs=5):
	# initialize lists for each type of loss
	gen_losses = list()
	discA_losses = list()
	discB_losses = list()

	# calculate the size of the smaller train set
	end_idx = max(len(trainA), len(trainB))
	for epoch in tqdm(range(start_epoch, num_epochs)):
		# shuffle trainA and trainB so that the images are not matched together
		np.random.shuffle(trainA)
		np.random.shuffle(trainB)
		total_loss = 0

		# train on the smaller number of train images 
		for idx in tqdm(range(end_idx)):
			# pick an image from each domain and turn into a torch tensor
			realA = trainA[idx%len(trainA)]
			realA = torch.FloatTensor(realA).reshape(1,3,realA.shape[0], realA.shape[1]).to(device)
			realB = trainB[idx%len(trainB)]
			realB = torch.FloatTensor(realB).reshape(1,3,realB.shape[0], realB.shape[1]).to(device)

			# run the network
			gen_loss, discA_loss, discB_loss = model.optimize_parameters(realA, realB)
			total_loss += gen_loss
			if idx %100 == 0:
				avg_loss = total_loss / (idx+1)
				tqdm.write(f'gen_loss = {avg_loss:.2f} \t discA_loss = {discA_loss:.2f} \t discB_loss = {discB_loss:.2f}')
		tqdm.write(f'gen_loss = {total_loss/end_idx:.2f} \t discA_loss = {discA_loss:.2f} \t discB_loss = {discB_loss:.2f}')
		torch.save(model.genAB.state_dict(), f'models/gen_AB_{epoch}.pt')
		torch.save(model.genBA.state_dict(), f'models/gen_BA_{epoch}.pt')

''' Test method that takes in test images from domain A and B and an epoch and runs 
the test images through the generators of that epoch
Input:
	testA (list of images): list of images from domain A
	testB (list of images): list of images from domain B
	epoch (int): represents which generator to load based on the epoch number
Output: None
'''
def test(testA, testB, epoch):
	np.random.shuffle(testA)
	np.random.shuffle(testB)
	genAB = Generator().to(device)
	genAB.load_state_dict(torch.load(f'models/gen_AB_{epoch}.pt', map_location=torch.device('cpu')))
	genAB.eval()
	genBA = Generator().to(device)
	genBA.load_state_dict(torch.load(f'models/gen_BA_{epoch}.pt', map_location=torch.device('cpu')))
	genBA.eval()
	num_test = 4
	fig,ax = plt.subplots(nrows=num_test,ncols=3)
	for i in range(num_test):
		realA = testA[i]
		input_A = torch.FloatTensor(testA[i]).reshape(1,3,testA[i].shape[0], testA[i].shape[1]).to(device)
		fake_B = genAB(input_A)
		rec_A = genBA(fake_B)
		rec_A = rec_A[0].detach().cpu().numpy()
		realA = (realA + 1)/2
		fake_B = fake_B[0].detach().cpu().numpy()
		h_size,w_size = fake_B.shape[1], fake_B.shape[2]
		output_B = [[(fake_B[0,r,c], fake_B[1,r,c], fake_B[2,r,c]) for c in range(w_size)] for r in range(h_size)]
		output_B = np.asarray(output_B)
		output_B = (output_B + 1)/2
		rec_A = (rec_A + 1) /2
		
		ax[i,0].imshow(realA)
		ax[i,1].imshow(output_B)
		ax[i,2].imshow(rec_A.transpose(1, 2, 0))
		
	plt.show()


if __name__ == '__main__':
	
	# preprocessing on the test data
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

	# ensure data input is from -1 to 1
	trainA = (trainA * 2) - 1
	trainB = (trainB * 2) - 1
	model = CycleGAN().to(device)

	train(model, trainA, trainB, start_epoch=0, num_epochs=100)
	test(trainA, trainB, epoch=9)




