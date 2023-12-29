
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset,DataLoader

import argparse
import pickle
import gzip

class TipAdapter(nn.Module):
	def __init__(self, NK, C):
		super(TipAdapter, self).__init__()
		self.fc = nn.Linear(C, NK, bias=False)
	#

	def forward(self, x):
		return self.fc(x)
	#
#

if __name__ == "__main__":
	
	parser = argparse.ArgumentParser(description='Train binary TIP-Adapter-F with CLIP feature csv files')
	parser.add_argument('--input-poswfile', type=str, help='input pos weight csv file (required)', required=True)
	parser.add_argument('--input-negwfile', type=str, help='input neg weight csv file (required)', required=True)
	parser.add_argument('--input-postfile', type=str, help='input pos finetune csv file (required)', required=True)
	parser.add_argument('--input-negtfile', type=str, help='input neg finetune csv file (required)', required=True)
	parser.add_argument('--train-batchsize', type=int, default=5)
	parser.add_argument('--train-learnrate', type=float, default=0.01)
	parser.add_argument('--train-epoch', type=int, default=25)
	parser.add_argument('--train-beta', type=float, default=1.0)
	parser.add_argument('--output-model', type=str, default='./TipAdapterF_model.pt', help='default ./TipAdapterF_model.pt')
	parser.add_argument('--output-info', type=str, default='./TipAdapterF_info.pkl.gz', help='default ./TipAdapterF_info.pkl.gz')
	parser.add_argument('--version', action='version', version='%(prog)s 1.0')

	args = parser.parse_args()

	device = "cuda" if torch.cuda.is_available() else "cpu"

	data_pos = np.loadtxt(args.input_poswfile, delimiter=',')
	K1,col = data_pos.shape
	data_neg = np.loadtxt(args.input_negwfile, delimiter=',')
	K2,col = data_neg.shape

	data = np.vstack((data_neg,data_pos))
	wrow, wcol = data.shape
	# Train weight loading
	print("Train base weight loaded")
	print("pos: {} {}".format(args.input_poswfile, K1))
	print("neg: {} {}".format(args.input_negwfile, K2))
	print("tot: {} x {}".format(wrow, wcol))
	print("")	

	cache = torch.from_numpy(data).type(torch.float32)
	cache = cache/cache.norm(dim=-1,keepdim=True)
	div_tensor = torch.from_numpy(np.array([K2,K1])).type(torch.float32)

	L_train = np.array([0 for i in range(K2)] + [1 for i in range(K1)])
	L_train = np.eye(2)[L_train]
	L_train = torch.from_numpy(L_train).type(torch.float32)

	model = nn.Linear(wrow, wcol, bias=False)
	model.weight = nn.Parameter(cache)
	model.to(device)
	# Trian weight to model

	data_pos = np.loadtxt(args.input_postfile, delimiter=',')
	row1,col = data_pos.shape
	label_pos = np.ones((row1,1))

	data_neg = np.loadtxt(args.input_negtfile, delimiter=',')
	row2,col = data_neg.shape
	label_neg = np.zeros((row2,1))

	data_train = np.vstack((data_pos,data_neg))
	label_train = np.vstack((label_pos,label_neg))
	print("Train data loaded")
	print("pos: {} {}".format(args.input_postfile, row1))
	print("neg: {} {}".format(args.input_negtfile, row2))
	print("")

	train_data = TensorDataset(torch.from_numpy(data_train).type(torch.float32),torch.from_numpy(label_train).type(torch.LongTensor))
	# Finetune data loading

	bsize = args.train_batchsize
	train_loader = DataLoader(train_data, batch_size=bsize, shuffle=True)
	optimizer = optim.Adam(model.parameters(), lr = args.train_learnrate)
	n_epoch = args.train_epoch
	beta = args.train_beta

	print("Train options")
	print("epoch: {} | batch size: {} | beta: {}".format(n_epoch, bsize ,beta))
	print("Train log")
	for i in range(n_epoch):

		total_loss = 0
		model.train()
		for batchidx, (X,y) in enumerate(train_loader):

			X,y = X.to(device),y.to(device)
			optimizer.zero_grad()

			X = X/X.norm(dim=-1,keepdim=True)
			affinity = (-beta * (1 - model(X))).exp()
			logits = torch.div(affinity @ L_train, div_tensor)

			loss = F.cross_entropy(logits,y.squeeze(dim=-1))
			loss.backward()
			optimizer.step()
			total_loss += loss.item()
		#
		print("epoch: {} | loss: {}".format(i, total_loss))
	
	#
	print("")
	torch.save(model.state_dict(), args.output_model)
	print("Model pt file saved as {}".format(args.output_model))
	# model train

	pkg_dict = {}
	pkg_dict['train_pos'] = data_pos
	pkg_dict['train_neg'] = data_neg
	pkg_dict['train_data']= data
	pkg_dict['train_K1']  = K1
	pkg_dict['train_K2']  = K2
	pkg_dict['train_row'] = wrow
	pkg_dict['train_col'] = wcol
	pkg_dict['train_beta'] = args.train_beta
	pickle.dump(pkg_dict,gzip.open(args.output_info,'wb'))
	print("Model info pkl saved as {}".format(args.output_info))
	print("")
	# train info logging

##
