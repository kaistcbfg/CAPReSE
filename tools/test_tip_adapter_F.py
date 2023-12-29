
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset,DataLoader

from sklearn import metrics

import argparse
import pickle
import gzip

def clf_tipAdapter(query, model, L_train, beta, div_tensor):

	X = torch.from_numpy(query).type(torch.float32)
	X = X.to(device)
	X = torch.from_numpy(query).type(torch.float32)
	X = X/X.norm(dim=-1,keepdim=True)

	with torch.no_grad(): affinity = (-beta * (1 - model(X))).exp()
	logits = torch.div(affinity @ L_train, div_tensor)
	#print(logits)
	clf = torch.argmax(logits, dim=1)

	return clf.cpu().numpy()
#

if __name__ == "__main__":
	
	parser = argparse.ArgumentParser(description='Test fine-tuned Tip-adapter-F')
	parser.add_argument('--model-ptpath', type=str, help='finetuned model weight (required)', required=True)
	parser.add_argument('--model-infopath', type=str, help='finetuned model info pkl.gz (required)', required=True)
	parser.add_argument('--input-postest', type=str, help='input pos test csv file (required)', required=True)
	parser.add_argument('--input-negtest', type=str, help='input neg test csv file (required)', required=True)
	parser.add_argument('--version', action='version', version='%(prog)s 1.0')

	args = parser.parse_args()

	device = "cuda" if torch.cuda.is_available() else "cpu"

	## Load trained data
	# row, col , K1, K2, beta 
	
	model_info_dict = pickle.load(gzip.open(args.model_infopath,'rb'))
	row = model_info_dict['train_row']
	col = model_info_dict['train_col']
	K1  = model_info_dict['train_K1']
	K2  = model_info_dict['train_K2']
	beta = model_info_dict['train_beta']
	print("Model info loaded. {}".format(args.model_infopath))

	div_tensor = torch.from_numpy(np.array([K2,K1])).type(torch.float32)
	model = nn.Linear(row, col, bias=False)
	model.weight = nn.Parameter(torch.load(args.model_ptpath)['weight'])
	model.eval()
	model.to(device)
	print("Model loaded. {}".format(args.model_ptpath))

	L_train = np.array([0 for i in range(K2)] + [1 for i in range(K1)])
	L_train = np.eye(2)[L_train]
	L_train = torch.from_numpy(L_train).type(torch.float32)
	#

	data_pos = np.loadtxt(args.input_postest, delimiter=',')
	K1,col = data_pos.shape
	data_neg = np.loadtxt(args.input_negtest, delimiter=',')
	K2,col = data_neg.shape
	X_test = np.vstack((data_neg,data_pos))
	y_test = np.array([0 for i in range(K2)] + [1 for i in range(K1)])

	y_pred = clf_tipAdapter(X_test,  model, L_train, beta, div_tensor)

	precision  = np.round(metrics.precision_score(y_test, y_pred),2)
	recall  = np.round(metrics.recall_score(y_test, y_pred),2)
	print("Precision: {} | Recall: {}".format(precision, recall))

	print("Neg True: {} | Pos True: {}".format(K2,K1))
	cnt1 = 0
	cnt2 = 0
	for i in range(K1+K2):
		if y_test[i] == 0 and y_pred[i] == 0: cnt1 += 1
		if y_test[i] == 1 and y_pred[i] == 1: cnt2 += 1
	print("Neg corr: {} | Pos corr: {}".format(cnt1, cnt2))
	print("")
##
