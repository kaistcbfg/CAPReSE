
import caprese

import numpy as np
from scipy.spatial import distance
import cooler

import cv2
from PIL import Image

import torch
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset,DataLoader
import clip

import argparse
import pickle
import gzip
import copy
import sys
import os

import warnings
warnings.filterwarnings("ignore")

def str2bool(v):
	if isinstance(v, bool): return v
	if v.lower() in ['yes', 'true', 't', 'y', '1']: return True
	elif v.lower() in ['no', 'false', 'f', 'n', '0']: return False
	else: raise argparse.ArgumentTypeError('Boolean value expected.')
#

def clf_tipAdapter(query, model, L_train, beta, div_tensor):

	X = torch.from_numpy(query).type(torch.float32)
	X = X.to(device)
	X = torch.from_numpy(query).type(torch.float32)
	X = X/X.norm(dim=-1,keepdim=True)

	with torch.no_grad(): affinity = (-beta * (1 - model(X))).exp()
	logits = torch.div(affinity @ L_train, div_tensor)
	clf = torch.argmax(logits, dim=1)

	return clf.cpu().numpy()
#

if __name__ == "__main__":

	parser = argparse.ArgumentParser(description='CAPReSE: Chromatin Anomaly Pattern REcognition and Size Estimation')
	parser.add_argument('--header', type=str, help='save file header', required=True)
	parser.add_argument('--input-file', type=str, help='input *.pkl.gz, *.cool, or *.mcool file', required=True)
	parser.add_argument('--input-format', type=str, default='pickle', help='default pickle, format: pickle (numpy array pickle) or mccol')
	parser.add_argument('--save-path', type=str, default='./output', help='')

	parser.add_argument('--chrom1', type=str, help='taget chr name', required=True)
	parser.add_argument('--chrom2', type=str, help='if chrom1==chrom2: cis-', required=True)
	parser.add_argument('--resolution', type=int, default=40000, help='bin resolution (default 40kb)')
	parser.add_argument('--clr-balance', type=str, default=False, help='cooler balance option (default False)')

	parser.add_argument('--imgproc-clipval', type=float, default=5, help='Hi-C contact map 255 scale conversion cutoff value (default 5)')
	parser.add_argument('--imgproc-sizefiltval', type=int, default=2, help='Contour size filter cutoff (default 2)')
	parser.add_argument('--imgproc-diagval', type=int, default=800000, help='Distance from diagonal (default 800000)')
	parser.add_argument('--imgproc-lowcov', type=float, default=0.3, help='0-1 Low coverage ratio cutoff (default 0.3)')
	parser.add_argument('--imgproc-cropsize', type=int, default=16, help='Half of boxsize of breakpoint centered crop(default 16)')

	parser.add_argument('--model-infopath', type=str, required=True, help='PATH to model info pkl.gz file')
	parser.add_argument('--model-ptpath', type=str, required=True, help='PATH to pytorch model pt file')
	parser.add_argument('--fdist-cutoff', type=float, default=3, help='Euclidean distance from train set (default 3)')	

	parser.add_argument('--visualize-flag', type=str2bool, default=True, help='Save detection result to png')
	parser.add_argument('--visualize-outdir', type=str, default='./output', help='png file save path')
	parser.add_argument('--visualize-boxsize', type=int, default=16, help='Half of boxsize of breakpoint centered crop (default 16')	

	parser.add_argument('--imgproc-getcrop', type=str2bool, default=False, help='Collect crop without clf')
	
	parser.add_argument('--version', action='version', version='%(prog)s 1.0')

	args = parser.parse_args()
	print("CAPReSE Tip-Adapter-F classifier version 1.0")

	## Load data
	if args.input_format == 'pickle':
		mat = pickle.load(gzip.open(args.input_file,'rb'))
	elif args.input_format == 'cool':
		clr = cooler.Cooler(args.input_file)
		bal = args.clr_balance
		if bal == False: bal = False
		if args.chrom1 == args.chrom2:  mat = c.matrix(balance=bal).fetch(args.chrom1)
		else: mat = c.matrix(balance=bal).fetch(args.chrom1, args.chrom2)
	elif args.input_format == 'mcool':
		clr = cooler.Cooler('{}::/resolutions/{}'.format(args.input_file, args.resolution))
		bal = args.clr_balance
		if bal == 'False': bal = False
		if args.chrom1 == args.chrom2:  mat = clr.matrix(balance=bal).fetch(args.chrom1)
		else: mat = clr.matrix(balance=bal).fetch(args.chrom1, args.chrom2)
	#
	print("Input loaded: {}".format(args.input_file))
	print("chr A: {} | chr B: {} | resolution {}".format(args.chrom1, args.chrom2, args.resolution))
	if args.input_format == 'cool': print("cooler balance: {}".format(args.clr_balance))

	## Output setting
	outputdir = '{}/{}'.format(args.save_path,args.header)
	if not os.path.exists(outputdir): os.mkdir(outputdir)
	
	if args.imgproc_getcrop == True:
		cropdir = '{}/{}/crop'.format(args.save_path,args.header)
		if not os.path.exists(cropdir): os.mkdir(cropdir)	
	#

	## Load trained data
	# row, col , K1, K2, beta 
	# model info pkl
	# model weight
	# distance standard set
	
	model_info_dict = pickle.load(gzip.open(args.model_infopath,'rb'))
	row = model_info_dict['train_row']
	col = model_info_dict['train_col']
	K1  = model_info_dict['train_K1']
	K2  = model_info_dict['train_K2']
	beta = model_info_dict['train_beta']
	rawdata = model_info_dict['train_data']
	print("Model info {} loaded.".format(args.model_infopath))

	device = "cuda" if torch.cuda.is_available() else "cpu"
	clip_model,preprocess = clip.load("ViT-B/32",device=device)

	div_tensor = torch.from_numpy(np.array([K2,K1])).type(torch.float32)

	adapter_model = nn.Linear(row, col, bias=False)
	adapter_model.weight = nn.Parameter(torch.load(args.model_ptpath)['weight'])
	adapter_model.eval()
	adapter_model.to(device)

	L_train = np.array([0 for i in range(K2)] + [1 for i in range(K1)])
	L_train = np.eye(2)[L_train]
	L_train = torch.from_numpy(L_train).type(torch.float32)
	print("Model pt {} loaded.".format(args.model_ptpath))

	## Image preporcess	
	cisFlag = (args.chrom1 == args.chrom2)
	clipimg, maskimg, boxlist = caprese.prep_img(mat, cisflag=cisFlag, clip_thresh=args.imgproc_clipval, size_cutoff=args.imgproc_sizefiltval, diag_thresh=args.imgproc_diagval, resolution=args.resolution)
	bkptimg = clipimg - (args.imgproc_clipval/2)
	vizimg = (clipimg * (255/args.imgproc_clipval)).astype(np.uint8) 
	vizimg = cv2.cvtColor(vizimg, cv2.COLOR_GRAY2BGR)

	covmask_img = caprese.make_lowCov_mask(mat, thresh=args.imgproc_lowcov)
	b,g,r = cv2.split(vizimg)
	vizimg = cv2.merge((covmask_img,g,maskimg))
	if args.visualize_flag == True: outimg = copy.copy(vizimg)

	outtxtfilename = '{}/{}_{}_{}_SVcall.txt'.format(outputdir,args.header,args.chrom1,args.chrom2)
	o = open(outtxtfilename,'w')
	callcnt = 0
	cropcnt = 0
	for idx,(x,x2,y,y2) in enumerate(boxlist):

		bkpt = caprese.check_contour_bkpt(bkptimg, (x,x2,y,y2))
		if bkpt != -1:
			cx,cy = bkpt
			cropimg = caprese.fix_crop_coord_BGR(bkpt, vizimg, args.imgproc_cropsize)
			with torch.no_grad():
				image = preprocess(Image.fromarray(cropimg)).unsqueeze(0).to(device)
				image_features = clip_model.encode_image(image)
				cropfeature = image_features.cpu().numpy()[0]

				if args.imgproc_getcrop == False:

					fdist = np.amax(np.min(distance.cdist([cropfeature],rawdata,metric='euclidean'),axis=1))
					if fdist <= args.fdist_cutoff:
						y_pred = clf_tipAdapter(np.array([cropfeature]), adapter_model, L_train, beta, div_tensor)
						if y_pred[0] == 1:
							outstr = '{}\t{}\t{}\t{}\t{}\t{}\t\n'.format(cx,cy,x,x2,y,y2)
							o.write(outstr)
							callcnt += 1
							if args.visualize_flag == True: outimg = caprese.mark_result(outimg, (cx,cy), (x,x2,y,y2), args.visualize_boxsize)
						# Final prediction and output	
					# Distance filter
				# DL CLF
				else:
					cv2.imwrite('{}/{}_{}_{}_{}.png'.format(cropdir,args.header,idx,cx,cy),cropimg)
					cropcnt += 1
					if args.visualize_flag == True: outimg = caprese.mark_result(outimg, (cx,cy), (x,x2,y,y2), args.visualize_boxsize)
				# Get img crop only

			# torch no_grad
		# Valid breakpoint processing
	# Valid contour processing
	o.close()
	if args.imgproc_getcrop == False: print("Result file saved at {} | {}".format(outtxtfilename, callcnt))
	else: print("Crops saved at {} | {}.".format(cropdir, cropcnt))

	if args.visualize_flag == True:
		outputfilename = '{}/{}_{}_{}_resMarked.png'.format(outputdir,args.header,args.chrom1,args.chrom2) 
		cv2.imwrite(outputfilename,outimg)
		print("Image saved at {}".format(outputfilename))
	#
	print("")

##
