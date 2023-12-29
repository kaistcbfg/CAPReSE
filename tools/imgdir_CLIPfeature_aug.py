
import numpy as np
import torch
import clip
import cv2
from PIL import Image

import warnings
warnings.filterwarnings("ignore")

import argparse
import glob
import sys
import os

device = "cuda" if torch.cuda.is_available() else "cpu"
model,preprocess = clip.load("ViT-B/32",device=device)

if __name__ == "__main__":
	
	parser = argparse.ArgumentParser(description='CLIP feature *.csv file generator')
	parser.add_argument('--input-dir', type=str, help='input dir for glob', required=True)
	parser.add_argument('--input-format', type=str, default='png', help='default png, image crop file format')
	parser.add_argument('--output-filename', type=str, default='./CLIPfeatures.csv', help='default ./CLIPfeatures.csv')
	parser.add_argument('--num-aug', type=int, default=0, help='default 0 (no aug) max 3, 0-3 range rotate augmentation apllied')
	parser.add_argument('--version', action='version', version='%(prog)s 1.0')
	
	args = parser.parse_args()

	imglist = glob.glob('{}/*.{}'.format(args.input_dir, args.input_format))
	imglist.sort()

	augnum = int(args.num_aug)
	if augnum > 3: augnum = 3
	elif augnum < 0: augnum = 0
	print("CLIP feature extraction: input:{}, imglen: {}, augnum: {}, output:{}".format(args.input_dir,len(imglist),augnum,args.output_filename))

	o = open(args.output_filename,'wt')
	for imgpath in imglist:

		cropimg = cv2.imread(imgpath)

		with torch.no_grad():

			if augnum == 0:
				image = preprocess(Image.fromarray(cropimg)).unsqueeze(0).to(device)
				image_features = model.encode_image(image)
				cropfeature = list(image_features.cpu().numpy()[0])
				cropfeaturestr = ",".join([str(i) for i in cropfeature])
				o.write(cropfeaturestr + '\n')

			else:
				for i in range(augnum+1):
					cropimg = np.rot90(cropimg,i)

					image = preprocess(Image.fromarray(cropimg)).unsqueeze(0).to(device)
					image_features = model.encode_image(image)
					cropfeature = list(image_features.cpu().numpy()[0])
					cropfeaturestr = ",".join([str(i) for i in cropfeature])
					o.write(cropfeaturestr + '\n')
				#
			#
		#
	#
	o.close()
	print("")
##

