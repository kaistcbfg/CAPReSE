
import numpy as np
import pandas as pd
import cv2
import mahotas

import copy

def gen_chrsize_dict(genome_info, resolution):

	chrsize_dict = {}
	f = open(genome_info)
	for line in f:
		size = int(int(line.split()[1])/resolution) + 1
		targetchrname = line.split()[0]
		chrsize_dict[targetchrname] = size
	#
	
	return chrsize_dict
#

def covnorm_df_to_array(df, chrsize_dict, resolution, chrname1, chrname2=False, cutoff=False, mirror=False):

	if chrname2 == False:
		bin_length1 = chrsize_dict[chrname1]
		contact_map = np.zeros((bin_length1, bin_length1))
	else:
		bin_length1 = chrsize_dict[chrname1]
		bin_length2 = chrsize_dict[chrname2]
		contact_map = np.zeros((bin_length1, bin_length2))

	plotlist = df[['frag1','frag2','capture_res']].to_records(index=False)
	for frag1, frag2, cap_res in plotlist:
		bin1 = int(int(frag1.split('.')[1])/resolution)
		bin2 = int(int(frag2.split('.')[1])/resolution)

		if cutoff == False:
		    contact_map[bin1][bin2] += cap_res
		    if chrname2 == False and mirror == True: contact_map[bin2][bin1] += cap_res
		else:
			if cap_res > cutoff:
				contact_map[bin1][bin2] += cap_res
				if chrname2 == False and mirror == True: contact_map[bin2][bin1] += cap_res
			#
		#
	#
	
	return contact_map
#

def make_lowCov_mask(mat,thresh=0.3):

	rowsum = np.sum(mat,axis=0)
	colsum = np.sum(mat,axis=1)

	row_thresh = np.mean(rowsum) * thresh
	col_thresh = np.mean(colsum) * thresh

	row_bin = (rowsum > row_thresh).astype(np.int_)
	col_bin = (colsum > col_thresh).astype(np.int_)
	covmask = np.outer(col_bin,row_bin)
	covmask = np.logical_not(covmask).astype(int)
	covmask_img =  (covmask * 255).astype(np.uint8)

	if thresh == 0: covmask_img = np.zeros(mat.shape).astype(np.uint8)

	return covmask_img
#

def prep_img(rawimg, cisflag, clip_thresh=5, size_cutoff=0, diag_thresh=2000000, resolution=40000):

	row,col = rawimg.shape
	if cisflag: rawimg = np.triu(rawimg)
	
	img = np.clip(rawimg, 0, clip_thresh)
	T_otsu = mahotas.otsu(img.astype(np.uint8))
	new_img = (np.logical_and(mahotas.majority_filter(img), img>T_otsu) * 255).astype(np.uint8)

	contours,_ = cv2.findContours(new_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

	boxlist = []
	for i in contours:
		x, y, w, h = cv2.boundingRect(np.array(i))
		x2 = x+w
		y2 = y+h
		if (w > size_cutoff or h > size_cutoff) and (np.amax(rawimg[y:y2,x:x2]) >= clip_thresh):
			if not cisflag: boxlist.append((x,x2,y,y2))
			else:
				if (not caprese.Liang_Barsky_line_rect_collision((x,y,x2,y2),(0,0,row,col))) and np.abs(x-y2) >= diag_thresh//resolution: boxlist.append((x,x2,y,y2))
		#
	#		

	return img, new_img, boxlist
#

def Liang_Barsky_line_rect_collision(boxcoord, linecoord):

	line_x_start, line_y_start, line_x_end, line_y_end = linecoord
	x, y, x2, y2 = boxcoord

	p = [-(line_x_end - line_x_start), (line_x_end - line_x_start), -(line_y_end - line_y_start), (line_y_end - line_y_start)]
	q = [line_x_start - x, x2 - line_x_start, line_y_start - y, y2 - line_y_start ]

	u1 = -np.inf
	u2 = np.inf

	for i in range(4):
		t = float(q[i])/p[i]
		if (p[i] < 0 and u1 < t): u1 = t
		elif (p[i] > 0 and u2 > t): u2 = t
	#

	if (u1 > u2 or u1 > 1 or u1 < 0):
		collision = False
	else:
		collision = True

	return collision
#

def find_validContours(img, mat, triu_flag=True, sizefilter=5, thresh=0.8, diag_thresh=1000000, resolution=40000):

	row,col = img.shape

	img_cont = copy.copy(img)
	if triu_flag: img_cont = np.triu(img_cont)
	contours,_ = cv2.findContours(img_cont, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

	boxlist = []
	for i in contours:
		x, y, w, h = cv2.boundingRect(np.array(i))
		x2 = x+w
		y2 = y+h
		if (w >= sizefilter or h >= sizefilter) and np.amax(mat[y:y2,x:x2]) > thresh:
			if not triu_flag: boxlist.append((x,x2,y,y2))
			else:
				if (not Liang_Barsky_line_rect_collision((x,y,x2,y2),(0,0,row,col))) and np.abs(x-y2) >= diag_thresh//resolution:
					boxlist.append((x,x2,y,y2))
				#
			#	
		#
	#	

	return boxlist
#

def kadane(arr, start, finish, n):

	Sum = 0
	maxSum = -999999999999
	i = None
	finish[0] = -1

	local_start = 0

	for i in range(n):
		Sum += arr[i]
		if Sum < 0:
			Sum = 0
			local_start = i + 1
		elif Sum > maxSum:
			maxSum = Sum
			start[0] = local_start
			finish[0] = i
		#
	#

	if finish[0] != -1: return maxSum

	maxSum = arr[0]
	start[0] = finish[0] = 0

	for i in range(1, n):
		if arr[i] > maxSum:
			maxSum = arr[i]
			start[0] = finish[0] = i
		#
	#

	return maxSum
#

def findMaxSum(M):

	ROW, COL = M.shape
	maxSum, finalLeft = -999999999999, None
	finalRight, finalTop, finalBottom = None, None, None
	left, right, i = None, None, None

	temp = [None] * ROW
	Sum = 0
	start = [0]
	finish = [0]

	for left in range(COL):

		temp = [0] * ROW

		for right in range(left, COL):

			for i in range(ROW): temp[i] += M[i][right]
			Sum = kadane(temp, start, finish, ROW)
			if Sum > maxSum:
				maxSum = Sum
				finalLeft = left
				finalRight = right
				finalTop = start[0]
				finalBottom = finish[0]
			#
		#
	#

	return finalLeft, finalRight, finalTop, finalBottom, maxSum
#

def select_quadCorner(grey_patch, boxcoord):

	gradcenter = (-1, -1)
	x, y, x2, y2 = boxcoord
	h = y2 - y
	w = x2 - x

	img_h, img_w = grey_patch.shape
	if not img_h%2 == 0: img_h += 1
	if not img_w%2 == 0: img_w += 1
	grey_patch = cv2.resize(grey_patch, (img_w, img_h))

	crop_up_half = np.hsplit(np.vsplit(grey_patch, 2)[0], 2)
	crop_down_half =  np.hsplit(np.vsplit(grey_patch, 2)[-1], 2)

	crop_up_left = crop_up_half[0]
	crop_up_right = crop_up_half[-1]
	crop_down_left = crop_down_half[0]
	crop_down_right = crop_down_half[-1]

	score_list = [np.sum(crop_up_left),np.sum(crop_up_right),np.sum(crop_down_left),np.sum(crop_down_right)]
	maxindex = np.argmax(score_list)	
	if maxindex == 0: gradcenter = (x, y)
	elif maxindex == 1: gradcenter = (x2, y)	
	elif maxindex == 2: gradcenter = (x, y2)
	elif maxindex == 3: gradcenter = (x2, y2)
	#	

	return gradcenter[0], gradcenter[1], maxindex
#

def check_contour_bkpt(rawnormmat, boxcoord):

	x,x2,y,y2 = boxcoord
	crop = rawnormmat[y:y2,x:x2]

	xstart, xend, ystart, yend, maxsum = findMaxSum(crop)
	if np.abs(yend-ystart) > 0 and np.abs(xend-xstart) > 0:
		newcrop = rawnormmat[y+ystart:y+yend,x+xstart:x+xend]
		center_x,center_y,bkptquad = select_quadCorner(newcrop, (x+xstart, y+ystart, x+xend, y+yend))
	else: return -1

	return (center_x,center_y)
#

def fix_crop_coord_BGR(bkptcoord, img, boxsize):

	bkptx, bkpty = bkptcoord
	height, width, ch = img.shape

	boxx = bkptx - boxsize
	boxy = bkpty - boxsize
	boxx2 = bkptx + boxsize
	boxy2 = bkpty + boxsize
		
	leftflag = False
	rightflag = False
	upflag = False
	downflag = False

	if boxx < 0:
		boxx = 0
		leftflag = True
	if boxx2 > width:
		boxx2 = width
		rightflag = True
	if boxy < 0:
		boxy = 0
		upflag = True
	if boxy2 > height:
		downflag = True
	#
	crop = img[boxy:boxy2,boxx:boxx2]

	if leftflag:
		exd_x = np.abs(bkptx - boxsize) 
		crop = np.pad(crop,((0,0),(exd_x,0),(0,0)),'constant',constant_values=0) 
	if rightflag:
		exd_x = (bkptx + boxsize)-width
		crop = np.pad(crop,((0,0),(0,exd_x),(0,0)),'constant',constant_values=0)
	if downflag:
		exd_y = (bkpty + boxsize)-height
		crop = np.pad(crop,((0,exd_y),(0,0),(0,0)),'constant',constant_values=0)
	if upflag:
		exd_y = np.abs(bkpty - boxsize)
		crop = np.pad(crop,((exd_y,0),(0,0),(0,0)),'constant',constant_values=0)
	#

	return crop.astype(np.uint8)
# 

def mark_result(outimg, bkpt, contour, boxsize):

	cx,cy = bkpt
	x,x2,y,y2 = contour
	outimg = cv2.rectangle(outimg, (cx-boxsize,cy-boxsize), (cx+boxsize,cy+boxsize), (0,255,255), 1)
	outimg = cv2.rectangle(outimg, (x,y), (x2,y2), (0,255,0), 1)

	return outimg
#

