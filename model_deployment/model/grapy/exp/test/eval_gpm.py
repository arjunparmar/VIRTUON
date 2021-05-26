import timeit
import numpy as np
from PIL import Image
import os
import sys
sys.path.append(os.path.abspath('model/grapy'))
sys.path.append(os.path.abspath('model/grapy/exp/test'))
import pandas as pd

# PyTorch includes
import torch
from torch.autograd import Variable
from torchvision import transforms
from torch.utils.data import DataLoader
import cv2

from dataloaders import cihp, atr, pascal
from networks import graph, grapy_net
from dataloaders import custom_transforms as tr

#
import argparse
import copy
import torch.nn.functional as F
from test_from_disk import eval_, eval_with_numpy


gpu_available = torch.cuda.is_available()
if gpu_available:
	device = torch.device("cuda")
else:
	device = torch.device("cpu")

label_colours = [(0,0,0)
				, (128,0,0), (255,0,0), (0,85,0), (170,0,51), (255,85,0), (0,0,85), (0,119,221), (85,85,0), (0,85,85), (85,51,0), (52,86,128), (0,128,0)
				, (0,0,255), (51,170,221), (0,255,255), (85,255,170), (170,255,85), (255,255,0), (255,170,0)]


def flip(x, dim):
	indices = [slice(None)] * x.dim()
	indices[dim] = torch.arange(x.size(dim) - 1, -1, -1,
								dtype=torch.long, device=x.device)
	return x[tuple(indices)]


def flip_cihp(tail_list):
	'''

	:param tail_list: tail_list size is 1 x n_class x h x w
	:return:
	'''
	# tail_list = tail_list[0]
	tail_list_rev = [None] * 20
	for xx in range(14):
		tail_list_rev[xx] = tail_list[xx].unsqueeze(0)
	tail_list_rev[14] = tail_list[15].unsqueeze(0)
	tail_list_rev[15] = tail_list[14].unsqueeze(0)
	tail_list_rev[16] = tail_list[17].unsqueeze(0)
	tail_list_rev[17] = tail_list[16].unsqueeze(0)
	tail_list_rev[18] = tail_list[19].unsqueeze(0)
	tail_list_rev[19] = tail_list[18].unsqueeze(0)
	return torch.cat(tail_list_rev, dim=0)


def flip_atr(tail_list):
	'''

	:param tail_list: tail_list size is 1 x n_class x h x w
	:return:
	'''
	# tail_list = tail_list[0]
	tail_list_rev = [None] * 18
	for xx in range(9):
		tail_list_rev[xx] = tail_list[xx].unsqueeze(0)
	tail_list_rev[10] = tail_list[9].unsqueeze(0)
	tail_list_rev[9] = tail_list[10].unsqueeze(0)
	tail_list_rev[11] = tail_list[11].unsqueeze(0)
	tail_list_rev[12] = tail_list[13].unsqueeze(0)
	tail_list_rev[13] = tail_list[12].unsqueeze(0)
	tail_list_rev[14] = tail_list[15].unsqueeze(0)
	tail_list_rev[15] = tail_list[14].unsqueeze(0)
	tail_list_rev[16] = tail_list[16].unsqueeze(0)
	tail_list_rev[17] = tail_list[17].unsqueeze(0)

	return torch.cat(tail_list_rev, dim=0)

def decode_labels(mask, num_images=1, num_classes=20):
	"""Decode batch of segmentation masks.

	Args:
	  mask: result of inference after taking argmax.
	  num_images: number of images to decode from the batch.
	  num_classes: number of classes to predict (including background).

	Returns:
	  A batch with num_images RGB images of the same size as the input.
	"""
	n, h, w = mask.shape
	assert(n >= num_images), 'Batch size %d should be greater or equal than number of images to save %d.' % (n, num_images)
	outputs = np.zeros((num_images, h, w, 3), dtype=np.uint8)
	for i in range(num_images):
	  img = Image.new('RGB', (len(mask[i, 0]), len(mask[i])))
	  pixels = img.load()
	  for j_, j in enumerate(mask[i, :, :]):
		  for k_, k in enumerate(j):
			  if k < num_classes:
				  pixels[k_,j_] = label_colours[k]
	  outputs[i] = np.array(img)
	return outputs

def gpm_segment(txt_file = './model/input/test_pairs.txt', classes = 20, hidden_graph_layers=256, resume_model = './model/grapy/data/models/CIHP_trained.pth', dataset = 'cihp', batch = 1, output_path = './model/input', cloth = False):

	if not cloth:
		img_list = pd.read_csv(txt_file, header = None, sep = " ").iloc[:,0].apply(lambda x: x[:-3]).to_list()
	else:
		img_list = pd.read_csv(txt_file, header=None, sep=" ").iloc[:, 1].apply(lambda x: x[:-3]).to_list()

	net = grapy_net.GrapyNet(n_classes=classes, os=16, hidden_layers=hidden_graph_layers)

	net.to(device)

	if not resume_model == '':
		x = torch.load(resume_model, map_location= device)
		net.load_state_dict(x)

		print('resume model:', resume_model)

	else:
		print('we are not resuming from any model')

	if dataset == 'cihp':
		val = cihp.VOCSegmentation
		val_flip = cihp.VOCSegmentation

		vis_dir = '/image-parse/'
		mat_dir = '/image-parse-new/'

		cate_lis1 = [[0], [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]]
		cate_lis2 = [[0], [1, 2, 4, 13], [5, 6, 7, 10, 11, 12], [3, 14, 15], [8, 9, 16, 17, 18, 19]]

	elif dataset == 'pascal':

		val = pascal.VOCSegmentation
		val_flip = pascal.VOCSegmentation

		vis_dir = '/pascal_output_vis/'
		mat_dir = '/pascal_output/'

		cate_lis1 = [[0], [1, 2, 3, 4, 5, 6]]
		cate_lis2 = [[0], [1], [2], [3, 4], [5, 6]]

	else:
		val = atr.VOCSegmentation
		val_flip = atr.VOCSegmentation

		vis_dir = '/atr_output_vis/'
		mat_dir = '/atr_output/'

		cate_lis1 = [[0], [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]]
		cate_lis2 = [[0], [1, 2, 3, 11], [4, 5, 7, 8, 16, 17], [14, 15], [6, 9, 10, 12, 13]]

	## multi scale
	scale_list=[1,0.5,0.75,1.25,1.5,1.75]
	testloader_list = []
	testloader_flip_list = []
	for pv in scale_list:
		composed_transforms_ts = transforms.Compose([
			tr.Scale_(pv),
			tr.Normalize_xception_tf(),
			tr.ToTensor_()])

		composed_transforms_ts_flip = transforms.Compose([
			tr.Scale_(pv),
			tr.HorizontalFlip(),
			tr.Normalize_xception_tf(),
			tr.ToTensor_()])

		voc_val = val(split='val', transform=composed_transforms_ts, cloth = cloth)
		voc_val_f = val_flip(split='val', transform=composed_transforms_ts_flip, cloth = cloth)

		testloader = DataLoader(voc_val, batch_size=batch, shuffle=False, num_workers=4, pin_memory=gpu_available)
		testloader_flip = DataLoader(voc_val_f, batch_size=batch, shuffle=False, num_workers=4, pin_memory=gpu_available)

		testloader_list.append(copy.deepcopy(testloader))
		testloader_flip_list.append(copy.deepcopy(testloader_flip))

	print("Eval Network")

	if not os.path.exists(output_path + vis_dir):
		os.makedirs(output_path + vis_dir)
	if not os.path.exists(output_path + mat_dir):
		os.makedirs(output_path + mat_dir)

	# start_time = timeit.default_timer()
	# One testing epoch
	total_iou = 0.0

	net.set_category_list(cate_lis1, cate_lis2)

	net.eval()

	if not os.path.exists(output_path + '/cloth-mask/'):
    				os.makedirs(output_path + '/cloth-mask/')
	if not os.path.exists(output_path + '/image-mask/'):
    				os.makedirs(output_path + '/image-mask/')

	with torch.no_grad():
		for ii, large_sample_batched in enumerate(zip(*testloader_list, *testloader_flip_list)):
			# print(ii)
			#1 0.5 0.75 1.25 1.5 1.75 ; flip:
			sample1 = large_sample_batched[:6]
			sample2 = large_sample_batched[6:]
			for iii,sample_batched in enumerate(zip(sample1,sample2)):

				inputs = sample_batched[0]['image']
				# inputs_f= sample_batched[1]['image']
				# inputs = torch.cat((inputs, inputs_f), dim=0)
				# labels = torch.cat((labels_single, labels_single_f), dim=0)

				if iii == 0:
					_,_,h,w = inputs.size()
				# assert inputs.size() == inputs_f.size()

				# Forward pass of the mini-batch
				inputs = Variable(inputs, requires_grad=False)

				with torch.no_grad():
					if gpu_available:
						inputs = inputs.to(device)
					# outputs = net.forward(inputs)
					# pdb.set_trace()

					outputs, outputs_aux = net.forward(inputs, training=False)

					# if dataset == 'cihp':
					# 	outputs = (outputs + flip(flip_cihp(outputs[1]), dim=-1)) / 2
					# elif dataset == 'pascal':
					# 	outputs = (outputs[0] + flip(outputs[1], dim=-1)) / 2
					# else:
					# 	outputs = (outputs[0] + flip(flip_atr(outputs[1]), dim=-1)) / 2


					# outputs = outputs.unsqueeze(0)
					# print(outputs.size())

					if iii>0:
						outputs = F.upsample(outputs,size=(h,w),mode='bilinear',align_corners=True)
						outputs_final = outputs_final + outputs
					else:
						outputs_final = outputs.clone()

			################ plot pic
			predictions = torch.max(outputs_final, 1)[1]
			# prob_predictions = torch.max(outputs_final,1)[0]
			results = predictions.cpu().numpy()
			# prob_results = prob_predictions.cpu().numpy()
			if not cloth:
				vis_res = decode_labels(results)
				image_mask = results[0, :, :].copy()
				image_mask[image_mask>0] = 255

				parsing_im = Image.fromarray(vis_res[0])
				parsing_im.save(output_path + vis_dir + '{}.png'.format(img_list[ii][:-1]))
				cv2.imwrite(output_path + mat_dir + '{}.png'.format(img_list[ii][:-1]), results[0,:,:])

				cv2.imwrite(output_path + '/image-mask/' + '{}.png'.format(img_list[ii][:-1]), image_mask)
			else:
				cloth_mask = results[0, :, :].copy()

				cloth_mask[cloth_mask == 5] = 255
				cloth_mask[cloth_mask == 6] = 255
				cloth_mask[cloth_mask == 7] = 255
				cloth_mask[cloth_mask != 255] = 0
				# cloth_mask_1 = (cloth_mask == 255).astype(int).reshape(256,192,1)
				# cloth_mask_1 = np.concatenate([cloth_mask_1, cloth_mask_1, cloth_mask_1], axis = -1)
				# print(cloth_mask_1.shape, "Cloth Mask Size")
				cloth_orig = cv2.imread(output_path + '/cloth/' + '{}.jpg'.format(img_list[ii][:-1]))
				# cloth_orig = cloth_orig*(cloth_mask_1)
				cloth_orig[cloth_mask == 0] = 255

				cv2.imwrite(output_path + '/cloth/' + '{}.jpg'.format(img_list[ii][:-1]), cloth_orig)
				cv2.imwrite(output_path + '/cloth-mask/' + '{}.jpg'.format(img_list[ii][:-1]), cloth_mask)


		# total_iou += utils.get_iou(predictions, labels)
	# end_time = timeit.default_timer()
	# print('time use for '+ str(ii) + ' is :' + str(end_time - start_time))

	# Eval
	# pred_path = output_path + mat_dir
	# eval_with_numpy(pred_path=pred_path, gt_path=gt_path,classes=classes, txt_file=txt_file, dataset=dataset)


if __name__ == '__main__':
	gpm_segment()
