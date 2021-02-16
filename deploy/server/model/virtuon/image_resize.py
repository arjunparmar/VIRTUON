import cv2 as cv
import torch
import torchvision.transforms as transforms
import os
def image_resize():
	base_dir_1 = os.path.abspath('./model/input/test/test/image')
	base_dir_2 = os.path.abspath('./model/input/test/test/cloth')
	# print(base_dir)
	image_list = os.listdir(base_dir_1)
	cloth_list = os.listdir(base_dir_2)
	
	for i in image_list:
		temp = cv.imread(os.path.join(base_dir_1, i))
		temp = cv.resize(temp, (192,256))
		cv.imwrite(os.path.join(base_dir_1, i), temp)

	for i in cloth_list:
		temp = cv.imread(os.path.join(base_dir_2, i))
		temp = cv.resize(temp, (192,256))
		cv.imwrite(os.path.join(base_dir_2, i), temp)

