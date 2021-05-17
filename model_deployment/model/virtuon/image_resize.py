import cv2 as cv
from PIL import Image
import torch
import torchvision.transforms as transforms
import os
def image_resize():
	base_dir_1 = os.path.abspath('./model/input/image')
	base_dir_2 = os.path.abspath('./model/input/cloth')
	# print(base_dir)
	image_list = os.listdir(base_dir_1)
	cloth_list = os.listdir(base_dir_2)
	transformation = transforms.Compose([transforms.Resize((256,192), interpolation = Image.BICUBIC)])
	
	for i in image_list:
		temp = Image.open(os.path.join(base_dir_1, i))
		# temp = cv.resize(temp, (192,256))
		temp = transformation(temp)
		temp.save(os.path.join(base_dir_1, i))

	for i in cloth_list:
		temp = Image.open(os.path.join(base_dir_2, i))
		# temp = cv.resize(temp, (192,256))
		temp = transformation(temp)
		temp.save(os.path.join(base_dir_2, i))

