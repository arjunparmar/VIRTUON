import cv2 as cv
import os
def image_resize():
	base_dir = os.path.abspath('./model/input/test/test/image')
	# print(base_dir)
	image_list = os.listdir(base_dir)
	for i in image_list:
		temp = cv.imread(os.path.join(base_dir, i))
		temp = cv.resize(temp, (192,256))
		cv.imwrite(os.path.join(base_dir, i), temp)
