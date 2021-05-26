import glob
import os
import sys

def mkdir(path):
	if not os.path.exists(path):
		os.makedirs(path)

def clear():

	base_dir = os.path.abspath('media')
	image_dir = os.path.join(base_dir, 'image')
	cloth_dir = os.path.join(base_dir, 'cloth')
	output_dir = os.path.join(base_dir, 'output')

	mkdir(base_dir)

	mkdir(image_dir)
	for i in os.listdir(image_dir):
		os.remove(os.path.join(image_dir, i))

	mkdir(cloth_dir)
	for i in os.listdir(cloth_dir):
		os.remove(os.path.join(cloth_dir, i))

	mkdir(output_dir)
	for i in os.listdir(output_dir):
		os.remove(os.path.join(output_dir, i))

if __name__ == '__main__':
    clear()
