import glob
import os
import sys

def clear():

	base_dir = os.path.abspath('media')
	image_dir = os.path.join(base_dir, 'image')
	cloth_dir = os.path.join(base_dir, 'cloth')
	output_dir = os.path.join(base_dir, 'output')

	for i in os.listdir(image_dir):
		os.remove(os.path.join(image_dir, i))

	for i in os.listdir(cloth_dir):
		os.remove(os.path.join(cloth_dir, i))

	for i in os.listdir(output_dir):
		os.remove(os.path.join(output_dir, i))

if __name__ == '__main__':
    clear()
