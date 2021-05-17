import glob
import os
import sys

def clear_media():

	base_dir = os.path.abspath('media')
	image_dir = os.path.join(base_dir, 'image')
	cloth_dir = os.path.join(base_dir, 'cloth')
	output_dir = os.path.join(base_dir, 'output')


	for i in os.listdir(image_dir):
		os.remove(i)

	for i in os.listdir(cloth_dir):
		os.remove(i)

	for i in os.listdir(output_dir):
		os.remove(i)

