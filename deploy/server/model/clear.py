'''
import os
import glob
import shutil

def clear():
    img_mask = glob.glob(os.path.abspath('model/input/image-mask/*'))
    cloth_mask = glob.glob(os.path.abspath('model/input/cloth-mask/*'))
    img_parse = glob.glob(os.path.abspath('model/input/image-parse/*'))
    img_parse_new = glob.glob(os.path.abspath('model/input/image-parse-new/*'))
    overlay = glob.glob(os.path.abspath('model/input/overlayed-TPS/*'))
    pose = glob.glob(os.path.abspath('model/input/pose/*'))
    gmm_result = glob.glob(os.path.abspath('model/input/result-dir/*'))
    warp_cloth = glob.glob(os.path.abspath('model/input/warp-cloth/*'))
    warp_mask = glob.glob(os.path.abspath('model/input/warp-mask/*'))

    input_image = glob.glob(os.path.abspath('model/input/image/*'))
    input_cloth = glob.glob(os.path.abspath('model/input/cloth/*'))

    try:
        shutil.rmtree('./media/image')
        # shutil.rmtree('./media/cloth')
        # shutil.rmtree('./media/output')
    except:
        pass
    try:
        # shutil.rmtree('./media/image')
        shutil.rmtree('./media/cloth')
        # shutil.rmtree('./media/output')
    except:
        pass
    try:
        # shutil.rmtree('./media/image')
        # shutil.rmtree('./media/cloth')
        shutil.rmtree('./media/output')
    except:
        pass

    for i in range(len(img_mask)):
        try:
            os.remove(input_image[i])
        except:
            pass
        try:
            os.remove(input_cloth[i])
        except:
            pass
        try:
            os.remove(img_mask[i])
        except:
            pass
        try:
            os.remove(cloth_mask[i])
        except:
            pass
        try:
            os.remove(img_parse[i])
        except:
            pass
        try:
            os.remove(img_parse_new[i])
        except:
            pass
        try:
            os.remove(overlay[i])
        except:
            pass
        try:
            os.remove(pose[i])
        except:
            pass
        try:
            os.remove(gmm_result[i])
        except:
            pass
        try:
            os.remove(warp_cloth[i])
        except:
            pass
        try:
            os.remove(warp_mask[i])
        except:
            pass
'''
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
