import os
import glob
import shutil

img_mask = glob.glob(os.path.abspath('input/image-mask/*'))
cloth_mask = glob.glob(os.path.abspath('input/cloth-mask/*'))
img_parse = glob.glob(os.path.abspath('input/image-parse/*'))
img_parse_new = glob.glob(os.path.abspath('input/image-parse-new/*'))
overlay = glob.glob(os.path.abspath('input/overlayed-TPS/*'))
pose = glob.glob(os.path.abspath('input/pose/*'))
gmm_result = glob.glob(os.path.abspath('input/result-dir/*'))
warp_cloth = glob.glob(os.path.abspath('input/warp-cloth/*'))
warp_mask = glob.glob(os.path.abspath('input/warp-mask/*'))

shutil.rmtree('./output')

for i in range(len(img_mask)):
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