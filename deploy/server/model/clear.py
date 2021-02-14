import os
import glob
import shutil

def clear():
    img_mask = glob.glob(os.path.abspath('input/test/test/image-mask/*'))
    cloth_mask = glob.glob(os.path.abspath('input/test/test/cloth-mask/*'))
    img_parse = glob.glob(os.path.abspath('input/test/test/image-parse/*'))
    img_parse_new = glob.glob(os.path.abspath('input/test/test/image-parse-new/*'))
    overlay = glob.glob(os.path.abspath('input/test/test/overlayed-TPS/*'))
    pose = glob.glob(os.path.abspath('input/test/test/pose/*'))
    gmm_result = glob.glob(os.path.abspath('input/test/test/result-dir/*'))
    warp_cloth = glob.glob(os.path.abspath('input/test/test/warp-cloth/*'))
    warp_mask = glob.glob(os.path.abspath('input/test/test/warp-mask/*'))
    try:
        shutil.rmtree('./static/output')
    except:
        pass

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


if __name__ == '__main__':
    clear()
