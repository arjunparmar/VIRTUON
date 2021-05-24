import os
import sys
sys.path.append(os.path.abspath('model/virtuon'))
sys.path.append(os.path.abspath('model/openpose'))
from Data_Gen import CPDataset, CPDataLoader
from GMM import test_GMM, GMM
from TOM import test_tom, UnetGenerator
import torch.nn as nn
from grapy.exp.test.eval_gpm import gpm_segment
import torch
# import tensorflow as tf
# import time
from image_resize import image_resize
from openpose import openpose
# from segment import segment
# from segment_cloth import segment_cloth
# import os
# from numba import cuda

import sys
sys.path.append(os.path.abspath('model/grapy'))

def virtuon():

    # segment_cloth()

    # torch.cuda.empty_cache()

    # tf.reset_default_graph()
    image_resize()
    # with tf.Session() as session:
    #     tf.reset_default_graph()
    # device = cuda.get_current_device()
    # device.reset()

    # segment()
    # time.sleep(2)
    #     session.close()

    with torch.no_grad():
        openpose()
    torch.cuda.empty_cache()


    with torch.no_grad():
        gpm_segment()
    torch.cuda.empty_cache()

    with torch.no_grad():
        gpm_segment(cloth=True)
    torch.cuda.empty_cache()


    # with tf.Session() as sess:

    # sess.close()

    # with tf.Session() as sess:

    # sess.close()
    # tf.reset_default_graph()

    # with torch.no_grad():
    dataset_gmm = CPDataset()
    dataset_loader_gmm = CPDataLoader(dataset_gmm, batch=4, workers=1)
    gmm_model = GMM()
    test_GMM(gmm_model, dataset_loader_gmm)
    tom_model = UnetGenerator(26, 4, 6, ngf=64, norm_layer=nn.InstanceNorm2d)
    dataset_tom = CPDataset("TOM")
    dataset_loader_tom = CPDataLoader(dataset_tom, batch=4, workers=1)
    test_tom(dataset_loader_tom, tom_model)


if __name__ == '__main__':
    virtuon()
