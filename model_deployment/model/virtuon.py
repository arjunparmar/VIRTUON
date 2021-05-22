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

from image_resize import image_resize
from openpose import openpose
import sys
sys.path.append(os.path.abspath('model/grapy'))

def virtuon():

    image_resize()

    with torch.no_grad():
        openpose()
    torch.cuda.empty_cache()


    with torch.no_grad():
        gpm_segment()
    torch.cuda.empty_cache()

    with torch.no_grad():
        gpm_segment(cloth=True)
    torch.cuda.empty_cache()

    dataset_gmm = CPDataset()
    dataset_loader_gmm = CPDataLoader(dataset_gmm, batch=4, workers=1 )
    gmm_model = GMM()
    test_GMM(gmm_model, dataset_loader_gmm)
    tom_model = UnetGenerator(26, 4, 6, ngf=64, norm_layer=nn.InstanceNorm2d)
    dataset_tom = CPDataset("TOM")
    dataset_loader_tom = CPDataLoader(dataset_tom, batch=4, workers=1, )
    test_tom(dataset_loader_tom, tom_model)


if __name__ == '__main__':
    virtuon()
