from __future__ import print_function, division
import os
from PIL import Image
from torch.utils.data import Dataset
from .mypath_cihp import Path
import pandas as pd
import random


class VOCSegmentation(Dataset):
    """
    CIHP dataset
    """

    def __init__(self,
                 base_dir=Path.db_root_dir('cihp'),
                 split='train',
                 transform=None,
                 flip=False,
                 cloth = False,
                 ):
        """
        :param base_dir: path to CIHP dataset directory
        :param split: train/val/test
        :param transform: transform to apply
        """
        super(VOCSegmentation).__init__()

        if not cloth:
            self._image_dir = os.path.join(base_dir, 'image')
        else:
            self._image_dir = os.path.join(base_dir, 'cloth')

        self.transform = transform

        if (not cloth):
            self.im_ids = pd.read_csv('./model/input/test_pairs.txt', sep = " ", header = None, ).iloc[:,0].to_list()
            self.im_ids = [x[:-4] for x in self.im_ids]
            self.images = pd.read_csv('./model/input/test_pairs.txt', sep = " ", header = None, ).iloc[:,0].to_list()
            self.images = [os.path.join(self._image_dir, x ) for x in self.images]
        else:
            self.im_ids = pd.read_csv('./model/input/test_pairs.txt', sep=" ", header=None, ).iloc[:, 1].to_list()
            self.im_ids = [x[:-4] for x in self.im_ids]
            self.images = pd.read_csv('./model/input/test_pairs.txt', sep=" ", header=None, ).iloc[:, 1].to_list()
            self.images = [os.path.join(self._image_dir, x) for x in self.images]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        _img= self._make_img_gt_point_pair(index)
        sample = {'image': _img}

        if self.transform is not None:
            sample = self.transform(sample)

        return sample

    def _make_img_gt_point_pair(self, index):

        _img = Image.open(self.images[index]).convert('RGB')  # return is RGB pic

        return _img

    def __str__(self):
        return 'CIHP(split=' + str(self.split) + ')'



