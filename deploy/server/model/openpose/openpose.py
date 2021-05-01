import cv2
import os
import sys

sys.path.append(os.path.abspath('model/openpose'))

import numpy as np
from src_pose.body import Body
import json
import torch
from torchvision import transforms
from PIL import Image, ImageDraw
from time import sleep
import matplotlib.pyplot as plt

base_dir = os.path.abspath('./model/input/test/test')
if not os.path.exists(os.path.join(base_dir, 'image')):
    				os.makedirs(os.path.join(base_dir, 'image'))
if not os.path.exists(os.path.join(base_dir, 'pose')):
    				os.makedirs(os.path.join(base_dir, 'pose'))
IMG_DIR = os.path.join(base_dir, 'image')


def openpose():
    body_estimation = Body(os.path.abspath('./model/openpose/model_pose/body_pose_model.pth'))


    list_of_image = os.listdir(IMG_DIR)
    # print(list_of_image)
    for img in list_of_image:
        img_name = os.path.join(IMG_DIR,img)
        json_name = os.path.join(IMG_DIR.replace('/image','/pose'), img[:-4]+"_keypoints.json")
        oriImg = cv2.imread(img_name)
        oriImg = cv2.resize(oriImg, (192, 256))
        candidate, subset = body_estimation(oriImg)

        candidate_ = candidate[:, :-1]
        subset_ = subset.flatten().tolist()
        # canvas = copy.deepcopy(oriImg)
        # n = candidate.shape[0]
        x = np.zeros((18, 3))

        j = 0
        for i in range(18):
            if subset_[i] == -1:
                x[i, :] = np.zeros((1, 3))
            else:
                x[i, :] = candidate_[j, :]
                j = j + 1

        pose_points = x.flatten().tolist()

        pose_dict = {
            "face_keypoints": [],
            "pose_keypoints": pose_points,
            "hand_right_keypoints": [],
            "hand_left_keypoints": []
        }

        people = [pose_dict]
        joints_json = {"version": 1.0, "people": people}
        with open(json_name, 'w') as joint_file:
            json.dump(joints_json, joint_file)
            
if __name__ == '__main__':
    openpose()
