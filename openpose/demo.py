import cv2
import matplotlib.pyplot as plt
import copy
import numpy as np
import json
import os
from src import model
from src import util
from src.body import Body
from src.hand import Hand

IMG_DIR = '/home/taksh/dev/python/pytorch-openpose/test'

def main():
    body_estimation = Body('model/body_pose_model.pth')

    list_of_image = os.listdir(IMG_DIR)

    for img in list_of_image:
        img_name = os.path.join(IMG_DIR,img)
        json_name = os.path.join(IMG_DIR.replace('/test','/pose'), img[:-4]+"_keypoints.json")

        oriImg = cv2.imread(img_name)
        candidate, subset = body_estimation(oriImg)
        candidate_ = candidate[:,:-1]
        subset_ = subset.flatten().tolist()
        canvas = copy.deepcopy(oriImg)
        n = candidate.shape[0]
        x = np.zeros((18,3))

        j = 0
        for i in range(18):
            if subset_[i] == -1:
                x[i,:] = np.zeros((1,3))
            else:
                x[i,:] = candidate_[j,:]
                j+=1

        pose_points = x.flatten().tolist()


        pose_dict = {
                "face_keypoints": [],
                      "pose_keypoints":pose_points ,
                      "hand_right_keypoints": [],
                      "hand_left_keypoints":[]
        }


        people   = [pose_dict]
        joints_json =  { "version": 1.0, "people": people }
        with  open(json_name, 'w') as joint_file:
                json.dump(joints_json, joint_file)

        canvas = util.draw_bodypose(canvas, candidate, subset)
        plt.imshow(canvas[:, :, [2, 1, 0]])
        plt.axis('off')
        plt.show()
if __name__ == '__main__':
    print("main")
    main()
