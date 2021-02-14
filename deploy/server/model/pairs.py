import os

base_dir_1 = os.path.abspath('media/')
base_dir_2 = os.path.abspath('model/input/test/test/')
img_dir = os.path.join(base_dir, 'image')
cloth_dir = os.path.join(base_dir, 'cloth')

img_name = os.listdir(img_dir)
cloth_name = os.listdir(cloth_dir)
for i in range(len(img_name)):
    os.rename(os.path.join(img_dir, img_name[i]), os.path.join(img_dir, "d{}".format(i)))
    os.rename(os.path.join(cloth_dir, cloth_name[i]), os.path.join(cloth_dir, "c{}".format(i)))