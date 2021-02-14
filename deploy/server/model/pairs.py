import os

def pairs():
    base_dir_1 = os.path.abspath('media')
    base_dir_2 = os.path.abspath('model/input/test/test/')
    img_dir_1 = os.path.join(base_dir_1, 'image')
    cloth_dir_1 = os.path.join(base_dir_1, 'cloth')
    img_dir_2 = os.path.join(base_dir_2, 'image')
    cloth_dir_2 = os.path.join(base_dir_2, 'cloth')


    img_name = os.listdir(img_dir_1)
    cloth_name = os.listdir(cloth_dir_1)
    for i in range(len(img_name)):
        os.rename(os.path.join(img_dir_1, img_name[i]), os.path.join(img_dir_2, "d{}.jpg".format(i)))
        os.rename(os.path.join(cloth_dir_1, cloth_name[i]), os.path.join(cloth_dir_2, "c{}.jpg".format(i)))

if __name__ == '__main__':
    pairs()