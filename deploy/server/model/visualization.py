from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

img_dir = os.path.abspath('input/test/test/image')
cloth_dir = os.path.abspath('input/test/test/cloth')
output_dir = os.path.abspath('output/try-on')

pairs = pd.read_csv(os.path.abspath('input/test/test_pairs.txt'), delimiter=" ", header=None).to_numpy()
for num in range(0, 4):
    img_name = pairs[num, 0]
    cloth_name = pairs[num, 1]

    img_path = os.path.join(img_dir, img_name)
    cloth_path = os.path.join(cloth_dir, cloth_name)
    output_path = os.path.join(output_dir, img_name)

    img = Image.open(img_path)
    cloth = Image.open(cloth_path)
    output = Image.open(output_path)

    img_array = np.asarray(img)
    cloth_array = np.asarray(cloth)
    output_array = np.asarray(output)

    # print(img_array.shape)
    final = np.concatenate([img_array, cloth_array, output_array], axis=1)

    try:
        final_list = np.concatenate([final_list, final], axis=0)
    except:
        final_list = final

    # final_list.append(final)

final = Image.fromarray(final_list)

final.save('output.jpg')
