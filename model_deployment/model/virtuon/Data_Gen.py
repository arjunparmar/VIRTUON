import torch
import os.path as osp
from PIL import Image, ImageDraw
from torchvision import transforms
import numpy as np
import torch.utils.data as data
import json


class CPDataset(data.Dataset):
    def __init__(self, stage='GMM', all_root=osp.abspath(''), data_path="./model/input/", mode="test", radius=5, img_height=256,
                 img_width=192):
        super(CPDataset, self).__init__()

        self.root = all_root

        self.data_root = osp.join(all_root, data_path)

        self.datamode = mode

        self.stage = stage

        self.data_list = "".join([mode, "_pairs.txt"])

        self.fine_height = img_height

        self.fine_width = img_width

        self.radius = radius

        self.data_path = osp.join(all_root, data_path)

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        self.transform_1 = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5), (0.5))
        ])

        self.transform_2 = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5), (0.5, 0.5))
        ])

        self.transform_3 = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        im_names = []
        c_names = []

        with open(osp.join(self.data_root, self.data_list), 'r') as f:
            for line in f.readlines():
                im_name, c_name = line.strip().split()
                im_names.append(im_name)
                c_names.append(c_name)

        self.im_names = im_names
        self.c_names = c_names

    def name(self):
        return "CPDataset"

    def __getitem__(self, index):
        c_name = self.c_names[index]
        im_name = self.im_names[index]
        if self.stage == "GMM":
            c = Image.open(osp.join(self.data_path, 'cloth', c_name))
            cm = Image.open(osp.join(self.data_path, 'cloth-mask', c_name)).convert('L')
        else:
            c = Image.open(osp.join(self.data_path, 'warp-cloth', im_name))
            cm = Image.open(osp.join(self.data_path, 'warp-mask', im_name)).convert('L')

        c = self.transform(c)
        cm_array = np.array(cm)
        cm_array = (cm_array >= 128).astype(np.float32)
        cm = torch.from_numpy(cm_array)
        cm.unsqueeze_(0)

        # person image
        im = Image.open(osp.join(self.data_path, 'image', im_name))
        im = self.transform(im)

        # LIP labels

        # [(0, 0, 0),    # 0=Background
        #  (128, 0, 0),  # 1=Hat
        #  (255, 0, 0),  # 2=Hair
        #  (0, 85, 0),   # 3=Glove
        #  (170, 0, 51),  # 4=SunGlasses
        #  (255, 85, 0),  # 5=UpperClothes
        #  (0, 0, 85),     # 6=Dress
        #  (0, 119, 221),  # 7=Coat
        #  (85, 85, 0),    # 8=Socks
        #  (0, 85, 85),    # 9=Pants
        #  (85, 51, 0),    # 10=Jumpsuits
        #  (52, 86, 128),  # 11=Scarf
        #  (0, 128, 0),    # 12=Skirt
        #  (0, 0, 255),    # 13=Face
        #  (51, 170, 221),  # 14=LeftArm
        #  (0, 255, 255),   # 15=RightArm
        #  (85, 255, 170),  # 16=LeftLeg
        #  (170, 255, 85),  # 17=RightLeg
        #  (255, 255, 0),   # 18=LeftShoe
        #  (255, 170, 0)    # 19=RightShoe
        #  (170, 170, 50)   # 20=Skin/Neck/Chest (Newly added after running dataset_neck_skin_correction.py)
        #  ]

        # load parsing image
        parse_name = im_name.replace('.jpg', '.png')
        im_parse = Image.open(osp.join(self.data_path, 'image-parse-new', parse_name)).convert('L')
        parse_array = np.array(im_parse)

        im_mask = Image.open(osp.join(self.data_path, 'image-mask', parse_name)).convert('L')
        mask_array = np.array(im_mask)

        parse_shape = (mask_array > 0).astype(np.float32)

        if self.stage == 'GMM':
            parse_head = (parse_array == 1).astype(np.float32) + (parse_array == 4).astype(np.float32) + (
                    parse_array == 13).astype(np.float32)

        else:
            parse_head = (parse_array == 1).astype(np.float32) + (parse_array == 2).astype(np.float32) + (
                    parse_array == 4).astype(np.float32) + (parse_array == 9).astype(np.float32) + (
                                 parse_array == 12).astype(np.float32) + (parse_array == 13).astype(np.float32) + (
                                 parse_array == 16).astype(np.float32) + (parse_array == 17).astype(np.float32)

        parse_cloth = (parse_array == 5).astype(np.float32) + (parse_array == 6).astype(np.float32) + (
                parse_array == 7).astype(np.float32)

        parse_shape_ori = Image.fromarray((parse_shape * 255).astype(np.uint8))

        parse_shape = parse_shape_ori.resize((self.fine_width // 16, self.fine_height // 16), Image.BILINEAR)

        parse_shape = parse_shape.resize((self.fine_width, self.fine_height), Image.BILINEAR)

        parse_shape_ori = parse_shape_ori.resize((self.fine_width, self.fine_height), Image.BILINEAR)

        shape_ori = self.transform_1(parse_shape_ori)

        shape = self.transform_1(parse_shape)

        phead = torch.from_numpy(parse_head)

        pcm = torch.from_numpy(parse_cloth)

        # Upper Cloth
        im_c = im * pcm + (1 - pcm)
        im_h = im * phead + (1 - phead)

        # load pose points
        pose_name = im_name.replace('.jpg', '_keypoints.json')
        with open(osp.join(self.data_path, 'pose', pose_name), 'r') as f:
            pose_label = json.load(f)
            pose_data = pose_label['people'][0]['pose_keypoints']
            pose_data = np.array(pose_data)
            pose_data = pose_data.reshape([-1, 3])

        point_num = pose_data.shape[0]
        pose_map = torch.zeros(point_num, self.fine_height, self.fine_width)

        r = self.radius

        im_pose = Image.new('L', (self.fine_width, self.fine_height))
        pose_draw = ImageDraw.Draw(im_pose)

        for i in range(point_num):
            one_map = Image.new('L', (self.fine_width, self.fine_height))
            draw = ImageDraw.Draw(one_map)
            pointx = pose_data[i, 0]
            pointy = pose_data[i, 1]

            if pointx > 1 and pointy > 1:
                draw.rectangle((pointx - r, pointy - r, pointx + r, pointy + r), 'white', 'white')
                pose_draw.rectangle((pointx - r, pointy - r, pointx + r, pointy + r), 'white', 'white')

            one_map = self.transform_1(one_map)
            pose_map[i] = one_map[0]

        im_pose = self.transform_1(im_pose)

        agnostic = torch.cat([shape, im_h, pose_map], 0)

        # if self.stage == 'GMM':
        #     im_g = Image.open(osp.join(self.root, 'grid.png'))
        #     im_g = self.transform(im_g)
        # else:
        #     im_g = ''

        pcm.unsqueeze_(0)

        result = {
            'c_name': c_name,
            'im_name': im_name,
            'cloth': c,
            'cloth_mask': cm,
            'image': im,
            'agnostic': agnostic,
            'parse_cloth': im_c,
            'shape': shape,
            'head': im_h,
            'pose_image': im_pose,
            # 'grid_image': im_g,
            'parse_cloth_mask': pcm,
            'shape_ori': shape_ori,
        }

        return result

    def __len__(self):
        return len(self.im_names)


class CPDataLoader(object):
    def __init__(self, dataset, shuffle=True, batch=4, workers=4):
        super(CPDataLoader, self).__init__()

        if shuffle:
            train_sampler = torch.utils.data.sampler.RandomSampler(dataset)
        else:
            train_sampler = None

        self.data_loader = torch.utils.data.DataLoader(
            dataset, batch_size=batch, shuffle=(train_sampler is None),
            num_workers=workers, pin_memory=False, sampler=train_sampler
        )
        self.dataset = dataset
        self.data_iter = self.data_loader.__iter__()

    def next_batch(self):
        try:
            batch = self.data_iter.__next__()
        except StopIteration:
            self.data_iter = self.data_loader.__iter__()
            batch = self.data_iter.__next__()

        return batch
