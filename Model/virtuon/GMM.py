import torch
import torch.nn as nn
from utils import save_images, save_checkpoint, board_add_images, TpsGridGen, dir, load_checkpoint
import time
import torch.nn.functional as F
from losses import GicLoss
import os
import os.path as osp


class FeatureL2Norm(torch.nn.Module):
    def __init__(self):
        super(FeatureL2Norm, self).__init__()

    def forward(self, feature):
        epsilon = 1e-6
        norm = torch.pow(torch.sum(torch.pow(feature, 2), 1) +
                         epsilon, 0.5).unsqueeze(1).expand_as(feature)
        return torch.div(feature, norm)


class FeatureExtraction(nn.Module):
    def __init__(self, input_nc, ngf=64, n_layers=3, use_dropout=False):
        super(FeatureExtraction, self).__init__()

        downconv = nn.Conv2d(input_nc, ngf, kernel_size=4, stride=2, padding=1)

        model = [downconv, nn.ReLU(True), nn.BatchNorm2d(ngf)]

        for i in range(n_layers):
            in_ngf = 2 ** i * ngf if 2 ** i * ngf < 512 else 512
            out_ngf = 2 ** (i + 1) * ngf if 2 ** i * ngf < 512 else 512
            downconv = nn.Conv2d(in_ngf, out_ngf, kernel_size=4, stride=2, padding=1)
            model.append(downconv)
            model.append(nn.ReLU(True))
            model.append(nn.BatchNorm2d(out_ngf))

        model.append(nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1))
        model.append(nn.ReLU(True))
        model.append(nn.BatchNorm2d(512))
        model.append(nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1))
        model.append(nn.ReLU(True))

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)


class FeatureCorrelation(nn.Module):
    def __init__(self):
        super(FeatureCorrelation, self).__init__()

    def forward(self, feature_A, feature_B):
        b, c, h, w = feature_A.size()

        feature_A = feature_A.transpose(2, 3).contiguous().view(b, c, h * w)
        feature_B = feature_B.contiguous().view(b, c, h * w).transpose(1, 2)

        feature_mul = torch.bmm(feature_B, feature_A)
        correlation_tensor = feature_mul.view(b, h, w, h * w).transpose(2, 3).transpose(1, 2)

        return correlation_tensor
        # return feature_mul


class FeatureRegression(nn.Module):
    def __init__(self, input_nc=512, output_dim=50, use_cuda=True):
        super(FeatureRegression, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(input_nc, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.Conv2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True)
        )
        self.linear = nn.Linear(64 * 4 * 3, output_dim)
        self.tanh = nn.Tanh()
        if use_cuda:
            self.conv.cuda()
            self.tanh.cuda()
            self.linear.cuda()

    def forward(self, x):
        x = self.conv(x)
        # x = x.view(x.size(0), -1)
        x = x.reshape(x.shape[0], -1)
        x = self.linear(x)
        x = self.tanh(x)
        return x


class GMM(nn.Module):
    def __init__(self, grid_size=5, fine_height=256, fine_width=192):
        super(GMM, self).__init__()
        self.extractionA = FeatureExtraction(22, ngf=64, n_layers=3)
        self.extractionB = FeatureExtraction(1, ngf=64, n_layers=3)
        self.l2norm = FeatureL2Norm()
        self.correlation = FeatureCorrelation()
        self.regression = FeatureRegression(input_nc=192, output_dim=2 * grid_size ** 2, use_cuda=True)
        self.gridGen = TpsGridGen(fine_height, fine_width, use_cuda=True, grid_size=grid_size)

    def forward(self, inputA, inputB):
        featureA = self.extractionA(inputA)
        featureB = self.extractionB(inputB)
        featureA = self.l2norm(featureA)
        featureB = self.l2norm(featureB)
        correlation = self.correlation(featureA.cuda(), featureB.cuda())

        theta = self.regression(correlation)
        grid = self.gridGen(theta)
        return grid, theta


def train_GMM(model, train_loader, board, lr=1e-4, keep_step=100000, decay_step=100000, save_count=500,
              display_count=100, checkpoint_dir="/content/checkpoint", name='GMM'):
    model.cuda()
    model.train()

    L1loss = nn.L1Loss()
    Gicloss = GicLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, betas=(0.5, 0.999))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,
                                                  lr_lambda=lambda step: 1.0 - (
                                                          max(0, step - keep_step) / float(decay_step + 1)))

    for step in range(keep_step + decay_step):
        iter_start_time = time.time()
        inputs = train_loader.next_batch()

        im = inputs['image'].cuda()  # full image of person
        im_pose = inputs['pose_image'].cuda()  # pose channels
        im_h = inputs['head'].cuda()  # person head Image
        shape = inputs['shape'].cuda()  # blurred binary mask of person and
        agnostic = inputs['agnostic'].cuda()  # person Representation for GMM
        c = inputs['cloth'].cuda()  # in shop cloths
        cm = inputs['cloth_mask'].cuda()  # in shop cloth mask
        im_c = inputs['parse_cloth'].cuda()  # GT for GMM
        im_g = inputs['grid_image'].cuda()  # grid image for Viz.
        pcm = inputs['parse_cloth_mask'].cuda()

        grid, theta = model(agnostic, cm)
        warped_cloth = F.grid_sample(c, grid, padding_mode='border', align_corners=True)
        warped_mask = F.grid_sample(cm, grid, padding_mode='zeros', align_corners=True)
        warped_grid = F.grid_sample(im_g, grid, padding_mode='zeros', align_corners=True)

        visuals = [[im_h, shape, im_pose],
                   [c, warped_cloth, im_c],
                   [warped_grid, (warped_cloth + im) * 0.5, im]]

        Lwarp = L1loss(warped_mask, pcm)

        Lgic = Gicloss(grid)

        Lgic = Lgic / (grid.shape[0] * grid.shape[1] * grid.shape[2])

        loss = Lwarp + 40 * Lgic

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (step + 1) % display_count == 0:
            board_add_images(board, 'combine', visuals, step + 1)
            board.add_scalar('loss', loss.item(), step + 1)
            board.add_scalar('40*Lgic', (40 * Lgic).item(), step + 1)
            board.add_scalar('Lwarp', Lwarp.item(), step + 1)
            t = time.time() - iter_start_time
            print('step: %8d, time: %.3f, loss: %4f, (40*Lgic): %.8f, Lwarp: %.6f' %
                  (step + 1, t, loss.item(), (40 * Lgic).item(), Lwarp.item()), flush=True)
        if (step + 1) % save_count == 0:
            save_checkpoint(model, os.path.join(
                checkpoint_dir, name, 'step_%06d.pth' % (step + 1)))


def test_GMM(model, test_loader, checkpoint_path=os.path.abspath('virtuon/PreTrainedModels'), name="GMM",
             model_name="PreTrainedGMM", result_dir=os.path.abspath('input/test/test')):
    model_path = osp.join(checkpoint_path, name, model_name + ".pth")
    load_checkpoint(model, model_path)

    model.cuda()
    model.eval()

    save_dir = osp.join(result_dir)

    if not osp.exists(save_dir):
        os.makedirs(save_dir)

    warp_cloth_dir = dir(save_dir, 'warp-cloth')

    warp_mask_dir = dir(save_dir, 'warp-mask')

    result_dir = dir(save_dir, 'result-dir')

    overlayed_TPS_dir = dir(save_dir, 'overlayed-TPS')

    # warped_grid_dir = dir(save_dir, 'warped_grid')

    for step, inputs in enumerate(test_loader.data_loader):
        iter_start_time = time.time()

        c_names = inputs['c_name']
        im_names = inputs['im_name']
        im = inputs['image'].cuda()
        # im_pose = inputs['pose_image'].cuda()
        # im_h = inputs['head'].cuda()
        # shape = inputs['shape'].cuda()
        agnostic = inputs['agnostic'].cuda()
        c = inputs['cloth'].cuda()
        cm = inputs['cloth_mask'].cuda()
        # im_c = inputs['parse_cloth'].cuda()
        # im_g = inputs['grid_image'].cuda()
        shape_ori = inputs['shape_ori']  # original body shape without blurring

        grid, theta = model(agnostic, cm)
        warped_cloth = F.grid_sample(c, grid, padding_mode='border', align_corners=True)
        warped_mask = F.grid_sample(cm, grid, padding_mode='zeros', align_corners=True)
        # warped_grid = F.grid_sample(im_g, grid, padding_mode='zeros', align_corners=True)
        overlay = 0.7 * warped_cloth + 0.3 * im

        # visuals = [[im_h, shape, im_pose],
        #            [c, warped_cloth, im_c],
        #            [warped_grid, (warped_cloth + im) * 0.5, im]]

        # save_images(warped_cloth, c_names, warp_cloth_dir)
        # save_images(warped_mask * 2 - 1, c_names, warp_mask_dir)
        save_images(warped_cloth, im_names, warp_cloth_dir)
        save_images(warped_mask * 2 - 1, im_names, warp_mask_dir)
        save_images(shape_ori.cuda() * 0.2 + warped_cloth *
                    0.8, im_names, result_dir)
        # save_images(warped_grid, im_names, warped_grid_dir)
        save_images(overlay, im_names, overlayed_TPS_dir)

        # if (step + 1) % 100 == 0:
        #     #     board_add_images(board, 'combine', visuals, step+1)
        #     t = time.time() - iter_start_time
        #     print('step: %8d, time: %.3f' % (step + 1, t), flush=True)
