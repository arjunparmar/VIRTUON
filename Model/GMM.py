import torch
import torch.nn as nn
from utils import save_images, save_checkpoint, load_checkpoint, board_add_images, board_add_image, tensor_for_board, tensor_list_for_board
from Data_Gen import CPDataset, CPDataLoader
import numpy as np
import time
import torch.nn.functional as F
from losses import DT, DT2, GicLoss
import os


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


class TpsGridGen(nn.Module):
    def __init__(self, out_h=256, out_w=192, use_regular_grid=True, grid_size=5, reg_factor=0, use_cuda=True):
        super(TpsGridGen, self).__init__()
        self.out_h, self.out_w = out_h, out_w
        self.reg_factor = reg_factor
        self.use_cuda = use_cuda

        # create grid in numpy
        self.grid = np.zeros([self.out_h, self.out_w, 3], dtype=np.float32)
        # sampling grid with dim-0 coords (Y)
        self.grid_X, self.grid_Y = np.meshgrid(
            np.linspace(-1, 1, out_w), np.linspace(-1, 1, out_h))
        # grid_X,grid_Y: size [1,H,W,1,1]
        self.grid_X = torch.FloatTensor(self.grid_X).unsqueeze(0).unsqueeze(3)
        self.grid_Y = torch.FloatTensor(self.grid_Y).unsqueeze(0).unsqueeze(3)
        if use_cuda:
            self.grid_X = self.grid_X.cuda()
            self.grid_Y = self.grid_Y.cuda()

        # initialize regular grid for control points P_i
        if use_regular_grid:
            axis_coords = np.linspace(-1, 1, grid_size)
            self.N = grid_size * grid_size
            P_Y, P_X = np.meshgrid(axis_coords, axis_coords)
            P_X = np.reshape(P_X, (-1, 1))  # size (N,1)
            P_Y = np.reshape(P_Y, (-1, 1))  # size (N,1)
            P_X = torch.FloatTensor(P_X)
            P_Y = torch.FloatTensor(P_Y)
            self.P_X_base = P_X.clone()
            self.P_Y_base = P_Y.clone()
            self.Li = self.compute_L_inverse(P_X, P_Y).unsqueeze(0)
            self.P_X = P_X.unsqueeze(2).unsqueeze(
                3).unsqueeze(4).transpose(0, 4)
            self.P_Y = P_Y.unsqueeze(2).unsqueeze(
                3).unsqueeze(4).transpose(0, 4)
            if use_cuda:
                self.P_X = self.P_X.cuda()
                self.P_Y = self.P_Y.cuda()
                self.P_X_base = self.P_X_base.cuda()
                self.P_Y_base = self.P_Y_base.cuda()

    def forward(self, theta):
        warped_grid = self.apply_transformation(
            theta, torch.cat((self.grid_X, self.grid_Y), 3))

        return warped_grid

    def compute_L_inverse(self, X, Y):
        N = X.size()[0]  # num of points (along dim 0)
        # construct matrix K
        Xmat = X.expand(N, N)
        Ymat = Y.expand(N, N)
        P_dist_squared = torch.pow(
            Xmat - Xmat.transpose(0, 1), 2) + torch.pow(Ymat - Ymat.transpose(0, 1), 2)
        # make diagonal 1 to avoid NaN in log computation
        P_dist_squared[P_dist_squared == 0] = 1
        K = torch.mul(P_dist_squared, torch.log(P_dist_squared))
        # construct matrix L
        O = torch.FloatTensor(N, 1).fill_(1)
        Z = torch.FloatTensor(3, 3).fill_(0)
        P = torch.cat((O, X, Y), 1)
        L = torch.cat((torch.cat((K, P), 1), torch.cat(
            (P.transpose(0, 1), Z), 1)), 0)
        Li = torch.inverse(L)
        if self.use_cuda:
            Li = Li.cuda()
        return Li

    def apply_transformation(self, theta, points):
        if theta.dim() == 2:
            theta = theta.unsqueeze(2).unsqueeze(3)
        # points should be in the [B,H,W,2] format,
        # where points[:,:,:,0] are the X coords
        # and points[:,:,:,1] are the Y coords

        # input are the corresponding control points P_i
        batch_size = theta.size()[0]
        # split theta into point coordinates
        Q_X = theta[:, :self.N, :, :].squeeze(3)
        Q_Y = theta[:, self.N:, :, :].squeeze(3)
        Q_X = Q_X + self.P_X_base.expand_as(Q_X)
        Q_Y = Q_Y + self.P_Y_base.expand_as(Q_Y)

        # get spatial dimensions of points
        points_b = points.size()[0]
        points_h = points.size()[1]
        points_w = points.size()[2]

        # repeat pre-defined control points along spatial dimensions of points to be transformed
        P_X = self.P_X.expand((1, points_h, points_w, 1, self.N))
        P_Y = self.P_Y.expand((1, points_h, points_w, 1, self.N))

        # compute weigths for non-linear part
        W_X = torch.bmm(self.Li[:, :self.N, :self.N].expand(
            (batch_size, self.N, self.N)), Q_X)
        W_Y = torch.bmm(self.Li[:, :self.N, :self.N].expand(
            (batch_size, self.N, self.N)), Q_Y)
        # reshape
        # W_X,W,Y: size [B,H,W,1,N]
        W_X = W_X.unsqueeze(3).unsqueeze(4).transpose(
            1, 4).repeat(1, points_h, points_w, 1, 1)
        W_Y = W_Y.unsqueeze(3).unsqueeze(4).transpose(
            1, 4).repeat(1, points_h, points_w, 1, 1)
        # compute weights for affine part
        A_X = torch.bmm(self.Li[:, self.N:, :self.N].expand(
            (batch_size, 3, self.N)), Q_X)
        A_Y = torch.bmm(self.Li[:, self.N:, :self.N].expand(
            (batch_size, 3, self.N)), Q_Y)
        # reshape
        # A_X,A,Y: size [B,H,W,1,3]
        A_X = A_X.unsqueeze(3).unsqueeze(4).transpose(
            1, 4).repeat(1, points_h, points_w, 1, 1)
        A_Y = A_Y.unsqueeze(3).unsqueeze(4).transpose(
            1, 4).repeat(1, points_h, points_w, 1, 1)

        # compute distance P_i - (grid_X,grid_Y)
        # grid is expanded in point dim 4, but not in batch dim 0, as points P_X,P_Y are fixed for all batch
        points_X_for_summation = points[:, :, :, 0].unsqueeze(
            3).unsqueeze(4).expand(points[:, :, :, 0].size() + (1, self.N))
        points_Y_for_summation = points[:, :, :, 1].unsqueeze(
            3).unsqueeze(4).expand(points[:, :, :, 1].size() + (1, self.N))

        if points_b == 1:
            delta_X = points_X_for_summation - P_X
            delta_Y = points_Y_for_summation - P_Y
        else:
            # use expanded P_X,P_Y in batch dimension
            delta_X = points_X_for_summation - \
                      P_X.expand_as(points_X_for_summation)
            delta_Y = points_Y_for_summation - \
                      P_Y.expand_as(points_Y_for_summation)

        dist_squared = torch.pow(delta_X, 2) + torch.pow(delta_Y, 2)
        # U: size [1,H,W,1,N]
        dist_squared[dist_squared == 0] = 1  # avoid NaN in log computation
        U = torch.mul(dist_squared, torch.log(dist_squared))

        # expand grid in batch dimension if necessary
        points_X_batch = points[:, :, :, 0].unsqueeze(3)
        points_Y_batch = points[:, :, :, 1].unsqueeze(3)
        if points_b == 1:
            points_X_batch = points_X_batch.expand(
                (batch_size,) + points_X_batch.size()[1:])
            points_Y_batch = points_Y_batch.expand(
                (batch_size,) + points_Y_batch.size()[1:])

        points_X_prime = A_X[:, :, :, :, 0] + \
                         torch.mul(A_X[:, :, :, :, 1], points_X_batch) + \
                         torch.mul(A_X[:, :, :, :, 2], points_Y_batch) + \
                         torch.sum(torch.mul(W_X, U.expand_as(W_X)), 4)

        points_Y_prime = A_Y[:, :, :, :, 0] + \
                         torch.mul(A_Y[:, :, :, :, 1], points_X_batch) + \
                         torch.mul(A_Y[:, :, :, :, 2], points_Y_batch) + \
                         torch.sum(torch.mul(W_Y, U.expand_as(W_Y)), 4)

        return torch.cat((points_X_prime, points_Y_prime), 3)


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