import torch
import torch.nn as nn

gpu_available = torch.cuda.is_available()
if gpu_available:
	device = torch.device("cuda")
else:
	device = torch.device("cpu")

class DT(nn.Module):
    def __init__(self):
        super(DT, self).__init__()

    def forward(self, x1, x2):
        dt = torch.abs(x1 - x2)
        return dt


class DT2(nn.Module):
    def __init__(self):
        super(DT2, self).__init__()

    def forward(self, x1, y1, x2, y2):
        dt = torch.sqrt(torch.mul(x1 - x2, x1 - x2) + torch.mul(y1 - y2, y1 - y2))
        return dt


class GicLoss(nn.Module):
    def __init__(self, fine_height=256, fine_width=192):
        super(GicLoss, self).__init__()

        self.dT = DT()

        self.fine_height = fine_height
        self.fine_width = fine_width

    def forward(self, grid):
        Gx = grid[:, :, :, 0]
        Gy = grid[:, :, :, 1]

        Gxcenter = Gx[:, 1:self.fine_height - 1, 1:self.fine_width - 1]
        # Gxup = Gx[:, 0:self.fine_height - 2, 1:self.fine_width - 1]
        # Gxdown = Gx[:, 2:self.fine_height, 1:self.fine_width - 1]
        Gxleft = Gx[:, 1:self.fine_height - 1, 0:self.fine_width - 2]
        Gxright = Gx[:, 1:self.fine_height - 1, 2:self.fine_width]

        Gycenter = Gy[:, 1:self.fine_height - 1, 1:self.fine_width - 1]
        Gyup = Gy[:, 0:self.fine_height - 2, 1:self.fine_width - 1]
        Gydown = Gy[:, 2:self.fine_height, 1:self.fine_width - 1]
        # Gyleft = Gy[:, 1:self.fine_height - 1, 0:self.fine_width - 2]
        # Gyright = Gy[:, 1:self.fine_height - 1, 2:self.fine_width]

        dtleft = self.dT(Gxleft, Gxcenter)
        dtright = self.dT(Gxright, Gxcenter)
        dtup = self.dT(Gyup, Gycenter)
        dtdown = self.dT(Gydown, Gycenter)

        return torch.sum(torch.abs(dtleft - dtright) + torch.abs(dtup - dtdown))


class VGGLoss(nn.Module):
    def __init__(self, layids=None):
        super(VGGLoss, self).__init__()
        self.vgg = Vgg19()
        self.vgg.to(device)
        self.criterion = nn.L1Loss()
        self.weights = [1.0 / 32, 1.0 / 16, 1.0 / 8, 1.0 / 4, 1.0]
        self.layids = layids

    def forward(self, x, y):
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        loss = 0
        if self.layids is None:
            self.layids = list(range(len(x_vgg)))
        for i in self.layids:
            loss += self.weights[i] * \
                    self.criterion(x_vgg[i], y_vgg[i].detach())
        return loss
