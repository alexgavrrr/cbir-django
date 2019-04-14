from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from cbir.legacy_utils import L2Norm
from cbir import CONFIG
import cbir


def detect(detector, image, max_keypoints, out=None):
    '''
    Detector object must have method 'detect()'
    '''
    kp = detector.detect(image)
    # kp = filter_points(kp)
    return sorted(kp,
                  key=lambda x: x.response,
                  reverse=True)[:max_keypoints]


def compute(model, image, patch_size, kp, expansion_coef=1.3,
            bs=64, use_cuda=True):
    '''
    Model must be a pytorch neural network
    '''
    if len(kp) <= 1:
        return None, None

    patches = [cv2.resize(cv2.getRectSubPix(image,
                                            (round(k.size * expansion_coef),
                                             round(k.size * expansion_coef)),
                                            k.pt),
                          (patch_size, patch_size)) for k in kp]
    # N_DRAW = 30
    # patches_to_draw = random.sample(patches, N_DRAW)
    # draw_patches(patches_to_draw)

    patches = Variable(torch.FloatTensor(np.array(patches).reshape(-1, 1, patch_size, patch_size)))
    it = int(patches.size()[0] / bs)
    r = patches.size()[0] % bs

    des = []
    for i in range(it):
        if use_cuda:
            des.append(model(patches[i * bs:(i + 1) * bs].cuda()).cpu().data.numpy())
        else:
            des.append(model(patches[i * bs:(i + 1) * bs]).cpu().data.numpy())

    if r > 1:
        rem = patches[it * bs:]
        if use_cuda:
            rem = rem.cuda()
        des.append(model(rem).cpu().data.numpy())

    return kp, np.concatenate(des, 0)


class L2net(nn.Module):
    def __init__(self):
        super(L2net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32, affine=False)
        self.conv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(32, affine=False)
        self.conv3 = nn.Conv2d(32, 64, 3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(64, affine=False)
        self.conv4 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn4 = nn.BatchNorm2d(64, affine=False)
        self.conv5 = nn.Conv2d(64, 128, 3, stride=2, padding=1)
        self.bn5 = nn.BatchNorm2d(128, affine=False)
        self.conv6 = nn.Conv2d(128, 128, 3, padding=1)
        self.bn6 = nn.BatchNorm2d(128, affine=False)
        self.conv7 = nn.Conv2d(128, 128, 8)
        self.bn7 = nn.BatchNorm2d(128, affine=False)

    def input_norm(self, x):
        flat = x.view(x.size(0), -1)
        mp = torch.mean(flat, dim=1)
        sp = torch.std(flat, dim=1) + 1e-7
        return ((x - mp.detach().unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand_as(x)) /
                sp.detach().unsqueeze(-1).unsqueeze(-1).unsqueeze(1).expand_as(x))

    def forward(self, x):
        x = self.input_norm(x)
        int1 = self.bn1(self.conv1(x))
        x = F.relu(int1)
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        x = F.relu(self.bn6(self.conv6(x)))
        x = self.bn7(self.conv7(x))
        int2 = x.view(x.size(0), -1)
        return L2Norm()(int2)


class HardNetPS(nn.Module):
    def __init__(self):
        super(HardNetPS, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=8)
        )

    def input_norm(self, x):
        flat = x.view(x.size(0), -1)
        mp = torch.mean(flat, dim=1)
        sp = torch.std(flat, dim=1) + 1e-7
        return ((x - mp.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand_as(x)) /
                sp.unsqueeze(-1).unsqueeze(-1).unsqueeze(1).expand_as(x))

    def forward(self, input):
        x_features = self.features(self.input_norm(input))
        x = x_features.view(x_features.size(0), -1)
        return L2Norm()(x)


class HardNetVanilla(nn.Module):
    """
        HardNet model definition
    """

    def __init__(self):
        super(HardNetVanilla, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32, affine=False),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32, affine=False),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64, affine=False),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64, affine=False),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128, affine=False),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128, affine=False),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Conv2d(128, 128, kernel_size=8, bias=False),
            nn.BatchNorm2d(128, affine=False),
        )

    def input_norm(self, x):
        flat = x.view(x.size(0), -1)
        mp = torch.mean(flat, dim=1)
        sp = torch.std(flat, dim=1) + 1e-7
        return ((x - mp.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand_as(x)) /
                sp.unsqueeze(-1).unsqueeze(-1).unsqueeze(1).expand_as(x))

    def forward(self, input):
        x_features = self.features(self.input_norm(input))
        x = x_features.view(x_features.size(0), -1)
        return L2Norm()(x)


class L2net_des:
    def __init__(self, max_keypoints=200, patch_size=32,
                 model_src=Path(cbir.ROOT) / 'pretrained/l2net_L_N+.pt',  # TODO: change path to model to be absolute.
                 use_cuda=True):

        print("Loading pretrained network...")
        self.use_cuda = use_cuda
        self.net = L2net()

        if CONFIG['cpu_required']:
            self.net.load_state_dict(torch.load(model_src, map_location='cpu'))
        else:
            self.net.load_state_dict(torch.load(model_src))

        if use_cuda:
            self.net.cuda()
        self.detector = cv2.xfeatures2d.SURF_create(350)
        self.max_keypoints = max_keypoints
        self.ps = patch_size

    def detect(self, image, out=None):
        return detect(self.detector, image, self.max_keypoints, out)

    def compute(self, image, kp, expansion_coef=1.3, bs=16):
        return compute(self.net, image, self.ps, kp, expansion_coef, bs, self.use_cuda)


class HardNetBrown_des:
    def __init__(self, model_src=Path(cbir.ROOT) / 'pretrained/6Brown/hardnetBr6.pth',  # TODO: change path to model to be absolute.
                 patch_size=32, max_keypoints=200, use_cuda=True):
        self.use_cuda = use_cuda
        self.detector = cv2.xfeatures2d.SURF_create(350)
        self.max_keypoints = max_keypoints
        self.net = HardNetVanilla()
        self.net.load_state_dict(torch.load(model_src)['state_dict'])
        if use_cuda:
            self.net.cuda()
        self.ps = patch_size

    def detect(self, image, out=None):
        return detect(self.detector, image, self.max_keypoints, out)

    def compute(self, image, kp, expansion_coef=1.3, bs=16):
        return compute(self.net, image, self.ps, kp, expansion_coef, bs, self.use_cuda)


class HardNetAll_des:
    def __init__(self, model_src=Path(cbir.ROOT) / 'pretrained/pretrained_all_datasets/HardNet++.pth',  # TODO: change path to model to be absolute.
                 patch_size=32, max_keypoints=500, use_cuda=True):
        self.use_cuda = use_cuda
        self.detector = cv2.xfeatures2d.SURF_create(350)
        self.max_keypoints = max_keypoints
        self.net = HardNetVanilla()

        if CONFIG['cpu_required']:
            self.net.load_state_dict(torch.load(model_src, map_location='cpu')['state_dict'])
        else:
            self.net.load_state_dict(torch.load(model_src)['state_dict'])

        if use_cuda:
            self.net.cuda()
        self.ps = patch_size

    def detect(self, image, out=None):
        return detect(self.detector, image, self.max_keypoints, out)

    def compute(self, image, kp, expansion_coef=1.3, bs=16):
        return compute(self.net, image, self.ps, kp, expansion_coef, bs, self.use_cuda)


class HardNetHPatches_des:
    def __init__(self, model_src=Path(cbir.ROOT) / 'pretrained/3rd_party/HardNetPS/HardNetPS.pth',  # TODO: change path to model to be absolute.
                 patch_size=32, max_keypoints=200, use_cuda=True):
        self.use_cuda = use_cuda
        self.detector = cv2.xfeatures2d.SURF_create(350)
        self.max_keypoints = max_keypoints
        self.net = HardNetPS()
        self.net.load_state_dict(torch.load(model_src))
        if use_cuda:
            self.net.cuda()
        self.ps = patch_size

    def detect(self, image, out=None):
        return detect(self.detector, image, self.max_keypoints, out)

    def compute(self, image, kp, expansion_coef=1.3, bs=16):
        return compute(self.net, image, self.ps, kp, expansion_coef, bs, self.use_cuda)


class SURF:
    def __init__(self, max_keypoints):
        self.surf = cv2.xfeatures2d.SURF_create(350, extended=True)
        self.max_keypoints = max_keypoints

    def detect(self, image, out=None):
        kp = self.surf.detect(image)
        return sorted(kp,
                      key=lambda x: x.response,
                      reverse=True)[:self.max_keypoints]

    def compute(self, image, kp):
        if len(kp) <= 1:
            return None, None
        return self.surf.compute(image, kp)


class SIFT:
    def __init__(self, max_keypoints):
        self.sift = cv2.xfeatures2d.SIFT_create(nfeatures=max_keypoints)
        self.max_keypoints = max_keypoints

    def detect(self, image, out=None):
        kp = self.sift.detect(image)
        return sorted(kp,
                      key=lambda x: x.response,
                      reverse=True)

    def compute(self, image, kp):
        if len(kp) <= 1:
            return None, None
        return self.sift.compute(image, kp)
