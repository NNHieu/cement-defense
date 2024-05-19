'''
This code is based on https://github.com/THUYimingLi/BackdoorBox/blob/main/core/attacks/WaNet.py
This is the implement of WaNet [1].

Reference:
[1] WaNet - Imperceptible Warping-based Backdoor Attack. ICLR 2021.
'''

import copy
from copy import deepcopy
import random

import torch
import numpy as np
import PIL
from PIL import Image
from torchvision.transforms import functional as TF
import torch.nn.functional as F
import torch.nn as nn
from torchvision.transforms import Compose


class AddTrigger:
    def __init__(self):
        pass

    def add_trigger(self, img, noise=False):
        """Add WaNet trigger to image.

        Args:
            img (torch.Tensor): shape (C, H, W).
            noise (bool): turn on noise mode, default is False

        Returns:
            torch.Tensor: Poisoned image, shape (C, H, W).
        """
        if noise:
            ins = torch.rand(1, self.h, self.h, 2) * self.noise_rescale - 1  # [-1, 1]
            grid = self.grid + ins / self.h
            grid = torch.clamp(self.grid + ins / self.h, -1, 1)
        else:
            grid = self.grid
        poison_img = nn.functional.grid_sample(img.unsqueeze(0), grid, align_corners=True).squeeze()  # CHW
        return poison_img



class AddCIFAR10Trigger(AddTrigger):
    """Add WaNet trigger to CIFAR10 image.

    Args:
        identity_grid (orch.Tensor): the poisoned pattern shape.
        noise_grid (orch.Tensor): the noise pattern.
        noise (bool): turn on noise mode, default is False.
        s (int or float): The strength of the noise grid. Default is 0.5.
        grid_rescale (int or float): Scale :attr:`grid` to avoid pixel values going out of [-1, 1].
            Default is 1.
        noise_rescale (int or float): Scale the random noise from a uniform distribution on the
            interval [0, 1). Default is 2.
    """

    def __init__(self, identity_grid, noise_grid, noise=False, s=0.5, grid_rescale=1, noise_rescale=2):
        super(AddCIFAR10Trigger, self).__init__()

        self.identity_grid = deepcopy(identity_grid)
        self.noise_grid = deepcopy(noise_grid)
        self.h = self.identity_grid.shape[2]
        self.noise = noise
        self.s = s
        self.grid_rescale = grid_rescale
        grid = self.identity_grid + self.s * self.noise_grid / self.h
        self.grid = torch.clamp(grid * self.grid_rescale, -1, 1)
        self.noise_rescale = noise_rescale

    def __call__(self, img):
        img = TF.pil_to_tensor(img)
        img = TF.convert_image_dtype(img, torch.float)
        img = self.add_trigger(img, noise=self.noise)
        img = img.numpy().transpose(1, 2, 0)
        img = Image.fromarray(np.clip(img*255,0,255).round().astype(np.uint8))
        # img = Image.fromarray(img.permute(1, 2, 0).numpy())
        return img

class WanetTriggerHandler(object):
    """
    """
    def __init__(self,
                 trigger_label,
                 image_shape,
                 k, 
                 s=0.5, grid_rescale=1, noise_rescale=2,
                 noise=False, 
                 ) -> None:
        ins = torch.rand(1, 2, k, k) * 2 - 1
        ins = ins / torch.mean(torch.abs(ins))
        array1d = torch.linspace(-1, 1, steps=image_shape[0])
        x, y = torch.meshgrid(array1d, array1d)
        identity_grid = torch.stack((y, x), 2)[None, ...]
        noise_grid = (
            F.upsample(ins, size=image_shape[0], mode="bicubic", align_corners=True)
            .permute(0, 2, 3, 1)
        )
        self.trigger_label = trigger_label
        self.add_trigger = AddCIFAR10Trigger(identity_grid, noise_grid, noise, s, grid_rescale, noise_rescale)


    def put_trigger(self, img: Image):
        img = self.add_trigger(img)
        return img

