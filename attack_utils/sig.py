#This script is for Sig attack callable transform

'''
This code is based on https://github.com/bboylyg/NAD

The original license:
License CC BY-NC

The update include:
    1. change to callable object
    2. change the way of trigger generation, use the original formulation.

# idea : set the parameter in initialization, then when the object is called, it will use the add_trigger method to add trigger
'''

from typing import Union
import torch
import numpy as np
from PIL import Image


class SigTriggerAttack(object):
    """
    Implement paper:
    > Barni, M., Kallas, K., & Tondi, B. (2019).
    > A new Backdoor Attack in CNNs by training set corruption without label poisoning.
    > arXiv preprint arXiv:1902.11237
    superimposed sinusoidal backdoor signal with default parameters
    """
    def __init__(self,
                 trigger_label,
                 image_shape,
                 delta : Union[int, float, complex, np.number, torch.Tensor] = 40,
                 f : Union[int, float, complex, np.number, torch.Tensor] = 6,
                 ) -> None:
        self.delta = delta
        self.f = f
        self.create_pattern(image_shape)
        self.trigger_label = trigger_label
    
    def create_pattern(self, img_shape):
        pattern = np.zeros(img_shape, dtype=np.float32)
        m = pattern.shape[1]
        for i in range(int(img_shape[0])):
              for j in range(int(img_shape[1])):
                    pattern[i, j] = self.delta * np.sin(2 * np.pi * j * self.f / m)
        self.pattern = pattern

    def put_trigger(self, img: Image):
        img = np.float32(np.array(img))
        img = np.uint32(img) + self.pattern
        img = np.uint8(np.clip(img, 0, 255))
        img = Image.fromarray(img)
        return img

