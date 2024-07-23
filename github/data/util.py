import os
import math
import random
import numpy as np
import torch
import cv2


def augment(img_list, hflip=True, rot=True):
    # horizontal flip OR rotate
    hflip = hflip and random.random() < 0.5
    vflip = rot and random.random() < 0.5
    rot90 = rot and random.random() < 0.5

    def _augment(img):
        if isinstance(img, list):
            if hflip:
                img = [image[:, ::-1, :] for image in img]
            if vflip:
                img = [image[:, :, ::-1] for image in img]
            if rot90:
                img = [image.transpose(0, 2, 1) for image in img]
        else:
            if hflip:
                img = img[:, ::-1, :]
            if vflip:
                img = img[:, :, ::-1]
            if rot90:
                img = img.transpose(0, 2, 1)
        return img

    return [_augment(img) for img in img_list]

