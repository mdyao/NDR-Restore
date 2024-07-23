import cv2
import torch
import numpy as np
from torch.utils import data as data
import glob
import os
import random
import torch.nn.functional as F

class Dataset(data.Dataset):

    def __init__(self, opt):
        super(Dataset, self).__init__()

        self.opt = opt

        self.lq_paths =sorted(glob.glob(os.path.join(self.opt['dataroot_lq'], '*')))
        self.gt_paths = sorted(glob.glob(os.path.join(self.opt['dataroot_gt'], '*')))

        print("successfully loaded validation images", len(self.lq_paths), len(self.gt_paths))

        assert len(self.gt_paths) == len(self.lq_paths)

    def __getitem__(self, index):
        GT_size = self.opt['GT_size']

        img_lq = cv2.imread(self.lq_paths[index]) / 255.0
        img_gt = cv2.imread(self.gt_paths[index]) / 255.0

        img_lq = img_lq[:,:,:3]
        img_gt = img_gt[:,:,:3]

        img_lq = img_lq.transpose(2,0,1)
        img_gt = img_gt.transpose(2,0,1)

        img_lq = torch.from_numpy(np.ascontiguousarray(img_lq)).float()
        img_gt = torch.from_numpy(np.ascontiguousarray(img_gt)).float()

        img_lq = img_lq.unsqueeze(0)
        h, w = img_lq.shape[-2:]
        pad_h = 8 - h % 8 if h % 8 != 0 else 0
        pad_w = 8 - w % 8 if w % 8 != 0 else 0

        # add padding
        img_lq = F.pad(img_lq, (0, pad_w, 0, pad_h), mode='reflect')
        img_lq = img_lq.squeeze(0)

        return {
            'lq': img_lq,
            'gt': img_gt
        }

    def __len__(self):
        return len(self.lq_paths)
