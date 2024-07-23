import cv2
import torch
import numpy as np
from torch.utils import data as data
import glob
import os
import data.util as util
import random


class Dataset(data.Dataset):

    def __init__(self, opt):
        super(Dataset, self).__init__()

        self.opt = opt
        self.lq_paths = []
        self.gt_paths = []

        # -------------------------------------------------
        # dehaze
        haze_lq_path = sorted(glob.glob(os.path.join(opt['dehaze_lq'], '*')))
        haze_gt_path = []
        for i in range(len(haze_lq_path)):
            if not os.path.exists(os.path.join(self.opt['dehaze_gt'], haze_lq_path[i].split('/')[-1][:4] + '.jpg')):
                haze_gt_path.append(os.path.join(self.opt['dehaze_gt'], haze_lq_path[i].split('/')[-1][:4] + '.png'))
            else:
                haze_gt_path.append(os.path.join(self.opt['dehaze_gt'], haze_lq_path[i].split('/')[-1][:4] + '.jpg'))

        assert len(haze_gt_path)==len(haze_lq_path)

        print("successfully loaded haze all:", len(haze_lq_path))
        haze_lq_path = haze_lq_path * self.opt['dehaze_ratio']
        haze_gt_path = haze_gt_path * self.opt['dehaze_ratio']
        print("successfully loaded haze all_repeat:", len(haze_lq_path))

        self.lq_paths += haze_lq_path
        self.gt_paths += haze_gt_path

        # -------------------------------------------------
        # derain
        rain_gt_path = sorted(glob.glob(os.path.join(self.opt['derain_gt'], '*')))
        rain_lq_path = sorted(glob.glob(os.path.join(self.opt['derain_lq'], '*')))

        assert len(rain_gt_path)==len(rain_lq_path)

        print("successfully loaded rain all:", len(rain_lq_path))
        rain_gt_path = rain_gt_path * self.opt['derain_ratio']
        rain_lq_path = rain_lq_path * self.opt['derain_ratio']
        print("successfully loaded rain all_repeat:", len(rain_lq_path))
        
        self.gt_paths += rain_gt_path
        self.lq_paths += rain_lq_path

        # -------------------------------------------------
        # denoise
        noise_gt_path = sorted(glob.glob(os.path.join(self.opt['denoise_gt'], '*')))
        # add denoise_flag
        ##################### Be careful of path name ##########################s
        for i in range(len(noise_gt_path)):
            noise_gt_path[i] = noise_gt_path[i]+'NOISE_FLAG'
        print("successfully loaded noise all:", len(noise_gt_path))
        noise_gt_path = noise_gt_path * self.opt['denoise_ratio']
        print("successfully loaded noise all_repeat:", len(noise_gt_path))
        
        
        self.gt_paths += noise_gt_path

        
    def _add_gaussian_noise(self, clean_patch, sigma):
            noise = np.random.randn(*clean_patch.shape) *sigma / 255.0
            noisy_patch = np.clip(clean_patch + noise , 0, 1)
            return noisy_patch

    def __getitem__(self, index):
        GT_size = self.opt['GT_size']

        if 'NOISE_FLAG' in self.gt_paths[index]:
            img_gt = cv2.imread(self.gt_paths[index][:-10])/255.0
            sigma = random.choice(self.opt['noise_levels'])
            img_lq = self._add_gaussian_noise(img_gt,sigma)
        else:
            img_gt = cv2.imread(self.gt_paths[index])/255.0
            img_lq = cv2.imread(self.lq_paths[index])/255.0

        img_gt = img_gt[:,:,:3]
        img_lq = img_lq[:,:,:3]

        img_lq = img_lq.transpose(2,0,1)
        img_gt = img_gt.transpose(2,0,1)

        _,h,w = img_gt.shape
        random_h = np.random.randint(0,h-GT_size)
        random_w = np.random.randint(0, w - GT_size)
        img_gt = img_gt[:,random_h:random_h+GT_size, random_w:random_w+GT_size]
        img_lq = img_lq[:,random_h:random_h+GT_size, random_w:random_w+GT_size]

        img_lq, img_gt = util.augment([img_lq, img_gt])

        img_lq = torch.from_numpy(np.ascontiguousarray(img_lq)).float()
        img_gt = torch.from_numpy(np.ascontiguousarray(img_gt)).float()

        return {
                'lq': img_lq,
                'gt': img_gt
            }

    def __len__(self):
        return len(self.gt_paths)

