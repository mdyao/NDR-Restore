import cv2
import torch
import numpy as np
from torch.utils import data as data
from os import path as osp
import glob
import os
import data.util as util
import random
import torch.nn.functional as F

class Dataset_Hazy(data.Dataset):

    def __init__(self, opt):
        super(Dataset_Hazy, self).__init__()

        self.opt = opt
        self.filename_tmpl = '{}'

        self.gt_paths =[]
        self.lq_paths =[]

        # haze
        lq_path = sorted(glob.glob(os.path.join(self.opt['dehaze_lq'], '*')))[:10]
        self.lq_paths += lq_path
        
        gt_path = []
        for i in range(len(lq_path)):
            gt_path.append(os.path.join(self.opt['dehaze_gt'], lq_path[i].split('/')[-1][:4] + '.png'))
        self.gt_paths += gt_path
        print("successfully loaded haze", len(lq_path))

        assert len(self.gt_paths) == len(self.lq_paths)

    def __getitem__(self, index):
        scale = self.opt['scale']
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


class Dataset_Rain(data.Dataset):

    def __init__(self, opt):
        super(Dataset_Rain, self).__init__()

        self.opt = opt

        # rain
        self.gt_paths = sorted(glob.glob(os.path.join(self.opt['derain_gt'], '*')))
        self.lq_paths = sorted(glob.glob(os.path.join(self.opt['derain_lq'], '*')))
        print("successfully loaded rain", len(self.lq_paths))

        assert len(self.gt_paths) == len(self.lq_paths)

    def __getitem__(self, index):
        scale = self.opt['scale']
        GT_size = self.opt['GT_size']

        scale = 1
        img_lq = cv2.imread(self.lq_paths[index]) / 255.0
        img_gt = cv2.imread(self.gt_paths[index]) / 255.0

        img_lq = img_lq[:,:,:3]
        img_gt = img_gt[:,:,:3]

        H, W, C = img_gt.shape
        H_r, W_r = H % scale, W % scale
        img_gt = img_gt[:H - H_r, :W - W_r, :]

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

class Dataset_Noise15(data.Dataset):

    def __init__(self, opt):
        super(Dataset_Noise15, self).__init__()

        self.opt = opt

        #-------------------------------------------------------------
        self.gt_paths = sorted(glob.glob(os.path.join(self.opt['denoise_gt'], '*')))[:10]

        print("successfully loaded_noise15", len(self.gt_paths))
        #-------------------------------------------------------------


    def _add_gaussian_noise(self, clean_patch, sigma):
            noise = np.random.randn(*clean_patch.shape) *sigma / 255.0
            noisy_patch = np.clip(clean_patch + noise , 0, 1)
            return noisy_patch

    def __getitem__(self, index):

        img_gt = cv2.imread(self.gt_paths[index])/255.0
        sigma = 15
        img_lq = self._add_gaussian_noise(img_gt,sigma)

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
        return len(self.gt_paths)



class Dataset_Noise25(data.Dataset):

    def __init__(self, opt):
        super(Dataset_Noise25, self).__init__()

        self.opt = opt

        #-------------------------------------------------------------
        self.gt_paths = sorted(glob.glob(os.path.join(self.opt['denoise_gt'], '*')))[:10]

        print("successfully loaded_noise25", len(self.gt_paths))
        #-------------------------------------------------------------

    def _add_gaussian_noise(self, clean_patch, sigma):
            noise = np.random.randn(*clean_patch.shape) *sigma / 255.0
            noisy_patch = np.clip(clean_patch + noise , 0, 1)
            return noisy_patch

    def __getitem__(self, index):

        img_gt = cv2.imread(self.gt_paths[index])/255.0
        sigma = 25
        img_lq = self._add_gaussian_noise(img_gt,sigma)

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
        return len(self.gt_paths)



class Dataset_Noise50(data.Dataset):

    def __init__(self, opt):
        super(Dataset_Noise50, self).__init__()

        self.opt = opt

        #-------------------------------------------------------------
        self.gt_paths = sorted(glob.glob(os.path.join(self.opt['denoise_gt'], '*')))[:10]
        print("successfully loaded_noise50", len(self.gt_paths))
        #-------------------------------------------------------------

    def _add_gaussian_noise(self, clean_patch, sigma):
            noise = np.random.randn(*clean_patch.shape) *sigma / 255.0
            noisy_patch = np.clip(clean_patch + noise , 0, 1)
            return noisy_patch

    def __getitem__(self, index):

        img_gt = cv2.imread(self.gt_paths[index])/255.0
        sigma = 50
        img_lq = self._add_gaussian_noise(img_gt,sigma)

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
        return len(self.gt_paths)


