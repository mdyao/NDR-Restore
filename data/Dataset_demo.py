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

class Dataset(data.Dataset):

    def __init__(self, opt):
        super(Dataset, self).__init__()

        self.opt = opt
        if os.path.isfile(self.opt['dataset_lq']):
            self.lq_paths = [self.opt['dataset_lq']]
        else:
            self.lq_paths = sorted(glob.glob(os.path.join(self.opt['dataset_lq'], '*')))
        #-------------------------------------------------------------

        # print("successfully loaded_images", len(self.lq_paths))
        #-------------------------------------------------------------

    def __getitem__(self, index):

        _, full_file_name = os.path.split(self.lq_paths[index])
        print(full_file_name)
        name_lq, _= os.path.splitext(full_file_name)

        img_lq = cv2.imread(self.lq_paths[index]) / 255.0
        img_lq = img_lq[:,:,:3]
        img_lq = img_lq.transpose(2,0,1)
        img_lq = torch.from_numpy(np.ascontiguousarray(img_lq)).float()

        # add padding
        img_lq = img_lq.unsqueeze(0)
        h, w = img_lq.shape[-2:]
        pad_h = 8 - h % 8 if h % 8 != 0 else 0
        pad_w = 8 - w % 8 if w % 8 != 0 else 0
        
        img_lq = F.pad(img_lq, (0, pad_w, 0, pad_h), mode='reflect')
        img_lq = img_lq.squeeze(0)

        return {
            'lq': img_lq,
            'name_lq': name_lq
        }

    def __len__(self):
        return len(self.lq_paths)
