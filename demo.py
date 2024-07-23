import os
import math
import argparse
import random
import logging

import torch
import torch.distributed as dist

import options.options as option
from utils import util
from data import create_dataloader, create_dataset
from models import create_model
import numpy as np
import cv2
from PIL import Image

def main():
    #### options
    parser = argparse.ArgumentParser()
    parser.add_argument('-opt', type=str, help='Path to option YMAL file.')
    args = parser.parse_args()
    opt = option.parse(args.opt, is_train=True)

    #### loading resume state if exists
    if opt['path'].get('resume_state', None):
        # distributed resuming: all load into default GPU
        device_id = torch.cuda.current_device()
        resume_state = torch.load(opt['path']['resume_state'],
                                  map_location=lambda storage, loc: storage.cuda(device_id))
        option.check_resume(opt, resume_state['iter'])  # check resume options
    else:
        resume_state = None

    # convert to NoneDict, which returns None for missing keys
    opt = option.dict_to_nonedict(opt)

    #### random seed
    seed = opt['train']['manual_seed']
    if seed is None:
        seed = random.randint(1, 10000)
    print('Random seed: {}'.format(seed))
    util.set_random_seed(seed)

    torch.backends.cudnn.benchmark = True
    # torch.backends.cudnn.deterministic = True

    #### create train and val dataloader
    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'demo':
            demo_set = create_dataset(dataset_opt)
            demo_loader = create_dataloader(demo_set, dataset_opt, opt, None)
            print('Number of demo images in [{:s}]: {:d}'.format(
                dataset_opt['name'], len(demo_set)))
        else:
            raise NotImplementedError('Phase [{:s}] is not recognized.'.format(phase))

    #### create model
    model = create_model(opt)

    # #### set save dir
    # util.mkdir_and_rename(opt['path']['root'])

    #### training
    for demo_data in demo_loader:
        model.feed_data_demo(demo_data)
        model.test()
        visuals = model.get_current_visuals_demo()
        out_img = visuals['out_img'].numpy()
        lq_img = visuals['lq_img'].numpy()
        name_lq = demo_data['name_lq'][0]

        c, h, w = lq_img.shape
        out_img = out_img[:c, :h, :w]

        out_img = out_img[::-1,:,:]
        out_img = out_img.transpose(1,2,0)
        out_img = np.clip(out_img,0,1)

        Image.fromarray((out_img*255).astype(np.uint8)).save(os.path.join(opt['path']['experiments_root'], name_lq+'.png'))



if __name__ == '__main__':
    main()
