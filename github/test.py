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

def compute_ssim(img1, img2):

    ssims = []
    for i in range(3):
        ssims.append(_ssim(img1[i], img2[i]))
    return np.array(ssims).mean()


def _ssim(img1, img2):
    """Calculate SSIM (structural similarity) for one channel images.

    It is called by func:`calculate_ssim`.

    Args:
        img1 (ndarray): Images with range [0, 255] with order 'HWC'.
        img2 (ndarray): Images with range [0, 255] with order 'HWC'.

    Returns:
        float: ssim result.
    """

    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) *
                (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                       (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()


def main():
    #### options
    parser = argparse.ArgumentParser()
    parser.add_argument('-opt', type=str, help='Path to option YMAL file.')
    parser.add_argument('--launcher', choices=['none', 'pytorch'], default='none',
                        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    opt = option.parse(args.opt, is_train=True)


    opt['dist'] = True

    #### loading resume state if exists
    if opt['path'].get('resume_state', None):
        device_id = torch.cuda.current_device()
        resume_state = torch.load(opt['path']['resume_state'],
                                  map_location=lambda storage, loc: storage.cuda(device_id))
        option.check_resume(opt, resume_state['iter'])  # check resume options
    else:
        resume_state = None

    #### mkdir and loggers
    if resume_state is None:
        util.mkdir_and_rename(
            opt['path']['experiments_root'])  # rename experiment folder if exists
        util.mkdirs((path for key, path in opt['path'].items() if not key == 'experiments_root'
                    and 'pretrain_model' not in key and 'resume' not in key))

    # config loggers. Before it, the log will not work
    util.setup_logger('base', opt['path']['log'], 'train_' + opt['name'], level=logging.INFO,
                    screen=True, tofile=True)
    util.setup_logger('val', opt['path']['log'], 'val_' + opt['name'], level=logging.INFO,
                    screen=True, tofile=True)
    logger = logging.getLogger('base')
    logger.info(option.dict2str(opt))

    # convert to NoneDict, which returns None for missing keys
    opt = option.dict_to_nonedict(opt)

    torch.backends.cudnn.benchmark = True
    # torch.backends.cudnn.deterministic = True

    #### create train and val dataloader
    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'test':
            val_set_hazy, val_set_rain, val_set_noise15, val_set_noise25, val_set_noise50 = create_dataset(dataset_opt)
            val_loader_hazy = create_dataloader(val_set_hazy, dataset_opt, opt, None)
            val_loader_rain = create_dataloader(val_set_rain, dataset_opt, opt, None)
            val_loader_noise15 = create_dataloader(val_set_noise15, dataset_opt, opt, None)
            val_loader_noise25 = create_dataloader(val_set_noise25, dataset_opt, opt, None)
            val_loader_noise50 = create_dataloader(val_set_noise50, dataset_opt, opt, None)

            
            logger.info('Number of val images in [{:s}]: {:d}'.format(
                dataset_opt['name'], len(val_loader_noise15)))
        else:
            raise NotImplementedError('Phase [{:s}] is not recognized.'.format(phase))
    # assert train_loader is not None

    #### create model
    model = create_model(opt)

    #### resume training
    if resume_state:
        logger.info('Resuming training from epoch: {}, iter: {}.'.format(
            resume_state['epoch'], resume_state['iter']))

        start_epoch = resume_state['epoch']
        current_step = resume_state['iter']
        model.resume_training(resume_state)  # handle optimizers and schedulers
    else:
        current_step = 0
        start_epoch = 0

    #### training
    # validation
    avg_psnr_hazy = 0.0
    avg_psnr_rain = 0.0
    avg_psnr_noise15 = 0.0
    avg_psnr_noise25 = 0.0
    avg_psnr_noise50 = 0.0
    avg_ssim_hazy = 0.0
    avg_ssim_rain = 0.0
    avg_ssim_noise15 = 0.0
    avg_ssim_noise25 = 0.0
    avg_ssim_noise50 = 0.0

    idx = 0
    for val_data in val_loader_hazy:
        idx += 1
        model.feed_data_test(val_data)
        model.test()
        visuals = model.get_current_visuals()
        out_img = visuals['out_img'].numpy()
        gt_img = visuals['gt_img'].numpy()

        c, h, w = gt_img.shape
        out_img = out_img[:c, :h, :w]

        def compute_psnr(img_orig, img_out, peak):
            mse = np.mean(np.square(img_orig - img_out))
            psnr = 10 * np.log10(peak * peak / mse)
            return psnr

        gt_img = np.clip(gt_img, 0, 1)
        out_img = np.clip(out_img, 0, 1)
        curr_psnr = compute_psnr(out_img, gt_img, 1)
        curr_ssim = compute_ssim(out_img * 255, gt_img * 255)

        avg_psnr_hazy += curr_psnr
        avg_ssim_hazy += curr_ssim
        if idx % 100 == 0:
            print('idx_HAZY', idx, curr_psnr,curr_ssim)
    avg_psnr_hazy = avg_psnr_hazy / idx
    avg_ssim_hazy = avg_ssim_hazy / idx
    # ————————————————————————————————————————————————————————————————————
    idx = 0
    for val_data in val_loader_rain:
        idx += 1
        model.feed_data_test(val_data)
        model.test()
        visuals = model.get_current_visuals()
        out_img = visuals['out_img'].numpy()
        gt_img = visuals['gt_img'].numpy()

        c, h, w = gt_img.shape
        out_img = out_img[:c, :h, :w]

        def compute_psnr(img_orig, img_out, peak):
            mse = np.mean(np.square(img_orig - img_out))
            psnr = 10 * np.log10(peak * peak / mse)
            return psnr

        gt_img = np.clip(gt_img, 0, 1)
        out_img = np.clip(out_img, 0, 1)
        curr_psnr = compute_psnr(out_img, gt_img, 1)
        curr_ssim = compute_ssim(out_img * 255, gt_img * 255)

        avg_psnr_rain += curr_psnr
        avg_ssim_rain += curr_ssim
        if idx % 50 == 0:
            print('idx_Rain', idx, curr_psnr,curr_ssim)
    avg_psnr_rain = avg_psnr_rain / idx
    avg_ssim_rain = avg_ssim_rain / idx
    # ————————————————————————————————————————————————————————————————————
    idx = 0
    for val_data in val_loader_noise15:
        idx += 1
        model.feed_data_test(val_data)
        model.test()
        visuals = model.get_current_visuals()
        out_img = visuals['out_img'].numpy()
        gt_img = visuals['gt_img'].numpy()

        c, h, w = gt_img.shape
        out_img = out_img[:c, :h, :w]

        def compute_psnr(img_orig, img_out, peak):
            mse = np.mean(np.square(img_orig - img_out))
            psnr = 10 * np.log10(peak * peak / mse)
            return psnr

        gt_img = np.clip(gt_img, 0, 1)
        out_img = np.clip(out_img, 0, 1)
        curr_psnr = compute_psnr(out_img, gt_img, 1)
        curr_ssim = compute_ssim(out_img * 255, gt_img * 255)

        avg_psnr_noise15 += curr_psnr
        avg_ssim_noise15 += curr_ssim
        if idx % 10 == 0:
            print('idx_Noise15', idx, curr_psnr, curr_ssim)
    avg_psnr_noise15 = avg_psnr_noise15 / idx
    avg_ssim_noise15 = avg_ssim_noise15 / idx
    # ————————————————————————————————————————————————————————————————————
    # ————————————————————————————————————————————————————————————————————
    idx = 0
    for val_data in val_loader_noise25:
        idx += 1
        model.feed_data_test(val_data)
        model.test()
        visuals = model.get_current_visuals()
        out_img = visuals['out_img'].numpy()
        gt_img = visuals['gt_img'].numpy()

        c, h, w = gt_img.shape
        out_img = out_img[:c, :h, :w]

        def compute_psnr(img_orig, img_out, peak):
            mse = np.mean(np.square(img_orig - img_out))
            psnr = 10 * np.log10(peak * peak / mse)
            return psnr

        gt_img = np.clip(gt_img, 0, 1)
        out_img = np.clip(out_img, 0, 1)
        curr_psnr = compute_psnr(out_img, gt_img, 1)
        curr_ssim = compute_ssim(out_img * 255, gt_img * 255)

        avg_psnr_noise25 += curr_psnr
        avg_ssim_noise25 += curr_ssim
        if idx % 10 == 0:
            print('idx_Noise25', idx, curr_psnr, curr_ssim)
    avg_psnr_noise25 = avg_psnr_noise25 / idx
    avg_ssim_noise25 = avg_ssim_noise25 / idx
    # ————————————————————————————————————————————————————————————————————
    # ————————————————————————————————————————————————————————————————————
    idx = 0
    for val_data in val_loader_noise50:
        idx += 1
        model.feed_data_test(val_data)
        model.test()
        visuals = model.get_current_visuals()
        out_img = visuals['out_img'].numpy()
        gt_img = visuals['gt_img'].numpy()

        c, h, w = gt_img.shape
        out_img = out_img[:c, :h, :w]

        def compute_psnr(img_orig, img_out, peak):
            mse = np.mean(np.square(img_orig - img_out))
            psnr = 10 * np.log10(peak * peak / mse)
            return psnr

        gt_img = np.clip(gt_img, 0, 1)
        out_img = np.clip(out_img, 0, 1)
        curr_psnr = compute_psnr(out_img, gt_img, 1)
        curr_ssim = compute_ssim(out_img * 255, gt_img * 255)

        avg_psnr_noise50 += curr_psnr
        avg_ssim_noise50 += curr_ssim
        if idx % 10 == 0:
            print('idx_Noise50', idx, curr_psnr)
    avg_psnr_noise50 = avg_psnr_noise50 / idx
    avg_ssim_noise50 = avg_ssim_noise50 / idx


    # ————————————————————————————————————————————————————————————————————
        # log
    logger_val = logging.getLogger('val')  # validation logger
    logger_val.info(
        'HAZY_PSNR: {:.4f} Rain_PSNR: {:.4f} Noise15_PSNR: {:.4f} Noise25_PSNR: {:.4f} Noise50_PSNR: {:.4f}.'.format(
            avg_psnr_hazy, avg_psnr_rain, avg_psnr_noise15, avg_psnr_noise25,
            avg_psnr_noise50))
    logger_val.info(
        ' HAZY_SSIM: {:.4f} Rain_SSIM: {:.4f} Noise15_SSIM: {:.4f} Noise25_SSIM: {:.4f} Noise50_SSIM: {:.4f}.'.format(
                avg_ssim_hazy, avg_ssim_rain, avg_ssim_noise15, avg_ssim_noise25,
            avg_ssim_noise50))
    # log))
    # ————————————————————————————————————————————————————————————————————




if __name__ == '__main__':
    main()
