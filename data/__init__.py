'''create dataset and dataloader'''
import logging
import torch
import torch.utils.data


def create_dataloader(dataset, dataset_opt, opt=None, sampler=None):
    phase = dataset_opt['phase']
    if phase == 'train':
        if opt['dist']:
            world_size = torch.distributed.get_world_size()
            num_workers = dataset_opt['n_workers']
            assert dataset_opt['batch_size'] % world_size == 0
            batch_size = dataset_opt['batch_size'] // world_size
            shuffle = False
        else:
            num_workers = dataset_opt['n_workers'] * len(opt['gpu_ids'])
            batch_size = dataset_opt['batch_size']
            shuffle = True
        return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                                           num_workers=num_workers, sampler=sampler, drop_last=True,
                                           pin_memory=False)
    else:
        return torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=6,
                                           pin_memory=False)


def create_dataset(dataset_opt):
    mode = dataset_opt['mode']
    if mode == 'NDR_train':
        from data.Dataset_train import Dataset as D
        dataset = D(dataset_opt)

        logger = logging.getLogger('base')
        logger.info('Dataset [{:s} - {:s}] is created.'.format(dataset.__class__.__name__,
                                                            dataset_opt['name']))
        return dataset
    elif mode =='NDR_val':
        from data.Dataset_val import Dataset as D
        dataset = D(dataset_opt)

        logger = logging.getLogger('base')
        
        logger.info('Dataset [{:s} - {:s}] is created.'.format(dataset.__class__.__name__,
                                                            dataset_opt['name']))
        return dataset
    elif mode=='NDR_test':
        from data.Dataset_test import Dataset_Hazy as D_hazy
        dataset_hazy = D_hazy(dataset_opt)

        from data.Dataset_test import Dataset_Rain as D_rain
        dataset_rain = D_rain(dataset_opt)

        from data.Dataset_test import Dataset_Noise15 as D_noise15
        dataset_noise15 = D_noise15(dataset_opt)

        from data.Dataset_test import Dataset_Noise25 as D_noise25
        dataset_noise25 = D_noise25(dataset_opt)

        from data.Dataset_test import Dataset_Noise50 as D_noise50
        dataset_noise50 = D_noise50(dataset_opt)

        logger = logging.getLogger('base')
        logger.info('Dataset [{:s} - {:s}] is created.'.format(dataset_hazy.__class__.__name__,
                                                            dataset_opt['name']))
        logger.info('Dataset [{:s} - {:s}] is created.'.format(dataset_rain.__class__.__name__,
                                                            dataset_opt['name']))
        logger.info('Dataset [{:s} - {:s}] is created.'.format(dataset_noise15.__class__.__name__,
                                                            dataset_opt['name']))
        logger.info('Dataset [{:s} - {:s}] is created.'.format(dataset_noise25.__class__.__name__,
                                                            dataset_opt['name']))
        logger.info('Dataset [{:s} - {:s}] is created.'.format(dataset_noise50.__class__.__name__,
                                                            dataset_opt['name']))
        return dataset_hazy, dataset_rain, dataset_noise15, dataset_noise25, dataset_noise50
        
    elif mode =='NDR_demo':
        from data.Dataset_demo import Dataset as D
        dataset = D(dataset_opt)

        logger = logging.getLogger('base')
        
        logger.info('Dataset [{:s} - {:s}] is created.'.format(dataset.__class__.__name__,
                                                            dataset_opt['name']))
        return dataset
    else:
        raise NotImplementedError('Dataset [{:s}] is not recognized.'.format(mode))
    
