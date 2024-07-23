import logging
logger = logging.getLogger('base')


def create_model(opt):
    model = opt['model']

    if model == 'NDR':
        from .NDR_model import NDR_Model as M
    else:
        raise NotImplementedError('Model [{:s}] not recognized.'.format(model))
    m = M(opt)
    logger.info('Model [{:s}] is created.'.format(m.__class__.__name__))
    return m
