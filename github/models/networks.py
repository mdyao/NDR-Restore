import torch
import logging
from models.modules.NDR import NDRN
import math
logger = logging.getLogger('base')


####################
# define network
####################
def define_G(opt):
    netG = NDRN()
    return netG
