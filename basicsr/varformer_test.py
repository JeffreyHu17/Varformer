import os
import os.path as osp
import logging
import time
import argparse
from collections import OrderedDict
import numpy as np

from options import options as option
import utils.util as util
from torchvision.transforms.functional import normalize
from archs import build_network
from utils.misc import gpu_is_available, get_device
import torch
from glob import glob
import tqdm
import cv2
import torch.nn.functional as F
from utils import tensor2img


#### options
parser = argparse.ArgumentParser()
parser.add_argument('-opt', type=str, required=True, help='Path to options YMAL file.')
opt = option.parse(parser.parse_args().opt, is_train=False)
opt = option.dict_to_nonedict(opt)

util.mkdirs(
    (path for key, path in opt['path'].items()
     if not key == 'experiments_root' and 'pretrain_model' not in key and 'resume' not in key))
util.setup_logger('base', opt['path']['log'], 'test_' + opt['name'], level=logging.INFO,
                  screen=True, tofile=True)
logger = logging.getLogger('base')
logger.info(option.dict2str(opt))

device = get_device()
model = build_network(opt['network_g']).to(device)
ckpt_path = opt['path']['pretrain_model']
checkpoint = torch.load(ckpt_path)


ckpt = checkpoint['params_ema']
model.load_state_dict(ckpt,strict=opt['path']['strict_load'])

model.eval()


#### Create test dataset and dataloader
test_loaders = []
for phase, dataset_opt in sorted(opt['datasets'].items()):
    paths_LQ = dataset_opt['dataroot_LQ']
    paths_GT = dataset_opt['dataroot_GT']

    crop_border = opt['crop_border'] if opt['crop_border'] is not None else opt['scale']
    need_GT = False if dataset_opt['dataroot_GT'] is None else True

    test_set_name = dataset_opt['name']
    logger.info('\nTesting [{:s}]...'.format(test_set_name))
    dataset_dir = osp.join(opt['path']['results_root'], test_set_name)
    util.mkdir(dataset_dir)

    img_path_list = glob(os.path.join(paths_LQ, "*.png")) + glob(os.path.join(paths_LQ, "*.jpg")) 
    
    for img_path in tqdm.tqdm(img_path_list):
        """ Load an image """
        img_name = os.path.basename(img_path)
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        H, W, C = img.shape
        img = img.astype(np.float32) / 255.
        img = cv2.resize(img, (opt['input_size'], opt['input_size']), interpolation=cv2.INTER_LINEAR)
        if img.shape[2] == 3:
            img = img[:, :, [2, 1, 0]]        
        img = torch.from_numpy(np.ascontiguousarray(np.transpose(img, (2, 0, 1)))).float()
        normalize(img, opt['mean'], opt['std'], inplace=True)

        model.eval()
        output, _ = model(img.unsqueeze(0).to(device))
        sr_img = output.detach().cpu()
        sr_img = tensor2img(sr_img)
        save_img_path = osp.join(dataset_dir, img_name)
        util.save_img(sr_img, save_img_path)
