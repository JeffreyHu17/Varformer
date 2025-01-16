import cv2
import math
import random
import numpy as np
import os.path as osp
from scipy.io import loadmat
from PIL import Image
import torch
import torch.utils.data as data
from torchvision.transforms.functional import normalize
from data import gaussian_kernels as gaussian_kernels
from data.transforms import augment, augment2
from data.data_util import paths_from_folder, brush_stroke_mask, random_ff_mask
from utils import FileClient, get_root_logger, imfrombytes, img2tensor, tensor2img, imwrite
from utils.registry import DATASET_REGISTRY
import data.util as util

@DATASET_REGISTRY.register()
class LQGTDataset(data.Dataset):


    def __init__(self, opt):
        super(LQGTDataset, self).__init__()
        self.opt = opt
        self.data_type = self.opt['data_type']
        self.paths_LQ, self.paths_GT = None, None
        self.sizes_LQ, self.sizes_GT = None, None
        self.LQ_env, self.GT_env = None, None  
        self.mean = opt.get('mean', [0.5, 0.5, 0.5])
        self.std = opt.get('std', [0.5, 0.5, 0.5])

        self.paths_GT, self.sizes_GT = util.get_image_paths(self.data_type, opt['dataroot_GT'])
        self.paths_LQ, self.sizes_LQ = util.get_image_paths(self.data_type, opt['dataroot_LQ'])

        assert self.paths_GT, 'Error: GT path is empty.'
        if self.paths_LQ and self.paths_GT:
            assert len(self.paths_LQ) == len(
                self.paths_GT
            ), 'GT and LQ datasets have different number of images - {}, {}.'.format(
                len(self.paths_LQ), len(self.paths_GT))
        self.random_scale_list = [1]

    

    def __getitem__(self, index):
        if self.data_type == 'lmdb':
            if (self.GT_env is None) or (self.LQ_env is None):
                self._init_lmdb()
        GT_path, LQ_path = None, None
        scale = self.opt['scale']
        GT_size = self.opt['GT_size'] if self.opt['GT_size'] is not None else self.opt['HQ_size']

        # get GT image
        GT_path = self.paths_GT[index]
        if self.data_type == 'lmdb':
            resolution = [int(s) for s in self.sizes_GT[index].split('_')]
        else:
            resolution = None
        img_GT = util.read_img(self.GT_env, GT_path, resolution)
        W,H,C = img_GT.shape

        # get LQ image
        if self.paths_LQ:
            LQ_path = self.paths_LQ[index]
            if self.data_type == 'lmdb':
                resolution = [int(s) for s in self.sizes_LQ[index].split('_')]
            else:
                resolution = None
            try:
                img_LQ = util.read_img(self.LQ_env, LQ_path, resolution)
            except:
                print(LQ_path)
        else:  
            if self.opt['phase'] == 'train':
                random_scale = random.choice(self.random_scale_list)
                H_s, W_s, _ = img_GT.shape

                def _mod(n, random_scale, scale, thres):
                    rlt = int(n * random_scale)
                    rlt = (rlt // scale) * scale
                    return thres if rlt < thres else rlt

                H_s = _mod(H_s, random_scale, scale, GT_size)
                W_s = _mod(W_s, random_scale, scale, GT_size)
                img_GT = cv2.resize(np.copy(img_GT), (W_s, H_s), interpolation=cv2.INTER_LINEAR)
                # force to 3 channels
                if img_GT.ndim == 2:
                    img_GT = cv2.cvtColor(img_GT, cv2.COLOR_GRAY2BGR)

            H, W, _ = img_GT.shape
            # using matlab imresize
            img_LQ = util.imresize_np(img_GT, 1 / scale, True)
            if img_LQ.ndim == 2:
                img_LQ = np.expand_dims(img_LQ, axis=2)

        if self.opt['phase'] == 'train':
            # if the image size is too small
            H, W, _ = img_GT.shape
            if H < GT_size or W < GT_size:
                img_GT = cv2.resize(np.copy(img_GT), (GT_size, GT_size),
                                    interpolation=cv2.INTER_LINEAR)
                # using matlab imresize
                img_LQ = util.imresize_np(img_GT, 1 / scale, True)
                if img_LQ.ndim == 2:
                    img_LQ = np.expand_dims(img_LQ, axis=2)

            H, W, C = img_LQ.shape
            LQ_size = GT_size // scale

            # randomly crop
            rnd_h = random.randint(0, max(0, H - LQ_size))
            rnd_w = random.randint(0, max(0, W - LQ_size))
            img_LQ = img_LQ[rnd_h:rnd_h + LQ_size, rnd_w:rnd_w + LQ_size, :]
            rnd_h_GT, rnd_w_GT = int(rnd_h * scale), int(rnd_w * scale)
            img_GT = img_GT[rnd_h_GT:rnd_h_GT + GT_size, rnd_w_GT:rnd_w_GT + GT_size, :]

            # augmentation - flip, rotate
            img_LQ, img_GT = util.augment([img_LQ, img_GT], self.opt['use_flip'],
                                          self.opt['use_rot'])
        if img_GT.shape[2] == 3:
            img_GT = img_GT[:, :, [2, 1, 0]]
            img_LQ = img_LQ[:, :, [2, 1, 0]]


        if self.opt['phase'] == 'val':
        # resize to in_size
            img_LQ = cv2.resize(img_LQ, (GT_size, GT_size), interpolation=cv2.INTER_LINEAR)
            img_GT = cv2.resize(img_GT, (GT_size, GT_size), interpolation=cv2.INTER_LINEAR)

        img_GT = torch.from_numpy(np.ascontiguousarray(np.transpose(img_GT, (2, 0, 1)))).float()
        img_LQ = torch.from_numpy(np.ascontiguousarray(np.transpose(img_LQ, (2, 0, 1)))).float()

        if LQ_path is None:
            LQ_path = GT_path


        normalize(img_LQ, self.mean, self.std, inplace=True)
        normalize(img_GT, self.mean, self.std, inplace=True)


        return {'in': img_LQ, 'gt': img_GT, 'lq_path': LQ_path, 'gt_path': GT_path}

    def __len__(self):
        return len(self.paths_GT)




@DATASET_REGISTRY.register()
class LQGTDataset3(data.Dataset): 


    def __init__(self, opt):
        super(LQGTDataset3, self).__init__()
        self.opt = opt
        self.data_type = self.opt['data_type']
        self.dir_num = self.opt['data_num']
        self.paths_LQ, self.paths_GT = None, None
        self.sizes_LQ, self.sizes_GT = None, None
        self.LQ_env, self.GT_env = None, None  
        self.mean = opt.get('mean', [0.5, 0.5, 0.5])
        self.std = opt.get('std', [0.5, 0.5, 0.5])
        self.haze = self.opt['haze']
        

        for i in range(self.dir_num):
            if i == 0:
                self.paths_GT, self.sizes_GT = util.get_image_paths(self.data_type, opt['dataroot_GT'])
                self.paths_LQ, self.sizes_LQ = util.get_image_paths(self.data_type, opt['dataroot_LQ'])
                
            else:
                self.paths_GT2, self.sizes_GT2 = util.get_image_paths(self.data_type, opt['dataroot_GT'+str(i)])
                self.paths_LQ2, self.sizes_LQ2 = util.get_image_paths(self.data_type, opt['dataroot_LQ'+str(i)])
                self.paths_GT =  self.paths_GT + self.paths_GT2 
                
                self.paths_LQ = self.paths_LQ + self.paths_LQ2 
                
                

        if self.haze == True:
                self.paths_LQ2, self.sizes_LQ2 = util.get_image_paths(self.data_type, opt['dataroot_LQ_z'])
                self.paths_GT2 = [ opt['dataroot_GT_z'] + '/'+ (lq.split('/')[-1]).split('_')[0]+'.jpg' for lq in self.paths_LQ2]
                self.paths_GT =  self.paths_GT + self.paths_GT2 
                
                self.paths_LQ = self.paths_LQ + self.paths_LQ2    
                        

        assert self.paths_GT, 'Error: GT path is empty.'
        if self.paths_LQ and self.paths_GT:
            assert len(self.paths_LQ) == len(
                self.paths_GT
            ), 'GT and LQ datasets have different number of images - {}, {}.'.format(
                len(self.paths_LQ), len(self.paths_GT))
        self.random_scale_list = [1]

    

    def __getitem__(self, index):
        if self.data_type == 'lmdb':
            if (self.GT_env is None) or (self.LQ_env is None):
                self._init_lmdb()
        GT_path, LQ_path = None, None
        scale = self.opt['scale']
        GT_size = self.opt['GT_size'] if self.opt['GT_size'] is not None else self.opt['HQ_size']

        # get GT image
        GT_path = self.paths_GT[index]
        if self.data_type == 'lmdb':
            resolution = [int(s) for s in self.sizes_GT[index].split('_')]
        else:
            resolution = None
        
        img_GT = util.read_img(self.GT_env, GT_path, resolution)
        W,H,C = img_GT.shape
        

        # get LQ image
        if self.paths_LQ:
            LQ_path = self.paths_LQ[index]
            if self.data_type == 'lmdb':
                resolution = [int(s) for s in self.sizes_LQ[index].split('_')]
            else:
                resolution = None
            try:
                img_LQ = util.read_img(self.LQ_env, LQ_path, resolution)
            except:
                print(LQ_path)
        else:  # down-sampling on-the-fly
            # randomly scale during training
            if self.opt['phase'] == 'train':
                random_scale = random.choice(self.random_scale_list)
                H_s, W_s, _ = img_GT.shape

                def _mod(n, random_scale, scale, thres):
                    rlt = int(n * random_scale)
                    rlt = (rlt // scale) * scale
                    return thres if rlt < thres else rlt

                H_s = _mod(H_s, random_scale, scale, GT_size)
                W_s = _mod(W_s, random_scale, scale, GT_size)
                img_GT = cv2.resize(np.copy(img_GT), (W_s, H_s), interpolation=cv2.INTER_LINEAR)
                # force to 3 channels
                if img_GT.ndim == 2:
                    img_GT = cv2.cvtColor(img_GT, cv2.COLOR_GRAY2BGR)

            H, W, _ = img_GT.shape
            # using matlab imresize
            img_LQ = util.imresize_np(img_GT, 1 / scale, True)
            if img_LQ.ndim == 2:
                img_LQ = np.expand_dims(img_LQ, axis=2)

        if self.opt['phase'] == 'train':
            # if the image size is too small
            H, W, _ = img_GT.shape
            if H < GT_size or W < GT_size:
                img_GT = cv2.resize(np.copy(img_GT), (GT_size, GT_size),
                                    interpolation=cv2.INTER_LINEAR)
                # using matlab imresize
                img_LQ = util.imresize_np(img_GT, 1 / scale, True)
                if img_LQ.ndim == 2:
                    img_LQ = np.expand_dims(img_LQ, axis=2)

            H, W, C = img_LQ.shape
            LQ_size = GT_size // scale

            # randomly crop
            rnd_h = random.randint(0, max(0, H - LQ_size))
            rnd_w = random.randint(0, max(0, W - LQ_size))
            img_LQ = img_LQ[rnd_h:rnd_h + LQ_size, rnd_w:rnd_w + LQ_size, :]
            rnd_h_GT, rnd_w_GT = int(rnd_h * scale), int(rnd_w * scale)
            img_GT = img_GT[rnd_h_GT:rnd_h_GT + GT_size, rnd_w_GT:rnd_w_GT + GT_size, :]

            # augmentation - flip, rotate
            img_LQ, img_GT = util.augment([img_LQ, img_GT], self.opt['use_flip'],
                                          self.opt['use_rot'])

        # BGR to RGB, HWC to CHW, numpy to tensor
        if img_GT.shape[2] == 3:
            img_GT = img_GT[:, :, [2, 1, 0]]
            img_LQ = img_LQ[:, :, [2, 1, 0]]


        if self.opt['phase'] == 'val':
        # resize to in_size
            img_LQ = cv2.resize(img_LQ, (GT_size, GT_size), interpolation=cv2.INTER_LINEAR)
            img_GT = cv2.resize(img_GT, (GT_size, GT_size), interpolation=cv2.INTER_LINEAR)

        img_GT = torch.from_numpy(np.ascontiguousarray(np.transpose(img_GT, (2, 0, 1)))).float()
        img_LQ = torch.from_numpy(np.ascontiguousarray(np.transpose(img_LQ, (2, 0, 1)))).float()

        if LQ_path is None:
            LQ_path = GT_path
        
        

        normalize(img_LQ, self.mean, self.std, inplace=True)
        normalize(img_GT, self.mean, self.std, inplace=True)


        return {'in': img_LQ, 'gt': img_GT, 'lq_path': LQ_path, 'gt_path': GT_path}

    def __len__(self):
        return len(self.paths_GT)



@DATASET_REGISTRY.register()
class LQGTDataset3_pre(data.Dataset): 

    def __init__(self, opt):
        super(LQGTDataset3_pre, self).__init__()
        self.opt = opt
        self.data_type = self.opt['data_type']
        self.dir_num = self.opt['data_num']
        self.paths_LQ, self.paths_GT = None, None
        self.sizes_LQ, self.sizes_GT = None, None
        self.LQ_env, self.GT_env = None, None  
        self.mean = opt.get('mean', [0.5, 0.5, 0.5])
        self.std = opt.get('std', [0.5, 0.5, 0.5])


        for i in range(self.dir_num):
            paths_GT2, sizes_GT2 = util.get_image_paths(self.data_type, opt['dataroot_GT'+str(i)])
            paths_LQ2, sizes_LQ2 = util.get_image_paths(self.data_type, opt['dataroot_LQ'+str(i)])
            if i == 0:
                self.paths_GT =  paths_GT2
                self.paths_LQ =  paths_LQ2
            else:
                self.paths_GT =  self.paths_GT + paths_GT2
                self.paths_LQ = self.paths_LQ + paths_LQ2

        assert self.paths_GT, 'Error: GT path is empty.'
        if self.paths_LQ and self.paths_GT:
            assert len(self.paths_LQ) == len(
                self.paths_GT
            ), 'GT and LQ datasets have different number of images - {}, {}.'.format(
                len(self.paths_LQ), len(self.paths_GT))
        self.random_scale_list = [1]

    def __getitem__(self, index):
        if self.data_type == 'lmdb':
            if (self.GT_env is None) or (self.LQ_env is None):
                self._init_lmdb()
        GT_path, LQ_path = None, None
        scale = self.opt['scale']
        GT_size = self.opt['GT_size'] if self.opt['GT_size'] is not None else self.opt['HQ_size']

        # get GT image
        GT_path = self.paths_GT[index]
        if self.data_type == 'lmdb':
            resolution = [int(s) for s in self.sizes_GT[index].split('_')]
        else:
            resolution = None
        img_GT = util.read_img(self.GT_env, GT_path, resolution)
        W,H,C = img_GT.shape

        # get LQ image
        if self.paths_LQ:
            LQ_path = self.paths_LQ[index]
            if self.data_type == 'lmdb':
                resolution = [int(s) for s in self.sizes_LQ[index].split('_')]
            else:
                resolution = None
            try:
                img_LQ = util.read_img(self.LQ_env, LQ_path, resolution)
            except:
                print(LQ_path)
        else:  # down-sampling on-the-fly
            # randomly scale during training
            if self.opt['phase'] == 'train':
                random_scale = random.choice(self.random_scale_list)
                H_s, W_s, _ = img_GT.shape

                def _mod(n, random_scale, scale, thres):
                    rlt = int(n * random_scale)
                    rlt = (rlt // scale) * scale
                    return thres if rlt < thres else rlt

                H_s = _mod(H_s, random_scale, scale, GT_size)
                W_s = _mod(W_s, random_scale, scale, GT_size)
                img_GT = cv2.resize(np.copy(img_GT), (W_s, H_s), interpolation=cv2.INTER_LINEAR)
                # force to 3 channels
                if img_GT.ndim == 2:
                    img_GT = cv2.cvtColor(img_GT, cv2.COLOR_GRAY2BGR)

            H, W, _ = img_GT.shape
            # using matlab imresize
            img_LQ = util.imresize_np(img_GT, 1 / scale, True)
            if img_LQ.ndim == 2:
                img_LQ = np.expand_dims(img_LQ, axis=2)

        if self.opt['phase'] == 'train':
            # if the image size is too small
            H, W, _ = img_GT.shape
            if H < GT_size or W < GT_size:
                img_GT = cv2.resize(np.copy(img_GT), (GT_size, GT_size),
                                    interpolation=cv2.INTER_LINEAR)
                # using matlab imresize
                img_LQ = util.imresize_np(img_GT, 1 / scale, True)
                if img_LQ.ndim == 2:
                    img_LQ = np.expand_dims(img_LQ, axis=2)

            H, W, C = img_LQ.shape
            LQ_size = GT_size // scale

            # randomly crop
            rnd_h = random.randint(0, max(0, H - LQ_size))
            rnd_w = random.randint(0, max(0, W - LQ_size))
            img_LQ = img_LQ[rnd_h:rnd_h + LQ_size, rnd_w:rnd_w + LQ_size, :]
            rnd_h_GT, rnd_w_GT = int(rnd_h * scale), int(rnd_w * scale)
            img_GT = img_GT[rnd_h_GT:rnd_h_GT + GT_size, rnd_w_GT:rnd_w_GT + GT_size, :]

            # augmentation - flip, rotate
            img_LQ, img_GT = util.augment([img_LQ, img_GT], self.opt['use_flip'],
                                          self.opt['use_rot'])

        # BGR to RGB, HWC to CHW, numpy to tensor
        if img_GT.shape[2] == 3:
            img_GT = img_GT[:, :, [2, 1, 0]]
            img_LQ = img_LQ[:, :, [2, 1, 0]]


        if self.opt['phase'] == 'val':
        # resize to in_size
            img_LQ = cv2.resize(img_LQ, (GT_size, GT_size), interpolation=cv2.INTER_LINEAR)
            img_GT = cv2.resize(img_GT, (GT_size, GT_size), interpolation=cv2.INTER_LINEAR)

        img_GT = torch.from_numpy(np.ascontiguousarray(np.transpose(img_GT, (2, 0, 1)))).float()
        img_LQ = torch.from_numpy(np.ascontiguousarray(np.transpose(img_LQ, (2, 0, 1)))).float()

        if LQ_path is None:
            LQ_path = GT_path

        normalize(img_LQ, self.mean, self.std, inplace=True)
        normalize(img_GT, self.mean, self.std, inplace=True)

        return {'in': img_LQ, 'gt': img_GT, 'lq_path': LQ_path, 'gt_path': GT_path}

    def __len__(self):
        return len(self.paths_GT)




@DATASET_REGISTRY.register()
class LQGTDataset3_weight(data.Dataset): 


    def __init__(self, opt):
        super(LQGTDataset3_weight, self).__init__()
        self.opt = opt
        self.data_type = self.opt['data_type']
        self.dir_num = self.opt['data_num']

        self.sizes_LQ, self.sizes_GT = 0, 0
        self.LQ_env, self.GT_env = None, None  
        self.mean = opt.get('mean', [0.5, 0.5, 0.5])
        self.std = opt.get('std', [0.5, 0.5, 0.5])
        self.haze = self.opt['haze']
        self.paths_LQs, self.paths_GTs = [], []        
        self.indices = []

        for i in range(self.dir_num):
            paths_GT, sizes_GT = util.get_image_paths(self.data_type, opt['dataroot_GT'+str(i)])
            paths_LQ, sizes_LQ = util.get_image_paths(self.data_type, opt['dataroot_LQ'+str(i)])
            self.paths_LQs.append(paths_LQ)
            self.paths_GTs.append(paths_GT)
            self.indices.append(list(range(len(paths_LQ))))
            if i == 0:    
                self.sizes_GT = sizes_GT
                self.sizes_LQ = sizes_LQ
            else:
                self.sizes_GT += sizes_GT
                self.sizes_LQ += sizes_LQ  

        if self.haze == True:
            paths_LQ2, sizes_LQ2 = util.get_image_paths(self.data_type, opt['dataroot_LQ_z'])
            paths_GT2 = [ opt['dataroot_GT_z'] + '/'+ (lq.split('/')[-1]).split('_')[0]+'.jpg' for lq in paths_LQ2]
            self.paths_LQs.append(paths_LQ2)
            self.paths_GTs.append(paths_GT2)
            self.indices.append(list(range(len(paths_LQ2))))                

        assert self.paths_GTs, 'Error: GT path is empty.'
        if self.paths_LQs and self.paths_GTs:
            assert self.sizes_GT == self.sizes_LQ , 'GT and LQ datasets have different number of images - {}, {}.'.format(
                self.sizes_LQ, self.sizes_GT)
        if len(self.opt['weights']) == 0:
            w = 0.1
            self.weights = [1.0 / len(self.paths_LQs) for i in range(len(self.paths_LQs))]
        else:
            self.weights = [w / sum(self.opt['weights']) for w in self.opt['weights']]
        
        self.ww = [0 for i in self.weights]
        self.shuffle_indices()

    def shuffle_indices(self, data_type_idx=None):
        """
        Shuffle the indices for the specified data type or all data types.
        """
        if data_type_idx is None:
            for idx_list in self.indices:
                random.shuffle(idx_list)
        else:
            random.shuffle(self.indices[data_type_idx])
        
    def __getitem__(self, index):
        data_type_idx = random.choices(range(len(self.weights)), weights=self.weights, k=1)[0]
        data_t_paths = self.paths_LQs[data_type_idx]
        gt_t_paths = self.paths_GTs[data_type_idx]
        idx = random.choices(range(len(gt_t_paths)), weights=[1] * len(gt_t_paths), k=1)[0] 
        
        GT_path, LQ_path = None, None
        GT_path = gt_t_paths[idx]
        LQ_path = data_t_paths[idx]
        scale = self.opt['scale']
        self.ww[data_type_idx] += 1
        
        GT_size = self.opt['GT_size'] if self.opt['GT_size'] is not None else self.opt['HQ_size']

        # get GT image

        if self.data_type == 'lmdb':
            resolution = [int(s) for s in self.sizes_GT[index].split('_')]
        else:
            resolution = None
        
        img_GT = util.read_img(self.GT_env, GT_path, resolution)
        img_LQ = util.read_img(self.LQ_env, LQ_path, resolution)
        
        
        W,H,C = img_GT.shape
        


        if self.opt['phase'] == 'train':
            # if the image size is too small
            H, W, _ = img_GT.shape
            if H < GT_size or W < GT_size:
                img_GT = cv2.resize(np.copy(img_GT), (GT_size, GT_size),
                                    interpolation=cv2.INTER_LINEAR)
                # using matlab imresize
                img_LQ = util.imresize_np(img_GT, 1 / scale, True)
                if img_LQ.ndim == 2:
                    img_LQ = np.expand_dims(img_LQ, axis=2)

            H, W, C = img_LQ.shape
            LQ_size = GT_size // scale

            # randomly crop
            rnd_h = random.randint(0, max(0, H - LQ_size))
            rnd_w = random.randint(0, max(0, W - LQ_size))
            img_LQ = img_LQ[rnd_h:rnd_h + LQ_size, rnd_w:rnd_w + LQ_size, :]
            rnd_h_GT, rnd_w_GT = int(rnd_h * scale), int(rnd_w * scale)
            img_GT = img_GT[rnd_h_GT:rnd_h_GT + GT_size, rnd_w_GT:rnd_w_GT + GT_size, :]

            # augmentation - flip, rotate
            img_LQ, img_GT = util.augment([img_LQ, img_GT], self.opt['use_flip'],
                                          self.opt['use_rot'])

        
        # BGR to RGB, HWC to CHW, numpy to tensor
        if img_GT.shape[2] == 3:
            img_GT = img_GT[:, :, [2, 1, 0]]
            img_LQ = img_LQ[:, :, [2, 1, 0]]


        if self.opt['phase'] == 'val':
        # resize to in_size
            img_LQ = cv2.resize(img_LQ, (GT_size, GT_size), interpolation=cv2.INTER_LINEAR)
            img_GT = cv2.resize(img_GT, (GT_size, GT_size), interpolation=cv2.INTER_LINEAR)

        img_GT = torch.from_numpy(np.ascontiguousarray(np.transpose(img_GT, (2, 0, 1)))).float()
        img_LQ = torch.from_numpy(np.ascontiguousarray(np.transpose(img_LQ, (2, 0, 1)))).float()

        if LQ_path is None:
            LQ_path = GT_path

        normalize(img_LQ, self.mean, self.std, inplace=True)
        normalize(img_GT, self.mean, self.std, inplace=True)

        return {'in': img_LQ, 'gt': img_GT, 'lq_path': LQ_path, 'gt_path': GT_path}

    def __len__(self):
        return sum(len(files) for files in self.paths_LQs)

