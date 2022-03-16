import torch.utils.data as Data
import numpy as np
from PIL import Image

class Dataset(Data.Dataset):
    'Dataset for CNN or GAN. set positions are always upper left corner'
    def __init__(self, list_paths,labels, transform, mode='CNN',param_mode={} ):
        'Initialization'
        self.list_paths = list_paths
        self.transform = transform
        self.mode = mode
        if mode=='GAN':
            if 'img_size' not in param_mode.keys(): raise ValueError('img_size not in param_mode ')
            if 'mask_size' not in param_mode.keys(): raise ValueError('mask_size not in param_mode ')
            if 'mask_mode' not in param_mode.keys(): param_mode['mask_mode']='random'
            self.img_size = param_mode['img_size']
            self.mask_size = param_mode['mask_size']
            self.mask_mode = param_mode['mask_mode']
            if self.mask_mode=="position":
                if 'positions' not in param_mode.keys(): raise ValueError('No positions set for position mask mode')
                self.positions = param_mode['positions']
        elif mode=='CNN':
            if not labels: raise ValueError('Missing labels value for CNN mode.')
            self.labels = labels
    def apply_random_mask(self, img):
        """Randomly masks image"""
        x1, y1 = np.random.randint(0, self.img_size - self.mask_size, 2) #upper left coordinate
        x2, y2 = x1 + self.mask_size, y1 + self.mask_size
        masked_part = img[:, x1:x2, y1:y2]
        masked_img = img.clone()
        masked_img[:, x1:x2, y1:y2] = 1

        return masked_img, masked_part, x1, y1

    def apply_center_mask(self, img):
        """Mask center part of image"""
        i = (self.img_size - self.mask_size) // 2 #upper left coordinate
        masked_part = img[:, i : i + self.mask_size, i : i + self.mask_size]
        masked_img = img.clone()
        masked_img[:, i : i + self.mask_size, i : i + self.mask_size] = 1

        return masked_img, masked_part, i, i

    def apply_position_mask(self, img, position):
        """Set masks image on specificed position (upper left corner)"""
        x1, y1 = position[0], position[1] #upper left coordinate
        x2, y2 = x1 + self.mask_size, y1 + self.mask_size
        masked_part = img[:, x1:x2, y1:y2]
        masked_img = img.clone()
        masked_img[:, x1:x2, y1:y2] = 1

        return masked_img, masked_part, x1, y1

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.list_paths)

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select and load sample
        path = self.list_paths[index]
        x = Image.open(path)
        img = self.transform(x)

        # Select mode
        if self.mode=='CNN':
            return img, self.labels[path]
        elif self.mode=='GAN':
            if self.mask_mode=='random': masked_img, masked_part, x1, y1 = self.apply_random_mask(img)
            elif self.mask_mode=='center': masked_img, masked_part, x1, y1 = self.apply_center_mask(img)
            elif self.mask_mode=='position': masked_img, masked_part, x1, y1 = self.apply_position_mask(img, self.positions[path])
            else: raise ValueError('mask_mode input : %s is incorrect.'%(self.mode))
            return img, masked_img, masked_part, x1, y1
        else: raise ValueError('Input mode : %s is incorrect.'%(self.mode))
