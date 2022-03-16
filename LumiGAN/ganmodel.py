# %%--  Imports
import torch
import torch.nn as nn
import numpy as np
# %%-
# %%--  Functions
def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0.0)
# %%-
# %%--  Generator
class Generator(nn.Module):
    ''' Generator model - Initial layer size 2**(p2_offset) and p2_deep number of layers and image to mask size difference'''
    def __init__(self, img_size, mask_size, p2_deep=2, p2_offset=6, channels=3, weights_init=True):
        super().__init__()
        def downsample(p2_diff):
            layers = []
            for i in np.arange(p2_offset,p2_offset+p2_deep+p2_diff+1,1):
                if i == p2_offset :
                    layers.append(nn.Conv2d(channels, 2**(i), 4, stride=2, padding=1))
                    layers.append(nn.LeakyReLU(0.2))
                    continue
                layers.append(nn.Conv2d(2**(i-1), 2**(i), 4, stride=2, padding=1))
                layers.append(nn.BatchNorm2d(2**(i), 0.8))
                layers.append(nn.LeakyReLU(0.2))
                layers.append(nn.Dropout(0.1))
            return layers
        def upsample(p2_diff):
            layers = []
            for i in np.arange(p2_offset+p2_deep+p2_diff,p2_offset+p2_diff-1,-1):
                if  i == p2_offset+p2_diff:
                    layers.append(nn.ConvTranspose2d((2**i), channels, 4, stride=2, padding=1))
                    layers.append(nn.Tanh())
                    continue
                layers.append(nn.ConvTranspose2d(2**(i), 2**(i-1), 4, stride=2, padding=1))
                layers.append(nn.BatchNorm2d(2**(i-1), 0.8))
                layers.append(nn.LeakyReLU(0.2))
                layers.append(nn.Dropout(0.1))
            return layers
        if not np.log2(img_size).is_integer(): raise ValueError("Imgage  size = %s is not a power of two."%(img_size))
        if not np.log2(mask_size).is_integer(): raise ValueError("Mask  size = %s is not a power of two."%(mask_size))
        if not img_size>mask_size: raise ValueError("Image size %s is smaller than Mask size %s"%(img_size,mask_size))
        p2_img= int(np.log2(img_size))
        p2_mask = int(np.log2(mask_size))
        p2_diff = p2_img-p2_mask
        if not p2_img>p2_diff+p2_deep: raise ValueError("Model is too deep. Reduce p2_deep")
        self.model = nn.Sequential(
            *downsample(p2_diff),
            *upsample(p2_diff),
        )
        if weights_init: self.apply(weights_init_normal)

    def forward(self, img):
        o = self.model(img)
        return o

# %%-
# %%--  Discriminator
class Discriminator(nn.Module):
    '''Discriminator model - Initial layer size 2**(p2_offset) and p2_deep number of layers'''
    def __init__(self, img_size, p2_deep=4, p2_offset=6, channels=3, weights_init=True):
        super().__init__()
        def block():
            layers = []
            for i in np.arange(p2_offset,p2_offset+p2_deep+1,1):
                if i == p2_offset :
                    layers.append(nn.Conv2d(channels, 2**(i), 4, stride=2, padding=1))
                    layers.append(nn.LeakyReLU(0.2))
                    continue
                layers.append(nn.Conv2d(2**(i-1), 2**(i), 4, stride=2, padding=1))
                layers.append(nn.BatchNorm2d(2**(i), 0.8))
                layers.append(nn.LeakyReLU(0.2))
                layers.append(nn.Dropout(0.1))
            return layers
        p2_mask = np.log2(img_size)
        if not p2_mask.is_integer(): raise ValueError("Image  size = %s is not a power of two."%(img_size))
        if not p2_mask>p2_deep: raise ValueError("Network is too deep (%s) for input image size %s"%(p2_deep,p2_mask))

        self.conv = nn.Sequential(*block())
        self.fc = nn.Sequential(
            nn.Linear(int(2**(p2_deep+p2_offset)*(2**(2*(p2_mask-p2_deep-1)))), int(2**(p2_deep))),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.1),
            nn.Linear(2**(p2_deep),1),
            nn.Tanh()
         )
        if weights_init: self.apply(weights_init_normal)

    def forward(self, img):
        o = self.conv(img)
        o = torch.flatten(o, 1)
        o = self.fc(o)
        return o
# %%-
