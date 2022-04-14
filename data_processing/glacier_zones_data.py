from data_processing.glacier_data import GlacierDataModule
from data_processing.glacier_data import GlacierDataset
import torch
from torchvision.transforms import functional as F
import numpy as np


class ToTensorZones(object):
    def __call__(self, image, target):
        image = F.to_tensor(image)
        target = torch.as_tensor(np.array(target))
        # value for NA area=0, stone=64, glacier=127, ocean with ice melange=254
        target[target == 0] = 0
        target[target == 64] = 1
        target[target == 127] = 2
        target[target == 254] = 3
        # class ids for NA area=0, stone=1, glacier=2, ocean with ice melange=3
        return image, target


class GlacierZonesDataset(GlacierDataset):
    def __init__(self, mode, augmentation, parent_dir, bright, wrap, noise, rotate, flip):
        super(GlacierZonesDataset, self).__init__(target="zones",
                                                  mode=mode,
                                                  augmentation=augmentation,
                                                  parent_dir=parent_dir,
                                                  bright=bright,
                                                  wrap=wrap,
                                                  noise=noise,
                                                  rotate=rotate,
                                                  flip=flip)

    def custom_to_tensor(self, image, target):
        to_tensor = ToTensorZones()
        image, mask = to_tensor(image=image, target=target)
        return image, mask


class GlacierZonesDataModule(GlacierDataModule):

    def __init__(self, batch_size, augmentation, parent_dir, bright, wrap, noise, rotate, flip):
        super(GlacierZonesDataModule, self).__init__(batch_size=batch_size,
                                                     target="zones",
                                                     augmentation=augmentation,
                                                     parent_dir=parent_dir,
                                                     bright=bright,
                                                     wrap=wrap,
                                                     noise=noise,
                                                     rotate=rotate,
                                                     flip=flip)

    def setup(self, stage=None):
        if stage == 'test' or stage is None:
            self.glacier_test = GlacierZonesDataset(mode='test', augmentation=self.augmentation,
                                                    parent_dir=self.parent_dir,
                                                    bright=self.bright,
                                                    wrap=self.wrap,
                                                    noise=self.noise,
                                                    rotate=self.rotate,
                                                    flip=self.flip)
        if stage == 'fit' or stage is None:
            self.glacier_train = GlacierZonesDataset(mode='train', augmentation=self.augmentation,
                                                     parent_dir=self.parent_dir,
                                                     bright=self.bright,
                                                     wrap=self.wrap,
                                                     noise=self.noise,
                                                     rotate=self.rotate,
                                                     flip=self.flip)
            self.glacier_val = GlacierZonesDataset(mode='val', augmentation=self.augmentation,
                                                   parent_dir=self.parent_dir,
                                                   bright=self.bright,
                                                   wrap=self.wrap,
                                                   noise=self.noise,
                                                   rotate=self.rotate,
                                                   flip=self.flip)
