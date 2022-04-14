from data_processing.glacier_data import GlacierDataModule
from data_processing.glacier_data import GlacierDataset
import torch
from torchvision.transforms import functional as F
import numpy as np


class ToTensorFront(object):
    def __call__(self, image, target):
        image = F.to_tensor(image)
        target = torch.as_tensor(np.array(target))
        # value for front=255, background=0
        target[target == 255] = 1
        target[target == 0] = 0
        # class ids for front=1, background=0
        return image, target


class GlacierFrontDataset(GlacierDataset):
    def __init__(self, mode, augmentation, parent_dir, bright, wrap, noise, rotate, flip):
        super(GlacierFrontDataset, self).__init__(target="fronts",
                                                  mode=mode,
                                                  augmentation=augmentation,
                                                  parent_dir=parent_dir,
                                                  bright=bright,
                                                  wrap=wrap,
                                                  noise=noise,
                                                  rotate=rotate,
                                                  flip=flip)

    def custom_to_tensor(self, image, target):
        to_tensor = ToTensorFront()
        image, mask = to_tensor(image=image, target=target)
        return image, mask


class GlacierFrontDataModule(GlacierDataModule):

    def __init__(self, batch_size, augmentation, parent_dir, bright, wrap, noise, rotate, flip):
        super(GlacierFrontDataModule, self).__init__(batch_size=batch_size,
                                                     target="fronts",
                                                     augmentation=augmentation,
                                                     parent_dir=parent_dir,
                                                     bright=bright,
                                                     wrap=wrap,
                                                     noise=noise,
                                                     rotate=rotate,
                                                     flip=flip)

    def setup(self, stage=None):
        if stage == 'test' or stage is None:
            self.glacier_test = GlacierFrontDataset(mode='test', augmentation=self.augmentation,
                                                    parent_dir=self.parent_dir,
                                                    bright=self.bright,
                                                    wrap=self.wrap,
                                                    noise=self.noise,
                                                    rotate=self.rotate,
                                                    flip=self.flip)
        if stage == 'fit' or stage is None:
            self.glacier_train = GlacierFrontDataset(mode='train', augmentation=self.augmentation,
                                                     parent_dir=self.parent_dir,
                                                     bright=self.bright,
                                                     wrap=self.wrap,
                                                     noise=self.noise,
                                                     rotate=self.rotate,
                                                     flip=self.flip)
            self.glacier_val = GlacierFrontDataset(mode='val', augmentation=self.augmentation,
                                                   parent_dir=self.parent_dir,
                                                   bright=self.bright,
                                                   wrap=self.wrap,
                                                   noise=self.noise,
                                                   rotate=self.rotate,
                                                   flip=self.flip)
