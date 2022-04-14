import pytorch_lightning as pl
from torch.utils.data import DataLoader
import os
import cv2
from torch.utils.data import Dataset
import torchvision
from data_processing.utils import MyWrap, Rotate, Bright, Noise
import numpy as np
import pickle


class GlacierDataset(Dataset):
    def __init__(self, target, mode, augmentation, parent_dir, bright, wrap, noise, rotate, flip):
        if mode == 'train':
            self.images_path = os.path.join(parent_dir, "data", "sar_images", "train")
            self.targets_path = os.path.join(parent_dir, "data", target, "train")
        elif mode == 'val':
            self.images_path = os.path.join(parent_dir, "data", "sar_images", "val")
            self.targets_path = os.path.join(parent_dir, "data", target, "val")
        elif mode == 'test':
            self.images_path = os.path.join(parent_dir, "data", "sar_images", "test")
            self.targets_path = os.path.join(parent_dir, "data", target, "test")
        self.imgs = os.listdir(self.images_path)
        self.labels = os.listdir(self.targets_path)
        # Sort so images and labels fit together
        self.imgs.sort()
        self.labels.sort()

        # Shuffle so that if limit_train/val/test_batch is set, not only all images from one glacier are used
        # However, always use same shuffle so that it is reproducible
        if not os.path.exists(os.path.join("data_processing", "data_splits")):
            os.makedirs(os.path.join("data_processing", "data_splits"))
        if not os.path.isfile(os.path.join("data_processing", "data_splits", "shuffle_" + mode + ".txt")):
            shuffle = np.random.permutation(len(self.imgs))
            with open(os.path.join("data_processing", "data_splits", "shuffle_" + mode + ".txt"), "wb") as fp:
                pickle.dump(shuffle, fp)
        else:
            # use already existing shuffle
            with open(os.path.join("data_processing", "data_splits", "shuffle_" + mode + ".txt"), "rb") as fp:
                shuffle = pickle.load(fp)
                # if lengths do not match, we need to create a new permutation
                if len(shuffle) != len(self.imgs):
                    shuffle = np.random.permutation(len(self.imgs))
                    with open(os.path.join("data_processing", "data_splits", "shuffle_" + mode + ".txt"), "wb") as fp:
                        pickle.dump(shuffle, fp)

        self.imgs = np.array(self.imgs)
        self.labels = np.array(self.labels)
        tmp = self.imgs[shuffle]
        self.imgs = tmp
        tmp = self.labels[shuffle]
        self.labels = tmp
        self.imgs = list(self.imgs)
        self.labels = list(self.labels)

        # assert both lists have the same length
        assert len(self.imgs) == len(self.labels), "You don't have the same number of images and masks"
        self.mode = mode
        self.augmentation = augmentation

        self.bright = bright
        self.wrap = wrap
        self.noise = noise
        self.rotate = rotate
        self.flip = flip

    def custom_to_tensor(self, image, target):
        return NotImplemented, NotImplemented

    def transform(self, image, mask):
        do_augmentation = self.augmentation
        # ToTensor automatically scales the input to [0, 1]
        image, mask = self.custom_to_tensor(image=image, target=mask)

        if self.mode == 'train' and do_augmentation:
            if np.random.random() >= (1 - self.flip):
                image = torchvision.transforms.functional.hflip(image)
                mask = torchvision.transforms.functional.hflip(mask)
            if np.random.random() >= (1 - self.rotate):
                rot = Rotate()
                image, mask = rot(image=image, target=mask.unsqueeze(0))
                mask = mask.squeeze(0)
            if np.random.random() >= (1 - self.bright):
                bright = Bright()
                image, mask = bright(image=image, target=mask)
            if np.random.random() >= (1 - self.wrap):
                wrap_transform = MyWrap()
                image, mask = wrap_transform(image=image, target=mask)
            if np.random.random() >= (1 - self.noise):
                noise = Noise()
                image, mask = noise(image=image, target=mask)

        # Z-Score Normalization
        norm = torchvision.transforms.Normalize(mean=0.3047126829624176, std=0.32187142968177795)
        image = norm(image)
        return image, mask

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img_name = self.imgs[idx]
        label_name = self.labels[idx]
        # Assert that image and label name match
        assert img_name.split("__")[0] == label_name.split("__")[0][:-6], "image and label name don't match. Image name: " + img_name + ". Label name: " + label_name
        assert img_name.split("__")[1] == label_name.split("__")[1], "image and label name don't match. Image name: " + img_name + ". Label name: " + label_name
        image = cv2.imread(os.path.join(self.images_path, img_name).__str__(), cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(os.path.join(self.targets_path, label_name).__str__(), cv2.IMREAD_GRAYSCALE)
        x, y = self.transform(image, mask)
        return x, y, img_name, label_name


class GlacierDataModule(pl.LightningDataModule):

    def __init__(self, batch_size, target, augmentation, parent_dir, bright, wrap, noise, rotate, flip):
        """
        :param batch_size: batch size
        :param target: Either 'zones' or 'front'. Tells which masks should be used.
        :param augmentation: Whether or not augmentation shall be performed
        """
        super().__init__()
        self.batch_size = batch_size
        self.target = target
        self.glacier_test = None
        self.glacier_train = None
        self.glacier_val = None
        self.augmentation = augmentation
        self.parent_dir = parent_dir

        self.bright = bright
        self.wrap = wrap
        self.noise = noise
        self.rotate = rotate
        self.flip = flip

    def setup(self, stage=None):
        pass

    def train_dataloader(self):
        return DataLoader(self.glacier_train, batch_size=self.batch_size, num_workers=4, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.glacier_val, batch_size=self.batch_size, num_workers=4, pin_memory=True)

    def test_dataloader(self):
        return DataLoader(self.glacier_test, batch_size=self.batch_size, num_workers=4, pin_memory=True)

    def prepare_data(self, *args, **kwargs):
        pass
