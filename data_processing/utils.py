import torch
import numpy as np
import tormentor
import torchvision


class Compose(object):
    """
    Class for chaining transforms together
    """
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class MyWrap(object):
    """
    Random wrap augmentation taken from tormentor
    """
    def __call__(self, image, target):
        # This augmentation acts like many simultaneous elastic transforms with gaussian sigmas set at varius harmonics
        wrap_rand = tormentor.Wrap.override_distributions(roughness=tormentor.random.Uniform(value_range=(.1, .7)),
                                                          intensity=tormentor.random.Uniform(value_range=(.0, 1.)))
        wrap = wrap_rand()
        image = wrap(image)
        mask = wrap(target, is_mask=True)
        return image, mask


class Rotate(object):
    """
    Random rotation augmentation
    """
    def __call__(self, image, target):
        random = np.random.randint(0, 3)
        angle = 90
        if random == 1:
            angle = 180
        elif random == 2:
            angle = 270
        image = torchvision.transforms.functional.rotate(image, angle=angle)
        mask = torchvision.transforms.functional.rotate(target, angle=angle)

        return image, mask


class Bright(object):
    """
    Random brightness adjustment augmentations
    """
    def __call__(self, image, target):
        bright_rand = tormentor.Brightness.override_distributions(
            brightness=tormentor.random.Uniform((-0.2, 0.2)))
        bright = bright_rand()
        image_transformed = image.clone()
        image_transformed = bright(image_transformed)
        # set NA areas back to zero
        image_transformed[image == 0] = 0.0

        return image_transformed, target


class Noise(object):
    """
    Random additive noise augmentation
    """
    def __call__(self, image, target):
        # add noise. It is a multiplicative gaussian noise so no need to set na areas back to zero again
        noise = torch.normal(mean=0, std=0.3, size=image.shape)
        image = image + image * noise
        image[image > 1.0] = 1.0
        image[image < 0.0] = 0.0

        return image, target