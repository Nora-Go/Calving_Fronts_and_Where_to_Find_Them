import os
import cv2
from torchvision.transforms import functional as F
import torch


if __name__ == "__main__":
    path = os.path.join(os.path.dirname(os.getcwd()), "data_raw", "sar_images", "train")
    imgs = os.listdir(path)
    list_imgs = []
    for img_name in imgs:
        image = cv2.imread(os.path.join(path, img_name).__str__(), cv2.IMREAD_GRAYSCALE)
        image = F.to_tensor(image)
        list_imgs.append(torch.flatten(image))
    torch_imgs = torch.cat(list_imgs, dim=0)
    mean = torch.mean(torch_imgs)
    std_dev = torch.std(torch_imgs)
    print(f"Mean of training data: {mean}")
    print(f"Standard deviation of training data: {std_dev}")
