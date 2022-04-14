from argparse import ArgumentParser
import os
import threading
from pathlib import Path
import numpy as np
from sklearn.model_selection import train_test_split
import pickle
import cv2


def extract_grayscale_patches(img, patch_size_tuple, offset=(0, 0), stride=(1, 1)):
    """ Extracts (typically) overlapping regular patches from a grayscale image
    Changing the offset and stride parameters will result in images
    reconstructed by reconstruct_from_grayscale_patches having different
    dimensions! Callers should pad and unpad as necessary!
    Adopted from: http://jamesgregson.ca/extract-image-patches-in-python.html
    :param img: (rows/height x columns/width ndarray) input image from which to extract patches
    :param patch_size_tuple: (2-element arraylike) patch_size_tuple of that patches as (h, w)
    :param offset: (2-element arraylike) offset of the initial point as (y, x)
    :param stride: (2-element arraylike) vertical and horizontal strides
    :return: patches, origin
        patches (ndarray): output image patches as (N,patch_size_tuple[0],patch_size_tuple[1]) array
        origin (2-tuple): array of y_coord and array of x_coord coordinates
    """
    px, py = np.meshgrid(np.arange(patch_size_tuple[1]), np.arange(patch_size_tuple[0]))

    # Get left, top (x, y) coordinates for the patches
    x_tmp = np.arange(offset[0], len(img) - patch_size_tuple[0] + 1, stride[0])
    y_tmp = np.arange(offset[1], len(img[0]) - patch_size_tuple[1] + 1, stride[1])

    # Return coordinate matrices from coordinate vectors
    # (e.g. x_tmp is [0, 1, 2] and y_tmp is [0, 1],
    # then x_coord is [[0, 1, 2], [0, 1, 2]] and y_coord is [[0, 1], [0, 1], [0, 1]]
    x_coord, y_coord = np.meshgrid(x_tmp, y_tmp)

    # Return a contiguous flattened array
    # In the example: x_coord becomes [0, 1, 2, 0, 1, 2] and y_coord [0, 1, 0, 1, 0, 1]
    x_coord = x_coord.ravel()
    y_coord = y_coord.ravel()

    # Get X Indices for each pixel per patch
    x_index_within_patch = np.tile(py[None, :, :], (x_coord.size, 1, 1))
    x_offset_in_image = np.tile(x_coord[:, None, None], (1, patch_size_tuple[0], patch_size_tuple[1]))
    x = x_index_within_patch + x_offset_in_image

    # Get Y Indices for each pixel per patch
    y_offset_in_image = np.tile(y_coord[:, None, None], (1, patch_size_tuple[0], patch_size_tuple[1]))
    y_index_within_patch = np.tile(px[None, :, :], (y_coord.size, 1, 1))
    y = y_offset_in_image + y_index_within_patch

    patches = img[x.ravel(), y.ravel()].reshape((-1, patch_size_tuple[0], patch_size_tuple[1]))
    return patches, (x_coord, y_coord)


def pad_image_and_masks(image, patch_size_tuple, overlap):
    """
    Pads the image on bottom and right such that the height and width are both multiples of the patch size if the
    overlap is 0 otherwise the (height - patch_size_tuple[0]) is a multiple of (patch_size_tuple[0] - overlap)
    and (width - patch_size_tuple[1]) are both multiples of (patch_size_tuple[1] - overlap)
    :param image: np array (height x width)
    :param patch_size_tuple: (integer, integer)
    :param overlap: integer
    :return: the padded image as np arrays
    """
    bottom = patch_size_tuple[0] - (len(image) % patch_size_tuple[0])
    bottom = bottom % patch_size_tuple[0]  # if bottom is exactly patch_size then now it is 0
    right = patch_size_tuple[1] - (len(image[0]) % patch_size_tuple[1])
    right = right % patch_size_tuple[1]  # if right is exactly patch_size then now it is 0
    if overlap > 0:
        bottom = (patch_size_tuple[0] - overlap) - ((len(image) - patch_size_tuple[0]) % (patch_size_tuple[0] - overlap))
        right = (patch_size_tuple[1] - overlap) - ((len(image[0]) - patch_size_tuple[1]) % (patch_size_tuple[1] - overlap))
    img_pad = np.zeros((len(image) + bottom, len(image[0]) + right), dtype=image.dtype)
    img_pad[:len(image), :len(image[0])] = image
    return img_pad, bottom, right


def thicken_fronts(mask):
    """
    Dilate front label with a 5x5 kernel.
    :param mask: the front label
    :return: the dilated front label
    """
    kernel = np.ones((5, 5), np.uint8)
    mask_line_pad = cv2.dilate(mask, kernel, iterations=1)
    return mask_line_pad


def preprocess(modality_dir, data_split_dir, files, patch_size, overlap):
    """
    Perform pre-processing steps.
    :param modality_dir: Either "sar_images" or "zones" or "fronts"
    :param data_split_dir: "test", "train" or "val"
    :param files: The location+name of all files to be processed
    :param patch_size: size of the patches that shall be extracted
    :param overlap: number of pixels that the patches shall overlap
    :return:
    """
    parent_dir = os.path.dirname(os.getcwd())
    for file in files:
        print(file)
        image = cv2.imread(file.__str__(), cv2.IMREAD_GRAYSCALE)

        # #####################################################################################################
        # Alter images
        # #####################################################################################################
        if modality_dir == "fronts":
            # Thicken the fronts
            image = thicken_fronts(image)

        # #####################################################################################################
        # Pad sides such that patch extraction works smoothly
        # #####################################################################################################
        stride = (patch_size - overlap, patch_size - overlap)
        patch_size_tuple = (patch_size, patch_size)

        image, bottom, right = pad_image_and_masks(image, patch_size_tuple=patch_size_tuple, overlap=overlap)

        # #####################################################################################################
        # Extract Patches
        # #####################################################################################################
        patches_image, coords_image = extract_grayscale_patches(image, patch_size_tuple=patch_size_tuple, stride=stride)
        img_name = os.path.split(file)[1][:-4]

        # #####################################################################################################
        # Store Patches with useful names
        # #####################################################################################################
        for j in range(len(patches_image)):
            # naming convention: nameOfTheOriginalImage__PaddedBottom_PaddedRight_NumberOfPatch_irow_icol.png
            # with x and y the x and y coordinates of the patch in the complete image
            add_to_name = '__' + str(bottom) + '_' + str(right) + '_' + str(j) + '_' \
                          + str(coords_image[0][j]) + '_' + str(coords_image[1][j]) + '.png'
            cv2.imwrite(os.path.join(parent_dir, "data", modality_dir, data_split_dir, img_name + add_to_name), patches_image[j])


def main(raw_data_dir, patch_size, overlap, overlap_test, overlap_val):
    """
    Split dataset and initiate pre-processing.
    :param raw_data_dir: name of the directory including the not processed data
    :param patch_size: size of the patches that shall be extracted
    :param overlap: number of pixels that the training patches shall overlap
    :param overlap_test: number of pixels that the test patches shall overlap
    :param overlap_val: number of pixels that the validation patches shall overlap
    :return:
    """
    parent_dir = os.path.dirname(os.getcwd())
    threads = []
    for modality_dir in ["sar_images", "zones", "fronts"]:
        if not os.path.exists(os.path.join(parent_dir, "data", modality_dir)):
            os.makedirs(os.path.join(parent_dir, "data", modality_dir))
        if not os.path.exists(os.path.join(parent_dir, "data", modality_dir, "val")):
            os.makedirs(os.path.join(parent_dir, "data", modality_dir, "val"))

        for data_split_dir in ["test", "train"]:
            if not os.path.exists(os.path.join(parent_dir, "data", modality_dir, data_split_dir)):
                os.makedirs(os.path.join(parent_dir, "data", modality_dir, data_split_dir))

            folder = sorted(Path(os.path.join(parent_dir, raw_data_dir, modality_dir, data_split_dir)).rglob('*.png'))
            files = [x for x in folder]

            if data_split_dir == "train":
                ###################################################################################
                #      Train data needs to be split into training and validation set
                ###################################################################################
                if not os.path.exists("data_splits"):
                    os.makedirs("data_splits")

                if not os.path.isfile(os.path.join("data_splits", "train_idx.txt")) \
                        or not os.path.isfile(os.path.join("data_splits", "val_idx.txt")):
                    data_idx = np.arange(len(files))
                    train_idx, val_idx = train_test_split(data_idx, test_size=0.1, random_state=1)

                    with open(os.path.join("data_splits", "train_idx.txt"), "wb") as fp:
                        pickle.dump(train_idx, fp)

                    with open(os.path.join("data_splits", "val_idx.txt"), "wb") as fp:
                        pickle.dump(val_idx, fp)
                else:
                    # use already existing split
                    with open(os.path.join("data_splits", "train_idx.txt"), "rb") as fp:
                        train_idx = pickle.load(fp)

                    with open(os.path.join("data_splits", "val_idx.txt"), "rb") as fp:
                        val_idx = pickle.load(fp)

                # Start preprocessing for both training and validation set
                t = threading.Thread(target=preprocess, args=(modality_dir, data_split_dir, [files[i] for i in train_idx], patch_size, overlap))
                threads.append(t)
                t.start()

                t = threading.Thread(target=preprocess, args=(modality_dir, "val", [files[i] for i in val_idx], patch_size, overlap_val))
                threads.append(t)
                t.start()

            else:
                ###################################################################################
                #      Test data
                ###################################################################################
                t = threading.Thread(target=preprocess, args=(modality_dir, data_split_dir, files, patch_size, overlap_test))
                threads.append(t)
                t.start()

    for t in threads:
        t.join()


if __name__ == '__main__':
    parser = ArgumentParser(add_help=False)
    parser.add_argument('--raw_data_dir', default="data_raw", help="The name of the directory, "
                                                                   "where the raw data is stored. "
                                                                   "Default is 'data_raw'.")
    parser.add_argument('--patch_size', default=256, help="The size of the extracted patches of an image.")
    parser.add_argument('--overlap', default=0, help="The overlap in the sliding window approach (patch extraction) for training data.")
    parser.add_argument('--overlap_test', default=128, help="The overlap in the sliding window approach (patch extraction) for test data.")
    parser.add_argument('--overlap_val', default=128, help="The overlap in the sliding window approach (patch extraction) for validation data.")
    hparams = parser.parse_args()

    parent_dir = os.path.dirname(os.getcwd())

    if not os.path.exists(os.path.join(parent_dir, "data")):
        os.makedirs(os.path.join(parent_dir, "data"))

    assert hparams.patch_size > hparams.overlap, "Choose an overlap that is smaller than your patch_size"

    main(raw_data_dir=hparams.raw_data_dir, patch_size=hparams.patch_size, overlap=hparams.overlap, overlap_test=hparams.overlap_test, overlap_val=hparams.overlap_val)
