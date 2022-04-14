import numpy as np
import os
import re
import pickle
import cv2
from einops import rearrange
from scipy.ndimage.filters import gaussian_filter
import skimage.measure
import skimage.color
from skimage.morphology import skeletonize
from fil_finder import FilFinder2D
import astropy.units as u

# ################################################################################################################
# POSTPROCESSING PUTS THE PATCHES TOGETHER, SUBSTRACTS THE PADDING
# AND CHOOSES THE CLASS WITH HIGHEST PROBABILITY AS PREDICTION.
# SECONDLY, THE FRONT LINE IS EXTRACTED FROM THE PREDICTION
# ################################################################################################################


def is_subarray(subarray, arr):
    """
    Test whether subarray is a subset of arr
    :param subarray: list of numbers
    :param arr: list of numbers
    :return: boolean
    """
    count = 0
    for element in subarray:
        if element in arr:
            count += 1
    if count == len(subarray):
        return True
    return False


def reconstruct_from_patches_and_binarize(src_directory, dst_directory, modality, threshold_front_prob):
    """
    Reconstruct the image from patches in src_directory and store them in dst_directory.
    The src_directory contains masks (patches = number_of_classes x height x width).
    The class with maximum probability will be chosen as prediction after averaging the probabilities across patches
    (if there is an overlap) and the image in dst_directory will only show the prediction (image = height x width)
    :param src_directory:  source directory which contains pickled patches (class x height x width)
    :param dst_directory: destination directory
    :param modality: Either "fronts" or "zones"
    :return: prediction (image = height x width)
    """

    assert modality == "fronts" or modality == "zones", "Modality must either be 'fronts' or 'zones'."

    patches = os.listdir(src_directory)
    list_of_names = []
    for patch_name in patches:
        list_of_names.append(os.path.split(patch_name)[1].split("__")[0])
    image_names = set(list_of_names)
    for name in image_names:
        print(f"File: {name}")
        # #####################################################################################################
        # Search all patches that belong to the image with the name "name"
        # #####################################################################################################
        pattern = re.compile(name)
        patches_for_image_names = [a for a in patches if pattern.match(a)]
        assert len(patches_for_image_names) > 0, "No patches found for image " + name
        patches_for_image = []  # Will be Number_Of_Patches x Number_Of_Classes x Height x Width
        irow = []
        icol = []
        padded_bottom = int(patches_for_image_names[0][:-4].split("_")[-5])
        padded_right = int(patches_for_image_names[0][:-4].split("_")[-4])

        for file_name in patches_for_image_names:
            # #####################################################################################################
            # Get the origin of the patches out of their names
            # #####################################################################################################
            # naming convention: nameOfTheOriginalImage__PaddedBottom_PaddedRight_NumberOfPatch_irow_icol.png

            # Mask patches are 3D arrays with class probabilities
            with open(os.path.join(src_directory, file_name), "rb") as fp:
                class_probabilities_array = pickle.load(fp)
                assert class_probabilities_array.ndim == 3, "Patch " + file_name + " has not enough dimensions (3 needed). Found: " + str(class_probabilities_array.ndim)
                if modality == "fronts":
                    assert len(class_probabilities_array) <= 2, "Patch " + file_name + " has too many classes (<2 needed). Found: " + str(len(class_probabilities_array))
                else:
                    assert len(class_probabilities_array) <= 4, "Patch " + file_name + " has too many classes (<4 needed). Found: " + str(len(class_probabilities_array))
                patches_for_image.append(class_probabilities_array)
            irow.append(int(os.path.split(file_name)[1][:-4].split("_")[-2]))
            icol.append(int(os.path.split(file_name)[1][:-4].split("_")[-1]))

        # Images are masks and store the probabilities for each class (patch = number_class x height x width)
        class_patches_for_image = []
        patches_for_image = [np.array(x) for x in patches_for_image]
        patches_for_image = np.array(patches_for_image)
        for class_layer in range(len(patches_for_image[0])):
            class_patches_for_image.append(patches_for_image[:, class_layer, :, :])

        class_probabilities_complete_image = []

        # #####################################################################################################
        # Reconstruct image (with number of channels = classes) from patches
        # #####################################################################################################
        for class_number in range(len(class_patches_for_image)):
            class_probability_complete_image, _ = reconstruct_from_grayscale_patches_with_origin(class_patches_for_image[class_number],
                                                                                                 origin=(irow, icol), use_gaussian=True)
            class_probabilities_complete_image.append(class_probability_complete_image)

        ######################################################################################################
        # Cut Padding
        ######################################################################################################
        if modality == "zones":
            class_probabilities_complete_image = np.array(class_probabilities_complete_image)
            class_probabilities_complete_image = class_probabilities_complete_image[:, :-padded_bottom, :-padded_right]
        else:
            class_probabilities_complete_image = rearrange(class_probabilities_complete_image, '1 h w -> h w')
            class_probabilities_complete_image = np.array(class_probabilities_complete_image)
            class_probabilities_complete_image = class_probabilities_complete_image[:-padded_bottom, :-padded_right]

        # #####################################################################################################
        # Get prediction from probabilities
        # #####################################################################################################
        if modality == "zones":
            # Choose class with highest probability as prediction
            prediction = np.argmax(class_probabilities_complete_image, axis=0)
        else:
            # Take a threshold to get the class
            prediction = class_probabilities_complete_image
            prediction[prediction > threshold_front_prob] = 1
            prediction[prediction <= threshold_front_prob] = 0

        # #####################################################################################################
        #  Convert [0, 1] to [0, 255] range
        # #####################################################################################################
        if modality == "fronts":
            prediction[prediction == 0] = 0
            prediction[prediction == 1] = 255
            assert (is_subarray(np.unique(prediction), [0, 255])), "Unique front values are not correct"
        else:
            prediction[prediction == 0] = 0
            prediction[prediction == 1] = 64
            prediction[prediction == 2] = 127
            prediction[prediction == 3] = 254
            assert (is_subarray(np.unique(prediction), [0, 64, 127, 254])), "Unique zone values are not correct"

        cv2.imwrite(os.path.join(dst_directory, name + '.png'), prediction)


def get_gaussian(patch_size, sigma_scale=1. / 8) -> np.ndarray:
    """
    Returns Gaussian map with size of patch and sig
    :param patch_size: The size of the image patches -> gaussian importance map will have the same size
    :param sigma_scale: A scaling factor
    :return: Gaussian importance map
    """
    tmp = np.zeros(patch_size)
    center_coords = [i // 2 for i in patch_size]
    sigmas = [i * sigma_scale for i in patch_size]
    tmp[tuple(center_coords)] = 1
    gaussian_importance_map = gaussian_filter(tmp, sigmas, 0, mode='constant', cval=0)
    gaussian_importance_map = gaussian_importance_map / np.max(gaussian_importance_map) * 1
    gaussian_importance_map = gaussian_importance_map.astype(np.float32)

    # gaussian_importance_map cannot be 0, otherwise we may end up with nans!
    gaussian_importance_map[gaussian_importance_map == 0] = np.min(
        gaussian_importance_map[gaussian_importance_map != 0])

    return gaussian_importance_map


def reconstruct_from_grayscale_patches_with_origin(patches, origin, use_gaussian, epsilon=1e-12):
    """Rebuild an image from a set of patches by averaging. The reconstructed image will have different dimensions than
    the original image if the strides and offsets of the patches were changed from the defaults!
    Adopted from: http://jamesgregson.ca/extract-image-patches-in-python.html
    :param patches: (ndarray) input patches as (N,patch_height,patch_width) array
    :param origin: (2-tuple) = row index and column index coordinates of each patch
    :param use_gaussian: Boolean to turn on Gaussian Importance Weighting
    :param epsilon: (scalar) regularization term for averaging when patches some image pixels are not covered by any patch
    :return image, weight
        image (ndarray): output image reconstructed from patches of size (max(origin[0])+patches.shape[1], max(origin[1])+patches.shape[2])
        weight (ndarray): output weight matrix consisting of the count of patches covering each pixel
    """
    patches = np.array(patches)
    origin = np.array(origin)
    patch_height = len(patches[0])
    patch_width = len(patches[0][0])
    img_height = np.max(origin[0]) + patch_height
    img_width = np.max(origin[1]) + patch_width

    out = np.zeros((img_height, img_width))
    wgt = np.zeros((img_height, img_width))
    if use_gaussian:
        scale_wgt = get_gaussian((patch_height, patch_width))
    else:
        scale_wgt = np.ones((patch_height, patch_width))

    for i in range(patch_height):
        for j in range(patch_width):
            out[origin[0]+i, origin[1]+j] += patches[:, i, j] * scale_wgt[i, j]
            wgt[origin[0] + i, origin[1] + j] += scale_wgt[i, j]

    return out / np.maximum(wgt, epsilon), wgt


def postprocess_zone_segmenation(mask):
    """
    Post-process zone segmentation by filling gaps in ocean region and creating cluster of ocean mask and removing clusters except for the largest -> left with one big ocean.
    :param mask: a numpy array representing the segmentation mask with 1 channel
    :return mask: a numpy array representing the filtered mask with 1 channel
    """

    # #############################################################################################
    # Fill Gaps in Ocean
    # #############################################################################################
    # get inverted ocean mask
    ocean_mask = mask == 254
    ocean_mask = np.invert(ocean_mask)
    labeled_image, num_cluster = skimage.measure.label(ocean_mask, connectivity=2, return_num=True)

    # extract largest cluster
    cluster_size = np.zeros(num_cluster + 1)
    for cluster_label in range(1, num_cluster + 1):
        cluster = labeled_image == cluster_label
        cluster_size[cluster_label] = cluster.sum()

    final_cluster = cluster_size.argmax()

    # create map of the gaps in ocean area
    gaps_mask = np.zeros_like(labeled_image)
    gaps_mask[labeled_image >= 1] = 1
    gaps_mask[labeled_image == final_cluster] = 0
    # fill gaps
    mask[gaps_mask == 1] = 254

    # #############################################################################################
    # Take largest connected component of ocean as ocean
    # #############################################################################################
    # Connected Component Analysis
    ocean_mask = mask >= 254  # Ocean (254)
    labeled_image, num_cluster = skimage.measure.label(ocean_mask, connectivity=2, return_num=True)
    if num_cluster == 0:
        return mask

    # extract largest cluster
    cluster_size = np.zeros(num_cluster + 1)  # +1 for background
    for cluster_label in range(1, num_cluster + 1):  # +1 as range(x, y) is exclusive for y
        cluster = labeled_image == cluster_label
        cluster_size[cluster_label] = cluster.sum()

    final_cluster = cluster_size.argmax()
    final_mask = labeled_image == final_cluster

    # overwrite small ocean cluster (254) with glacier value (127) (it is not important with what value these are
    # filled, as these pixels are not at the boundary between ocean and glacier anymore and hence do not contribute to
    # the front delineation)
    mask[mask == 254] = 127
    mask[final_mask] = 254

    return mask


def extract_front_from_zones(zone_mask, front_length_threshold):
    """
    Extract front prediction from zone segmentation by choosing the boundary between glacier and ocean as front and deleting to short fronts.
    :param zone_mask: zone segmentation prediction
    :param front_length_threshold: Threshold for deletion of too short front predictions
    :return: the front prediction
    """
    # detect edge between ocean and glacier
    mask_mi = np.pad(zone_mask, ((1, 1), (1, 1)), mode='constant')
    mask_le = np.pad(zone_mask, ((1, 1), (0, 2)), mode='constant')
    mask_ri = np.pad(zone_mask, ((1, 1), (2, 0)), mode='constant')
    mask_do = np.pad(zone_mask, ((0, 2), (1, 1)), mode='constant')
    mask_up = np.pad(zone_mask, ((2, 0), (1, 1)), mode='constant')

    front = np.logical_and(mask_mi == 254, np.logical_or.reduce((mask_do == 127, mask_up == 127, mask_ri == 127, mask_le == 127)))
    front = front[1:-1, 1:-1].astype(float)

    # delete too short fronts
    labeled_front, num_cluster = skimage.measure.label(front, connectivity=2, return_num=True)
    if num_cluster == 0:
        return front * 255

    for cluster_label in range(1, num_cluster + 1):  # +1 as range(x, y) is exclusive for y
        cluster = labeled_front == cluster_label
        cluster_size = cluster.sum()
        if cluster_size <= front_length_threshold:
            front[labeled_front == cluster_label] = 0
        else:
            front[labeled_front == cluster_label] = 1

    front *= 255
    return front


def postprocess_front_segmenation(complete_predicted_mask, threshold_front_length):
    """
    Post-process the front segmentation by skeletonization, filament extraction, and deletion of too short fronts
    :param complete_predicted_mask: front segmentation prediction
    :param threshold_front_length: Threshold for deletion of too short front predictions
    :return: the post-processed front prediction
    """
    if len(np.unique(complete_predicted_mask)) == 1:
        print(f"No front predicted {np.unique(complete_predicted_mask)}")
        return complete_predicted_mask
    skeleton = skeletonize(complete_predicted_mask)
    fil = FilFinder2D(skeleton, distance=None, mask=skeleton)
    fil.preprocess_image(skip_flatten=True)
    fil.create_mask(use_existing_mask=True)
    fil.medskel(verbose=False)
    fil.analyze_skeletons(skel_thresh=5 * u.pix)
    # find longest path through the skeleton and delete all other branches
    skeleton_longpaths = fil.skeleton_longpath
    # delete fronts that are too short
    labeled_skeleton_longpaths, num_cluster = skimage.measure.label(skeleton_longpaths, connectivity=2, return_num=True)
    if num_cluster == 0:
        return skeleton_longpaths

    for cluster_label in range(1, num_cluster + 1):  # +1 as range(x, y) is exclusive for y
        cluster = labeled_skeleton_longpaths == cluster_label
        cluster_size = cluster.sum()
        if cluster_size <= threshold_front_length:
            skeleton_longpaths[labeled_skeleton_longpaths == cluster_label] = 0
        else:
            skeleton_longpaths[labeled_skeleton_longpaths == cluster_label] = 1
    return skeleton_longpaths
