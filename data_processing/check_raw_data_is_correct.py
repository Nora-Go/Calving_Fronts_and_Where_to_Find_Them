import os
import numpy as np
import cv2


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


if __name__ == "__main__":

    # ####################################################################################################
    # Check all folders exist
    # ####################################################################################################
    parent_dir = os.path.dirname(os.getcwd())
    assert os.path.exists(os.path.join(parent_dir, "data_raw")), "There is no data_raw folder in the top layer of this project. " \
                                                                 "Please make sure, you downloaded the data and put it into a folder called " \
                                                                 "'data_raw' that is directly under your project folder."

    for modality_dir in ["sar_images", "zones", "fronts"]:
        assert os.path.exists(os.path.join(parent_dir, "data_raw", modality_dir)), "One of the subfolders 'sar_images', 'zones' or 'fronts' " \
                                                              "is missing in the data_raw folder."

        for data_split_dir in ["test", "train"]:
            assert os.path.exists(os.path.join(parent_dir, "data_raw", modality_dir, data_split_dir)), \
                "Please check if each subfolder 'sar_images', 'zones' or 'fronts' in data_raw has again the subfolders " \
                "'test' and 'train'. One must be missing."

    for data_split_dir in ["test", "train"]:
        sar_images = sorted(os.listdir(os.path.join(parent_dir, "data_raw", "sar_images", data_split_dir)))
        fronts = sorted(os.listdir(os.path.join(parent_dir, "data_raw", "fronts", data_split_dir)))
        zones = sorted(os.listdir(os.path.join(parent_dir, "data_raw", "zones", data_split_dir)))

        assert len(sar_images) == len(fronts) and len(sar_images) == len(
            fronts), "You don't have the same number of images and masks"

        if data_split_dir == "train":
            assert len(sar_images) == 559, \
                f"There should be XXX images in the training set. You have: {len(sar_images)}."
        else:
            assert len(sar_images) == 122, \
                f"There should be YYY images in the training set. You have: {len(sar_images)}."

        for sar_image, front, zone in zip(sar_images, fronts, zones):
            # ####################################################################################################
            # Check whether front, zone and sar files fit together
            # ####################################################################################################
            assert str(sar_image)[:-4] == str(front)[:-10] and str(sar_image)[:-4] == str(zone)[:-10], \
                "The names of the files of the sar_image, the front and the zones don't fit together."

            # ####################################################################################################
            # Check that all files are pngs
            # ####################################################################################################
            assert str(sar_image)[-4:] == ".png" or str(sar_image)[-4:] == ".PNG", "Sar Image is not a PNG."
            assert str(front)[-4:] == ".png" or str(front)[-4:] == ".PNG", "Front Image is not a PNG."
            assert str(zone)[-4:] == ".png" or str(zone)[-4:] == ".PNG", "Zone Image is not a PNG."

            sar_image_array = cv2.imread(os.path.join(parent_dir, "data_raw", "sar_images", data_split_dir, sar_image).__str__(), cv2.IMREAD_GRAYSCALE)
            front_array = cv2.imread(os.path.join(parent_dir, "data_raw", "fronts", data_split_dir, front).__str__(), cv2.IMREAD_GRAYSCALE)
            zone_array = cv2.imread(os.path.join(parent_dir, "data_raw", "zones", data_split_dir, zone).__str__(), cv2.IMREAD_GRAYSCALE)

            # ####################################################################################################
            # Check whether masks have correct values
            # ####################################################################################################
            assert np.array_equiv(np.unique(front_array), [0, 255]), "Front Mask does not have the correct values (should have 0 and 255), file:" + str(front)
            assert is_subarray(np.unique(zone_array), [0, 64, 127, 254]), \
                "Zones Mask does not have the correct values (should have 0, 64, 127, 254 or a subset of this), file:" + str(zone)

            # ####################################################################################################
            # Check whether masks and image have the same size
            # ####################################################################################################
            assert sar_image_array.ndim == 2 and zone_array.ndim == 2 and front_array.ndim == 2, \
                "The image and masks should be two dimensional, but they are not."
            assert len(sar_image_array) == len(front_array) and len(sar_image_array) == len(zone_array), \
                "The height of image and masks don't fit together."
            assert len(sar_image_array[0]) == len(front_array[0]) and len(sar_image_array[0]) == len(zone_array[0]), \
                "The width of image and masks don't fit together."

    print("All is looking good!")
