import cv2.cv2
import pandas as pd
import numpy as np
import cv2
import glob
from skimage.registration import phase_cross_correlation
from skimage.io import imread
import skimage.io as io
import matplotlib.pyplot as plt
from PIL import Image

path_to_files = 'C:/Users/Karola/Desktop/ZTDT_termowizja/'
visible_lst = []
thermal_list = []
reference_point_coordintes = []
dim = 0


def click_event(event, x, y, flags, params):
    """Display and save coordinates for references points """
    # checking for left mouse clicks
    if event == cv2.EVENT_LBUTTONDOWN:
        # displaying the coordinates
        # on the Shell
        print(x, ' ', y)
        reference_point_coordintes.append([x, y])

        # displaying the coordinates
        # on the image window
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(first_image, str(x) + ',' +
                    str(y), (x, y), font,
                    1, (255, 0, 0), 2)
        cv2.imshow('image', first_image)


def load_vis_images(path):
    """Loads images registered in visual light via classic digital camera"""
    path += 'VIS/*.JPG'
    for img in glob.glob(path):
        image = cv2.imread(img, 1)
        dim = int(image.shape[0] / 4)
        dim2 = int(image.shape[1] / 4)
        resized_image = cv2.resize(image, (dim2, dim))

        visible_lst.append(resized_image)

    return visible_lst, dim, dim2


def load_thermal_images(path, dim, dim2):
    """Loads images registered via thermal camera"""
    path += 'THERMO/*.PNG'
    for img in glob.glob(path):
        image = cv2.imread(img, 1)
        resized_image = cv2.resize(image, (dim, dim2))

        thermal_list.append(resized_image)
    return thermal_list


if __name__ == '__main__':
    print("Loading images ....")
    *x, first_dim, second_dim = load_vis_images(path_to_files)
    load_thermal_images(path_to_files, second_dim, first_dim)

    for index in range(len(visible_lst)):
        if index == 0:
            print("Visible ref image coordinates")
            first_image = visible_lst[index]
            cv2.imshow('image', first_image)
            cv2.setMouseCallback('image', click_event)
            cv2.waitKey(0)

            print("Thermal ref image coordinates ....")
            first_image = thermal_list[index]
            cv2.imshow('image', first_image)
            cv2.setMouseCallback('image', click_event)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    vis = reference_point_coordintes[0:4]
    termo = reference_point_coordintes[4:9]

    sift = cv2.xfeatures2d.SIFT_create()

    keypoints_1, descriptors_1 = sift.detectAndCompute(visible_lst[0], None)
    keypoints_2, descriptors_2 = sift.detectAndCompute(thermal_list[0], None)

    # feature matching
    bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)

    matches = bf.match(descriptors_1, descriptors_2)
    matches = sorted(matches, key=lambda x: x.distance)

    img3 = cv2.drawMatches(visible_lst[0], keypoints_1, thermal_list[0], keypoints_2, matches[:50], thermal_list[0], flags=2)
    plt.imshow(img3), plt.show()
