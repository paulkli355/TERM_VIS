import cv2.cv2
import imreg_dft as imreg_dft
import kwargs as kwargs
import pandas as pd
import numpy as np
import cv2
import glob
import imreg as imreg
import imreg_dft as imreg_dft
import scipy
from skimage.registration import phase_cross_correlation
from skimage.io import imread
import skimage.io as io
import matplotlib.pyplot as plt
from PIL import Image as im
import os

path_to_files = 'C:/Users/Karola/Desktop/Pythong/'
visible_lst = []
thermal_list = []
reference_point_coordintes = []

dim = 0
dim2 = 0


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
        cv2.circle(first_image, (x, y), radius=5, color=(0, 0, 255), thickness=-1)
        cv2.imshow('image', first_image)


def load_vis_images(path):
    """Loads images registered in visual light via classic digital camera"""
    path += 'VIS43/*.JPG'
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

    vis_mask = np.zeros((visible_lst[0].shape[0], visible_lst[0].shape[1]))
    cv2.circle(vis_mask, (vis[0][0], vis[0][1]), radius=5, color=(255, 255, 255), thickness=-1)
    cv2.circle(vis_mask, (vis[1][0], vis[1][1]), radius=5, color=(255, 255, 255), thickness=-1)
    cv2.circle(vis_mask, (vis[2][0], vis[2][1]), radius=5, color=(255, 255, 255), thickness=-1)
    cv2.circle(vis_mask, (vis[3][0], vis[3][1]), radius=5, color=(255, 255, 255), thickness=-1)

    termo_mask = np.zeros((thermal_list[0].shape[0], thermal_list[0].shape[1]))
    cv2.circle(termo_mask, (termo[0][0], termo[0][1]), radius=5, color=(255, 255, 255), thickness=-1)
    cv2.circle(termo_mask, (termo[1][0], termo[1][1]), radius=5, color=(255, 255, 255), thickness=-1)
    cv2.circle(termo_mask, (termo[2][0], termo[2][1]), radius=5, color=(255, 255, 255), thickness=-1)
    cv2.circle(termo_mask, (termo[3][0], termo[3][1]), radius=5, color=(255, 255, 255), thickness=-1)

    cv2.imshow('image', vis_mask)
    cv2.waitKey(0)
    cv2.imshow('image', termo_mask)
    cv2.waitKey(0)

    MAX_FEATURES = 500
    GOOD_MATCH_PERCENT = 0.15
    #from config import folder_path_aligned_images


    def match_img(im1, im2):
        # Convert images to grayscale
        im1_gray = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
        im2_gray = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)

        # Detect ORB features and compute descriptors.
        orb = cv2.ORB_create(MAX_FEATURES)
        keypoints_1, descriptors_1 = orb.detectAndCompute(im1_gray, None)
        keypoints_2, descriptors_2 = orb.detectAndCompute(im2_gray, None)

        # Match features.
        matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
        matches = matcher.match(descriptors_1, descriptors_2, None)

        # Sort matches by score
        matches_dict = {}
        #matches_dict.keys() = [matches]
        #matches_dict.values() = list(matches).apply(lambda x: x.distance)
        #matches_dist = list(map(lambda x: x.distance, matches))
        #matches = sorted(matches, key=matches_dist, reverse=True)
        matches = sorted(matches, key=lambda x: x.distance, reverse=False)

        # Remove not so good matches
        num_good_matches = int(len(matches) * GOOD_MATCH_PERCENT)
        matches = matches[:num_good_matches]

        # Draw top matches
        im_matches = cv2.drawMatches(im1, keypoints_1, im2, keypoints_2, matches, None)
        cv2.imwrite(os.path.join("C:/Users/Karola/Desktop/Pythong/NEW/02.jpg"), im_matches)

        # Extract location of good matches
        points_1 = np.zeros((len(matches), 2), dtype=np.float32)
        points_2 = np.zeros((len(matches), 2), dtype=np.float32)

        for i, match in enumerate(matches):
            points_1[i, :] = keypoints_1[match.queryIdx].pt
            points_2[i, :] = keypoints_2[match.trainIdx].pt

        # Find homography
        h, mask = cv2.findHomography(points_1, points_2, cv2.RANSAC)

        # Use homography
        height, width, channels = im2.shape
        im1_reg = cv2.warpPerspective(im1, h, (width, height))

        return im1_reg, h
    #@classmethod
    def align_img(template_path, raw_img_path, result_img_path):
        # Read reference image
        ref_filename = template_path
        print("Reading reference image: ", ref_filename)
        im_reference = cv2.imread(ref_filename, cv2.IMREAD_COLOR)

        # Read image to be aligned
        im_filename = raw_img_path
        print("Reading image to align: ", im_filename)
        im = cv2.imread(raw_img_path, cv2.IMREAD_COLOR)

        print("Aligning images ...")
        # Registered image will be resorted in im_reg.
        im_reg, h = match_img(im, im_reference)

        # Write aligned image to disk.
        print("Saving aligned image : ", result_img_path)
        cv2.imwrite(result_img_path, im_reg)

        return result_img_path


    x = align_img("C:/Users/Karola/Desktop/Pythong/VIS43/02.jpg", "C:/Users/Karola/Desktop/Pythong/TER/02.jpg", "C:/Users/Karola/Desktop/Pythong/NEW/02.jpg")
    #im2, scale, angle, (t0, t1) = imreg.similarity(vis_mask, termo_mask)
    #imreg_dft.imreg.similarity(vis_mask, termo_mask, numiter=1, order=3, constraints=None, filter_pcorr=0, exponent='inf',
    #                           reports=None)
    #x = imreg_dft.imreg.transform_img(vis_mask, scale=1.0, angle=0.0, tvec=(0, 0), mode='constant', bgval=None, order=1)

    #imreg_dft.imreg.imshow(vis_mask, termo_mask, x, cmap=None, fig=None, **kwargs)

'''
    cv2.imshow('image', vis_mask)
    cv2.waitKey(0)

    cv2.imshow('image', termo_mask)
    cv2.waitKey(0)

sift = cv2.xfeatures2d.SIFT_create()

image8bit = cv2.normalize(vis_mask, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')
image8bit2 = cv2.normalize(termo_mask, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')

keypoints_1, descriptors_1 = sift.detectAndCompute(image8bit, None)
keypoints_2, descriptors_2 = sift.detectAndCompute(image8bit2, None)

# feature matching
bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)

matches = bf.match(descriptors_1, descriptors_2)
matches = sorted(matches, key=lambda x: x.distance)

img3 = cv2.drawMatches(image8bit, keypoints_1, image8bit2, keypoints_2, matches[:50], image8bit2,
                       flags=2)
plt.imshow(img3), plt.show()
'''
