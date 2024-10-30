import cv2
import numpy as np
import os

PATH = '/home/aroychoudhury/Stanford_thyroid/thyroidultrasoundcineclip/activation_pool_combined/'
MASK_PATH = '/home/aroychoudhury/Stanford_thyroid/thyroidultrasoundcineclip/masks_new/'
total_important = 0
total_mask = 0
total_intersection = 0
files = os.listdir(PATH)
for file in files:
    activation = cv2.resize(cv2.imread(PATH + file), (64,64))
    mask = cv2.resize(cv2.imread(MASK_PATH + file), (64, 64))
    activation[activation <= 127] = 0
    activation[activation > 127] = 1
    mask_nz = np.count_nonzero(mask)
    total_mask += mask_nz
    important = np.count_nonzero(activation)
    mask[activation == 0] = 0
    intersection = np.count_nonzero(mask)
    total_intersection += intersection
    total_important += important
    #print(file, important, intersection)


print(total_important, total_intersection, total_mask, total_intersection/total_important, total_intersection/total_mask)
