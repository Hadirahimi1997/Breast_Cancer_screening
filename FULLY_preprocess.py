import random

from PIL import Image
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
import scipy.signal
import math


def checkLRFlip(image):

    # Get number of rows and columns in the image.
    nrows, ncols = image.shape
    x_center = ncols // 2
    y_center = nrows // 2

    # Sum down each column.
    col_sum = image.sum(axis=0)
    # Sum across each row.
    row_sum = image.sum(axis=1)

    left_sum = sum(col_sum[0:x_center])
    right_sum = sum(col_sum[x_center:-1])

    if left_sum < right_sum:
        LR_flip = True
    else:
        LR_flip = False

    return LR_flip






folder = os.listdir('/home/hadi/niloofar.shabani/datan/fully_training')
folder1 = '/home/hadi/niloofar.shabani/datan/fully_training/'
for name in (folder):
    file_name= name
    img = Image.open(os.path.join(folder1, file_name))
    img = np.asarray(img)

    # resize images for crop borders from sides
    l = 0.01
    r = 0.01
    u = 0.06
    d = 0.06

    nrows, ncols = img.shape

    # Get the start and end rows and columns
    l_crop = int(ncols * l)
    r_crop = int(ncols * (1 - r))
    u_crop = int(nrows * u)
    d_crop = int(nrows * (1 - d))
    img = img[u_crop:d_crop, l_crop:r_crop]

    #binarised image
    thresh = 0.2
    maxval = 255
    imge = np.zeros(img.shape, np.uint8)
    imge[img >= thresh] = maxval

    #we expand the boundaries of the white contours in the mask with morphology. This ensures that we really capture the entire region of any artefacts

    ksize = (23, 23)
    operation = "open"
    kernel = cv2.getStructuringElement(shape=cv2.MORPH_RECT, ksize=ksize)

    if operation == "open":
        imge = cv2.morphologyEx(imge, cv2.MORPH_OPEN, kernel)
    elif operation == "close":
        imge = cv2.morphologyEx(imge, cv2.MORPH_CLOSE, kernel)

    # Then dilate
    imge = cv2.morphologyEx(imge, cv2.MORPH_DILATE, kernel)

    #find biggest contour(breast area) then convert other countors to black(background)
    contours, hierarchy = cv2.findContours(imge,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

    # Keep only the largest contour (object) in the foreground
    max_contour = max(contours, key=cv2.contourArea)

    # Convert all other objects of foreground to black
    for contour in contours:
        if contour is not max_contour:
            cv2.drawContours(imge, [contour], 0, (0, 0, 0), -1)

    imgg = img.copy()
    imgg[imge == 0] = 0

    #calhe filter for image enhancment
    #clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    #mgg = clahe.apply(imgg)

    #Horizontal flip(convert all images direction left to right)
    lr_flip = checkLRFlip(imgg)
    if lr_flip:
        f_img = np.fliplr(imgg)
    elif not lr_flip:
        f_img = imgg

    #save
    path = '/home/hadi/niloofar.shabani/datan/preprocessed fully/'
    filenamee = name
    f_img = Image.fromarray(f_img)

    # Save the image with the new path and filename
    f_img.save(path + filenamee)

