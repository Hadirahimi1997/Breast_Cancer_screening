import random

from PIL import Image
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
import scipy.signal
import math


def checkLRFlip(roiname):
        namelist = os.listdir('/..../trainingmammo')
        folder1 = '/..../trainingmammo'
        for name in namelist:
            if (name[0:29]) == (roiname[0:29]):
            # Get number of rows and columns in the image.
                img = Image.open(os.path.join(folder1, name))
                img = np.asarray(img)
                nrows, ncols = img.shape
                x_center = ncols // 2
                y_center = nrows // 2

                # Sum down each column.
                col_sum = img.sum(axis=0)
                # Sum across each row.
                row_sum = img.sum(axis=1)

                left_sum = sum(col_sum[0:x_center])
                right_sum = sum(col_sum[x_center:-1])

                if left_sum < right_sum:
                    LR_flip = True
                else:
                    LR_flip = False

                return LR_flip







folder = os.listdir('/..../trainingmask')
folder1 = '/..../trainingmask'
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


    #Horizontal flip(convert all images direction left to right)
    lr_flip = checkLRFlip(name)
    if lr_flip:
        f_img = np.fliplr(img)
    elif not lr_flip:
        f_img = img

    #save
    path = '/..../dest'
    filenamee = name
    f_img = Image.fromarray(f_img)

    # Save the image with the new path and filename
    f_img.save(path + filenamee)

