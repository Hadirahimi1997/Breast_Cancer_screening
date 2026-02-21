################################################################################
                           #imports#
################################################################################
from PIL import Image
import numpy as np
import scipy.signal
import math
import csv
import os
import numpy as np
import cv2
from tqdm import tqdm

################################################################################
# extracting patches#
################################################################################
# detecting the central point (x,y) of the abnormal tissue (ROI)
def centeral_point(full_mg,ROI):
    img1 = np.asarray(full_mg)
    img2 = np.asarray(ROI)
    (n, m) = img1.shape
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    edges = scipy.signal.convolve2d(img2, kernel, 'valid')
    indices = np.where(edges == [255])
    lenght = (len(indices[0]))
    coordinates = zip(indices[1], indices[0])
    sumx = 0
    sumy = 0
    for point in (coordinates):
        (x, y) = point
        sumx = sumx + x
        sumy = sumy + y
    (cen_x, cen_y) = (math.ceil(sumx / lenght), math.ceil(sumy / lenght))
    return (indices, (cen_x, cen_y))

###################################
#extracting 224*224 patch from the center of ROI.
def centerpatch(center, full_mg):
    c_abnormal_patch = []
    x = center[0]
    y = center[1]
    crop_rectangle = (x - 112, y - 112, x + 112, y + 112)
    cropped_im = full_mg.crop(crop_rectangle)
    cropped_im = np.asarray(cropped_im)
    c_abnormal_patch.append(cropped_im)
    return (c_abnormal_patch)

###################################
#extracting random patches from ROI
def random_abnormalpatch(center,edge, full_mg):
    r_abnormal_patch = []
    for point in (edge):
        cols = edge[1]
        rows = edge[0]

    def posy():
        """
        if abs(max(cols) - min(cols)) > abs(max(rows) - min(rows)) * 1.70 or abs(max(rows) - min(rows)) > abs(max(cols) - min(cols)) * 1.70:
        """
        rand_index = np.random.randint(len(rows))

        if points != (center[0], center[1]):
            points.append((cols[rand_index], rows[rand_index]))
        """
        else:
            point1 = (center[0] + 112, center[1])
            point2 = (center[0], center[1] + 112)
            point3 = (center[0] - 112, center[1])
            point4 = (center[0], center[1] - 112)
            points.append(point1)
            points.append(point2)
            points.append(point3)
            points.append(point4)
        """

    # Choose 4 random points from ROI pixels in breast region
    points = []
    while len(points) < 7:
        posy()

    for k in range(len(points)):
        (h, d) = (points[k])
        crop_point = (h - 112, d - 112, h + 112, d + 112)
        croped_point = full_mg.crop(crop_point)
        croped_point = np.asarray(croped_point)
        r_abnormal_patch.append(croped_point)
    return (r_abnormal_patch)

###################################
#Extracting patches from normal tissues (non-ROI) 

def random_normalpatch(helplist,full_mg):
    folder1 = '/..../MASK/'
    folder2 = '/..../MAMMO/'
    mammo_img = cv2.imread(os.path.join(folder2, full_mg),cv2.IMREAD_GRAYSCALE)
    images = []
    for roi in helplist:
        roi = cv2.imread(os.path.join(folder1, roi))
        images.append(roi)
    mask_combined = np.zeros_like(images[0])
    for i in range(len(images)):
        mask_combined[images[i] > 0] = 255

    mask_combined = cv2.cvtColor(mask_combined, cv2.COLOR_BGR2GRAY)
    # Invert mask image (ROI is white)
    mask_inv = cv2.bitwise_not(mask_combined)

    breast_region = cv2.bitwise_and(mammo_img, mammo_img, mask=mask_inv)

    _, mam_img = cv2.threshold(breast_region, 10, 255, cv2.THRESH_BINARY)

    # Find non-zero pixels in breast region

    rows,cols = np.nonzero(mam_img)
    # Choose 15 random points from non-ROI pixels in breast region
    points = []
    while len(points) < 3:
        rand_index = np.random.randint(len(rows))
        pixel_val = mask_combined[rows[rand_index], cols[rand_index]]
        if pixel_val != 255:
            points.append((cols[rand_index], rows[rand_index]))
    mammo_img = Image.fromarray(mammo_img)
    r_normal_patch = []
    for k in range(len(points)):
        (h, d) = (points[k])
        crop_point = (h - 112, d - 112, h + 112, d + 112)
        croped_point = mammo_img.crop(crop_point)
        croped_point = np.asarray(croped_point)
        r_normal_patch.append(croped_point)
    return(r_normal_patch)


###################################
###################################
###################################
#extract patches based on modality (lcc, lmlo, rcc, rmlo)
# LABELS ARE BASED ON Normal and Abnormal(Benign Calc, Benign Mass, Malignant Calc, Malignant Mass)
#5 classes at all
def find_all_patches_in_directory():
    label_c_lcc=[]
    label_c_lmlo=[]
    label_c_rcc=[]
    label_c_rmlo=[]

    label_a_lcc=[]
    label_a_lmlo=[]
    label_a_rcc=[]
    label_a_rmlo=[]

    label_n_lcc=[]
    label_n_lmlo=[]
    label_n_rcc=[]
    label_n_rmlo=[]

    lcc_center_abnormal=[]
    lcc_abnormal_patch=[]
    lcc_normal_patch=[]

    lmlo_center_abnormal=[]
    lmlo_abnormal_patch=[]
    lmlo_normal_patch=[]

    rcc_center_abnormal=[]
    rcc_abnormal_patch=[]
    rcc_normal_patch=[]

    rmlo_center_abnormal=[]
    rmlo_abnormal_patch=[]
    rmlo_normal_patch=[]

    namelist = os.listdir('/..../MAMMO/')
    namelist = sorted(namelist, key=lambda namelist: int(namelist.split('_')[2]))
    #namelist = [name.split("_")[2] for name in namelist]
    roiname_list = os.listdir('/..../MASK/')
    f = open('/..../description.csv')
    csv_f = csv.reader(f)
    temp = []
    temp2 = []
    temp3 = []
    for row in (csv_f):
        temp.append(row[0])
        temp2.append(row[14])
        temp3.append(row[2])
    for name in tqdm(namelist):
        print(name)
        helplist = []
        helplist2 = []
        helplist3 = []
        c = []
        n = []
        a = []
        for roiname in roiname_list:
            if (name[0:29]) == (roiname[0:29]):
                helplist.append(roiname)
        for obj in helplist:
            obj = obj.split('.')[0]
            for i in range(len(temp)):
                a = temp[i].split('/')[0]
                if a in obj:
                    helplist2.append(temp2[i])
                    helplist3.append(temp3[i])
        n = random_normalpatch(helplist,name)

        if 'LEFT_CC' in name:
            lcc_normal_patch.append(n)
            for i in range(3):
                label_n_lcc.append(0)


        elif 'LEFT_MLO' in name:
            lmlo_normal_patch.append(n)
            for i in range(3):
                label_n_lmlo.append(0)


        elif 'RIGHT_CC' in name:
            rcc_normal_patch.append(n)
            for i in range(3):
                label_n_rcc.append(0)


        elif 'RIGHT_MLO' in name:
            rmlo_normal_patch.append(n)
            for i in range(3):
                label_n_rmlo.append(0)


        if (len(helplist)) > 1:
            for i in range(len(helplist)):
                folder1 = '/..../MAMMO/'
                filename1 = name
                full_mg = Image.open(os.path.join(folder1, filename1))
                folder2 = '/..../MASK/'
                filename2 = helplist[i]
                ROI = Image.open(os.path.join(folder2, filename2))
                (edge, center) = centeral_point(full_mg, ROI)

                c = centerpatch(center, full_mg)
                x = helplist2[i]
                z = helplist3[i]
                if 'LEFT_CC' in name:
                    lcc_center_abnormal.append(c)
                    if x[0:1] == 'B' and z[0:1] == 'c':
                        label_c_lcc.append(1)
                    elif x[0:1] == 'M' and z[0:1] == 'c':
                        label_c_lcc.append(2)
                    elif x[0:1] == 'B' and z[0:1] == 'm':
                        label_c_lcc.append(3)
                    elif x[0:1] == 'M' and z[0:1] == 'm':
                        label_c_lcc.append(4)
                elif 'LEFT_MLO' in name:
                    lmlo_center_abnormal.append(c)
                    if x[0:1] == 'B' and z[0:1] == 'c':
                        label_c_lmlo.append(1)
                    elif x[0:1] == 'M' and z[0:1] == 'c':
                        label_c_lmlo.append(2)
                    elif x[0:1] == 'B' and z[0:1] == 'm':
                        label_c_lmlo.append(3)
                    elif x[0:1] == 'M' and z[0:1] == 'm':
                        label_c_lmlo.append(4)
                elif 'RIGHT_CC' in name:
                    rcc_center_abnormal.append(c)
                    if x[0:1] == 'B' and z[0:1] == 'c':
                        label_c_rcc.append(1)
                    elif x[0:1] == 'M' and z[0:1] == 'c':
                        label_c_rcc.append(2)
                    elif x[0:1] == 'B' and z[0:1] == 'm':
                        label_c_rcc.append(3)
                    elif x[0:1] == 'M' and z[0:1] == 'm':
                        label_c_rcc.append(4)
                elif 'RIGHT_MLO' in name:
                    rmlo_center_abnormal.append(c)
                    if x[0:1] == 'B' and z[0:1] == 'c':
                        label_c_rmlo.append(1)
                    elif x[0:1] == 'M' and z[0:1] == 'c':
                        label_c_rmlo.append(2)
                    elif x[0:1] == 'B' and z[0:1] == 'm':
                        label_c_rmlo.append(3)
                    elif x[0:1] == 'M' and z[0:1] == 'm':
                        label_c_rmlo.append(4)

                a = random_abnormalpatch(center,edge, full_mg)
                if 'LEFT_CC' in name:
                    lcc_abnormal_patch.append(a)
                    if x[0:1] == 'B' and z[0:1] == 'c':
                        for r in range(7):
                            label_a_lcc.append(1)
                    elif x[0:1] == 'M' and z[0:1] == 'c':
                        for r in range(7):
                            label_a_lcc.append(2)
                    elif x[0:1] == 'B' and z[0:1] == 'm':
                        for r in range(7):
                            label_a_lcc.append(3)
                    elif x[0:1] == 'M' and z[0:1] == 'm':
                        for r in range(7):
                            label_a_lcc.append(4)
                elif 'LEFT_MLO' in name:
                    lmlo_abnormal_patch.append(a)
                    if x[0:1] == 'B' and z[0:1] == 'c':
                        for r in range(7):
                            label_a_lmlo.append(1)
                    elif x[0:1] == 'M' and z[0:1] == 'c':
                        for r in range(7):
                            label_a_lmlo.append(2)
                    elif x[0:1] == 'B' and z[0:1] == 'm':
                        for r in range(7):
                            label_a_lmlo.append(3)
                    elif x[0:1] == 'M' and z[0:1] == 'm':
                        for r in range(7):
                            label_a_lmlo.append(4)
                elif 'RIGHT_CC' in name:
                    rcc_abnormal_patch.append(a)
                    if x[0:1] == 'B' and z[0:1] == 'c':
                        for r in range(7):
                            label_a_rcc.append(1)
                    elif x[0:1] == 'M' and z[0:1] == 'c':
                        for r in range(7):
                            label_a_rcc.append(2)
                    elif x[0:1] == 'B' and z[0:1] == 'm':
                        for r in range(7):
                            label_a_rcc.append(3)
                    elif x[0:1] == 'M' and z[0:1] == 'm':
                        for r in range(7):
                            label_a_rcc.append(4)
                elif 'RIGHT_MLO' in name:
                    rmlo_abnormal_patch.append(a)
                    if x[0:1] == 'B' and z[0:1] == 'c':
                        for r in range(7):
                            label_a_rmlo.append(1)
                    elif x[0:1] == 'M' and z[0:1] == 'c':
                        for r in range(7):
                            label_a_rmlo.append(2)
                    elif x[0:1] == 'B' and z[0:1] == 'm':
                        for r in range(7):
                            label_a_rmlo.append(3)
                    elif x[0:1] == 'M' and z[0:1] == 'm':
                        for r in range(7):
                            label_a_rmlo.append(4)




        elif (len(helplist)) == 1:
            folder1 = '/..../MAMMO/'
            filename1 = name
            full_mg = Image.open(os.path.join(folder1, filename1))
            folder2 = '/..../MASK/'
            filename2 = helplist[0]
            ROI = Image.open(os.path.join(folder2, filename2))
            (edge, center) = centeral_point(full_mg, ROI)
            img2 = np.asarray(ROI)
            (width, length) = img2.shape
            c = centerpatch(center, full_mg)
            x = helplist2[0]
            z = helplist3[0]
            if 'LEFT_CC' in name:
                lcc_center_abnormal.append(c)
                if x[0:1] == 'B' and z[0:1] == 'c':
                    label_c_lcc.append(1)
                elif x[0:1] == 'M' and z[0:1] == 'c':
                    label_c_lcc.append(2)
                elif x[0:1] == 'B' and z[0:1] == 'm':
                    label_c_lcc.append(3)
                elif x[0:1] == 'M' and z[0:1] == 'm':
                    label_c_lcc.append(4)
            elif 'LEFT_MLO' in name:
                lmlo_center_abnormal.append(c)
                if x[0:1] == 'B' and z[0:1] == 'c':
                    label_c_lmlo.append(1)
                elif x[0:1] == 'M' and z[0:1] == 'c':
                    label_c_lmlo.append(2)
                elif x[0:1] == 'B' and z[0:1] == 'm':
                    label_c_lmlo.append(3)
                elif x[0:1] == 'M' and z[0:1] == 'm':
                    label_c_lmlo.append(4)
            elif 'RIGHT_CC' in name:
                rcc_center_abnormal.append(c)
                if x[0:1] == 'B' and z[0:1] == 'c':
                    label_c_rcc.append(1)
                elif x[0:1] == 'M' and z[0:1] == 'c':
                    label_c_rcc.append(2)
                elif x[0:1] == 'B' and z[0:1] == 'm':
                    label_c_rcc.append(3)
                elif x[0:1] == 'M' and z[0:1] == 'm':
                    label_c_rcc.append(4)
            elif 'RIGHT_MLO' in name:
                rmlo_center_abnormal.append(c)
                if x[0:1] == 'B' and z[0:1] == 'c':
                    label_c_rmlo.append(1)
                elif x[0:1] == 'M' and z[0:1] == 'c':
                    label_c_rmlo.append(2)
                elif x[0:1] == 'B' and z[0:1] == 'm':
                    label_c_rmlo.append(3)
                elif x[0:1] == 'M' and z[0:1] == 'm':
                    label_c_rmlo.append(4)

            a = random_abnormalpatch(center,edge, full_mg)
            if 'LEFT_CC' in name:
                lcc_abnormal_patch.append(a)
                if x[0:1] == 'B' and z[0:1] == 'c':
                    for r in range(7):
                        label_a_lcc.append(1)
                elif x[0:1] == 'M' and z[0:1] == 'c':
                    for r in range(7):
                        label_a_lcc.append(2)
                elif x[0:1] == 'B' and z[0:1] == 'm':
                    for r in range(7):
                        label_a_lcc.append(3)
                elif x[0:1] == 'M' and z[0:1] == 'm':
                    for r in range(7):
                        label_a_lcc.append(4)
            elif 'LEFT_MLO' in name:
                lmlo_abnormal_patch.append(a)
                if x[0:1] == 'B' and z[0:1] == 'c':
                    for r in range(7):
                        label_a_lmlo.append(1)
                elif x[0:1] == 'M' and z[0:1] == 'c':
                    for r in range(7):
                        label_a_lmlo.append(2)
                elif x[0:1] == 'B' and z[0:1] == 'm':
                    for r in range(7):
                        label_a_lmlo.append(3)
                elif x[0:1] == 'M' and z[0:1] == 'm':
                    for r in range(7):
                        label_a_lmlo.append(4)
            elif 'RIGHT_CC' in name:
                rcc_abnormal_patch.append(a)
                if x[0:1] == 'B' and z[0:1] == 'c':
                    for r in range(7):
                        label_a_rcc.append(1)
                elif x[0:1] == 'M' and z[0:1] == 'c':
                    for r in range(7):
                        label_a_rcc.append(2)
                elif x[0:1] == 'B' and z[0:1] == 'm':
                    for r in range(7):
                        label_a_rcc.append(3)
                elif x[0:1] == 'M' and z[0:1] == 'm':
                    for r in range(7):
                        label_a_rcc.append(4)
            elif 'RIGHT_MLO' in name:
                rmlo_abnormal_patch.append(a)
                if x[0:1] == 'B' and z[0:1] == 'c':
                    for r in range(7):
                        label_a_rmlo.append(1)
                elif x[0:1] == 'M' and z[0:1] == 'c':
                    for r in range(7):
                        label_a_rmlo.append(2)
                elif x[0:1] == 'B' and z[0:1] == 'm':
                    for r in range(7):
                        label_a_rmlo.append(3)
                elif x[0:1] == 'M' and z[0:1] == 'm':
                    for r in range(7):
                        label_a_rmlo.append(4)



    return (lcc_center_abnormal,lmlo_center_abnormal,rcc_center_abnormal,rmlo_center_abnormal,lcc_abnormal_patch,lmlo_abnormal_patch,rcc_abnormal_patch,rmlo_abnormal_patch,lcc_normal_patch,lmlo_normal_patch,rcc_normal_patch,rmlo_normal_patch,label_c_lcc,label_c_lmlo,label_c_rcc,label_c_rmlo,label_a_lcc,label_a_lmlo,label_a_rcc,label_a_rmlo,label_n_lcc,label_n_lmlo,label_n_rcc,label_n_rmlo)


(lcc_center_abnormal,lmlo_center_abnormal,rcc_center_abnormal,rmlo_center_abnormal,lcc_abnormal_patch,lmlo_abnormal_patch,rcc_abnormal_patch,rmlo_abnormal_patch,lcc_normal_patch,lmlo_normal_patch,rcc_normal_patch,rmlo_normal_patch,label_c_lcc,label_c_lmlo,label_c_rcc,label_c_rmlo,label_a_lcc,label_a_lmlo,label_a_rcc,label_a_rmlo,label_n_lcc,label_n_lmlo,label_n_rcc,label_n_rmlo) = find_all_patches_in_directory()


###################################
###################################
###################################
def general(lcc_center_abnormal,lmlo_center_abnormal,rcc_center_abnormal,rmlo_center_abnormal,lcc_abnormal_patch,lmlo_abnormal_patch,rcc_abnormal_patch,rmlo_abnormal_patch,lcc_normal_patch,lmlo_normal_patch,rcc_normal_patch,rmlo_normal_patch,label_c_lcc,label_c_lmlo,label_c_rcc,label_c_rmlo,label_a_lcc,label_a_lmlo,label_a_rcc,label_a_rmlo,label_n_lcc,label_n_lmlo,label_n_rcc,label_n_rmlo):
    patch_left_cc = []
    label_left_cc = []

    for i in range(len(lcc_center_abnormal)):
        patch_left_cc.append(lcc_center_abnormal[i][0])
    for i in range(len(label_c_lcc)):
        label_left_cc.append(label_c_lcc[i])
    for i in range(len(lcc_abnormal_patch)):
        for j in range(7):
            patch_left_cc.append(lcc_abnormal_patch[i][j])
    for i in range(len(label_a_lcc)):
        label_left_cc.append(label_a_lcc[i])
    for i in range(len(lcc_normal_patch)):
        for j in range(3):
            patch_left_cc.append(lcc_normal_patch[i][j])
    for i in range(len(label_n_lcc)):
        label_left_cc.append(label_n_lcc[i])

    patch_right_cc = []
    label_right_cc = []

    for i in range(len(rcc_center_abnormal)):
        patch_right_cc.append(rcc_center_abnormal[i][0])
    for i in range(len(label_c_rcc)):
        label_right_cc.append(label_c_rcc[i])
    for i in range(len(rcc_abnormal_patch)):
        for j in range(7):
            patch_right_cc.append(rcc_abnormal_patch[i][j])
    for i in range(len(label_a_rcc)):
        label_right_cc.append(label_a_rcc[i])
    for i in range(len(rcc_normal_patch)):
        for j in range(3):
            patch_right_cc.append(rcc_normal_patch[i][j])
    for i in range(len(label_n_rcc)):
        label_right_cc.append(label_n_rcc[i])

    patch_left_mlo= []
    label_left_mlo= []

    for i in range(len(lmlo_center_abnormal)):
        patch_left_mlo.append(lmlo_center_abnormal[i][0])
    for i in range(len(label_c_lmlo)):
        label_left_mlo.append(label_c_lmlo[i])
    for i in range(len(lmlo_abnormal_patch)):
        for j in range(7):
            patch_left_mlo.append(lmlo_abnormal_patch[i][j])
    for i in range(len(label_a_lmlo)):
        label_left_mlo.append(label_a_lmlo[i])
    for i in range(len(lmlo_normal_patch)):
        for j in range(3):
            patch_left_mlo.append(lmlo_normal_patch[i][j])
    for i in range(len(label_n_lmlo)):
        label_left_mlo.append(label_n_lmlo[i])


    patch_right_mlo= []
    label_right_mlo= []

    for i in range(len(rmlo_center_abnormal)):
        patch_right_mlo.append(rmlo_center_abnormal[i][0])
    for i in range(len(label_c_rmlo)):
        label_right_mlo.append(label_c_rmlo[i])
    for i in range(len(rmlo_abnormal_patch)):
        for j in range(7):
            patch_right_mlo.append(rmlo_abnormal_patch[i][j])
    for i in range(len(label_a_rmlo)):
        label_right_mlo.append(label_a_rmlo[i])
    for i in range(len(rmlo_normal_patch)):
        for j in range(3):
            patch_right_mlo.append(rmlo_normal_patch[i][j])
    for i in range(len(label_n_rmlo)):
        label_right_mlo.append(label_n_rmlo[i])

    return (patch_left_cc,patch_left_mlo,patch_right_cc,patch_right_mlo,label_left_cc,label_left_mlo,label_right_cc,label_right_mlo)


(patch_left_cc,patch_left_mlo,patch_right_cc,patch_right_mlo,label_left_cc,label_left_mlo,label_right_cc,label_right_mlo) = general(lcc_center_abnormal,lmlo_center_abnormal,rcc_center_abnormal,rmlo_center_abnormal,lcc_abnormal_patch,lmlo_abnormal_patch,rcc_abnormal_patch,rmlo_abnormal_patch,lcc_normal_patch,lmlo_normal_patch,rcc_normal_patch,rmlo_normal_patch,label_c_lcc,label_c_lmlo,label_c_rcc,label_c_rmlo,label_a_lcc,label_a_lmlo,label_a_rcc,label_a_rmlo,label_n_lcc,label_n_lmlo,label_n_rcc,label_n_rmlo)

############################################################################################################
##################################################################################
########################################################



patch_left_cc=np.array(patch_left_cc)
patch_right_cc=np.array(patch_right_cc)
patch_left_mlo=np.array(patch_left_mlo)
patch_right_mlo=np.array(patch_right_mlo)


label_left_cc=np.array(label_left_cc)
label_right_cc=np.array(label_right_cc)
label_left_mlo=np.array(label_left_mlo)
label_right_mlo=np.array(label_right_mlo)


np.save('patch_left_cc4.npy', patch_left_cc)
np.save('patch_right_cc4.npy', patch_right_cc)
np.save('patch_left_mlo4.npy', patch_left_mlo)
np.save('patch_right_mlo3.npy', patch_right_mlo)

np.save('label_left_cc4.npy', label_left_cc)
np.save('label_right_cc4.npy', label_right_cc)
np.save('label_left_mlo4.npy', label_left_mlo)
np.save('label_right_mlo4.npy', label_right_mlo)


print("number of patch_left_cc is:", len(patch_left_cc))
print("number of patch_right_cc is:", len(patch_right_cc))
print("number of patch_left_mlo is:", len(patch_left_mlo))
print("number of patch_right_mlo is:", len(patch_right_mlo))
