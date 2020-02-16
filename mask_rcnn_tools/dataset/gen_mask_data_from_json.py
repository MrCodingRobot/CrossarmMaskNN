# Author: Eduardo Davalos
# Date: 10/29/2019

#https://stackoverflow.com/questions/37177811/crop-rectangle-returned-by-minarearect-opencv-python

"""
The goal of this code is to utilize the hand-labeled images
from the training and validation dataset to create a
high-quality mask dataset.
"""

import json
import os
import cv2
import numpy as np
import imutils
import tqdm

#-------------------------------------------------------------------------------
# Constants

#train_path = r"C:\Users\daval\Documents\GitHub\CrossarmMaskNN\Mask_RCNN\samples\crossarm\crossarm_dataset\crossarm\train"
#val_path = r"C:\Users\daval\Documents\GitHub\CrossarmMaskNN\Mask_RCNN\samples\crossarm\crossarm_dataset\crossarm\val"

#train_json = os.path.join(train_path, "via_region_data.json")
#val_json = os.path.join(val_path, "via_region_data.json")

json_file = r"C:\Users\daval\Documents\GitHub\CrossarmMaskNN\Mask_RCNN\samples\crossarm\crossarm_dataset\annotation_conversion\1DataSet4.json"
img_dir = r"C:\Users\daval\Documents\GitHub\CrossarmMaskNN\Mask_RCNN\samples\crossarm\crossarm_dataset\1DataSet4\ds\img"

#--------------------------------------------------------------------------------
# Utilities

def crop_min_area_rect(img, rect):

    # rotate img
    angle = rect[2]
    rows,cols = img.shape[0], img.shape[1]
    M = cv2.getRotationMatrix2D((cols/2,rows/2),angle,1)
    img_rot = cv2.warpAffine(img,M,(cols,rows))

    # rotate bounding box
    rect0 = (rect[0], rect[1], 0.0) 
    box = cv2.boxPoints(rect0)
    pts = np.int0(cv2.transform(np.array([box]), M))[0]    
    pts[pts < 0] = 0

    # crop
    img_crop = img_rot[pts[1][1]:pts[0][1], 
                       pts[1][0]:pts[2][0]]

    return img_crop

def crop_min_area_rect2(img, rect):

    w,h = rect[1]
    w = int(w)
    h = int(h)

    box = cv2.boxPoints(rect)
    box = np.int0(box)

    new_pts = np.float32([[0,0],[w,0],[0,h],[w,h]])
    old_pts = np.float32([box[1],box[2],box[0],box[3]])

    M = cv2.getPerspectiveTransform(old_pts, new_pts)
    dst = cv2.warpPerspective(img, M, (w,h))

    return dst

#-------------------------------------------------------------------------------

with open(json_file) as json_file:
    data = json.load(json_file)

    for data_label in tqdm.tqdm(data):
        
        filename = data[data_label]['filename']
        img_path = os.path.join(img_dir, filename)
        img = cv2.imread(img_path)

        #print("Filename: {}".format(filename))

        for counter, region in enumerate(data[data_label]["regions"]):

            xs = region["shape_attributes"]["all_points_x"]
            ys = region["shape_attributes"]["all_points_y"]
        
            pts = []

            for i in range(len(xs)):
                pts.append((xs[i], ys[i]))
            
            pts = np.asarray(pts)
            #print(pts)

            rect = cv2.minAreaRect(pts)
            #print(rect)

            w,h = rect[1]
            w = int(w)
            h = int(h)

            #img_crop = crop_min_area_rect(img, rect)
            img_crop = crop_min_area_rect2(img, rect)

            if h > w:
                img_crop = imutils.resize(img_crop, height = int(h * 0.6))
                img_crop = imutils.rotate_bound(img_crop, angle=-90)
            else:
                img_crop = imutils.resize(img_crop, width = int(w * 0.6))
            
            h,w,d = img_crop.shape
            size = h*w
            #print("size: {}".format(size))

            if size < 10000:
                continue

            #cv2.imshow("output", img_crop)
            #cv2.waitKey(0)
            #cv2.destroyAllWindows()

            save_file = r"C:\Users\daval\Documents\GitHub\CrossarmMaskNN\Mask_RCNN\samples\crossarm\crossarm_dataset\crossarm\perfect_mask_2\{}_i{}.JPG".format(filename[0:-4], counter)
            #print(save_file)
            cv2.imwrite(save_file, img_crop)


