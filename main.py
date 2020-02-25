# Library Imports
import pathlib

# Third-Party Imports
import cv2
import imutils
import numpy as np
from keras.models import load_model

# Local Imports
import tools.classes as clss
import global_variables as gv

#------------------------------------------------------------------
# Main Code

##################### INPUT #######################
 
# MaskRCNN Parameters
maskrcnn_input = r"E:\CrossarmMaskNN\crossarm\crossarm_dataset\1DataSet4\ds\img\_1.jpg"
maskrcnn_parameters = {"weights_path": r"C:\Users\daval\Documents\GitHub\CrossarmMaskNN\tools\models\segmentation\epoch_0386.h5",
					   "cropping_ratio": 0.5,
					   "shared_mask_ratio_threshold": 30,
					   "only_long_crossarms": True,
					   "long_crossarm_w_h_ratio_threshold": 0.10}

# crack_classifier Parameters
crack_classifier_parameters = {"model_path": r"C:\Users\daval\Documents\GitHub\CrossarmMaskNN\tools\models\classification\bothmodel.h5"}

#################### LOADING MODELS #######################

maskrcnn_model = clss.MaskRCNN(maskrcnn_parameters)
crack_classifier_model = clss.CrackClassifier(crack_classifier_parameters)

#################### OUTPUT #####################

crossarm_images_list = maskrcnn_model.predict(maskrcnn_input)

# Showing the original input for comparision
original_image = cv2.imread(maskrcnn_input)
cv2.imshow("Original Image", imutils.resize(original_image, height=700))

# Predicting list of images
classifications = crack_classifier_model.predict_image_list(crossarm_images_list)




