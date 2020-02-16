# Library Imports

# Third-Party Imports
import cv2
import imutils
import numpy as np
from keras.models import load_model

# Local Imports
#import Mask_RCNN.samples.crossarm.global_variables as gv
#import Mask_RCNN.samples.crossarm.classes as classes
import mask_rcnn_tools.classes.classes as classes
#from mask_rcnn_tools.classes.result_manager import ResultManager
import mask_rcnn_tools.global_variables as gv


from Mask_RCNN.mrcnn.config import Config
from Mask_RCNN.mrcnn import model as modellib, utils

#------------------------------------------------------------------
# Classes

class CrossarmConfig(Config):
	
	"""
	Configuration for training on the toy  dataset.
	Derives from the base Config class and overrides some values.
	"""
	
	# Give the configuration a recognizable name
	NAME = "crossarm"

	# We use a GPU with 12GB memory, which can fit two images.
	# Adjust down if you use a smaller GPU.
	GPU_COUNT = 1
	#IMAGES_PER_GPU = 2
	IMAGES_PER_GPU = 1 # Needed to train all the layers

	# Number of classes (including background)
	NUM_CLASSES = 1 + 1	 # Background + crossarm

	# Number of training steps per epoch
	STEPS_PER_EPOCH = 100

	# Skip detections with < 90% confidence
	DETECTION_MIN_CONFIDENCE = 0.9
	
	# Use small validation steps since the epoch is small
	VALIDATION_STEPS = 5

class InferenceConfig(CrossarmConfig):
	# Set batch size to 1 since we'll be running inference on
	# one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
	GPU_COUNT = 1
	IMAGES_PER_GPU = 1

#------------------------------------------------------------------
# Functions

def MaskRCNN(maskrcnn_parameters):

	"""
	Mask-RCNN Section of the Crossarm Detection Pipeline
	"""

	# Mask-RCNN Setup

	config = InferenceConfig()
	config.display()

	# Loading Mask-RCNN Model

	model = modellib.MaskRCNN(mode="inference", config=config,
							model_dir=gv.DEFAULT_LOGS_DIR)
	model.load_weights(maskrcnn_parameters["weights_path"], by_name=True)

	# Feeding image into Mask-RCNN Model
	image = cv2.imread(maskrcnn_parameters["image_path"])
	results = model.detect([image], verbose=1)

	# Converting results into more useful data
	r = results[0]
	result_manager = classes.ResultManager(image, r, maskrcnn_parameters["image_path"])
	crossarm_images_list = result_manager.get_crossarm_images()

	return crossarm_images_list, result_manager

def CrackClassifier(crackclassifier_parameters):

	crack_no_crack_classifier = load_model()
	crack_no_crack_classifier.compile(loss="binary_crossentropy",
									optimizer="adam",
									metrics=["accuracy"])

	classifications = []

	for counter, test_image in enumerate(crossarm_images_list):

		test_image_saved = test_image
		test_image = cv2.resize(test_image,(128,128))
		test_image = np.reshape(test_image,[1,128,128,3])

		classification = crack_no_crack_classifier.predict_classes(test_image)[0][0]

		print(classification)

		if classification == 0:
			tag = "Cracked"
		else:
			tag = "No cracked"

		classifications.append(tag)

		cv2.imshow("Instance {} - Classified as {}".format(counter, tag), test_image_saved)

	return classifications

#------------------------------------------------------------------
# Main Code

# MaskRCNN Parameters
maskrcnn_parameters = {"image_path": r"C:\Users\daval\Documents\GitHub\CrossarmMaskNN\Mask_RCNN\samples\crossarm\crossarm_dataset\1DataSet4\ds\img\_174.jpg",
					   "weights_path": r"C:\Users\daval\Documents\GitHub\CrossarmMaskNN\Mask_RCNN\samples\crossarm\weights\all\epoch_0386.h5"}

# Showing Original Image for reference
original_image = cv2.imread(maskrcnn_parameters["image_path"])
cv2.imshow("Original Image", imutils.resize(original_image, height=700))

# Getting the output of the neural network and other statistics
crossarm_images_list, result_manager = MaskRCNN(maskrcnn_parameters)

crackclassifier_parameters = {"images_list": crossarm_images_list,
							  "model_filename": "bothmodel.h5"}

classifications = CrackClassifier(crackclassifier_parameters)

cv2.waitKey(0)
cv2.destroyAllWindows()




