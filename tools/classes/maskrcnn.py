# Common Core Library Imports
import os
import sys
import pathlib

# Third-Parthy Library Imports
import cv2
import keras

# Local Imports
ROOT_DIR = os.path.abspath("../../")
sys.path.append(ROOT_DIR)
sys.path.append(os.path.dirname(__file__)) # Appendings this file's path to PATH

# Modules written by us
import instance_data
import result_manager
import global_variables as gv

# Modules written by MaskRCNN Team
from Mask_RCNN.mrcnn.config import Config
from Mask_RCNN.mrcnn import model as modellib, utils

#------------------------------------------------------------------
# Class

class MaskRCNN():

    def __init__(self, dict_para):

        # Checking that weights_path is valid
        assert pathlib.Path(dict_para["weights_path"]).is_file() is True, "Mask-RCNN - Invalid Weights Path"

        # If valid parameters, store and continue
        self.dict_para = dict_para

        # Mask-RCNN Setup

        config = InferenceConfig()
        config.display()

        # Loading Mask-RCNN Model
 
        self.model = modellib.MaskRCNN(mode="inference", config=config,
                                       model_dir=gv.DEFAULT_LOGS_DIR)
        self.model.load_weights(dict_para["weights_path"], by_name=True)
        
        return None

    def __call__(self, image_path):

        return self.predict(image_path)

    def predict(self, image_path):

        # Checking that input image is valid
        assert pathlib.Path(image_path).is_file() is True, "Mask-RCNN - Invalid Image Path"

        # Feeding image into Mask-RCNN Model
        image = cv2.imread(image_path)
        self.results = self.model.detect([image], verbose=1)

        # Converting results into more useful data
        r = self.results[0]
        self.result_manager = result_manager.ResultManager(self.dict_para)
        self.result_manager.input(image, r, image_path)
        crossarm_images_list = self.result_manager.get_crossarm_images()

        return crossarm_images_list

#------------------------------------------------------------------
# Parasidic Classes (small and insignificant but needed to run MaskRCNN)

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
# Running Code

