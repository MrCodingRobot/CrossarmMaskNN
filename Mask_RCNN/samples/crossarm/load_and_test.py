#---------------------------------------------------------------------------------------
# Imports

import cv2
import numpy as np
import argparse
import os
import sys
import imutils
import time
import pickle
from pathlib import Path
import tqdm

import train_maskrcnn
import classes
import global_variables as glo_var

# Import Mask RCNN
sys.path.append(glo_var.ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import model as modellib, utils

#---------------------------------------------------------------------------------------
# Information

# Usage Example: 
# python test_crossarm.py --weights=C:\Users\daval\Desktop\COLLEGE\Graduate_Year_1\master_project\mask-rcnn\Mask_RCNN-master\samples\crossarm\weights\mask_rcnn_crossarm_15_epoch.h5 --image=C:\Users\daval\Desktop\COLLEGE\Graduate_Year_1\master_project\mask-rcnn\Mask_RCNN-master\samples\crossarm\crossarm_dataset\crossarm\test\DJI_0027.JPG

#---------------------------------------------------------------------------------------
# Classes

class InferenceConfig(train_maskrcnn.CrossarmConfig):
	# Set batch size to 1 since we'll be running inference on
	# one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
	GPU_COUNT = 1
	IMAGES_PER_GPU = 1
	
#---------------------------------------------------------------------------------------
# Methods (Functions)
	
#---------------------------------------------------------------------------------------
# Main Code
if __name__ == '__main__':
	
	# Setup
	config = InferenceConfig()
	config.display()

	# Parsing Parameters
	parser = argparse.ArgumentParser()
	parser.add_argument("--image", required=True)
	parser.add_argument("--weights", required=True)
	parser.add_argument("--save_pickle", required=False, default=False)
	parser.add_argument("--save_output_image", required=False, default=False)
	parser.add_argument("--display", required=False, default=False)
	args = parser.parse_args()

	path_object = Path(args.image)
	args.save_pickle = (args.save_pickle == "True" or args.save_pickle == True)
	args.save_output_image = (args.save_output_image == "True" or args.save_output_image == True)
	args.display = (args.display == "True" or args.display == True)

	# Checking if path_object exists
	if path_object.exists() is False or os.path.isabs(str(path_object)) is False:
		#print("Given image path is relative, making it absolute")
		alternative_path = os.path.join(os.getcwd(),args.image)
		path_object = Path(alternative_path)
		#print("New path: {}".format(str(path_object)))
		if path_object.exists() is False:
			raise RuntimeError("--image argument value does not exists")

	# Loading model
	weights_path = args.weights
	model = modellib.MaskRCNN(mode="inference", config=config,
							  model_dir=glo_var.DEFAULT_LOGS_DIR)				  
	model.load_weights(weights_path, by_name=True)

	# Handling path_object
	if path_object.is_file(): # Single image
		image_path_list = [args.image]
	elif path_object.is_dir(): # Directory
		directory_list = list(path_object.iterdir())
		#print(directory_list)
		image_path_list = [str(path) for path in directory_list if str(path).endswith(".JPG")]

	for image_path in tqdm.tqdm(image_path_list):
		
		# Loading image
		print("LOADING FOLLOWING IMAGE: {}".format(image_path))
		image = cv2.imread(image_path)

		# Creating predictions
		start = time.time()
		results = model.detect([image], verbose=1)
		end = time.time()

		duration = end - start
		print("Duration to create mask: {}".format(duration))

		# Visualizing predictions
		r = results[0]
		
		if args.save_pickle is True:
			if sys.platform.startswith("win32"):
				pickle.dump(r, open("saved_pickles\{}.pickle".format(Path(image_path).stem), "ab"))
			elif sys.platform.startswith("linux"):
				pickle.dump(r, open("saved_pickles/{}.pickle".format(Path(image_path).stem), "ab"))
		
		result_manager = classes.ResultManager(image, r, image_path)
		
		if args.display is True:
			result_manager.display()
			cv2.waitKey(0)
			cv2.destroyAllWindows()
			
		if args.save_output_image is True:
			result_manager.save_output_image()

						  
						  