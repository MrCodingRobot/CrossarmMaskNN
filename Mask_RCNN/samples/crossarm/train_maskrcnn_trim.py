"""
Mask R-CNN
Train on the crossarm dataset

Copyright (c) 2018 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla
Edited by Eduardo Davalos

------------------------------------------------------------

Usage: import the module (see Jupyter notebooks for examples), or run from
	   the command line as such:

	# Train a new model starting from pre-trained COCO weights
	python train_maskrcnn_trim.py --weights=coco --dataset=/path/to/dataset/
	
Usage: tensorboard --logdir={address to logs} --host=127.0.0.1
"""

# Windows Training Command
# python train_maskrcnn_trim --dataset=C:\Users\daval\Desktop\COLLEGE\Graduate_Year_1\master_project\mask-rcnn\Mask_RCNN-master\samples\crossarm\crossarm_dataset\crossarm --weights=coco

# Linux Training Command
# python train_maskrcnn_trim --dataset=/home/paperspace/Documents/Mask_RCNN-master/samples/crossarm/crossarm_dataset/crossarm --weights=coco

import os
import sys
import json
import datetime
import numpy as np
import skimage.draw
import argparse
from pathlib import Path

import email_notification
import global_variables as glo_var

# Import Mask RCNN
sys.path.append(glo_var.ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import model as modellib, utils

# Path to trained weights file
COCO_WEIGHTS_PATH = os.path.join(glo_var.ROOT_DIR, "mask_rcnn_coco.h5")

#-------------------------------------------------------------------------------
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

class CrossarmDataset(utils.Dataset):

	def load_crossarm(self, dataset_dir, subset):
		"""
		Load a subset of the Crossarm dataset.
		dataset_dir: Root directory of the dataset.
		subset: Subset to load: train or val
		"""
		
		# Add classes. We have only one class to add.
		self.add_class("crossarm", 1, "crossarm")

		# Train or validation dataset?
		assert subset in ["train", "val"]
		dataset_dir = os.path.join(dataset_dir, subset)

		# Load annotations
		# VGG Image Annotator (up to version 1.6) saves each image in the form:
		# { 'filename': '28503151_5b5b7ec140_b.jpg',
		#	'regions': {
		#		'0': {
		#			'region_attributes': {},
		#			'shape_attributes': {
		#				'all_points_x': [...],
		#				'all_points_y': [...],
		#				'name': 'polygon'}},
		#		... more regions ...
		#	},
		#	'size': 100202
		# }
		# We mostly care about the x and y coordinates of each region
		# Note: In VIA 2.0, regions was changed from a dict to a list.
		annotations = json.load(open(os.path.join(dataset_dir, "via_region_data.json")))
		annotations = list(annotations.values())  # don't need the dict keys

		# The VIA tool saves images in the JSON even if they don't have any
		# annotations. Skip unannotated images.
		annotations = [a for a in annotations if a['regions']]

		# Add images
		for a in annotations:
			# Get the x, y coordinaets of points of the polygons that make up
			# the outline of each object instance. These are stores in the
			# shape_attributes (see json format above)
			# The if condition is needed to support VIA versions 1.x and 2.x.
			if type(a['regions']) is dict:
				polygons = [r['shape_attributes'] for r in a['regions'].values()]
			else:
				polygons = [r['shape_attributes'] for r in a['regions']] 

			# load_mask() needs the image size to convert polygons to masks.
			# Unfortunately, VIA doesn't include it in JSON, so we must read
			# the image. This is only managable since the dataset is tiny.
			image_path = os.path.join(dataset_dir, a['filename'])
			image = skimage.io.imread(image_path)
			height, width = image.shape[:2]

			self.add_image(
				"crossarm",
				image_id=a['filename'],	 # use file name as a unique image id
				path=image_path,
				width=width, height=height,
				polygons=polygons)

	def load_mask(self, image_id):
		"""
		Generate instance masks for an image.
	    Returns:
		masks: A bool array of shape [height, width, instance count] with one mask per instance.
		class_ids: a 1D array of class IDs of the instance masks.
		"""
		
		# If not a crossarm dataset image, delegate to parent class.
		image_info = self.image_info[image_id]
		if image_info["source"] != "crossarm":
			return super(self.__class__, self).load_mask(image_id)

		# Convert polygons to a bitmap mask of shape
		# [height, width, instance_count]
		info = self.image_info[image_id]
		mask = np.zeros([info["height"], info["width"], len(info["polygons"])],
						dtype=np.uint8)
		for i, p in enumerate(info["polygons"]):
			# Get indexes of pixels inside the polygon and set them to 1
			rr, cc = skimage.draw.polygon(p['all_points_y'], p['all_points_x'])
			mask[rr, cc, i] = 1

		# Return mask, and array of class IDs of each instance. Since we have
		# one class ID only, we return an array of 1s
		return mask.astype(np.bool), np.ones([mask.shape[-1]], dtype=np.int32)

	def image_reference(self, image_id):
		"""Return the path of the image."""
		info = self.image_info[image_id]
		if info["source"] == "crossarm":
			return info["path"]
		else:
			super(self.__class__, self).image_reference(image_id)

#---------------------------------------------------------------------------------
# Methods

def train(model, epoch, layers):
	"""Train the model."""
	
	# Training dataset.
	dataset_train = CrossarmDataset()
	dataset_train.load_crossarm(args.dataset, "train")
	dataset_train.prepare()

	# Validation dataset
	dataset_val = CrossarmDataset()
	dataset_val.load_crossarm(args.dataset, "val")
	dataset_val.prepare()
	
	if layers == "heads":
		print("Training network heads")
		model.train(dataset_train, dataset_val,
					learning_rate=config.LEARNING_RATE,
					epochs=epoch,
					layers='heads')
	else:
		print("Training all layers")
		model.train(dataset_train, dataset_val,
					learning_rate=config.LEARNING_RATE / 10,
					epochs=epoch,
					layers='all')
	
	subject = "NEURAL NETWORK DONE TRAINING"
	text = "YAY your neural work training is done"
	email_notification.send_email(subject,text)
				
	return None

#------------------------------------------------------------------------------
# Main Code

if __name__ == '__main__':

	# Parse command line arguments
	parser = argparse.ArgumentParser(
		description='Train Mask R-CNN to detect crossarms.')
	parser.add_argument('--dataset', required=True,
						metavar="/path/to/crossarm/dataset/",
						help='Directory of the Crossarm dataset')
	parser.add_argument('--weights', required=True,
						metavar="/path/to/weights.h5",
						help="Path to weights .h5 file or 'coco'")
	parser.add_argument("--layers", required=True,
						help="'all' or 'heads' layer training")
	parser.add_argument("--epoch", required=True,
						 help="Number of desired epochs")
	parser.add_argument('--logs', required=False,
					default=glo_var.DEFAULT_LOGS_DIR,
					metavar="/path/to/logs/",
					help='Logs and checkpoints directory (default=logs/)')
	args = parser.parse_args()
	
	args.epoch = int(args.epoch)

	print("Weights: ", args.weights)
	print("Dataset: ", args.dataset)
	print("Layers: ", args.layers)
	print("Epoch: ", args.epoch)
	print("Logs: ", args.logs)
	
	# Validating arguments
	assert args.epoch >= 0 or args.epoch < 200, "Epoch value is not within range (1-200)"
	assert args.layers == "all" or args.layers == "heads", "layers only can be 'all' or 'heads'" 
	
	path_object = Path(args.dataset) 
	if path_object.exists() is False:
		print("Given dataset path: {} does not exists, trying relative path".format(str(path_object)))
		alternative_path = os.path.join(os.getcwd(),args.image)
		path_object = Path(alternative_path)
		assert path_object.exists(), "Relative path: {} does not exists".format(str(path_object))

	# Create model
	config = CrossarmConfig()
	model = modellib.MaskRCNN(mode="training", config=config,
							  model_dir=args.logs)

	# Select weights file to load
	if args.weights.lower() == "coco":
		weights_path = COCO_WEIGHTS_PATH
		# Download weights file
		if not os.path.exists(weights_path):
			utils.download_trained_weights(weights_path)
	elif args.weights.lower() == "last":
		# Find last trained weights
		weights_path = model.find_last()
	elif args.weights.lower() == "imagenet":
		# Start from ImageNet trained weights
		weights_path = model.get_imagenet_weights()
	else:
		weights_path = args.weights

	# Load weights
	print("Loading weights ", weights_path)
	if args.weights.lower() == "coco":
		# Exclude the last layers because they require a matching
		# number of classes
		model.load_weights(weights_path, by_name=True, exclude=[
			"mrcnn_class_logits", "mrcnn_bbox_fc",
			"mrcnn_bbox", "mrcnn_mask"])
	else:
		model.load_weights(weights_path, by_name=True)

	# Train or evaluate
	try:
		train(model, args.epoch, args.layers)
	except:
		subject = "NEURAL NETWORK TRAINING FAILURE"
		text = "NEURAL NETWORK TRAINING FAILURE"
		email_notification.send_email(subject,text)
		