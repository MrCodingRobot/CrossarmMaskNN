# Common Core Libraries
import sys
import os

#---------------------------------------------------------------------

"""
This module contains all the necessary global variables of the network_controller.
"""

RASPBERRY_PI_EMAIL = "raspberry.pi.water.stations@gmail.com"
RASPBERRY_PI_EMAIL_PASSWORD = 'WaterStation'

PERSONAL_EMAIL = "davalosaeduardo@gmail.com"

CLASS_NAMES = ['BG', 'crossarm', 'bicycle', 'car', 'motorcycle', 'airplane',
               'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
               'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
               'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
               'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
               'kite', 'baseball bat', 'baseball glove', 'skateboard',
               'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
               'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
               'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
               'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
               'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
               'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
               'teddy bear', 'hair drier', 'toothbrush']
			   

ROOT_DIR = os.path.abspath("../../")
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")

if sys.platform.startswith("win32"):
	DATASET_DIR = os.path.join(ROOT_DIR, r"samples\crossarm\crossarm_dataset\crossarm")
elif sys.platform.startswith("linux"):
	DATASET_DIR = os.path.join(ROOT_DIR, "samples/crossarm/crossarm_dataset/crossarm")

SHARED_MASK_RATIO_THRESHOLD = 30
