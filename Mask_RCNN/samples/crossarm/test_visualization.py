import cv2
import pickle
import numpy as np
import imutils
from pathlib import Path
import argparse
import os
import tqdm
import sys

import classes

#-------------------------------------------------------------------------------------

#----------------------------------------------------------------------------------------------
# Main Code

parser = argparse.ArgumentParser()
parser.add_argument("--image", required=True)
args = parser.parse_args()

path_object = Path(args.image)

# Checking if path_object exists
if path_object.exists() is False or os.path.isabs(str(path_object)) is False:
	#print("Given image path is relative, making it absolute")
	alternative_path = os.path.join(os.getcwd(),args.image)
	path_object = Path(alternative_path)
	#print("New path: {}".format(str(path_object)))
	if path_object.exists() is False:
		raise RuntimeError("--image argument value does not exists")

# Handling path_object
if path_object.is_file(): # Single image
	image_path_list = [args.image]
elif path_object.is_dir(): # Directory
	directory_list = list(path_object.iterdir())
	#print(directory_list)
	image_path_list = [str(path) for path in directory_list if str(path).endswith(".JPG")]

for image_path in tqdm.tqdm(image_path_list):

	image = cv2.imread(image_path)
	
	if sys.platform.startswith("win32"):
		with open('saved_pickles\{}.pickle'.format(Path(image_path).stem), 'rb') as handle:
			r = pickle.load(handle)
	elif sys.platform.startswith("linux"):
		with open('saved_pickles/{}.pickle'.format(Path(image_path).stem), 'rb') as handle:
			r = pickle.load(handle)
		
	classes.ResultManager(image, r)
	cv2.waitKey(0)
	cv2.destroyAllWindows()