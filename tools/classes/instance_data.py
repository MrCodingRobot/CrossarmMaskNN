# Imports
import cv2
import numpy as np
import imutils
from pathlib import Path
import sys
import collections
import os

# Local Imports
ROOT_DIR = os.path.abspath("../../")
sys.path.append(ROOT_DIR)

import global_variables as gv

#-----------------------------------------------------------------------------------------
# Class

class InstanceData():

	def __init__(self, data):
	
		# Data: box, mask, class_id, score
		self.box, self.mask, self.class_id, self.score = data
		self.label = gv.CLASS_NAMES[self.class_id]
		self.unique = True
		
		self.make_mask_np_friendly()
	
		return None
		
	def __repr__(self):
		return repr((self.label, self.score, self.box, self.mask_pixel_count))
		
	def __gt__(self, other):
		assert isinstance(other, InstanceData)
		return self.mask_pixel_count > other.mask_pixel_count
		
	def __lt__(self, other):
		assert isinstance(other, InstanceData)
		return self.mask_pixel_count < other.mask_pixel_count
		
	def make_mask_np_friendly(self):
		
		# Changing True/False to 255/0
		self.mask = np.where(self.mask == 1, 255, 0)
		self.mask = self.mask.astype('uint8')
		
		# Dilating mask
		kernel = np.ones((10,10), np.uint8)
		self.mask = cv2.dilate(self.mask, kernel, iterations=4)
		
		self.mask_pixel_count = cv2.countNonZero(self.mask)
		
		return None
		
	def apply_mask(self, image):
	
		image_copy = image.copy()
		
		for c in range(3):
			image_copy[:,:,c] = np.where(self.mask == 255, image_copy[:,:,c], 0)
			
		return image_copy

	def apply_contour(self, image):
	
		image_copy = image.copy()
		r,c,d = image_copy.shape
		contour_mask = np.zeros( (r,c) )
		cv2.fillPoly(contour_mask, pts=self.cnts, color=(255,255,255))
		
		for c in range(3):
			image_copy[:,:,c] = np.where(contour_mask == 255, image_copy[:,:,c], 0)
	
		return image_copy
		