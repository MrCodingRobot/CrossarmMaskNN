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
sys.path.append(os.path.dirname(__file__)) # Appendings this file's path to PATH

import instance_data
import global_variables as gv

#-----------------------------------------------------------------------------------------
# Class

class ResultManager():

	def __init__(self, dict_para):
	
		self.dict_para = dict_para
		
		return None

	def input(self, image, results, image_path):

		self.image = image
		self.image_path = image_path
		self.no_instance_flag = False
		
		boxes = results['rois']
		masks = results['masks']
		class_ids = results['class_ids']
		scores = results['scores']
	
		number_of_instances = boxes.shape[0]
		self.instance_list = []
		
		if not number_of_instances:
			print("NO INSTANCES TO DISPLAY")
			image = imutils.resize(image, width=700)
			#cv2.imshow("Input", image)
			#cv2.waitKey(0)
			self.no_instance_flag = True
			return None
			
		for i in range(number_of_instances):
			self.instance_list.append(instance_data.InstanceData((boxes[i],masks[:,:,i],class_ids[i],scores[i])))
	
		self.sort_instances()
		self.check_for_repeat_instances()
		self.generate_contours()
		self.crop_crossarms()

		return None

	def sort_instances(self, key = None):
	
		self.instance_list = sorted(self.instance_list, key=lambda instance: instance.mask_pixel_count, reverse=True)
		
		return None
		
	def check_for_repeat_instances(self):
	
		checking_mask = None
		self.to_be_removed_instances = []
		
		for counter, instance in enumerate(self.instance_list):
			if type(checking_mask) == type(None):
				checking_mask = instance.mask
			else:
				current_instance_nonzero = cv2.countNonZero(instance.mask)
				
				shared_mask = cv2.bitwise_and(checking_mask, instance.mask)
				shared_nonzero = cv2.countNonZero(shared_mask)
				
				ratio = shared_nonzero/current_instance_nonzero * 100
				
				print("Instance {} - Ratio: {}".format(counter, ratio))
				
				if ratio > self.dict_para["shared_mask_ratio_threshold"]: # gv.SHARED_MASK_RATIO_THRESHOLD = 30
					print("Removed {} instance due to high sharing value".format(counter))
					instance.unique = False
		
		return None
		
	def generate_contours(self):
		
		for instance in self.instance_list:
			
			instance.cnts = cv2.findContours(instance.mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
			instance.cnts = imutils.grab_contours(instance.cnts)
	
		return None
	
	def shrink_rect(self, rect):

		#print("Rect: {}".format(rect))
		
		w,h = rect[1]

		if w > h:
			shrink_rect = ((rect[0][0], rect[0][1]),(w, int(h * self.dict_para["cropping_ratio"])), rect[2])
		else: # h > w
			shrink_rect = ((rect[0][0], rect[0][1]),(int(w * self.dict_para["cropping_ratio"]), h), rect[2])

		#print("Shrink Rect: {}".format(shrink_rect))

		return shrink_rect

	def crop_min_area_rect(self, img, rect):

		w,h = rect[1]
		w = int(w)
		h = int(h)

		box = cv2.boxPoints(rect)
		box = np.int0(box)

		new_pts = np.float32([[0,0],[w,0],[0,h],[w,h]])
		old_pts = np.float32([box[1],box[2],box[0],box[3]])

		M = cv2.getPerspectiveTransform(old_pts, new_pts)
		dst = cv2.warpPerspective(img, M, (w,h))

		if h > w:
			dst = imutils.resize(dst, height = int(h*0.6))
			dst = imutils.rotate_bound(dst, angle=-90)
		else:
			dst = imutils.resize(dst, width = int(w*0.6))

		return dst

	def get_rect_ratio(self, rect):

		w,h = rect[1]
		w = int(w)
		h = int(h)

		if w > h:
			return h/w
		return w/h 

	def crop_crossarms(self):

		if self.dict_para["only_long_crossarms"]:
			
			print("\nATTEMPTING TO REMOVE SHORT CROSSARMS")
			to_be_removed = []

		for instance in self.instance_list:

			rect = cv2.minAreaRect(instance.cnts[0])
			rect = self.shrink_rect(rect)

			if self.dict_para["only_long_crossarms"] is True:

				h_w_ratio = self.get_rect_ratio(rect)

				print("HW ratio: {}".format(h_w_ratio))

				if h_w_ratio > self.dict_para["long_crossarm_w_h_ratio_threshold"]:
					print("REMOVED")
					to_be_removed.append(instance)

			instance.cropped_image = self.crop_min_area_rect(self.image, rect) 

		if self.dict_para["only_long_crossarms"] is True:
			self.instance_list = [instance for instance in self.instance_list if instance not in to_be_removed]

		return None

	def show_crossarms(self):

		for counter, instance in enumerate(self.instance_list):

			cv2.imshow("Instance {} - {}".format(counter, instance.label), instance.cropped_image)

		return None
	
	def get_crossarm_images(self):

		return [instance.cropped_image for instance in self.instance_list]