# Imports
import cv2
import numpy as np
import imutils
from pathlib import Path
import sys
import collections
import os

# Local Imports
from instance_data import InstanceData

#----------------------------------------------------------------------------------
# Constants

SHARED_MASK_RATIO_THRESHOLD = 30

#-----------------------------------------------------------------------------------------
# Class

class ResultManager():

	def __init__(self, image, results, image_path):
	
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
			self.instance_list.append(InstanceData((boxes[i],masks[:,:,i],class_ids[i],scores[i])))
	
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
				
				if ratio > SHARED_MASK_RATIO_THRESHOLD:
					print("Removed {} instance due to high sharing value".format(counter))
					instance.unique = False
		
		return None
		
	def generate_contours(self):
		
		for instance in self.instance_list:
			
			instance.cnts = cv2.findContours(instance.mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
			instance.cnts = imutils.grab_contours(instance.cnts)
	
		return None
	
	def crop_crossarms(self):

		for instance in self.instance_list:

			rect = cv2.minAreaRect(instance.cnts[0])
			instance.cropped_image = self.crop_min_area_rect(self.image, rect) 

		return None

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

	def show_crossarms(self):

		for counter, instance in enumerate(self.instance_list):

			cv2.imshow("Instance {} - {}".format(counter, instance.label), instance.cropped_image)

		return None
	
	def get_crossarm_images(self):

		return [instance.cropped_image for instance in self.instance_list]

	#----------------------------------------------------------------------------------

	def modify_contours(self):
	
		for instance in self.instance_list:
			
			# Rectangle approximate contours
			instance.cnts, instance.dim = self.rotated_rect(instance.cnts)

			
		return None
		
	def display_instances_contours(self):
	
		for counter, instance in enumerate(self.instance_list):
		
			print("instance: {}".format(instance.cnts))
			
			#w, h = find_rect_w_and_h(instance.cnts[0])
			w, h = instance.dim
			w = int(w)
			h = int(h)

			c = instance.cnts[0]

			new_pts = np.float32([[0,0],[w,0],[0,h],[w,h]])
			old_pts = np.float32([c[1],c[2],c[0],c[3]]) # might not work for all images
			M = cv2.getPerspectiveTransform(old_pts, new_pts)
			dst = cv2.warpPerspective(self.image, M, (w,h))

			if h > w:
				dst = imutils.resize(dst, height = int(h * 0.6))
				dst = imutils.rotate_bound(dst, angle=90)
			else:
				dst = imutils.resize(dst, width = int(w * 0.6))

			cv2.imshow("Insta {} - {} - {:.2f}".format(counter, instance.label, instance.score), dst)

		return None
		
	def display_instances_mask(self):
	
		for counter, instance in enumerate(self.instance_list):
		
			# Apply masking
			image = instance.apply_mask(self.image)
			
			# Getting ROI
			y1, x1, y2, x2 = instance.box
			m = 0 # Margin
			roi_image = image[y1-m:y2+m, x1-m:x2+m]
			
			# Resizing ROI
			h,w,d = roi_image.shape
			if h > w:
				roi_image = imutils.resize(roi_image, height=700)
			else:
				roi_image = imutils.resize(roi_image, width=700)
			
			# Display instance
			cv2.imshow("Instance {} - {} - {:.2f}".format(counter, instance.label, instance.score), roi_image)
			
		return None
		
	def display_original_contours(self):
	
		cnts_image = self.image.copy()
	
		# Drawing the good instances in green
		# Drawing the bad instances in red
		
		for instance in self.instance_list:
		
			if instance.unique is True:
				color = (0,255,0) # green
			else:
				color = (0,0,255) # red
			
			# Drawing the contours onto image
			cv2.drawContours(cnts_image, instance.cnts, -1, color, 10)
		
		# Display contour image
		resized_cnts_image = imutils.resize(cnts_image, width=700)
		cv2.imshow("Contours", resized_cnts_image)
	
		return None
		
	def approximate_contour_rectangle(self, cnts):
	
		rect_cnts = []
		
		for c in cnts:
		
			# approximate the contour
			peri = cv2.arcLength(c, True)
			approx = cv2.approxPolyDP(c, 0.02 * peri, True)
			
			# if our approximated contour has four points, then we
			# can assume that we have found our crossarm
			if len(approx) == 4:
				rect_cnts.append(approx)
				
		return rect_cnts
		
	def rotated_rect(self, cnts):
	
		boxes = []
		
		for c in cnts:
		
			rect = cv2.minAreaRect(c)
			#print("Rect: {}".format(rect))
			#print("Dim: {}".format(rect[1]))
			box = cv2.boxPoints(rect)
			box = np.int0(box)
			boxes.append(box)
			
		return boxes, rect[1]
	
	def display(self):
	
		self.display_original_contours()
		self.display_instances_contours()
		
		return None
		
	def save_output_image(self):

		if self.no_instance_flag is True:
			return None
	
		for counter, instance in enumerate(self.instance_list):
		
			#print("instance: {}".format(instance.cnts))
			
			#w, h = find_rect_w_and_h(instance.cnts[0])
			w, h = instance.dim
			w = int(w)
			h = int(h)

			c = instance.cnts[0]

			new_pts = np.float32([[0,0],[w,0],[0,h],[w,h]])
			old_pts = np.float32([c[1],c[2],c[0],c[3]]) # might not work for all images
			M = cv2.getPerspectiveTransform(old_pts, new_pts)
			dst = cv2.warpPerspective(self.image, M, (w,h))

			if h > w:
				dst = imutils.resize(dst, height = int(h * 0.6))
				dst = imutils.rotate_bound(dst, angle=90)
			else:
				dst = imutils.resize(dst, width = int(w * 0.6))
			
			path_object = Path(self.image_path)
			if sys.platform.startswith("win32"):
				filename = "crossarm_dataset\crossarm\mask\{}_i{}.JPG".format(path_object.stem, counter)
			elif sys.platform.startswith("linux"):
				filename = "crossarm_dataset/crossarm/mask/{}_i{}.JPG".format(path_object.stem, counter)
				
			cv2.imwrite(filename, dst)			
	
		return None

