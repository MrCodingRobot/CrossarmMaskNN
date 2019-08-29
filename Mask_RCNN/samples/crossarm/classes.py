import cv2
import numpy as np
import imutils
from pathlib import Path
import sys

import global_variables as glo_var
	   
#----------------------------------------------------------------------------------

class InstanceData():

	def __init__(self, data):
	
		# Data: box, mask, class_id, score
		self.box, self.mask, self.class_id, self.score = data
		self.label = glo_var.CLASS_NAMES[self.class_id]
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
		
class ResultManager():

	def __init__(self, image, results, image_path):
	
		self.image = image
		self.image_path = image_path
		
		boxes = results['rois']
		masks = results['masks']
		class_ids = results['class_ids']
		scores = results['scores']
	
		number_of_instances = boxes.shape[0]
		self.instance_list = []
		
		if not number_of_instances:
			print("NO INSTANCES TO DISPLAY")
			image = imutils.resize(image, width=700)
			cv2.imshow("Input", image)
			cv2.waitKey(0)
			return None
			
		for i in range(number_of_instances):
			self.instance_list.append(InstanceData((boxes[i],masks[:,:,i],class_ids[i],scores[i])))
	
		self.sort_instances()
		self.check_for_repeat_instances()
		self.generate_contours()
		self.modify_contours()
		
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
				
				if ratio > glo_var.SHARED_MASK_RATIO_THRESHOLD:
					print("Removed {} instance due to high sharing value".format(counter))
					instance.unique = False
		
		return None
		
	def generate_contours(self):
		
		for instance in self.instance_list:
			
			instance.cnts = cv2.findContours(instance.mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
			instance.cnts = imutils.grab_contours(instance.cnts)
	
		return None
		
	def modify_contours(self):
	
		for instance in self.instance_list:
			
			# Rectangle approximate contours
			instance.cnts = self.rotated_rect(instance.cnts)
			
		return None
		
	def display_instances_contours(self):
	
		for counter, instance in enumerate(self.instance_list):
		
			# Apply masking
			image = instance.apply_contour(self.image)
			
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
			box = cv2.boxPoints(rect)
			box = np.int0(box)
			boxes.append(box)
			
		return boxes
	
	
	def display(self):
	
		self.display_instances_contours()
		self.display_original_contours()
		
		return None
		
	def save_output_image(self):
	
		for counter, instance in enumerate(self.instance_list):
		
			image = instance.apply_contour(self.image)
			
			y1, x1, y2, x2 = instance.box
			m = 0 # Margin
			roi_image = image[y1-m:y2+m, x1-m:x2+m]
			
			path_object = Path(self.image_path)
			if sys.platform.startswith("win32"):
				filename = "crossarm_dataset\crossarm\mask\{}_i{}.JPG".format(path_object.stem, counter)
			elif sys.platform.startswith("linux"):
				filename = "crossarm_dataset/crossarm/mask/{}_i{}.JPG".format(path_object.stem, counter)
				
			cv2.imwrite(filename, roi_image)			
	
		return None