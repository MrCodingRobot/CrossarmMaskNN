import cv2
import numpy as np
import imutils
from pathlib import Path
import sys
import collections
import os

#sys.path.append(".")
#import global_variables as glo_var

#----------------------------------------------------------------------------------
# Constants

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

SHARED_MASK_RATIO_THRESHOLD = 30

#----------------------------------------------------------------------------------

class InstanceData():

	def __init__(self, data):
	
		# Data: box, mask, class_id, score
		self.box, self.mask, self.class_id, self.score = data
		self.label = CLASS_NAMES[self.class_id]
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

	def shrink_rect(self, rect):

		#print("Rect: {}".format(rect))
		
		w,h = rect[1]

		if w > h:
			shrink_rect = ((rect[0][0], rect[0][1]),(w, int(h * 0.5)), rect[2])
		else: # h > w
			shrink_rect = ((rect[0][0], rect[0][1]),(int(w * 0.5), h), rect[2])

		#print("Shrink Rect: {}".format(shrink_rect))

		return shrink_rect

	def crop_min_area_rect(self, img, rect):

		rect = self.shrink_rect(rect)

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

		h,w,d = dst.shape
		size = h*w
		if size < 10000:
			dst = None

		return dst

	def show_crossarms(self):

		for counter, instance in enumerate(self.instance_list):

			cv2.imshow("Instance {} - {}".format(counter, instance.label), instance.cropped_image)

		return None
	
	def get_crossarm_images(self):

		return [instance.cropped_image for instance in self.instance_list if instance.cropped_image is not None]

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

