import os
import cv2
import time
import PIL
import numpy as np
import tensorflow as tf
from ultralytics import YOLO
from termcolor import colored

pt = "/models/yolov8n-face.pt"
yaml = "/models/yolov8n_face.yaml"

galleryInput = "/INPUTS/GALLERY/"
queryInput =  "/INPUTS/QUERY/"

galleryOutput = "/OUTPUTS/YOLO/GALLERY/"
queryOutput = "/OUTPUTS/YOLO/QUERY/"
class Yolov8Face(object):
	def __init__(self, yaml_file, pt_file):
		self.yaml_file = yaml_file
		self.pt_file = pt_file
		self.iscroppingdone = False
		self.acceptedFormat = set(('jpg','png','jpeg', 'png', 'bmp', 'gif'))
		# initialize yolo model
		self.model = YOLO(self.yaml_file)
		self.model = YOLO(self.pt_file)

	def _checkFormat(self, filename):
		extension = filename.split('.')[-1]
		return extension in self.acceptedFormat

	def _initialCheck(self, folder):
		filelist = os.listdir(folder)
		extension = []
		for item in filelist:
			if not item.split('.')[-1] in extension:
				extension.append(item.split('.')[-1])
		return len(set(extension).intersection(self.acceptedFormat)) == len(set(extension))
		
	def _cropSingle(self, filename, confidence, device='cpu'):
		#confidence, iou, and device, can be changed obtaining relevant results
		results = self.model(source=filename, conf=confidence, iou=0.70, device=device, 
			visualize=False, save_crop=False, box=False, verbose=True)
		return results[0]

	def _saveCroppedImage(self, bbobj, filename, output, verbose=True):
	
		if not os.path.isdir(output):
			os.mkdir(os.path.join(output))

		# checking for image name format (either absolute or not)
		if len(filename.split('/')) > 1:
			splitted = str(filename.split('/')[-1])
		else:
			splitted = filename
		outName = str(splitted.split('.')[0])

		img = np.array(PIL.Image.open(filename))
		for i, bb in enumerate(bbobj.boxes.xyxy):
			# cropping
			x1,y1,x2,y2 = np.int64(bb)
			cropped = img[y1:y2, x1:x2, :]
			# saving
			im = PIL.Image.fromarray(cropped)
			tmpOutName = f'{outName}_{i}.png'
			if verbose:
				print(colored(f'Now saving: {tmpOutName}', 'green'))

			im.save(os.path.join(output, tmpOutName))


	def cropFolder(self, folder, output, confidence):
		if not self._initialCheck(folder):
			print('*** (WARNING) Some not valid image file format found!')
		else:
			print(colored('Initial check done!', 'green'))

		time.sleep(1)

		for item in os.listdir(folder):
			filename = os.path.join(folder, item)
			try:
				bbobj = self._cropSingle(filename, confidence=confidence)
				self._saveCroppedImage(bbobj, filename, output)
			except:
				print(f'*** Some error with saving: {filename}')
		self.iscroppingdone = True
		print('\n')



detector = Yolov8Face(yaml, pt)

detector.cropFolder(queryInput, queryOutput, confidence=0.21)
detector.cropFolder(galleryInput, galleryOutput, confidence=0.21)
