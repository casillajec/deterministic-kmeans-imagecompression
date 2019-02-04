import numpy as np
from kmeans import rgb_distance
import cv2

def ptp_idm(im1, im2):
	"""
	Image Diversion Meassurement: a way to meassure the diversion 
	between two images by meassuring the rgb distance of both pixel
	by pixel and reducing to a numerical value
	
	Arguments:
	im1: numpy 3d numerical array
	im2: numpy 3d numerical array
	
	Output:
	idm: float
	"""
	
	if(im1.shape != im2.shape):
		error_msg = 'Images must have the same dimensions for a ptp'
		error_msg += 'image diversion meassurement and are'
		error_msg += '{} and {}'
		raise ValueError(error_msg.format(im1.shape, im2.shape))
	
	# Container for pixel distances	
	distances = np.ndarray(im1.shape, dtype = np.float32)
	
	# Calculate distances for every pixel
	for i in range(distances.shape[0]):
		for j in range(distances.shape[1]):
			distances[i][j] = rgb_distance(im1[i][j], im2[i][j])
	
	# Calculate the Image Diversion Measurement	by adding distances	
	idm = np.sum(distances)
	
	return idm
