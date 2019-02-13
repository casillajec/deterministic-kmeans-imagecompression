from kmeans import k_means
from rgb_distance import rgb_distance
from uniform_mode_dist import uniform_mode_dist_init
from pixel_to_hashable import pixel_to_str, str_to_pixel
import numpy as np
import argparse
import time
import cv2
import os
import utils

import resource

def compress_image(im_path, k, init_f = uniform_mode_dist_init):
	"""
	returns the original and compressed version of an image together
	with time profile data
	
	Arguments:
	im_path: string
	k: int
	init_f: function of 2d array x 1d array x int x function -> 2d array
	
	Output:
	image: numpy 2d numerical array
	compressed_image numpy 2d numerical array
	mse: float
	time_profile: dict of string -> float
	"""
	# Read image
	image = cv2.imread(im_path)
	# Store original shape
	original_shape = image.shape
	# Reshape into a numpy 2d array
	image = image.reshape([image.shape[0] * image.shape[1], 3])
	
	# Run k-means
	t0 = time.time()
	c_means, clusters, mse, time_profile = k_means(image, k, rgb_distance, init_f, pixel_to_str, str_to_pixel)
	t1 = time.time()
	time_profile['k_means'] = t1 - t0
	
	# Create compressed image
	compressed_image = np.zeros(image.shape, dtype = np.uint8)
	for i in range(compressed_image.shape[0]):
		compressed_image[i] = c_means[clusters[i]].astype(np.uint8)
		
	# Return to original shape
	image = image.reshape(original_shape).astype(np.uint8)
	compressed_image = compressed_image.reshape(original_shape).astype(np.uint8)
	
	return image, compressed_image, mse, time_profile

if __name__ == '__main__':
	
	# Script arguments
	ap = argparse.ArgumentParser(
		description = 'Compress an image with specified amount of colors'
	)
	ap.add_argument(
		'-i',
		'--image',
		required = True,
		help = 'Path to image'
	)
	ap.add_argument(
		'-c',
		'--colors',
		required = True,
		type = int,
		nargs = '+',
		help = 'Number of colors'
	)
	ap.add_argument(
		'-t',
		'--time',
		action = 'store_true',
		help = 'Print time profile'
	)
	ap.add_argument(
		'-mse',
		action = 'store_true',
		help = 'Print MSE'
	)
	ap.add_argument(
		'-v',
		'--verbosity',
		type = int,
		default = 1,
		help = 'Verbosity level'
	)
	args = ap.parse_args()
	
	# Define constants
	IM_PATH = args.image
	K = args.colors
	global vlevel
	utils.vlevel = args.verbosity
	
	# Image data
	im_name = IM_PATH.split('/')[-1].split('.')[:-1][0]
	
	print('Compressing', im_name)
	for k in K:
		print(k, 'colors')
		image, compressed_image, mse, time_profile = compress_image(IM_PATH, k)
		
		# Store original and resulting image in png format
		if(not os.path.isdir('./compressed')):
			os.mkdir('./compressed')
		
		if(not os.path.exists('./compressed/{}_original.png'.format(im_name))):
			cv2.imwrite('./compressed/{}_original.png'.format(im_name), image)
		cv2.imwrite('./compressed/{}_{}colors.png'.format(im_name, k), compressed_image)
		
		if(args.time):
			print('Time profile')
			for metric, value in time_profile.items():
				print('{}: {}'.format(metric, value))
				
		if(args.mse):
			print('MSE:', mse)
	
	print('memory:', resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1000, 'MB')
	
	
	
