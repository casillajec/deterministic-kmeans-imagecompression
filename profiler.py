from image_compressor import compress_image
from fft import deterministic_fft
from uniform_mode_dist import uniform_mode_dist_init
from random_init import random_init
import cv2
import os

class DataRow(object):
	
	def __init__(self, im_name, init_method, colors, mse, time_profile):
		self.im_name = im_name
		self.init_method = init_method
		self.colors = colors
		self.mse = mse
		self.unique_mapping = time_profile['unique_mapping']
		self.init_point_selection = time_profile['init_point_selection']
		self.unique_demapping = time_profile['unique_demapping']
		self.k_means = time_profile['k_means']
		
def best_random(n_cases = 100):
	best_mse = float('inf')
	best_compressed = None
	best_time_profile = None
	
	for _ in range(n_cases):
		image, compressed, mse, time_profile = compress_image(im_path, c, random_init)
		
		# Select best mse
		if(mse < best_mse):
			best_mse = mse
			best_compressed = compressed
			best_time_profile = time_profile
			
	return image, best_compressed, best_mse, best_time_profile
	
def generate_csv(out_path, data_rows):
	
	time_unit = 's'
	string = ''
	string += 'im_name,init_method,colors,mse,unique_mapping_time({0}),init_point_selection_time({0}),unique_demapping_time({0}),k_means_time({0})\n'.format(time_unit)
	
	for row in data_rows:
		string += row.im_name + ','
		string += row.init_method + ','
		string += row.colors + ','
		string += row.mse + ','
		string += row.unique_mapping + ','
		string += row.init_point_selection + ','
		string += row.unique_demapping + ','
		string += row.kmeans + '\n'
		
	with open(out_path, 'w') as f:
		f.write(string)
	

if __name__ == '__main__':
	
	if(not os.path.exists('profile_images')):
		raise Exception('\'profile_images\' folder not found')
	
	im_paths = ['./profile_images/' + file for file in os.listdir('profile_images')]
	colors = [4, 8, 16, 32, 64]
	
	out_path = 'profile.csv'
	data_rows = []
	n_cases = len(colors) * len(im_paths)
	i = 1
	
	for c in colors:
		for im_path in im_paths:
			print('Case{}/{}'.format(i, n_cases))
			i = i+1
			im_name = im_path.split('/')[-1].split('.')[:-1][0]
			# Random init
			image, compressed, mse, time_profile = best_random(100)
			data_rows.append(DataRow(im_name, 'random', c, mse, time_profile))
			cv2.imwrite('./compressed/{}_{}_{}colors.png'.format(im_name, 'random', c), compressed)
			
			# FFT init
			_, compressed, mse, time_profile = compress_image(im_path, c, deterministic_fft)
			data_rows.append(DataRow(im_name, 'fft', c, mse, time_profile))
			cv2.imwrite('./compressed/{}_{}_{}colors.png'.format(im_name, 'fft', c), compressed)
			
			# UMDI
			_, compressed, mse, time_profile = compress_image(im_path, c, uniform_mode_dist_init)
			data_rows.append(DataRow(im_name, 'umdi', c, mse, time_profile))
			cv2.imwrite('./compressed/{}_{}_{}colors.png'.format(im_name, 'umdi', c), compressed)
			
			if(not os.path.exists('./compressed/{}_original.png'.format(im_name))):
				cv2.imwrite('./compressed/{}_original.png'.format(im_name), image)
			
	generate_csv(out_file, data_rows)
	
	
