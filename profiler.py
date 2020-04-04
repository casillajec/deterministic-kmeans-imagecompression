from image_compressor import compress_image
from initializers.fft import deterministic_fft
from initializers.uniform_mode_dist import uniform_mode_dist_init
from initializers.random_init import random_init
import utils
from collections import defaultdict
import cv2
import os

class DataRow(object):

	def __init__(self, im_name, init_method, colors, mse, aid, time_profile):
		self.im_name = im_name
		self.init_method = init_method
		self.colors = colors
		self.mse = mse
		self.aid = aid
		self.unique_mapping = time_profile['unique_mapping']
		self.init_point_selection = time_profile['init_point_selection']
		self.unique_demapping = time_profile['unique_demapping']
		self.k_means = time_profile['k_means']

def best_random(n_cases = 100):
	best_mse = float('inf')
	best_mse_aid = ''
	best_aid = float('inf')
	best_aid_mse = ''
	best_compressed = None
	best_compressed_aid = None
	total_time_profile = defaultdict(float)

	for _ in range(n_cases):
		image, compressed, mse, aid, time_profile = compress_image(im_path, c, random_init)

		# Select best mse
		if(mse < best_mse):
			best_mse = mse
			best_mse_aid = aid
			best_compressed = compressed

		# Select best aid
		if(aid < best_aid):
			best_aid = aid
			best_aid_mse = mse
			best_compressed_aid = compressed

		# add times
		for key, value in time_profile.items():
			total_time_profile[key] += value

	return image, best_compressed, best_mse, best_mse_aid, best_compressed_aid, best_aid_mse, best_aid, total_time_profile,

def generate_csv(out_path, data_rows):

	time_unit = 's'
	string = ''
	string += 'im_name,init_method,colors,mse,aid,unique_mapping_time({0}),init_point_selection_time({0}),unique_demapping_time({0}),k_means_time({0})\n'.format(time_unit)

	for row in data_rows:
		string += row.im_name + ','
		string += row.init_method + ','
		string += str(row.colors) + ','
		string += str(row.mse) + ','
		string += str(row.aid) + ','
		string += str(row.unique_mapping) + ','
		string += str(row.init_point_selection) + ','
		string += str(row.unique_demapping) + ','
		string += str(row.k_means) + '\n'

	with open(out_path, 'w') as f:
		f.write(string)


if __name__ == '__main__':
	if(os.name == 'nt'):
		folder_sep = '\\'
	elif(os.name == 'posix'):
		folder_sep = '/'
	else:
		print('I can only run on Linux and Windows')
		exit()

	if(not os.path.exists('profile_images')):
		raise Exception('\'profile_images\' folder not found')

	im_paths = ['profile_images{}'.format(folder_sep) + file for file in os.listdir('profile_images')]
	#colors = [4, 8, 16, 32, 64]
	colors = [16]

	out_path = 'profile.csv'
	compressed_file_template = 'profile_compressed{}{}_{}_{}colors.png'
	data_rows = []
	n_cases = len(colors) * len(im_paths)
	i = 1

	utils.vlevel = 1

	for c in colors:
		for im_path in im_paths:
			print('Case{}/{}'.format(i, n_cases))
			i = i+1
			im_name = im_path.split(folder_sep)[-1].split('.')[:-1][0]

			# Random init
			image, compressed_mse, mse_mse, mse_aid, compressed_aid, aid_mse, aid_aid, time_profile = best_random(10)
			data_rows.append(DataRow(im_name, 'random_mse', c, mse_mse, mse_aid, time_profile))
			cv2.imwrite(compressed_file_template.format(folder_sep, im_name, 'random_mse', c), compressed_mse)

			data_rows.append(DataRow(im_name, 'random_aid', c, aid_mse, aid_aid, time_profile))
			cv2.imwrite(compressed_file_template.format(folder_sep, im_name, 'random_aid', c), compressed_aid)

			# FFT init
			_, compressed, mse, aid, time_profile = compress_image(im_path, c, deterministic_fft)
			data_rows.append(DataRow(im_name, 'fft', c, mse, aid, time_profile))
			cv2.imwrite(compressed_file_template.format(folder_sep, im_name, 'fft', c), compressed)

			# UMDI
			_, compressed, mse, aid, time_profile = compress_image(im_path, c, uniform_mode_dist_init)
			data_rows.append(DataRow(im_name, 'umdi', c, mse, aid, time_profile))
			cv2.imwrite(compressed_file_template.format(folder_sep, im_name, 'umdi', c), compressed)

			if(not os.path.exists('profile_compressed{}{}_original.png'.format(folder_sep, im_name))):
				cv2.imwrite('profile_compressed{}{}_original.png'.format(folder_sep, im_name), image)

	generate_csv(out_path, data_rows)


