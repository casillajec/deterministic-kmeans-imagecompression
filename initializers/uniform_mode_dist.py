import numpy as np
from .fft import get_mode

def uniform_mode_dist_init(unique_dpts, el_count, n, distance_f):
	"""
	Uniform Mode Distance Initialization
	it calculates the distance from all unique points to the mode and
	returns points that are uniformely spaced to the mode, including the
	mode and excluding the last point

	Arguments:
	unique_dpts: numpy 2d numerical array
	el_count: numpy 1d numerical array
	n: int
	distance_f: function of datapoint x datapoint -> float

	Output:
	init_points: numpy 2d numerical array
	"""

	# Initialize data holders
	init_points = np.ndarray(
		shape = [n, unique_dpts.shape[1]],
		dtype = unique_dpts.dtype
	)
	distances = np.ndarray(
		shape = [unique_dpts.shape[0]],
		dtype = np.float32
	)

	# Get the mode and make it first init point
	mode = get_mode(unique_dpts, el_count)
	init_points[0] = mode

	# Calculate distances and get min and max distances
	min_dis = float('inf')
	max_dis = float('-inf')
	for i in range(distances.shape[0]):
		dist = distance_f(mode, unique_dpts[i])
		distances[i] = dist

		if(dist > max_dis):
			max_dis = dist

	# Stablish distance within points (tresh)
	# Note that we are multiplying max_dis by 0.8 to get 80% of the
	# distances, that is to neglect the points which are the furthest
	# from the mode
	# similar results were obtained with max_dis/(n+2) but this seemed
	# way too arbitrary
	tresh = (max_dis*0.8)/(n)

	# Sort points by their distance to the mode
	sort_idxs = distances.argsort()
	unique_dpts = unique_dpts[sort_idxs]
	distances = distances[sort_idxs]

	# In case the loop ends and n points weren't found
	# diminish the treshold and start again
	j = 1
	while( j != n ):
		j = 1
		last = 0
		# Search points that are evenly spaced
		for i in range(0, distances.shape[0]):
			if(j == n):
				break
			elif(distances[i] >= last + tresh):
				last = distances[i]
				init_points[j] = unique_dpts[i]
				j += 1

		tresh = tresh*0.8

	return init_points


def get_mode_idx(unique_data, data_count):
	"""
	returns mode row from data array and a list with unique elements

	Arguments:
	data: numpy 2d numerical array

	Output:
	mode: numpy 1d numerical array
	unique_elems: numpy 2d numerical array
	"""
	max_count = 0
	mode_idx = -1
	for i in range(unique_data.shape[0]):
		if(data_count[i] > max_count):
			mode_idx = i
			max_count = data_count[i]

	return mode_idx
