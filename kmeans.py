import numpy as np
import time
from collections import defaultdict

EPS_F32 = np.finfo(np.float32).eps

def k_means(data, k, distance_f, init_f, datap_to_hashable, hashable_to_datap):
	"""
	k-means implementation
	
	Arguments:
	data: numpy 2d numerical array
	k: int
	distance_f: function of datapoint x datapoint -> float
	init_f: function of 2d array x 1d array x int x function -> 2d array
	
	Output:
	c_means: numpy 2d numerical array
	clusters: numpy 1d numerical array
	mse: float
	time_profile: dict of string -> float
	"""
	# Dict for holding execution time values
	time_profile = {}
	
	# Check that the amount of clusters is less or equal than the
	# total amount of points
	if(k > data.shape[0]):
		error_msg = 'The amount of clusters has to be less'
		error_msg += 'or equal than the amount of total data points.\n'
		error_msg += 'Clusters: {}\nTotal datapoints: {}.'
		raise ValueError(error_msg.format(k, data.shape[0]))
		
	# Get unique datapoints, their mapping to the original dataset
	# and element count for faster clusterization
	t0 = time.time()
	unique_datap, el_count, mapping = get_uniques_mapping(data, datap_to_hashable, hashable_to_datap)
	t1 = time.time()
	time_profile['unique_mapping'] = t1 - t0
	
	# Initialize cluster randomly
	t0 = time.time()
	c_means = init_f(unique_datap, el_count, k, distance_f).astype(np.float32)
	t1 = time.time()
	time_profile['init_point_selection'] = t1 - t0
	
	# Initial clusterization
	clusters = clusterize(unique_datap, c_means, distance_f)
	
	# First mse
	mse = get_mse(unique_datap, clusters, c_means, distance_f)
	
	while(True):
		old_means = c_means
		old_mse = mse
		
		# Update means
		c_means = get_means(unique_datap, clusters, old_means)
		dif = np.absolute(c_means - old_means)
		
		# If means didn't change, break the loop
		if(np.all(dif < EPS_F32)):
			break
			
		# Reclusterize
		clusters = clusterize(unique_datap, c_means, distance_f)
		
		# MSE
		mse = get_mse(unique_datap, clusters, c_means, distance_f)
		
		# If mse doesn't change, break the loop
		if(old_mse - mse < EPS_F32):
			break
		"""
		# If MSE doesn't change too much or increases, break the loop
		mse_change = (mse - old_mse)/old_mse
		print(mse_change)
		
		if(mse_change > 0 or abs(mse_change) < 0.001):
			mse_stop_
		"""
	# Remapping unique clusters to original dataset
	t0 = time.time()
	clusters_mapping = np.ndarray(
		shape = [data.shape[0]],
		dtype = np.int32
	)
	for i in range(len(mapping)):
		for idx in mapping[i]:
			clusters_mapping[idx] = clusters[i]
	t1 = time.time()
	time_profile['unique_demapping'] = t1 - t0
	
	return c_means, clusters_mapping, mse, time_profile
	
def clusterize(data, c_means, distance_f):
	"""
	given a list of datapoints and a list of means it returns a new 
	categorization of the data by clusters with c_means as central points
	
	Arguments:
	data: numpy 2d numerical array
	c_means: numpy 2d numerical array
	
	Outut:
	clusters: numpy 1d anumerical rray
	"""
	# Extract k
	k = c_means.shape[0]
	
	# Initialize cluster categorization array
	clusters = np.ndarray(
		shape = [data.shape[0]],
		dtype = np.int32
	)
	
	# For every datapoint search the cluster it is the nearest to
	# and assign the index(cluster id) to the clusters categorization
	# array
	for j, datap in enumerate(data):
		min_dis = float('inf')
		min_idx = -1
		
		for i in range(k):
			cluster_i_dis = distance_f(datap, c_means[i])
			if(cluster_i_dis < min_dis):
				min_dis = cluster_i_dis
				min_idx = i
				
		clusters[j] = min_idx
	
	return clusters
	
def get_means(data, clusters, old_means):
	"""
	given a list of datapoints, a cluster categorization of these and
	it's central points, it returns the new means of this cluster
	categorization
	
	Arguments:
	data: numpy 2d numerical array
	clusters: numpy 2d numerical array
	old_means: numpy 2d numerical array
	
	Output:
	new_means numpy 2d numerical array
	"""
	new_means = old_means.copy()
	mean_count = np.ones(old_means.shape[0], dtype = np.int64)
	
	for i in range(data.shape[0]):
		idx = clusters[i]
		new_means[idx] = new_means[idx] + data[i]
		mean_count[idx] += 1
		
	for i in range(new_means.shape[0]):
		new_means[i] = new_means[i] / mean_count[i]
	
	return new_means
	
def get_mse(data, clusters, c_means, distance_f):
	"""
	given a list of datapoints, a cluster categorization of these and
	it's central points, it returns the Mean Square Error, a meassurement
	of the ''quality'' of this categorization
	
	Arguments:
	data: numpy 2d numerical array
	clusters: numpy 2d numerical array
	c_means: numpy 2d numerical array
	
	Outpu:
	mse: float
	"""
	mse = 0
	for i in range(data.shape[0]):
		mse += np.power(distance_f(data[i], c_means[clusters[i]]), 2)
		
	return mse
	
def get_uniques_mapping(data, datap_to_hashable, hashable_to_datap):
	"""
	returns unique datapoints together with a mapping for their original
	position in the dataset and the count for each element
	
	Arguments:
	data: numpy 2d numerical array
	
	Output:
	unique_elems: numpy 2d numerical array
	count: numpy 2d numerical array
	mapping: python list of list of int
	"""
	element_mapping = defaultdict(list)
	
	for i, datap in enumerate(data):
		element_mapping[datap_to_hashable(datap)].append(i)
	
	n_uniques = len(element_mapping)
	unique_elems = np.ndarray(
		shape = [n_uniques, data.shape[1]],
		dtype = data.dtype
	)
	count = np.ndarray(
		shape = [n_uniques],
		dtype = np.int32
	)
	mapping = []
	
	for i, map_pair in enumerate(element_mapping.items()):
		elem = hashable_to_datap(map_pair[0])
		unique_elems[i] = elem
		mapping.append(map_pair[1])
		count[i] = len(map_pair[1])
		
	return unique_elems, count, mapping
	
	
if __name__ == '__main__':
	print(k_means(
		[[1, 2, 3],
		 [2, 3, 4],
		 [3, 4, 5],
		 [4, 5, 6]], 2))
