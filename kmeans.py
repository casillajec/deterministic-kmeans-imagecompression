import numpy as np
import time
from fft import deterministic_fft

def k_means(data, k):
	"""
	k-means implementation
	
	Arguments:
	data: numpy 2d numerical array
	k: int
	
	Output:
	c_means: numpy 2d numerical array
	clusters: numpy 1d numerical array
	mse: float
	"""
	
	t0 = time.time()
	# Initialize cluster means randomly
	c_means = deterministic_fft(data, k, rgb_distance).astype(np.float32)
	t1 = time.time()
	print('Time for FFT:', t1-t0, 's')
	
	# Initial clusterization
	clusters = clusterize(data, c_means)
	
	while(True):
		
		# Update means
		old_means = c_means
		c_means = get_means(data, clusters, old_means)
		
		# If means didn't change, break the loop
		if(np.all(c_means - old_means < np.finfo(c_means.dtype).eps)):
			break
			
		# Reclusterize
		clusters = clusterize(data, c_means)
		
	# Calculate clusters mse
	mse = get_mse(data, clusters, c_means)
	
	return c_means, clusters, mse
	
def clusterize(data, c_means):
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
			cluster_i_dis = rgb_distance(datap, c_means[i])
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
	new_means = old_means
	mean_count = np.ones(old_means.shape[0], dtype = np.int64)
	
	for i in range(data.shape[0]):
		idx = clusters[i]
		new_means[idx] = new_means[idx] + data[i]
		mean_count[idx] += 1
		
	for i in range(new_means.shape[0]):
		new_means[i] = new_means[i] / mean_count[i]
	
	return new_means
	
def get_mse(data, clusters, c_means):
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
		mse += np.power(rgb_distance(data[i], c_means[clusters[i]]), 2)
		
	return mse
	
def rgb_distance(p1, p2):
	"""
	given two rgb pixels, returns the rgb distance from the first to
	the second
	
	Arguments:
	p1: numpy 1d numerical array with shape [3]
	p2: numpy 1d numerical array with shape [3]
	
	Output:
	dis: float
	"""
	r = (int(p1[0])+int(p2[0]))/2
	s = np.array([2+(r/256), 4, (2+(255-r))/256], dtype = np.int32)
	px = p1 - p2
	
	dis = np.sqrt( np.sum( int(px[i])*int(px[i])*s[i] for i in range(3) ) )
	
	return dis
	
	
if __name__ == '__main__':
	print(k_means(
		[[1, 2, 3],
		 [2, 3, 4],
		 [3, 4, 5],
		 [4, 5, 6]], 2))
