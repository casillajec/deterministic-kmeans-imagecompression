import numpy as np

def k_means(data, k):
	"""
	k-means implementation
	
	Arguments:
	data: numpy 2d numerical array
	k: int
	
	Output:
	c_means: numpy 2d numerical array
	clusters: numpy 2d numerical array
	mse: float
	"""
	data = np.array(data)
	
	# Initialize cluster means randomly
	c_means = data[np.random.choice(data.shape[0], size = k, replace = False)]
	
	# Initial clusterization
	clusters = clusterize(data, c_means)
	
	while(True):
		
		# Update means
		old_means = c_means
		c_means = get_means(data, clusters, old_means)
		
		# If means didn't change, break the loop
		if(np.all(c_means == old_means)):
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
	clusters: numpy 2d anumerical rray
	"""
	clusters = None
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
	new_means = None
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
	mse = None
	return mse
	
print(k_means(
	[[1, 2, 3],
	 [2, 3, 4],
	 [3, 4, 5],
	 [4, 5, 6]], 2))
