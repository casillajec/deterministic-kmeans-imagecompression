import numpy as np

def deterministic_fft(data, n, distance_f):
	"""
	deterministic Farthest First Traversal implementation with n points
	using mode as starting point
	
	Arguments:
	data: numpy 2d numerical array
	n: int
	distance_f: function of datapoint x datapoint -> float
	
	Output:
	n_traversed: numpy 2d numerical array
	"""
	# Check the amount of points to traverse are less or equal than the
	# total amount of points
	if(n > data.shape[0]):
		error_msg = 'The amount of points to traverse has to be less'
		error_msg += 'or equal than the amount of total points.\n'
		error_msg += 'Points to traverse: {}\nTotal points: {}.'
		raise ValueError(error_msg.format(n, data.shape[0]))
	
	# Create n_traversed array
	n_traversed = np.ndarray([n, data.shape[1]], dtype = data.dtype)
	
	# Select mode as the starting point
	# Also extract array with unique datapoints
	start_p, unique_dpts = get_mode(data)
	
	# Add startng point to traversed list and remove from datapoints
	idx = np.where(np.all(unique_dpts == start_p, axis = 1))
	n_traversed[0] = unique_dpts[idx]
	unique_dpts = np.delete(unique_dpts, idx, axis = 0)
	
	# Find rest of the data points
	for i in range(1, n):
		max_dis = float('-inf')
		max_dis_idx = -1
		
		# Find point with max distance to the already traversed points
		for k in range(0, unique_dpts.shape[0]):
			k_dis = 0
			for u in range(0, i):
				k_dis += distance_f(unique_dpts[k], n_traversed[u])
				
			if(k_dis > max_dis):
				max_dis = k_dis
				max_dis_idx = k
				
		n_traversed[i] = unique_dpts[max_dis_idx]
		unique_dpts = np.delete(unique_dpts, max_dis_idx, axis = 0)
	
	return n_traversed
	
def get_mode(data):
	"""
	returns mode row from data array and a list with unique elements
	
	Arguments:
	data: numpy 2d numerical array
	
	Output:
	mode: numpy 1d numerical array
	unique_elems: numpy 2d numerical array
	"""
	element_count = {}
	
	for datap in data:
		key = str(datap)
		if(element_count.get(key)):
			element_count[key] += 1
		else:
			element_count[str(datap)] = 1
	
	max_count = 0
	mode = None
	unique_elems = []
	for elem, count in element_count.items():
		elem = np.fromstring(elem[1:-1], dtype = data.dtype, sep = ' ')
		unique_elems.append(elem)
		if(count > max_count):
			mode = elem
	
	unique_elems = np.array(unique_elems, dtype = data.dtype)
	
	return mode, unique_elems

if __name__ == '__main__':	
	def euc_distance(p1, p2):
		
		return np.sqrt(np.sum(np.power(p2 - p1, 2)))
		
	a = np.array(
		[[0, 0, 0],
		[10, 10, 0],
		[0, 0, 10],
		[10, 0, 0],
		[5, 0, 5],
		[5, 5, 5]]
	)
	b = deterministic_fft(a, 3, euc_distance)
	
	idx = np.where(np.all(a == b[0], axis = 1))
	a = np.delete(a, idx, axis = 0)
	print('path: \n', b)
	print('\ndistances:')
	
	for i in range(b.shape[0]-1):
		string = ''
		for j in range(i+1):
			string += str(b[j]) + ' and '
		string = string[:-5]
		
		for p in a:
			distance = 0
			for j in range(i+1):
				distance += euc_distance(p, b[j])
				
			print(string, 'to', p, distance)
		
		idx = np.where(np.all(a == b[i+1], axis = 1))
		a = np.delete(a, idx, axis = 0)
