import numpy as np

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
	r = (float(p1[0]) + float(p2[0]))/2
	s = np.array([2+(r/256), 4, (2+(255-r))/256], dtype = np.float32)
	px = (p1 - p2).astype(np.float32)
	
	return np.sqrt(np.sum(px*px*s))
