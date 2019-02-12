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
	# Cast pixels to float
	p1 = p1.astype(np.float32)
	p2 = p2.astype(np.float32)
	
	r = (p1[0] + p2[0])/2
	s = np.array([2+(r/256), 4, (2+(255-r))/256], dtype = np.float32)
	px = (p1 - p2)
	
	return np.sqrt(np.sum(px*px*s))
	
def rgb_distance_multipixel(p1, p2):
	p1 = p1.astype(np.float32)
	p2 = p2.astype(np.float32)

if __name__ == '__main__':
	
	p1 = np.array([255, 255, 255], dtype = np.uint8)
	p2 = np.array([1, 1, 1], dtype = np.uint8)
	
	ps = np.array(
		[[1,1,1],
		 [2,2,2],
		 [3,3,3]], dtype = np.uint8)
	
	"""
	print(p1 - ps)
	print(ps - p1)
	
	p1 = p1.astype(np.float32)
	ps = ps.astype(np.float32)
	
	print(p1 - ps)
	print(ps - p1)
	"""
	
	print('\nasd')
	print(p1-p2)
	print(rgb_distance(p1, p2))
	print(p2-p1)
	print(rgb_distance(p2, p1))
