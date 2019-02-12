import numpy as np
import time

def rgb_distance(p1, p2):
	"""
	Arguments:
	p1: numpy 1d/2d numerical array
	p2: numpy 1d/2d numerical array
	
	Output:
	dis: float/numpy 1d numerical array
	"""
	p1 = p1.astype(np.float32)
	p2 = p2.astype(np.float32)
	
	if(p1.ndim == 1 and p2.ndim == 1):
		r = (p1[0] + p2[0])/2
		s = np.array([2+(r/256), 4, (2+(255-r))/256], dtype = np.float32)
		px = (p1 - p2)
		dis = np.sqrt(np.sum(px*px*s))
	else:
		p1, p2 = np.atleast_2d(p1, p2)
		r = (p1[:, 0] + p2[:, 0])/2
		s = np.array([[2+(v/256), 4, (2+(255-v))/256] for v in r], dtype = np.float32)
		px = p2 - p1
		dis = np.sqrt(np.sum(px*px*s, axis = 1))
		
	return dis
	
def test_func(p1, ps):
	res = []
	
	for p2 in ps:
		res.append(rgb_distance(p1, p2))
		
	return np.array(res)
	
def test_func2(ps1, ps2):
	res = []
	
	for i in range(len(ps1)):
		res.append(rgb_distance(ps1[i], ps2[i]))
		
	return np.array(res)

if __name__ == '__main__':
	N = 1000
	p2p_t = 0
	mp_t = 0
	print('One vs Multiple pixels')
	for _ in range(N):
		p1 = np.random.randint(0, 255, [3], np.uint8)
		ps1 = np.random.randint(0, 255, [32, 3], np.uint8)
		
		t0 = time.perf_counter()
		p2p_res = test_func(p1, ps1)
		t1 = time.perf_counter()
		mp_res = rgb_distance_multipixel2(p1, ps1)
		t2 = time.perf_counter()
		
		if(np.any(p2p_res != mp_res)):
			print('Not equal outputs')
		
		p2p_t += (t1 - t0)
		mp_t += (t2 - t1)
	
	print('p2p:', p2p_t, 's')
	print('mp :', mp_t, 's')
	
	p2p_t = 0
	mp_t = 0
	print('\nOne vs One pixels')
	for _ in range(N):
		p1 = np.random.randint(0, 255, [3], np.uint8)
		p2 = np.random.randint(0, 255, [3], np.uint8)
		t0 = time.perf_counter()
		p2p_res = rgb_distance(p1, p2)
		t1 = time.perf_counter()
		mp_res = rgb_distance_multipixel2(p1, p2)
		t2 = time.perf_counter()
		
		if(np.any(p2p_res != mp_res)):
			print('Not equal outputs')
		
		p2p_t += (t1 - t0)
		mp_t += (t2 - t1)
	
	print('p2p:', p2p_t, 's')
	print('mp :', mp_t, 's')
	
	all_good = True
	for _ in range(N):
		ps1 = np.random.randint(0, 255, [32, 3], np.uint8)
		ps2 = np.random.randint(0, 255, [32, 3], np.uint8)
		all_good = np.all(test_func2(ps1, ps2) == rgb_distance_multipixel2(ps1, ps2)) and all_good
	
	if(all_good):
		print('\nAll tests are good!')
	else:
		print('\nsome tests were not good ):')
