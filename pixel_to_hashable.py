import numpy as np
import time

def pixel_to_str(pixel):
	"""
	returns pixel representation as hashable string
	
	Arguments:
	pixel: numpy 1d numerical array with shape [3]
	
	Output:
	string: string
	"""
	string = ''
	for p in pixel:
		string += str(p).zfill(3)
	
	return string
	
def str_to_pixel(str_pixel):
	"""
	returns string representation of a pixel as the original pixel
	
	Arguments:
	str_pixel: string
	
	Output: numpy 1d numerical array with shape [3]
	"""
	
	pixel = np.zeros([3], dtype = np.uint8)
	
	for i in range(3):
		pixel[i] = int(str_pixel[3*i:3*i+3])
		
	return pixel
	
def pixel_to_int(pixel):
	"""
	returns pixel representation as hashable integer
	
	Arguments:
	pixel: numpy 1d numerical array with shape [3]
	
	Output:
	integer: int
	"""
	integer = 0

	for i in range(3):
		integer += pixel[i]<<(i*8)

	return integer
	
	
def int_to_pixel(int_pixel):
	"""
	returns int representation of a pixel as the original pixel
	
	Arguments:
	int_pixel: int
	
	Output: numpy 1d numerical array with shape [3]
	"""
	
	pixel = np.zeros([3], dtype = np.uint8)
	
	for i in range(3):
		pixel[i] = (int_pixel & (0x0000FF << i*8)) >> i*8
	
	return pixel

if __name__ == '__main__':
	times_np = []
	times_str = []
	times_int = []
	number = 100000

	for _ in range(number):
		pixel = np.random.randint(0, 256, [3], dtype = np.uint8)
		
		t0 = time.perf_counter()
		res_str = str_to_pixel(pixel_to_str(pixel))
		t1 = time.perf_counter()
		res_np = np.fromstring(str(pixel)[1:-1], dtype = np.uint8, sep = ' ')
		t2 = time.perf_counter()
		res_int = int32_to_pixel(pixel_to_int32(pixel))
		t3 = time.perf_counter()
		
		if(np.any(res_str != res_np) or np.any(res_int != res_np)):
			raise ValueError('Error in the conversion or deconversion')
		
		times_str.append(t1 - t0)
		times_np.append(t2 - t1)
		times_int.append(t3 - t2)
		

	t_str = np.mean(times_str)*1000
	t_np  = np.mean(times_np)*1000
	t_int = np.mean(times_int)*1000

	print('str  :', t_str, 'ms')
	print('int  :', t_int, 'ms')
	print('numpy:', t_np,  'ms')
	print('str speedup:', t_np/t_str)
	print('int speedup:', t_np/t_int)

	str_dict = {}
	int_dict = {}
	for _ in range(number):
		pixel = np.random.randint(0, 256, [3], dtype = np.uint8)
		
		str_dict[pixel_to_str(pixel)] = 1
		int_dict[pixel_to_int32(pixel)] = 1
		s_pixel = pixel
		
	times_str = []
	times_int = []
		
	for _ in range(number):
		
		string = pixel_to_str(s_pixel)
		integr = pixel_to_int32(s_pixel)
		t0 = time.perf_counter()
		str_dict[string]
		t1 = time.perf_counter()
		int_dict[integr]
		t2 = time.perf_counter()
		
		times_str.append(t1 - t0)
		times_int.append(t2 - t1)
		
	mt_str = np.mean(times_str)*1000
	mt_int = np.mean(times_int)*1000

	print('mem str:', mt_str, 'ms')
	print('mem int:', mt_int, 'ms')
