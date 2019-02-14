import numpy as np

def random_init(unique_datap, el_count, k, distance_f):
	
	return (unique_datap[np.random.choice(unique_datap.shape[0], k, replace = False)])
