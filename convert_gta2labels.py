from PIL import Image
import numpy as np
import pickle
import os
import sys
import time
import datetime

path = '../cygan/ds_small/trainB_labels'
out = '../cygan/ds_small/trainB_labels'
with open('map_rgb2class.pkl', 'rb') as f:
	map_rgb2class = pickle.load(f)
prev_time=time.time()
mean_period = 0
allBls = [f for f in os.listdir(path) if f.endswith('png')]
total = len(allBls)
done = 0
for bl in allBls:
	img = Image.open(os.path.join(path,bl)).convert('RGB')
	arr = np.asarray(img)
	encoded = np.zeros((arr.shape[0],arr.shape[1])).astype('uint8')
	for k,v in map_rgb2class.items():
		encoded[np.logical_and(np.logical_and(arr[:,:,0]==k[0],arr[:,:,1]==k[1]),arr[:,:,2]==k[2])]=v

	tosave = Image.fromarray(encoded)
	tosave.save(os.path.join(out,bl))
	mean_period += time.time() - prev_time
	prev_time = time.time()
	done += 1
	# print('ETA: %s' % (datetime.timedelta(seconds=(total-done)*mean_period/done)))


