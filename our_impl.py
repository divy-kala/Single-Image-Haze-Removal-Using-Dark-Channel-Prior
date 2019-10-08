import numpy as np
import cv2
import sys
import heapq

file_name = sys.argv[1]


I = cv2.imread(file_name, cv2.IMREAD_COLOR).astype("float64")/255 

#print(I[500:550, 500:550, :])

(h, w, n_colors) = I.shape
dark = np.zeros([h, w])
PATCH_SIZE = 15 #must be an odd numebr

step = PATCH_SIZE >> 1

num_pixels = h * w
topvalues = max(num_pixels // 10000, 10)

HEAP = []
for i in range(h):
	for j in range(w):
		left_limit = max(j - step, 0)
		right_limit = min(j + step, w-1)
		top_limit = max(i - step, 0)
		bottom_limit = min(i+step, h-1)
		B = I[top_limit:bottom_limit, left_limit:right_limit, 0]
		G = I[top_limit:bottom_limit, left_limit:right_limit, 1]
		R = I[top_limit:bottom_limit, left_limit:right_limit, 2]
		
		min_color = np.minimum(np.minimum(B, G), R)
		min_val = np.min(min_color)
		dark[i, j] = min_val
		if len(HEAP) == topvalues:
			if HEAP[0][0] < min_val:
				heapq.heappop(HEAP)
				heapq.heappush(HEAP, (min_val, i, j))
		else:
			heapq.heappush(HEAP, (min_val, i, j))


#ATM_LIGHT = max((sum(I[i, j, :]) for _, i, j in HEAP))
max_val = -1
#optimization: only need to check n/2 leaves, because leaves of a heap contain the max value
#ATM_LIGHT = np.zeros([1, 3])
for _, i, j in HEAP:
	sm = sum(I[i, j, :])
	if sm > max_val:
		max_val = sm
		ATM_LIGHT = I[i, j, :]
		
				
#transmission
omega = 0.95
temp_I = np.empty(I.shape,I.dtype)
for i in range(3):
	temp_I[:, :, i] = I[:, :, i]/ATM_LIGHT[i]

dark_trans = np.zeros([h, w])	
for i in range(h):
	for j in range(w):
		left_limit = max(j - step, 0)
		right_limit = min(j + step, w-1)
		top_limit = max(i - step, 0)
		bottom_limit = min(i+step, h-1)
		B = temp_I[top_limit:bottom_limit, left_limit:right_limit, 0]
		G = temp_I[top_limit:bottom_limit, left_limit:right_limit, 1]
		R = temp_I[top_limit:bottom_limit, left_limit:right_limit, 2]
		
		min_color = np.minimum(np.minimum(B, G), R)
		min_val = np.min(min_color)
		dark_trans[i, j] = min_val

transmission = 1 - omega*dark_trans
t0 = 0.1
#ATM_LIGHT = np.array([ATM_LIGHT])

#scene_radiance = ( I - ATM_LIGHT ) / cv2.max(t0, transmission )  + ATM_LIGHT	

scene_radiance = np.empty(I.shape, I.dtype)
for i in range(0,3):
	scene_radiance[:,:,i] = (I[:,:,i]-ATM_LIGHT[i])/transmission + ATM_LIGHT[i]     
cv2.imshow(file_name, I)        
cv2.imshow("scene_radiance", scene_radiance)
cv2.imshow("dark", dark)
cv2.imshow("trans", transmission)
cv2.waitKey()
 
