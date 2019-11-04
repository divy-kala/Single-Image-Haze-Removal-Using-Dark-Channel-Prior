import numpy as np
import cv2
import sys
import heapq


def Guidedfilter(im,p,r,eps):
    mean_I = cv2.boxFilter(im,cv2.CV_64F,(r,r)) 
    mean_p = cv2.boxFilter(p, cv2.CV_64F,(r,r)) 
    mean_Ip = cv2.boxFilter(im*p,cv2.CV_64F,(r,r)) 
    cov_Ip = mean_Ip - mean_I*mean_p 

    mean_II = cv2.boxFilter(im*im,cv2.CV_64F,(r,r)) 
    var_I   = mean_II - mean_I*mean_I 

    a = cov_Ip/(var_I + eps) 
    b = mean_p - a*mean_I 

    mean_a = cv2.boxFilter(a,cv2.CV_64F,(r,r)) 
    mean_b = cv2.boxFilter(b,cv2.CV_64F,(r,r)) 

    q = mean_a*im + mean_b 
    return q 

def TransmissionRefine(im,et):
    gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY) 
    gray = np.float64(gray)/255 
    r = 60 
    eps = 0.0001 
    t = Guidedfilter(gray,et,r,eps) 

    return t 

file_name = sys.argv[1]

src = cv2.imread(file_name, cv2.IMREAD_COLOR)
I = src.astype("float64")/255 

#print(I[500:550, 500:550, :])

(h, w, n_colors) = I.shape
dark = np.zeros([h, w])
PATCH_SIZE = 3 #must be an odd numebr

step = PATCH_SIZE // 2

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
omega = 0.75
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
transmission = TransmissionRefine(src, transmission)

scene_radiance = np.empty(I.shape, I.dtype)
for i in range(0,3):
	scene_radiance[:,:,i] = (I[:,:,i]-ATM_LIGHT[i])/transmission + ATM_LIGHT[i]
     
scene_radiance_e = scene_radiance + 0.04    
	     
cv2.imshow("scene_radiance_exposed", scene_radiance_e)
cv2.imwrite("scene_exposed_"+file_name, scene_radiance_e*255)
     
cv2.imshow("scene_radiance", scene_radiance)
#cv2.imwrite("scene"+file_name, scene_radiance*255)


cv2.imshow("trans", transmission)
cv2.imwrite("trans"+file_name, transmission*255)


#cv2.imwrite("dark"+file_name, dark*255)
cv2.imshow("dark", dark)

cv2.imshow(file_name, I)  

cv2.waitKey()
 
