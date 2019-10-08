import numpy as np
import cv2

I = cv2.imread("im4.jpeg", cv2.IMREAD_COLOR).astype("float64")/255 

#print(I[500:550, 500:550, :])

(h, w, n_colors) = I.shape
dark = np.zeros([h, w])
PATCH_SIZE = 3

i, j = 0, 0

#print(h, w)
while i < h:
    j = 0
    while j < w:
        #print(i, j)
        '''tmp = I[i:(i+PATCH_SIZE), j:(j+PATCH_SIZE), 0]
        #print(tmp) 
        min_val_b = np.min( tmp )
        min_val_g = np.min( I[i:(i+PATCH_SIZE), j:(j+PATCH_SIZE), 1] )
        min_val_r = np.min( I[i:(i+PATCH_SIZE), j:(j+PATCH_SIZE), 2] )
        '''
        tmp = np.minimum( I[i:(i+PATCH_SIZE), j:(j+PATCH_SIZE), 0], I[i:(i+PATCH_SIZE), j:(j+PATCH_SIZE), 1] )
        tmp = np.minimum( tmp, I[i:(i+PATCH_SIZE), j:(j+PATCH_SIZE), 2])
        '''
        print(min_val_b, min_val_g, min_val_r)
        min_val = np.min([min_val_b, min_val_g, min_val_r]) 
        '''
        min_val = np.min(tmp)
        #print(min_val)
        x, y = tmp.shape
        dark[i:(i+x), j:(j+y)] = min_val
        j += PATCH_SIZE
    i += PATCH_SIZE

#dark = (dark * 255).astype(np.int8)
cv2.imshow("dark", dark)
cv2.waitKey()
