from PIL import Image
import numpy as np

img = Image.open('bochum_000000_001519_gtFine_labelIds.jpg')
arr = np.asarray(img)[:,:,0]
# arr = arr[:,:,np.newaxis]
arr = arr.copy()
ignore_label = 255
label_mapping = {-1: ignore_label, 0: ignore_label, 
                  1: ignore_label, 2: ignore_label, 
                  3: ignore_label, 4: ignore_label, 
                  5: ignore_label, 6: ignore_label, 
                  7: 0, 8: 1, 9: ignore_label, 
                  10: ignore_label, 11: 2, 12: 3, 
                  13: 4, 14: ignore_label, 15: ignore_label, 
                  16: ignore_label, 17: 5, 18: ignore_label, 
                  19: 6, 20: 7, 21: 8, 22: 9, 23: 10, 24: 11,
                  25: 12, 26: 13, 27: 14, 28: 15, 
                  29: ignore_label, 30: ignore_label, 
                  31: 16, 32: 17, 33: 18, 35:ignore_label}
copy = arr.copy()

for k, v in label_mapping.items():
	arr[copy==k] = v

label = np.zeros((arr.shape[0], arr.shape[1], 19), dtype=float)
for i in range(19):
      label[:, :, i][arr == i] = 1




print(label[0])