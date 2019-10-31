from PIL import Image
import numpy as np
img = Image.open('044_00142.png')
arr = np.asarray(img.convert('RGB'))
tosave = Image.fromarray(arr.astype('uint8'))
tosave.save('saved.png')