import h5py
import numpy as np
import matplotlib.pyplot as plt
import cv2

path = "/home/public/sar/outputs/img_120.h5"

file = h5py.File(path, 'r')
image = np.array(file['SarImage'])
image = image[0] + 1j * image[1]
sar_image = image.astype(np.complex64, copy=False)
file.close()

image_abs = np.abs(sar_image)
image_abs += 1e-9
out_image = 10 * np.log10(image_abs / image_abs.max())
plt.imsave('gsi_output.png', out_image, cmap=plt.cm.Greys_r, vmin=-40, vmax=0)

img = cv2.imread('gsi_output.png', 0)
equ = cv2.equalizeHist(img)
# Pay attention to change 'dsize' according to the image shape
equ = cv2.resize(equ, (1000, 1000), interpolation=cv2.INTER_AREA)

cv2.imshow('Equalized Image', equ)
cv2.waitKey(0)  # Wait for a key press to close the window
cv2.destroyAllWindows()  # Close the window
