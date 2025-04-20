import h5py
import tifffile
import numpy as np

path = "/path/to/img_0.h5"

file = h5py.File(path, 'r')
image = np.array(file['SarImage'])
image = image[0] + 1j * image[1]
sar_image = image.astype(np.complex64, copy=False)
file.close()

image_abs = np.abs(sar_image)
image_abs += 1e-9
out_image = 10 * np.log10(image_abs / image_abs.max())

# Normalize to [0, 255] range for uint8
vmin, vmax = -40, 0
log_image = np.clip(out_image, vmin, vmax)
scaled_image = ((log_image - vmin) / (vmax - vmin)) * 255
out_image = scaled_image.astype('uint8')

# Save using TiffWriter
with tifffile.TiffWriter('gsi_output.tif') as tif:
    tif.write(out_image, photometric='minisblack')
