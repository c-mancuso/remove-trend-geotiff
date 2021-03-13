"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Remove regional trend from potential field data using least squares
fitting of 3D polynomial plane. For example, removing regional trend
from magnetic data. Uses GDAL package.

Written in Python 3.8.5
Christopher Mancuso
Last updated March 13 2021
IN:
	GeoTIFF image (works best if square dimensions)
	-filename (see raster_name below)(do not include file extention)
OUT:
	GeoTIFF image
	-filename_trend_removed.tiff
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
import gdal
import itertools
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve2d

raster_name='mag_example'

def polyfit2d(x, y, z, order=3):
    ncols = (order + 1)**2
    G = np.zeros((x.size, ncols))
    ij = itertools.product(range(order+1), range(order+1))
    for k, (i,j) in enumerate(ij):
        G[:,k] = x**i * y**j
    m, _, _, _ = np.linalg.lstsq(G, z)
    return m

def polyval2d(x, y, m):
    order = int(np.sqrt(len(m))) - 1
    ij = itertools.product(range(order+1), range(order+1))
    z = np.zeros_like(x)
    for a, (i,j) in zip(m, ij):
        z += a * x**i * y**j
    return z

output_raster_name=raster_name+'_trend_removed.tif'
tiff_file = gdal.Open(raster_name+'.tif')

geotransform = tiff_file.GetGeoTransform()
projection = tiff_file.GetProjection()
band = tiff_file.GetRasterBand(1)    
xsize = band.XSize
ysize = band.YSize

array = band.ReadAsArray()
tiff_file = None #close it
band = None #close it

"""
OPS

"""
print(array.shape)
width=array.shape[0]
height=array.shape[1]
num_data = width*height
x,y=[],[]

for i in range(width):			#grid
	for j in range(height):
		x.append(i)
		y.append(j)

x=np.asarray(x)
y=np.asarray(y)
z = array.flatten()

print(x)
# Fit a 3rd order, 2d polynomial
m = polyfit2d(x,y,z, 3)

# Evaluate it on a grid...
nx, ny = width, height
xx, yy = np.meshgrid(np.linspace(x.min(), x.max(), nx), np.linspace(y.min(), y.max(), ny))
zz = polyval2d(xx, yy, m)

plt.imshow(array,aspect='auto');plt.title('Input');plt.colorbar();plt.show()
plt.imshow(zz,aspect='auto');plt.title('Trend');plt.colorbar();plt.show()
out_array=(array[:height,:height]-zz[:height,:height])
plt.imshow(out_array,aspect='auto');plt.title('Residual');plt.colorbar();plt.show()


#5.
driver = gdal.GetDriverByName('GTiff')
new_tiff = driver.Create(output_raster_name,xsize,ysize,1,gdal.GDT_Int16)
new_tiff.SetGeoTransform(geotransform)
new_tiff.SetProjection(projection)
new_tiff.GetRasterBand(1).WriteArray(out_array)
new_tiff.FlushCache() #Saves to disk 
new_tiff = None #closes the file

