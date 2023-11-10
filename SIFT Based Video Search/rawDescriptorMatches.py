import numpy as np
import scipy.io
import glob
from scipy import misc
import matplotlib.pyplot as plt
from displaySIFTPatches import displaySIFTPatches
from selectRegion import roipoly
from getPatchFromSIFTParameters import getPatchFromSIFTParameters
from skimage.color import rgb2gray
import matplotlib.cm as cm
from skimage import io
import pylab as pl
import pdb

# specific frame dir and siftdir
framesdir = 'frames/'
siftdir = 'sift/'

fname1 = siftdir + 'friends_0000000289.jpeg.mat'
fname2 = siftdir + 'friends_0000000290.jpeg.mat'
imname1 = framesdir + 'friends_0000000289.jpeg'

mat = scipy.io.loadmat(fname1)
numfeats = mat['descriptors'].shape[0]
print(f'Numfeats1: {numfeats}')

im = io.imread(imname1)

# now show how to select a subset of the features using polygon drawing.
print('use the mouse to draw a polygon, right click to end it')
pl.imshow(im)
MyROI = roipoly(roicolor='r')
Ind = MyROI.getIdx(im, mat['positions'])

# Ind contains the indices of the SIFT features whose centers fall
# within the selected region of interest.
# Note that these indices apply to the *rows* of 'descriptors' and
# 'positions', as well as the entries of 'scales' and 'orients'
# now display the same image but only in the polygon.

fig=plt.figure()
bx=fig.add_subplot(111)
bx.imshow(im)
coners = displaySIFTPatches(mat['positions'][Ind,:], mat['scales'][Ind,:], mat['orients'][Ind,:])

for j in range(len(coners)):
    bx.plot([coners[j][0][1], coners[j][1][1]], [coners[j][0][0], coners[j][1][0]], color='g', linestyle='-', linewidth=1)
    bx.plot([coners[j][1][1], coners[j][2][1]], [coners[j][1][0], coners[j][2][0]], color='g', linestyle='-', linewidth=1)
    bx.plot([coners[j][2][1], coners[j][3][1]], [coners[j][2][0], coners[j][3][0]], color='g', linestyle='-', linewidth=1)
    bx.plot([coners[j][3][1], coners[j][0][1]], [coners[j][3][0], coners[j][0][0]], color='g', linestyle='-', linewidth=1)
bx.set_xlim(0, im.shape[1])
bx.set_ylim(0, im.shape[0])
plt.gca().invert_yaxis()
plt.show()

##Euclidian distance
mat2 = scipy.io.loadmat(fname2)
numfeats2 = mat2['descriptors'].shape[0]
print(f'Numfeats2: {numfeats}')


imname2 = framesdir + 'friends_0000000290.jpeg'
im = io.imread(imname2)

print('Image 2')
pl.imshow(im)

loss = np.ones((len(Ind), numfeats2), dtype='float')
Ind2 = []

for j in range(len(Ind)):
    for k in range(numfeats2):
        loss_temp = mat['descriptors'][Ind[j], :] - mat2['descriptors'][k, :]
        loss_temp = loss_temp**2

        loss[j, k] = np.sum(loss_temp)
    print(f'Line {j}, Min loss = {np.min(loss[j, :])} for the descriptor index: {np.argmin(loss[j, :])}')
    Ind2.append(np.argmin(loss[j, :]))


fig=plt.figure()
bx=fig.add_subplot(111)
bx.imshow(im)
coners = displaySIFTPatches(mat2['positions'][Ind2,:], mat2['scales'][Ind2,:], mat2['orients'][Ind2,:])

for j in range(len(coners)):
    bx.plot([coners[j][0][1], coners[j][1][1]], [coners[j][0][0], coners[j][1][0]], color='g', linestyle='-', linewidth=1)
    bx.plot([coners[j][1][1], coners[j][2][1]], [coners[j][1][0], coners[j][2][0]], color='g', linestyle='-', linewidth=1)
    bx.plot([coners[j][2][1], coners[j][3][1]], [coners[j][2][0], coners[j][3][0]], color='g', linestyle='-', linewidth=1)
    bx.plot([coners[j][3][1], coners[j][0][1]], [coners[j][3][0], coners[j][0][0]], color='g', linestyle='-', linewidth=1)
bx.set_xlim(0, im.shape[1])
bx.set_ylim(0, im.shape[0])
plt.gca().invert_yaxis()
plt.show()

