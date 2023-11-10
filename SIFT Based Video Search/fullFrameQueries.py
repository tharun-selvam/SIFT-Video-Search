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

idx = np.load('idx_updated.npy')
centroids = np.load('centroids_updated.npy')

framesdir = 'frames/'
siftdir = 'sift/'
fnames = glob.glob(siftdir + '*.mat')
fnames = [i[-27:] for i in fnames]
fnames50 = fnames[:50]


def build_histogram_of_image(image_name, visual_words):
    """
        image name (string): represents the name of the image
        visual_words (ndarray (K, )): represents the centers of visual_words
    """

    image_path = siftdir + image_name
    mat = scipy.io.loadmat(image_path)
    X = mat['descriptors']

    K = visual_words.shape[0]
    histogram = np.zeros(K, dtype=int)

    temp = np.zeros((X.shape[0], K), dtype=float)
    for i in range(X.shape[0]):
        distance = []
        for j in range(visual_words.shape[0]):
            temp = X[i] - visual_words[j]
            temp = temp**2
            distance.append(np.sum(temp))
        a = np.argmin(distance)
        histogram[a] += 1

    normalised_histogram = histogram / np.sum(histogram)

    return histogram, normalised_histogram

#
# histogram_combined = np.empty((1, centroids.shape[0]), dtype=float)
# normalised_histogram_combined = np.empty((1, centroids.shape[0]), dtype=float)
#
# for i in range(len(fnames50)):
#
#     print(f'{i} images completed of {len(fnames50)}')
#
#     histogram, normalised_histogram = build_histogram_of_image(fnames50[i], centroids)
#
#     histogram = np.expand_dims(histogram, axis=0)
#     normalised_histogram = np.expand_dims(normalised_histogram, axis=0)
#
#     histogram_combined = np.concatenate((histogram_combined, histogram), axis=0)
#     normalised_histogram_combined = np.concatenate((normalised_histogram_combined, normalised_histogram), axis=0)
#
# histogram_combined = np.delete(histogram_combined, 0, 0)
# normalised_histogram_combined = np.delete(normalised_histogram_combined, 0, 0)
#
# np.save('histogram_images_combined_updated_50_images', histogram_combined)
# np.save('normalised_histogram_images_combined_updated_50_images', normalised_histogram_combined)


normalised_histogram_images_combined = np.load('normalised_histogram_images_combined_updated_50_images.npy')
histogram_images_combined = np.load('histogram_images_combined_updated_50_images.npy')


def get_the_best_frames(image_idx, num_images, normalised_histogram_combined):
    """
        image_idx: the index of the images wrt fnames list
        num_images: the num_of_images needed to display

        return: a list containing the best matched images
    """

    m = normalised_histogram_combined.shape[0]

    # broadcasted array containing the same size as normalised_histogram_combined
    broadcasted_array = np.tile(normalised_histogram_combined[image_idx], (m, 1))

    result_array = broadcasted_array * normalised_histogram_combined

    result_array = np.sum(result_array, axis=1)
    sorted_array = np.argsort(result_array)

    return sorted_array[:num_images]

image_idx = 35
matched_frames = get_the_best_frames(image_idx, 5, normalised_histogram_images_combined)


fname = siftdir + fnames50[image_idx]
mat = scipy.io.loadmat(fname)

imname = framesdir + fnames50[image_idx][:-4]
im = io.imread(imname)

fig = plt.figure()
ax = fig.add_subplot()
ax.imshow(im)
plt.show()

for i in matched_frames:
    fname = siftdir + fnames50[i]
    mat = scipy.io.loadmat(fname)

    imname = framesdir + fnames50[i][:-4]
    im = io.imread(imname)

    fig = plt.figure()
    ax = fig.add_subplot()
    ax.imshow(im)
    plt.show()

















