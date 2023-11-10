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


def shuffle_images_for_data(filename_list, num_images):
    '''
        filename_list (list) = the list containing the string of all the images' names
        num_images (int) =  the number of images that is required for extracting descriptors
    '''

    # Randomly reorder the indices of examples

    filename_list = np.array(filename_list)

    randidx = np.random.permutation(len(filename_list))

    # Take the first K examples as centroids
    filename_list = filename_list[randidx[:num_images]]

    return filename_list


#
# framesdir = 'frames/'
# siftdir = 'sift/'
#
# # Get a list of all the .mat file in that directory. There is one .mat file per image.
# # Preparation of data
# fnames = glob.glob(siftdir + '*.mat')
# fnames = [i[-27:] for i in fnames]
#
# fname1 = siftdir + fnames[0]
# mat1 = scipy.io.loadmat(fname1)
#
# X_data = mat1['descriptors']
#
# count = mat1['descriptors'].shape[0]
# image_decriptors_count_list = [count]
#
# lower_limit, upper_limit = 3, 4
# image_decriptors_count_list_updated = []
#
# mat_updated = np.empty((1, 128), dtype=float)
# count = 0
# num_of_images = 20
#
# # an array to store the corresponding image name and the descriptor index with respect to that image
# descriptor_info = np.empty((1, 2), dtype='<U25')
#
# # choosing random images from all the images
# # fnames = shuffle_images_for_data(fnames, num_of_images)
#
#
# for i in range(0, num_of_images):
#     fname = siftdir + fnames[i]
#     mat = scipy.io.loadmat(fname)
#     numfeats = mat['descriptors'].shape[0]
#
#
#     for j in range(numfeats):
#         if mat['scales'][j] > lower_limit and mat['scales'][j] < upper_limit:
#             desciptors_row_correct_shape = np.expand_dims(mat['descriptors'][j], 0)
#             mat_updated = np.concatenate((mat_updated, desciptors_row_correct_shape), axis=0)
#             temp_descriptor_info = np.array([[fnames[i], j]], dtype='<U35')
#             descriptor_info = np.concatenate((descriptor_info, temp_descriptor_info), axis=0)
#             count+=1
#
#     image_decriptors_count_list_updated.append(count)
#
#
# print(f'No.of descriptors is: {count} | No.of images scanned: {num_of_images} | Scale varies from {lower_limit} to {upper_limit}'   )
#
# # X_data now houses the required descriptors that fall in the scale range
# # also, we can use image_decriptors_count_list_updated to tell which image a
# # particular descriptor comes from which image.
# mat_updated = np.delete(mat_updated, 0, 0)
# descriptor_info = np.delete(descriptor_info, 0, 0)
# X_data = mat_updated
#
# np.save('descriptors_20_images.npy', X_data)
# np.save('descriptors_info_20_images.npy', descriptor_info)
#

def get_image_info(list_of_count, index):
    '''
        list_of_count: the list containing the cumulative frequency of the descriptors' count
        index: the index of the descriptor in the X_data

        return
        image_index = returns the index of the image's name as per fnames list
        descriptor_index = the index of the particular descriptor belonging to the corresponding image
    '''

    for i in range(len(list_of_count)):
        if index < list_of_count[i]:
            image_index = i
            if i == 0:
                descriptor_index = index
            else:
                descriptor_index = index - list_of_count[i]

    return image_index, descriptor_index


def find_closest_centroids(X, centroids):
    """
    Computes the centroid memberships for every example

    Args:
        X (ndarray): (m, n) Input values
        centroids (ndarray): (K, n) centroids

    Returns:
        idx (array_like): (m,) closest centroids

    """

    # Set K
    K = centroids.shape[0]

    # You need to return the following variables correctly
    idx = np.zeros(X.shape[0], dtype=int)

    ### START CODE HERE ###

    temp = np.zeros((X.shape[0], K), dtype=float)
    for i in range(X.shape[0]):
        distance = []
        for j in range(centroids.shape[0]):
            temp = X[i] - centroids[j]
            temp = temp ** 2
            distance.append(np.sum(temp))
        idx[i] = np.argmin(distance)

    ### END CODE HERE ###

    return idx


def compute_centroids(X, idx, K):
    """
    Returns the new centroids by computing the means of the
    data points assigned to each centroid.

    Args:
        X (ndarray):   (m, n) Data points
        idx (ndarray): (m,) Array containing index of closest centroid for each
                       example in X. Concretely, idx[i] contains the index of
                       the centroid closest to example i
        K (int):       number of centroids

    Returns:
        centroids (ndarray): (K, n) New centroids computed
    """

    # Useful variables
    m, n = X.shape

    # You need to return the following variables correctly
    centroids = np.zeros((K, n), dtype=float)

    ### START CODE HERE ###
    for i in range(K):
        # count = 0.0
        # for j in range(m):
        #     if (idx[j] == i):
        #         # print(centroids[i])
        #         centroids[i] = centroids[i] + X[j]
        #         # print(centroids[i])
        #         count += 1.0
        # # print(centroids[i])
        # centroids[i] = centroids[i] / count
        points = X[idx == i]
        centroids[i] = np.mean(points, axis=0)

    ### END CODE HERE ##

    return centroids


def run_kMeans(X, initial_centroids, max_iters=10):
    """
    Runs the K-Means algorithm on data matrix X, where each row of X
    is a single example
    """

    # Initialize values
    m, n = X.shape
    K = initial_centroids.shape[0]
    centroids = initial_centroids
    previous_centroids = centroids
    idx = np.zeros(m)
    plt.figure(figsize=(8, 6))

    # Run K-Means
    for i in range(max_iters):
        # Output progress
        print("K-Means iteration %d/%d" % (i, max_iters - 1))

        # For each example in X, assign it to the closest centroid
        # print(idx)
        idx = find_closest_centroids(X, centroids)

        # Given the memberships, compute new centroids
        centroids = compute_centroids(X, idx, K)
    return centroids, idx


def kMeans_init_centroids(X, K):
    """
    This function initializes K centroids that are to be
    used in K-Means on the dataset X

    Args:
        X (ndarray): Data points
        K (int):     number of centroids/clusters

    Returns:
        centroids (ndarray): Initialized centroids
    """

    # Randomly reorder the indices of examples
    randidx = np.random.permutation(X.shape[0])

    # Take the first K examples as centroids
    centroids = X[randidx[:K]]

    return centroids


# # k-means clustering
# K = 100
# max_iters = 200
# initial_centroids = kMeans_init_centroids(X_data, K)
# centroids, idx = run_kMeans(X_data, initial_centroids, max_iters)
#
# np.save('centroids_updated.npy', centroids)
# np.save('idx_updated.npy', idx)

def see_patches_together(cluster_num, descriptor_info, idx):
    """
        cluster_num (int): specifies the cluster number to be viewed
        descriptor_info (ndarray): contains the information of the descriptor like image name and descriptor number
        idx (ndarray): contains the cluster numbers of the corresponding descriptor
    """

    count = 1
    images = []
    for j in range(len(idx)):
        if idx[j] == cluster_num:
            # print all the descriptors with j
            image_name, patch_num = descriptor_info[j, 0], descriptor_info[j, 1]
            patch_num = int(patch_num)
            print(f'Cluster number: {cluster_num} | image name: {image_name} | point no: {count}')
            fname = siftdir + image_name
            mat = scipy.io.loadmat(fname)

            imname = framesdir + image_name[:-4]
            im = io.imread(imname)

            img_patch = getPatchFromSIFTParameters(mat['positions'][patch_num, :], mat['scales'][patch_num],
                                                   mat['orients'][patch_num], rgb2gray(im))
            plt.imshow(img_patch, cmap=cm.Greys_r)

            count += 1
            images.append(img_patch)

    # Calculate the number of rows and columns for the subplot grid
    num_images = len(images)
    rows = int(num_images ** 0.5)  # Square root of the number of images rounded down
    cols = (num_images + rows - 1) // rows  # Round up the number of columns

    # Create a grid of subplots
    fig, axs = plt.subplots(rows, cols)

    # Loop through the subplots and display the images
    for i, ax in enumerate(axs.flat):
        if i < num_images:
            ax.imshow(images[i])
            ax.axis('off')  # Turn off axis labels and ticks for cleaner display
        else:
            ax.axis('off')  # Turn off empty subplots (if there are fewer images than subplots)

    # Adjust the layout and spacing of the subplots
    plt.tight_layout()

    # Show the plot with all the images
    plt.show()


def see_individual_patches(cluster_num, descriptor_info, idx):
    """
        cluster_num (int): specifies the cluster number to be viewed
        descriptor_info (ndarray): contains the information of the descriptor like image name and descriptor number
        idx (ndarray): contains the cluster numbers of the corresponding descriptor
    """
    count = 1
    images = []
    for j in range(len(idx)):
        if idx[j] == cluster_num:
            # print all the descriptors with j
            image_name, patch_num = descriptor_info[j, 0], descriptor_info[j, 1]
            patch_num = int(patch_num)
            print(f'Cluster number: {cluster_num} | image no: {image_name} | point no: {count}')
            fname = siftdir + image_name
            mat = scipy.io.loadmat(fname)

            imname = framesdir + image_name[:-4]
            im = io.imread(imname)

            print('imname = %s contains %d total features, each of dimension %d' % (
            imname, numfeats, mat['descriptors'].shape[1]))

            fig = plt.figure()
            ax = fig.add_subplot()
            ax.imshow(im)
            coners = displaySIFTPatches(mat['positions'][patch_num:patch_num + 1, :],
                                        mat['scales'][patch_num:patch_num + 1, :],
                                        mat['orients'][patch_num:patch_num + 1, :])

            for j in range(len(coners)):
                ax.plot([coners[j][0][1], coners[j][1][1]], [coners[j][0][0], coners[j][1][0]], color='g',
                        linestyle='-', linewidth=1)
                ax.plot([coners[j][1][1], coners[j][2][1]], [coners[j][1][0], coners[j][2][0]], color='g',
                        linestyle='-', linewidth=1)
                ax.plot([coners[j][2][1], coners[j][3][1]], [coners[j][2][0], coners[j][3][0]], color='g',
                        linestyle='-', linewidth=1)
                ax.plot([coners[j][3][1], coners[j][0][1]], [coners[j][3][0], coners[j][0][0]], color='g',
                        linestyle='-', linewidth=1)
            ax.set_xlim(0, im.shape[1])
            ax.set_ylim(0, im.shape[0])
            plt.gca().invert_yaxis()

            count += 1

            # List of images (you can replace these with your own images)
            images.append(im)

            # Calculate the number of rows and columns for the subplot grid
    num_images = len(images)
    rows = int(num_images ** 0.5)  # Square root of the number of images rounded down
    cols = (num_images + rows - 1) // rows  # Round up the number of columns

    # Create a grid of subplots
    fig, axs = plt.subplots(rows, cols)

    # Loop through the subplots and display the images
    for i, ax in enumerate(axs.flat):
        if i < num_images:
            ax.imshow(images[i])
            ax.axis('off')  # Turn off axis labels and ticks for cleaner display
        else:
            ax.axis('off')  # Turn off empty subplots (if there are fewer images than subplots)

    # Adjust the layout and spacing of the subplots
    plt.tight_layout()

    # Show the plot with all the images
    plt.show()

# print('The K-Means Clustering is complete and ready for viewing....')
# print(f'No.of visual words is {K}')
#
# want_to_continue = 'y'
# while want_to_continue == 'y':
#
#     cluster_no = int(input(f'\nEnter the cluster number you would like to see (0-{K-1}): '))
#     see_patches = input(f'If you want to see all patches at the same time enter y or else enter n: ')
#     print()
#
#     if see_patches == 'y':
#         see_patches_together(cluster_no, descriptor_info, idx)
#     elif see_patches == 'n':
#         see_individual_patches(cluster_no, descriptor_info, idx)
#     else:
#         print('Enter either y or n\n')
#
#     want_to_continue = input('If you want to see more clusters type y or else type n: ')


# # displaying the images
# for i in range(K):
#     count = 1
#     images = []
#     for j in range(len(idx)):
#         if idx[j] == i:
#             # print all the descriptors with j
#             image_name, patch_num = descriptor_info[j, 0], descriptor_info[j, 1]
#             patch_num = int(patch_num)
#             print(f'Cluster number: {i} | image no: {image_name} | point no: {count}')
#             fname = siftdir + image_name
#             mat = scipy.io.loadmat(fname)
#
#             imname = framesdir + image_name[:-4]
#             im = io.imread(imname)
#
#             img_patch = getPatchFromSIFTParameters(mat['positions'][patch_num,:], mat['scales'][patch_num], mat['orients'][patch_num], rgb2gray(im))
#             plt.imshow(img_patch, cmap=cm.Greys_r)
#             # plt.show()
#
#             # print('imname = %s contains %d total features, each of dimension %d' %(imname, numfeats, mat['descriptors'].shape[1]))
#             #
#             # fig = plt.figure()
#             # ax = fig.add_subplot()
#             # ax.imshow(im)
#             # coners = displaySIFTPatches(mat['positions'][patch_num:patch_num+1,:], mat['scales'][patch_num:patch_num+1,:], mat['orients'][patch_num:patch_num+1,:])
#             #
#             # for j in range(len(coners)):
#             #     ax.plot([coners[j][0][1], coners[j][1][1]], [coners[j][0][0], coners[j][1][0]], color='g', linestyle='-', linewidth=1)
#             #     ax.plot([coners[j][1][1], coners[j][2][1]], [coners[j][1][0], coners[j][2][0]], color='g', linestyle='-', linewidth=1)
#             #     ax.plot([coners[j][2][1], coners[j][3][1]], [coners[j][2][0], coners[j][3][0]], color='g', linestyle='-', linewidth=1)
#             #     ax.plot([coners[j][3][1], coners[j][0][1]], [coners[j][3][0], coners[j][0][0]], color='g', linestyle='-', linewidth=1)
#             # ax.set_xlim(0, im.shape[1])
#             # ax.set_ylim(0, im.shape[0])
#             # plt.gca().invert_yaxis()
#
#             count += 1
#             # plt.show()
#
#             # List of images (you can replace these with your own images)
#             # images = [image1, image2, image3, ...]  # Replace with your images or image paths
#             images.append(img_patch)
#
#             # Calculate the number of rows and columns for the subplot grid
#     num_images = len(images)
#     rows = int(num_images**0.5)  # Square root of the number of images rounded down
#     cols = (num_images + rows - 1) // rows  # Round up the number of columns
#
#     # Create a grid of subplots
#     fig, axs = plt.subplots(rows, cols)
#
#     # Loop through the subplots and display the images
#     for i, ax in enumerate(axs.flat):
#         if i < num_images:
#             ax.imshow(images[i])
#             ax.axis('off')  # Turn off axis labels and ticks for cleaner display
#         else:
#             ax.axis('off')  # Turn off empty subplots (if there are fewer images than subplots)
#
#     # Adjust the layout and spacing of the subplots
#     plt.tight_layout()
#
#     # Show the plot with all the images
#     plt.show()
#
