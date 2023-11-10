• loadDataExample.py (ipynb): Run this first and make sure you understand the data format. It is
a script that shows a loop of data files, and how to access each descriptor. It also shows how to use
some of the other functions below. You can also run the code using the jupyter notebook.
• displaySIFTPatches.py: given SIFT descriptor info, it draws the patches on top of an image
• getPatchFromSIFTParameters.py: given SIFT descriptor info, it extracts the image patch itself and
returns as a single image
• selectRegion.py: given an image and list of feature positions, it allows a user to draw a polygon
showing a region of interest, and then returns the indices within the list of positions that fell within
the polygon.
• dist2.py: a fast implementation of computing pairwise distances between two matrices for which each
row is a data point

Tips: overview of framework requirements
The basic framework will require these components:
• Compute nearest raw SIFT descriptors. Use the Euclidean distance between SIFT descriptors to
determine which are nearest among two images’ descriptors. That is, “match” features from one image
to the other, without quantizing to visual words.
• Form a visual vocabulary. Cluster a large, representative random sample of SIFT descriptors from
some portion of the frames using k-means. Let the k centers be the visual words. The value of k
is a free parameter; for this data something like k=1500 should work, but feel free to play with this
parameter [For Matlab, see Matlab’s kmeans function, or provided kmeansML.m code. For Python, see
kmeans function in sklearn, scipy, opencv etc.]. Note: You may run out of memory if you use
all the provided SIFT descriptors to build the vocabulary.
• Map a raw SIFT descriptor to its visual word. The raw descriptor is assigned to the nearest visual
word. [see provided dist2.m (Matlab) or dist2.py (Python) code for fast distance computations].
• Map an image’s features into its bag-of-words histogram. The histogram for image Ij is a k-dimensional
vector:
F(Ij ) = [freq1,j , freq2,j , ..., freqk,j ]
where each entry freqi,j counts the number of occurrences of the i-th visual word in that image, and
k is the number of total words in the vocabulary. In other words, a single image’s list of n SIFT
descriptors yields a k-dimensional bag of words histogram.
• Compute similarity scores. Compare two bag-of-words histograms using the normalized scalar product:
S(Ii, Ij ) = F(Ii) · F(Ij ) / ||F(Ii)|| ||F(Ij)|| = 1/||F(Ii)|| ||F(Ij)|| . Σm=1,k (FREQ m,i . FREQ m,j)
where S() is the similarity score. ||F(Ii)|| is the L2 norm of F(Ii).
• Sort the similarity scores between a query histogram and the histograms associated with the rest of
the images in the video. Pull up the images associated with the M most similar examples.