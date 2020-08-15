#!/usr/bin/env python
import numpy as np
import cv2
import os
import time
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

'''
Function: extract_features(img_path, hess_th, des_size)

Inputs:
    - img_path : Path to vocabulary images
    - hess_th : Hessian matrix determinant threshold for SURF features
    - des_size : Features descrpitor size

Returns:
    - A numpy array that contains all of the detected features

'''
def extract_features(img_path, hess_th, des_size):
    features = np.empty((0, des_size))
    for imagepath in sorted(img_path):
        # Capture frame-by-frame
        frame = cv2.imread(str(imagepath))

        ## Extract SURF features
        surf = cv2.xfeatures2d.SURF_create(hess_th)
        kp, des = surf.detectAndCompute(frame, None)
        features = np.append(features, des, axis=0)

        ## Plot SURF features on image
        img2 = cv2.drawKeypoints(frame, kp, None)
        plt.imshow(img2), plt.show()

    return features

'''
Function: cluster_vocabulary(voc, num_clusters)

Inputs:
    - voc : Numpy array that contains the features to be clustered (num_features x descriptor_size)
    - num_clusters : Number of clusters for K-means

Returns:
    - The codebook in a numpy array form (num_visual_words x descriptor_size)

'''
def cluster_vocabulary(voc, num_clusters):

    ## Kmeans clustering
    kmeans = KMeans(n_clusters=num_clusters)
    kmeans.fit(voc)
    print("Clustering done!")

    words = kmeans.cluster_centers_

    return words