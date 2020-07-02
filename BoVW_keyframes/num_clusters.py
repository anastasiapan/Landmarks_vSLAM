#!/usr/bin/env python
## Resolve cv2 and ROS conflict of python versions
import sys
sys.path.remove('/opt/ros/melodic/lib/python2.7/dist-packages')

## opencv and openni libraries
import cv2
from primesense import openni2
from primesense import _openni2 as c_api

import glob
import numpy as np
from sklearn.cluster import KMeans
from sklearn.datasets.samples_generator import make_blobs

import matplotlib.pyplot as plt

features = np.load('../V_SLAM_fcn/codebook/DTU_codebook.npy') # Bag of visual Words

## Choose some values using the elbow rule
Sum_of_squared_distances = []
print(features.shape[0])
K = range(10, 2000,200)

for k in K:
    print(k)
    km = KMeans(n_clusters=k)
    km = km.fit(features)
    Sum_of_squared_distances.append(km.inertia_)

    plt.plot(k, km.inertia_, 'bx-')
    plt.xlabel('k')
    plt.title('Optimal k using elbow rule')
    plt.pause(0.05)

plt.show()

cv2.destroyAllWindows()
