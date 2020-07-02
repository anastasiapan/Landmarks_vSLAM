#!/usr/bin/env python
## Resolve cv2 and ROS conflict of python versions
import sys
sys.path.remove('/opt/ros/melodic/lib/python2.7/dist-packages')

## opencv and openni libraries
import cv2
import numpy as np
import matplotlib.pyplot as plt

from codebook_fcn import calculate_cost

histograms = np.load('../V_SLAM_fcn/codebook/histograms_b326_exp25.npy', allow_pickle=True).reshape(1, 1)
histograms = histograms[0,0]

re_hist = np.load('../V_SLAM_fcn/codebook/tfidf_histograms_b326_exp25.npy', allow_pickle=True).reshape(1, 1)
re_hist = re_hist[0,0]

## random example for cone 1
hist_arr = np.array(list(histograms.values()))
hist_arr = hist_arr.reshape(hist_arr.shape[0],hist_arr.shape[2])

## random example for cone 1
re_hist_arr = np.array(list(re_hist.values()))

eucl, cos = calculate_cost(hist_arr)
eucl1, cos1 = calculate_cost(re_hist_arr)
fig, (ax1, ax2) = plt.subplots(1, 2)
ax1.imshow(cos, cmap='magma')
ax2.imshow(cos1, cmap='magma')
plt.show()

