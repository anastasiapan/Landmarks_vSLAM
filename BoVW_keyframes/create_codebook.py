#!/usr/bin/env python
## Resolve cv2 and ROS conflict of python versions
import sys
sys.path.remove('/opt/ros/melodic/lib/python2.7/dist-packages')

## opencv and openni libraries
import cv2

import glob
import numpy as np

from codebook_fcn import *

## Path to keyframes
img_path=glob.glob("./325_keyframes/*.png")
vocab_path=glob.glob("./vocabulary_images/*.png")

## SURF features
des_size = 64 # descriptor size
hess_th = 500

new_vocabulary = False

if new_vocabulary:
    ## Detect image features and create visual vocabulary
    #features = extract_features(vocab_path, hess_th, des_size)
    #np.save('../V_SLAM_fcn/codebook/DTU_codebook', features)

    ## Cluster the detected SURF features
    features = np.load('../V_SLAM_fcn/codebook/DTU_codebook.npy')
    print(features.shape)
    v_words = cluster_vocabulary(features, num_clusters = 100)
    np.save('../V_SLAM_fcn/codebook/Visual_Words100', v_words)
else:
    v_words = np.load('../V_SLAM_fcn/codebook/Visual_Words100.npy')
    print(v_words.shape)

## Create keyframes histograms
codebook = create_codebook(img_path, des_size, hess_th, v_words)
np.save('../V_SLAM_fcn/codebook/histograms_b325_exp25', codebook.hist_cbook)
np.save('../V_SLAM_fcn/codebook/tfidf_histograms_b325_exp25', codebook.re_hist)

