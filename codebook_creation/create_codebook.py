#!/usr/bin/env python
## opencv and openni libraries
import cv2

import glob
import numpy as np

from codebook_fcn import *

## Path to vocabulary images
vocab_path=glob.glob("./vocabulary_images/*.png")

## SURF features
des_size = 64 # descriptor size 64 if SURF/SIFT, 32 if ORB
hess_th = 500

## Number of Visual words - clusters of features for K-means
k = 100

## Detect image features and create visual vocabulary
features = extract_features(vocab_path, hess_th, des_size)
v_words = cluster_vocabulary(features, num_clusters = k)

## Save the codebook as a numpy array
np.save('../V_SLAM_fcn/codebook/Visual_Words100', v_words)