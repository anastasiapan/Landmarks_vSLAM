 #!/usr/bin/env python
import numpy as np

global codebook
codebook =  np.load('./V_SLAM_fcn/codebook/Visual_Words100.npy') ## codebook

## Parameters
global img_proc_param
img_proc_param = {"hess_th": 500, ## Hessian threshold for SURF features
              "match_thres": 50, ## Matching threshold percentile for the tracker
              "exp_pct": 0.5} ## Percentage for bounding box expansion

## Spatial filter radius
global r
r = 2.

## Transformation matrix camera to imu
global T_cam_imu
T_cam_imu = np.array([0.375, 0, 0.18]).reshape(3,1)

## Visual data association parameters
global min_samples # Minimum number of samples to go through frame comparison 
min_samples = 10

global bow_thres # BoVW matching percentage threshold to declare a match for each comparison
bow_thres = 75

global loop_cl_thres # Loop closure detection -- Sampler threshold -- number of good matches/total matches
loop_cl_thres = 70