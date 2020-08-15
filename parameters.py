 #!/usr/bin/env python
import numpy as np

global codebook
codebook =  np.load('./V_SLAM_fcn/codebook/Visual_Words100.npy') ## codebook

## Parameters
'''
img_proc_param : Parameters for image processing
	* hess_th : Hessian matrix determinant threshold
	* match_thres : Matching threshold to track objects
	* exp_pct : Percentage to expand x and y coordinates of the bounding box to find features
'''
global img_proc_param
img_proc_param = {"hess_th": 500, ## Hessian threshold for SURF features
              "match_thres": 50, ## Matching threshold percentile for the tracker
              "exp_pct": 0.5} ## Percentage for bounding box expansion

## Spatial filter radius
global r
r = 1.

## Transformation matrix camera to imu
global T_cam_imu
T_cam_imu = np.array([0.375, 0, 0.18]).reshape(3,1)

## Data association parameters
## Minimum number of samples to go through frame comparison
global min_samples  
min_samples = 10

## BoVW matching percentage threshold to declare a match for each comparison
global bow_thres 
bow_thres = 85

## Loop closure detection -- Sampler threshold -- number of good matches/total matches
global loop_cl_thres 
loop_cl_thres = 70