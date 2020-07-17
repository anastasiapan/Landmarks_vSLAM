 #!/usr/bin/env python
import numpy as np

global codebook
codebook =  np.load('./V_SLAM_fcn/codebook/Visual_Words100.npy') ## codebook

## Parameters
global parameters
parameters = {"hess_th": 500, ## Hessian threshold for SURF features
              "match_thres": 65, ## Matching threshold percentile for the tracker
              "exp_pct": 0.5} ## Percentage for bounding box expansion

## Spatial filter radius
global r
r = 1.

## Transformation matrix camera to imu
global T_cam_imu
T_cam_imu = np.array([0.375, 0, 0.18]).reshape(3,1)