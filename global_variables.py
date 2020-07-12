 #!/usr/bin/env python3
import numpy as np

global codebook
codebook =  np.load('./V_SLAM_fcn/codebook/Visual_Words100.npy') ## codebook

## Parameters
global parameters
parameters = {"hess_th": 500, ## Hessian threshold for SURF features
              "match_thres": 40, ## Matching threshold percentile for the tracker
              "exp_pct": 0.5} ## Percentage for bounding box expansion