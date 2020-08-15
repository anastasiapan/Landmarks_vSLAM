 #!/usr/bin/env python
import numpy as np

'''
Function: cloud_object_center(dmap, obj_cent)
Calculates 3D coordinates of the bounding box's center point

Inputs:
    - dmap : Depth map from the RGB-D camera
    - obj_cent : Center of bounding box formatted as a list: [x, y]

Returns/class members:
    - 3D coordinates of the center point in meters [x,y,z]
'''
def cloud_object_center(dmap, obj_cent):

    ## Camera clibration results
    fx = 496.02
    fy = 500.062
    cx = 328.67
    cy = 215.21

    ## 2D points to 3D Cartesian coordinates
    z = dmap[int(obj_cent[1])][int(obj_cent[0])]  # z
    x = -(int(obj_cent[0]) - cx) * z / fx  # x
    y = -(int(obj_cent[1]) - cy) * z / fy  # y

    pcl_point = [x/1000,y/1000,z/1000]

    return pcl_point