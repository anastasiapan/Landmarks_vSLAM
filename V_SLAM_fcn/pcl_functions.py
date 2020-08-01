 #!/usr/bin/env python
import numpy as np

def depth_to_cloud(dmap, height, width):

    points = np.zeros((3,height,width), dtype=np.float32)

    ## Camera matrix
    fx = 542.87
    fy = 540.68
    cx = 322.9
    cy = 234.2

    for row in range(height):
        for col in range(width):
            points[0][row][col] = dmap[row][col] # z
            points[1][row][col] = -(col - cx) * dmap[row][col] / fx # x
            points[2][row][col] = -(row - cy) * dmap[row][col] / fy # y

    return points

def pcl(dmap):

    height = dmap.shape[0]
    height = dmap.shape[1]
    points = np.zeros((height, width,3), dtype=np.float32)

    ## Camera matrix
    fx = 542.87
    fy = 540.68
    cx = 322.9
    cy = 234.2

    col = np.arange(height)
    row = np.arange(width)
    points[:,:,0] = dmap
    points[:, :, 1] = (col - cx) * dmap / fx
    points[:, :, 2] = (row - cy) * dmap / fy


    return points

def objects_median(cloud, boxS, boxE, ratio):
    num = len(boxS) # number of objects
    med = []

    for i in range(num): # iterate for every object
        obj_cent = [(boxS[i][0] + boxE[i][0])/2, (boxS[i][1] + boxE[i][1])/2] # center pixel of the area the object is located

        x1 = int(obj_cent[0] - ratio*(obj_cent[0] - boxS[i][0]))
        y1 = int(obj_cent[1] - ratio*(obj_cent[1] - boxS[i][1]))
        x2 = int(obj_cent[0] + ratio*(boxE[i][0] - obj_cent[0]))
        y2 = int(obj_cent[1] + ratio*(boxE[i][1] - obj_cent[1]))
        median_z = np.median(cloud[0,y1:y2,x1:x2])
        median_x = np.median(cloud[1,y1:y2,x1:x2])
        median_y = np.median(cloud[2,y1:y2,x1:x2])

        med.append([median_x/1000,median_y/1000,median_z/1000])
    cx = 322.9
    cy = 234.2

    #obj_cent = [(boxS[i][0] + boxE[i][0])/2, (boxS[i][1] + boxE[i][1])/2]
    z = dmap[int(obj_cent[1])][int(obj_cent[0])]  # z
    x = -(int(obj_cent[0]) - cx) * z / fx  # x
    y = -(int(obj_cent[1]) - cy) * z / fy  # y
    return med

def cloud_object_center(dmap, obj_cent):

    ## Camera matrix
    ## primesense
    #fx = 542.87
    #fy = 540.68
    #cx = 322.9
    #cy = 234.2


    ## orbbec astra
    fx = 496.02
    fy = 500.062
    cx = 328.67
    cy = 215.21

    #obj_cent = [(boxS[i][0] + boxE[i][0])/2, (boxS[i][1] + boxE[i][1])/2]
    ## Extract a 20x20 box
    #boxx = 
    #limX = np.arange(int(obj_cent[0])-5, int(obj_cent[0])+5)
    #limY = np.arange(int(obj_cent[1])-5, int(obj_cent[1])+5)
    z = dmap[int(obj_cent[1])][int(obj_cent[0])]  # z
    #z = dmap[int(obj_cent[1])-5:int(obj_cent[1])+5][int(obj_cent[0])-5:int(obj_cent[0])+5]
    #z = z.reshape(z.size)
    x = -(int(obj_cent[0]) - cx) * z / fx  # x
    y = -(int(obj_cent[1]) - cy) * z / fy  # y
    #x = -(limX - cx) * z / fx  # x
    #y = -(limY - cy) * z / fy  # y

    #pcl_point = [np.mean(x)/1000,np.mean(y)/1000,np.mean(z)/1000]
    pcl_point = [x/1000,y/1000,z/1000]

    return pcl_point