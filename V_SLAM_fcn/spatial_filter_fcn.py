 #!/usr/bin/env python
import numpy as np
import parameters
import rospy
import operator
from ROS_pub import *

'''
Class: spatial_filter
Sorts out outliers in case there is a false match

Inputs:
    - lmkObsv : Dictionary of recent landmarks observations
    - correct_hist : Keyframes histograms
    - tracked_histograms : Up-to-date tracked frames which contain landmarks
    - lmk_gp : global poses of landmarks calculated by the Cartographer system
    - lmk_obsv_poses : available global poses of landmarks
    - spatial_filter_text : Spatial filter result to be printed on screen
    - lmk_publisher : Landmarks to be sent to the Cartographer's pose graph
    - lmk_id : current counter for landmarks ids

Returns/class members:
    It updates : lmk_obsv_poses, lmk_id, correct_hist, tracked_histograms, spatial_filter_text
    - in_area : Boolean variable that indicates if the matched observation is within range
    - occupied : Boolean variable that indicates if the observed landmark is on a free space
    - lmk_publisher : Landmarks to be sent to the Cartographer's pose graph
'''
class spatial_filter():

    '''
    Class memeber function: insert_cartographer_poses
    Use landmarks global poses returned from the Cartographer if available
    '''
    def insert_cartographer_poses(self):
        for i, key in enumerate(self.lmk_obsv_poses):
            try:
                self.lmk_obsv_poses[key] = np.array(self.lmk_gp[i]).reshape(3,1)
            except:
                pass

        return self

    '''
    Class memeber function: pose_check
    Update boolean variables - when True:
        * in_area : Observation and match are close
        * occupied : Area of current observation occupied
    '''
    def pose_check(self, lmkPose, lmk_id):
        if lmk_id in self.lmk_obsv_poses:
            prevPose = self.lmk_obsv_poses[lmk_id]
            self.in_area = (lmkPose[0] - prevPose[0])**2 + (lmkPose[1] - prevPose[1])**2 <= parameters.r**2
            
        obj_class = lmk_id.split('_')
        obj_class = obj_class[0]  
        for prev_landmark in self.lmk_obsv_poses:
            prev_id = prev_landmark.split('_')
            prev_id = prev_id[0]
            if lmk_id != prev_landmark and prev_id == obj_class:
                prevPose = self.lmk_obsv_poses[prev_landmark]
                self.occupied = (lmkPose[0] - prevPose[0])**2 + (lmkPose[1] - prevPose[1])**2 <= parameters.r**2
                if self.occupied:
                    break
        return self

    '''
    Class memeber function: filtering
    Perform the Spatial filter checks and update the keyframes and tracks:
        * Loop closure: 
            Check if the global pose is close
            YES : Correct loop closure - send to pose graph
            NO : False loop closure - discard
        * New landmark: 
            Check if the area of the observation is free
            YES : New landmark
            NO : Not accepted as a new landmark - discard
    '''
    def filtering(self, lmk_publisher, correct_hist, tracked_histograms):
        for key in self.lmkObsv:
            if len(self.lmkObsv[key]) > parameters.min_samples:
                lmk = self.lmkObsv[key]
                gpose = np.array(list(map(operator.itemgetter(2), lmk)))
                gpose = gpose[:,:,0].transpose()
                gpose = np.true_divide(gpose.sum(1),(gpose!=0).sum(1)).reshape(3,1)
                spatial_filter.pose_check(self,gpose, key)

                if self.in_area and not self.occupied:
                    self.spatial_filter_text = 'Valid observation'
                    for i in range(len(lmk)):
                        lmk_gpose = lmk[i][2]
                        lmk_list = landmark_pub(lmk[i], key)
                        lmk_publisher.publish(lmk_list)

                    if key not in self.lmk_obsv_poses:
                        self.lmk_obsv_poses[key] = gpose

                elif not self.in_area and not self.occupied: 
                    self.spatial_filter_text = 'match out of range - free space - new object'                   
                    new_id = key.split('_')
                    self.lmk_id += 1
                    new_id = new_id[0] + '_' + str(self.lmk_id)
                    for i in range(len(lmk)):
                        lmk_gpose = lmk[i][2]
                        lmk_list = landmark_pub(lmk[i], new_id)
                        lmk_publisher.publish(lmk_list)

                    if new_id not in self.lmk_obsv_poses:
                        self.lmk_obsv_poses[new_id] = gpose

                    self.lmk_copy[new_id] = self.lmkObsv[key]
                    self.keyframes_copy[new_id] = correct_hist[key]
                    self.tracked_hists[new_id] = tracked_histograms[key]

                    del self.lmk_copy[key]
                    del self.keyframes_copy[key]
                    del self.tracked_hists[key]

                else:
                    self.spatial_filter_text = 'match out of range - space occupied - ignore'
                    del self.lmk_copy[key]
                    del self.keyframes_copy[key]
                    del self.tracked_hists[key]
        
        self.lmkObsv = self.lmk_copy
        self.lmkObsv = {}

    def __init__(self, lmkObsv, correct_hist, tracked_histograms, lmk_gp, lmk_obsv_poses, spatial_filter_text, lmk_publisher, lmk_id):
        
        self.lmk_copy = dict(lmkObsv)
        self.lmkObsv = lmkObsv
        self.keyframes_copy = dict(correct_hist)
        self.tracked_hists = dict(tracked_histograms)
        self.lmk_gp = lmk_gp
        self.lmk_obsv_poses = lmk_obsv_poses
        self.in_area = True
        self.occupied = False
        self.spatial_filter_text = spatial_filter_text
        self.lmk_id = lmk_id


        spatial_filter.insert_cartographer_poses(self)
        spatial_filter.filtering(self, lmk_publisher, correct_hist, tracked_histograms)