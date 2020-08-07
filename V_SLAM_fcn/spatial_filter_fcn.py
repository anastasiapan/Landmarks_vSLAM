 #!/usr/bin/env python
import numpy as np
import parameters
import rospy
import operator
from ROS_pub import *

class spatial_filter():

    def insert_cartographer_poses(self):
        for i, key in enumerate(self.lmk_obsv_poses):
            try:
                self.lmk_obsv_poses[key] = np.array(self.lmk_gp[i]).reshape(3,1)
            except:
                pass

        return self

    def pose_check(self, lmkPose, lmk_id):
        print(lmk_id)
        if lmk_id in self.lmk_obsv_poses:
            print('Landmark observed before')
            prevPose = self.lmk_obsv_poses[lmk_id]
            self.in_area = (lmkPose[0] - prevPose[0])**2 + (lmkPose[1] - prevPose[1])**2 <= parameters.r**2
            print((lmkPose[0] - prevPose[0])**2 + (lmkPose[1] - prevPose[1])**2) 
            
        obj_class = lmk_id.split('_')
        obj_class = obj_class[0]  
        for prev_landmark in self.lmk_obsv_poses:
            prev_id = prev_landmark.split('_')
            prev_id = prev_id[0]
            if lmk_id != prev_landmark and prev_id == obj_class:
                print('Landmark ' + prev_landmark + ' pose check: ')
                prevPose = self.lmk_obsv_poses[prev_landmark]
                print(prevPose)
                self.occupied = (lmkPose[0] - prevPose[0])**2 + (lmkPose[1] - prevPose[1])**2 <= parameters.r**2
                print((lmkPose[0] - prevPose[0])**2 + (lmkPose[1] - prevPose[1])**2)
                if self.occupied:
                    break

        print('in area: ' + str(self.in_area))
        print('occupied: ' + str(self.occupied))
        return self

    def filtering(self, lmk_publisher, correct_hist, tracked_histograms):
        for key in self.lmkObsv:
            if len(self.lmkObsv[key]) > parameters.min_samples:
                lmk = self.lmkObsv[key]
                gpose = np.array(list(map(operator.itemgetter(2), lmk)))
                gpose = gpose[:,:,0].transpose()
                gpose = np.true_divide(gpose.sum(1),(gpose!=0).sum(1)).reshape(3,1)
                print('- - - - - -- - -- - - - - -- - - - -- - - ')
                print('I saw ' + key + ' with a global pose of:')
                print(gpose)
                spatial_filter.pose_check(self,gpose, key)

                if self.in_area and not self.occupied:
                    print('Valid observation')
                    self.spatial_filter_text = 'Valid observation'
                    print(' Published id: ' + key)
                    for i in range(len(lmk)):
                        lmk_gpose = lmk[i][2]
                        lmk_list = landmark_pub(lmk[i], key)
                        lmk_publisher.publish(lmk_list)

                    if key not in self.lmk_obsv_poses:
                        print('not in cartographer - insert manually calculated global pose')
                        print(gpose)
                        self.lmk_obsv_poses[key] = gpose

                elif not self.in_area and not self.occupied:
                    print('matched object is out of range - unoccupied space - new object')  
                    self.spatial_filter_text = 'match out of range - free space - new object'                   
                    new_id = key.split('_')
                    self.lmk_id += 1
                    new_id = new_id[0] + '_' + str(self.lmk_id)
                    print(' Published id: ' + new_id)
                    for i in range(len(lmk)):
                        lmk_gpose = lmk[i][2]
                        lmk_list = landmark_pub(lmk[i], new_id)
                        lmk_publisher.publish(lmk_list)

                    if new_id not in self.lmk_obsv_poses:
                        print('not in cartographer - insert manually calculated global pose')
                        print(gpose)
                        self.lmk_obsv_poses[new_id] = gpose

                    self.lmk_copy[new_id] = self.lmkObsv[key]
                    self.keyframes_copy[new_id] = correct_hist[key]
                    self.tracked_hists[new_id] = tracked_histograms[key]

                    del self.lmk_copy[key]
                    del self.keyframes_copy[key]
                    del self.tracked_hists[key]

                else:
                    print('match out of range - space occupied - ignore')
                    self.spatial_filter_text = 'match out of range - space occupied - ignore'
                    del self.lmk_copy[key]
                    del self.keyframes_copy[key]
                    del self.tracked_hists[key]
                print('- - - - - -- - -- - - - - -- - - - -- - - ')
        
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