#!/usr/bin/env python
import rospy
from geometry_msgs.msg import PoseStamped
from visualization_msgs.msg import MarkerArray, Marker
from scipy.spatial.transform import Rotation as rot
import numpy as np

class robot_global_pose:
    def __init__(self):
        self.glob_pose = PoseStamped()
        self.trans = np.array([])
        rospy.Subscriber('robot_global_pose', PoseStamped, self.pose_callback)

    def pose_callback(self,pose):
        self.glob_pose = pose

    def trans_mat(self):
        #curr_pose = t_stamp == self.glob_pose.header.stamp
        invalid_quat = self.glob_pose.pose.orientation.x == 0 and self.glob_pose.pose.orientation.y == 0 and self.glob_pose.pose.orientation.z == 0 and self.glob_pose.pose.orientation.w == 0
        if invalid_quat:
            self.trans = np.array([])
        else:
            quat = [self.glob_pose.pose.orientation.x, self.glob_pose.pose.orientation.y, self.glob_pose.pose.orientation.z, self.glob_pose.pose.orientation.w]
            eul = rot.from_quat(quat).as_euler('zxy', degrees=False)
            theta = eul[0]
            self.trans = np.array([[np.cos(theta), -np.sin(theta), self.glob_pose.pose.position.x], [np.sin(theta), np.cos(theta), self.glob_pose.pose.position.y], [0, 0, 1]])

class landmarks_global_pose:
    def __init__(self):
        self.landmarks = MarkerArray()
        self.landmarks_global_poses = []
        rospy.Subscriber('landmark_poses_list', MarkerArray, self.global_pose_callback)

    def global_pose_callback(self,msg):
        self.landmarks = msg

    def lmk_gPoses_list(self):
        lmk_obsv_tot = len(self.landmarks.markers)
        self.landmarks_global_poses = [[0]*1]*lmk_obsv_tot
        for i in range(lmk_obsv_tot):
            lmk_cartographerID = self.landmarks.markers[i].id
            self.landmarks_global_poses[lmk_cartographerID] = [self.landmarks.markers[i].pose.position.x, self.landmarks.markers[i].pose.position.y, 1]
