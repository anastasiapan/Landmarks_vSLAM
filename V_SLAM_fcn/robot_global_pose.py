#!/usr/bin/env python
import rospy
from geometry_msgs.msg import PoseStamped
from scipy.spatial.transform import Rotation as rot
import numpy as np

class robot_global_pose:

    def __init__(self):
        self.glob_pose = PoseStamped()
        self.trans = np.array([])
        #rospy.init_node('rob_pose_sub')
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