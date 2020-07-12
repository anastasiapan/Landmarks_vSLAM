 #!/usr/bin/env python3
import numpy as np
import rospy

from cartographer_ros_msgs.msg import LandmarkEntry, LandmarkList
frame_id = 'imu_frame'

## Transformation between camera and imu_frame
t = np.array([0.375, 0, 0.18]).reshape(3,1)

def landmark_pub(landmark_obsv, lmk_id):
    landmark_list = LandmarkList()
    landmark = LandmarkEntry()

    landmark.id = lmk_id
    lmk_obsv = landmark_obsv[lmk_id]
    pose = np.asarray(lmk_obsv[1]).reshape(3,1)
    timestamp = lmk_obsv[0]
    # rospy.Time.from_sec(int(ts) / 1e9 if integer

    ## Calculate position w.r.t. imu frame
    landmark.tracking_from_landmark_transform.position.x = pose[2] + t[0]
    landmark.tracking_from_landmark_transform.position.y = pose[0]
    landmark.tracking_from_landmark_transform.position.z = pose[1] + t[2]

    ## Rotation - not used
    landmark.tracking_from_landmark_transform.orientation.x = 0
    landmark.tracking_from_landmark_transform.orientation.y = 0
    landmark.tracking_from_landmark_transform.orientation.z = 0
    landmark.tracking_from_landmark_transform.orientation.w = 1

    landmark.translation_weight = 1.0
    landmark.rotation_weight = 0.0
    landmark_list.landmarks.append(landmark)

    landmark_list.header.stamp = landmark_obsv
    landmark_list.header.frame_id = frame_id
        
    return landmark_list