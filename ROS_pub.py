 #!/usr/bin/env python
import numpy as np
import rospy

from cartographer_ros_msgs.msg import LandmarkEntry, LandmarkList
from global_variables import r, T_cam_imu
frame_id = 'imu_frame'

def landmark_pub(lmk_obsv, lmk_id):
    landmark_list = LandmarkList()
    landmark = LandmarkEntry()

    landmark.id = lmk_id
    pose = np.asarray(lmk_obsv[1]).reshape(3,1)
    timestamp = lmk_obsv[0]
    # rospy.Time.from_sec(int(ts) / 1e9 if integer

    ## Calculate position w.r.t. imu frame
    landmark.tracking_from_landmark_transform.position.x = pose[2] + T_cam_imu[0]
    landmark.tracking_from_landmark_transform.position.y = pose[0]
    z = pose[1] + T_cam_imu[2]
    landmark.tracking_from_landmark_transform.position.z = z if z>0 else 0

    ## Rotation - not used
    landmark.tracking_from_landmark_transform.orientation.x = 0
    landmark.tracking_from_landmark_transform.orientation.y = 0
    landmark.tracking_from_landmark_transform.orientation.z = 0
    landmark.tracking_from_landmark_transform.orientation.w = 1

    landmark.translation_weight = 1.0
    landmark.rotation_weight = 0.0
    landmark_list.landmarks.append(landmark)

    landmark_list.header.stamp = timestamp
    landmark_list.header.frame_id = frame_id
        
    return landmark_list

def spatial_filter(lmk_poses_list, lmkPose, lmk_id):
    in_area = True
    occupied = False
    if lmk_id in lmk_poses_list:
        print('Landmark observed before')
        prevPose = lmk_poses_list[lmk_id]
        in_area = (lmkPose[0] - prevPose[0])**2 + (lmkPose[1] - prevPose[1])**2 <= r**2
        print((lmkPose[0] - prevPose[0])**2 + (lmkPose[1] - prevPose[1])**2)

    obj_class = lmk_id.split('_')
    obj_class = obj_class[0]
    for prev_landmark in lmk_poses_list:
        prev_id = prev_landmark.split('_')
        prev_id = prev_id[0]
        if lmk_id != prev_landmark and prev_id == obj_class:
            print('Landmark ' + prev_landmark + ' pose check: ')
            prevPose = lmk_poses_list[prev_landmark]
            print(prevPose)
            occupied = (lmkPose[0] - prevPose[0])**2 + (lmkPose[1] - prevPose[1])**2 <= r**2
            if occupied:
                break
            print((lmkPose[0] - prevPose[0])**2 + (lmkPose[1] - prevPose[1])**2)

    print('in area: ' + str(in_area))
    print('occupied: ' + str(occupied))

    return in_area, occupied