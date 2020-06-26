 
#!/usr/bin/env python
import numpy as np
import rospy

from cartographer_ros_msgs.msg import LandmarkEntry, LandmarkList
frame_id = 'imu_frame'

## Transformation between camera and imu_frame
t = np.array([0.375, 0, 0.18]).reshape(3,1)
#rot = np.array([[0,1,0],[0,0,1],[1,0,0]])

def landmark_pub(med, ids,timestamp):
    landmark_list = LandmarkList()

    if len(ids) == 0:

        landmark_list = LandmarkList(rospy.Header(stamp=timestamp, frame_id=frame_id), [])

    else:
        for i in range(len(ids)):
            landmark = LandmarkEntry()
            landmark.id = ids[i]
            pose = np.asarray(med[i]).reshape(3,1)
            med_imu_x = pose[2] + t[0]
            med_imu_y = pose[0]
            med_imu_z = pose[1] + t[2]
            landmark.tracking_from_landmark_transform.position.x = med_imu_x
            landmark.tracking_from_landmark_transform.position.y = med_imu_y
            landmark.tracking_from_landmark_transform.position.z = med_imu_z

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
