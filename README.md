# Landmark-based Visual SLAM

Landmark-based Visual SLAM 🤖🤖🤖  
Currently the system is not 100% V-SLAM, it is aided by LIDAR scans too and detects landmarks with YOLO - still awesome though 😄  

Written by: 
  * Anastasia Panaretou https://github.com/anastasiapan
  * Phillip Mastrup https://github.com/PMastrup

Landmarks are detected with You-Only-Look-Once - YOLO object detection  
Graph SLAM by cartographer-project  

Cool videos of the results


Completed as a master thesis at Technical University of Denmark in collaboration with Mobile Industrial Robots  
Systems that we use:  
https://github.com/ultralytics/yolov5  
https://github.com/cartographer-project/cartographer  
Robot provided by Mobile Industrial Robots - MiR100  
https://www.mobile-industrial-robots.com/da/solutions/robots/mir100/  
You also need RGB-D camera and ROS (we used ROS Melodic)  

3D models used as landmarks found at:  
https://app.ignitionrobotics.org/OpenRobotics  

Attempt for a setting up and use tutorial 😅😅😅

1. First download and set-up ROS  
   http://wiki.ros.org/ROS/Installation  

2. Then setup Cartographer System  
   https://google-cartographer-ros.readthedocs.io/en/latest/compilation.html  

3. Cartographer system modifications  
   We did not create a package for this. You can just merge the files under 'cartographer_ros' as the directories  
   they appear to be. They should be under the path: *catkin_ws/src/cartographer_ros/cartographer_ros*  
   Only one source file has been slightly modified to visualise landmarks with 3D models - completely optional  

3. Setup YOLO by ultralytics https://github.com/ultralytics/yolov5  
   We trained for our own objects - currently traffic cones and fire extinguishers  
   If you don't feel like training COCO can be used  
   
4. Since an RGB-D camera is used you need to setup OpenNI library from pip     
   `pip install openni`  
   `pip install primesense`  
   
5. This method is based on Bag-of-Visual-Words and SURF  
   If not familiar with this method watch this amazing video:  
   https://www.youtube.com/watch?v=a4cFONdc6nc  
   Whatever vocabulary can be used (based on SURF)  
   OR create your own (which we did) following the instructions in `codebook_creation/create_codebook.py`  

That's all!

PS. This is a master's project. None of us is a computer scientist so our code might not be perfect  
but we do believe there are some cool ideas in it and the results look awesome 😄😊
