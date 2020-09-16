# Landmark-based Visual SLAM

Landmark-based Visual SLAM 🤖🤖🤖  
The idea behind the project is to perform landmark-based visual SLAM detecting static objects as landmarks with object detection  
Currently the system is not 100% V-SLAM, it uses LIDAR scans too and detects landmarks with YOLO - still awesome though 😄  

Written by: 
  * Anastasia Panaretou https://github.com/anastasiapan
  * Phillip Mastrup https://github.com/PMastrup

Landmarks are detected with You-Only-Look-Once - YOLO object detection  
Graph SLAM by cartographer-project  

Cool videos of the results  
https://www.youtube.com/watch?v=k3LRF8AGJbI&t=49s  
https://www.youtube.com/watch?v=6s2ePNo1SqA&t=27s  

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
   We did not create a package for this. You can just merge the files under 'cartographer_ros' in the directories  
   they appear to be. They should be under the path: *catkin_ws/src/cartographer_ros/cartographer_ros*  
   Only one source file has been slightly modified to visualise landmarks with 3D models - completely optional  

3. Setup YOLO by ultralytics https://github.com/ultralytics/yolov5  
   We trained for our own objects - currently traffic cones and fire extinguishers  
   If you don't feel like training, COCO can be used  
   
4. Find your camera drivers:  
   We used astra from Orbecc  
   https://astra-wiki.readthedocs.io/en/latest/installation.html  

5. Since an RGB-D camera is used you need to setup OpenNI library from pip     
   `pip install openni`  
   `pip install primesense`  
   
6. This method is based on Bag-of-Visual-Words and SURF  
   If not familiar with this method watch this amazing video:  
   https://www.youtube.com/watch?v=a4cFONdc6nc 
   and check out this repo  
   https://github.com/ovysotska/in_simple_english/blob/master/bag_of_visual_words.ipynb  
   Whatever vocabulary can be used (based on SURF)  
   OR create your own (which we did) following the instructions in `codebook_creation/create_codebook.py`  

That's all! Now you can run our superprogram like this:

Before running it please become familiar with cartographer's ROS integration:  
https://google-cartographer-ros.readthedocs.io/en/latest/index.html  

* Start the detector with python : `python detect_landmarks_vslam.py`  
* Launch cartographer : `roslaunch cartographer_ros vslam_2D_landmarks.launch`  
* If you want to test on a bagfile : `roslaunch cartographer_ros landmarks_2D.launch bag_filename:=path/to/your/bagfile.bag`  

PS. This is a master's project. We were both students 🎓🎓🎓 when we wrote it so our code might not be  
perfect but we do believe there are some cool ideas in it and the results look awesome 😄😊
