#!/usr/bin/env python
## Resolve cv2 and ROS conflict of python versions
#import sys
#sys.path.remove('/opt/ros/melodic/lib/python2.7/dist-packages')
import cv2
import argparse

import torch.backends.cudnn as cudnn

from utils import google_utils
from utils.datasets import *
from utils.utils import *

import operator
import numpy as np

## Find best match
from V_SLAM_fcn.visual_tracker_fcn_update import *
from V_SLAM_fcn.robot_global_pose import *
import global_variables

width = 640
height = 480

online_flag = True ## Run online or from a video
#----------------------------------------------------------------------------------#

## ROS landmark publisher ---------------------------------------------------------#
if online_flag:
    #sys.path.append('/opt/ros/melodic/lib/python2.7/dist-packages')
    import rospy
    from cartographer_ros_msgs.msg import LandmarkEntry, LandmarkList
    from ROS_pub import *
    TOPIC = '/v_landmarks'
    ## ROS landmark publisher node
    rospy.init_node('landmark_publisher', anonymous=True)
    lmk_pub = rospy.Publisher(TOPIC, LandmarkList, queue_size=1000)
    ## robot global pose listener node
    #rospy.init_node('robot_global_pose')
    #global_pose_listener = tf.TransformListener()
    #rate = rospy.Rate(10.0)
#----------------------------------------------------------------------------------#

def detect(save_img=False):
    ## Writing output video
    fps = 30
    codec = cv2.VideoWriter_fourcc(*'XVID')
    output_path = './rt_kf_sampler.avi'
    out_vid_write = cv2.VideoWriter(output_path, codec, fps, (width, height))

    out, source, weights, view_img, save_txt, imgsz = \
        opt.output, opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size
    webcam = source == '0' or source.startswith('rtsp') or source.startswith('http') or source.endswith('.txt')

    # Initialize
    device = torch_utils.select_device(opt.device)
    if os.path.exists(out):
        shutil.rmtree(out)  # delete output folder
    os.makedirs(out)  # make new output folder
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    google_utils.attempt_download(weights)
    model = torch.load(weights, map_location=device)['model'].float()  # load to FP32
    # torch.save(torch.load(weights, map_location=device), weights)  # update model if SourceChangeWarning
    # model.fuse()
    model.to(device).eval()
    if half:
        model.half()  # to FP16

    # Second-stage classifier
    classify = False
    if classify:
        modelc = torch_utils.load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model'])  # load weights
        modelc.to(device).eval()

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = True
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz)
    else:
        save_img = True
        dataset = LoadImages(source, img_size=imgsz)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names

    fps_disp = 0.0

    # Run inference
    t0 = time.time()
    img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
    _ = model(img.half() if half else img) if device.type != 'cpu' else None  # run once

    ## Init
    first_detection = True
    lmk_id = 0
    old_objects = {}
    old_num = 0
    codebook_match = {}
    txt = " "
    correct_hist = {}
    min_samples = 20
    rbt_glb_pose = robot_global_pose() ## robot global pose
    lmk_gb = np.zeros((3,1))
    lmk_obsv_poses = {}

    ## Main loop
    for path, img, im0s, d_map, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0

        dmap = d_map
        if online_flag and dmap is not None:
            d4d = np.uint8(dmap.astype(float) * 255 / 2 ** 12 - 1)  # Correct the range. Depth images are 12bits
            d4d = 255 - cv2.cvtColor(d4d, cv2.COLOR_GRAY2RGB)
            timestamp = rospy.Time.now()
            rbt_glb_pose.trans_mat()

        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        ## Online operation data
        online_data = {'flag': online_flag,
                       'timestamp': 0 if not online_flag else timestamp,
                       'depth_map': 0 if not online_flag else dmap,
                       'robot_global_pose': 0 if not online_flag else rbt_glb_pose.trans}
        # Inference
        t1 = torch_utils.time_synchronized()
        pred = model(img, augment=opt.augment)[0]

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)

        # Apply Classifier
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0 = path[i], '%g: ' % i, im0s[i].copy()
            else:
                p, s, im0 = path, '', im0s

            im_rgb = cv2.cvtColor(im0, cv2.COLOR_BGR2RGB)

            save_path = str(Path(out) / Path(p).name)
            s += '%gx%g ' % img.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  #  normalization gain whwh

            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += '%g %ss, ' % (n, names[int(c)])  # add to string

                # Write results
                for *xyxy, conf, cls in det:
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        with open(save_path[:save_path.rfind('.')] + '.txt', 'a') as file:
                            file.write(('%g ' * 5 + '\n') % (cls, *xywh))  # label format

                    if save_img or view_img:  # Add bbox to image
                        label = '%s %.2f' % (names[int(cls)], conf)
                        #plot_one_box(xyxy, im_rgb, label=label, color=colors[int(cls)], line_thickness=3)
                        plot_one_box(xyxy, im_rgb, label=label, color=(0,255,0), line_thickness=3)

            ## Check if any detections happen
            detections = pred[0]
            objects = detections
            if det is not None:
                ## number of detections:
                num_d = len(detections)
                for i in range(num_d):
                    det_info = detections[i].data.cpu().tolist()
                    bbox = [int(j) for j in det_info[0:4]]  ## bbox = [x1 y1 x2 y2]

                    ## If running online check if the object is within range
                    if online_flag:
                        obj_cent = [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2]  ## [x y] center pixel of the area the object is located
                        ## Optimal operation range: 0.6-5m
                        #print(dmap[int(obj_cent[1]), int(obj_cent[0])])
                        im_rgb = cv2.circle(im_rgb, (int(obj_cent[0]), int(obj_cent[1])), 5, (0,255,0), -1)
                        if online_flag: d4d = cv2.circle(d4d, (int(obj_cent[0]), int(obj_cent[1])), 5, (0,0,255), -1)
                        if dmap[int(obj_cent[1]), int(obj_cent[0])] < 600 or dmap[int(obj_cent[1]), int(obj_cent[0])] > 3500:
                            objects = torch.cat([detections[0:i], detections[i+1:]])

                if not objects.cpu().tolist(): objects = None

            ## Track and sample landmarks
            if objects is not None:
                if first_detection:
                    lmk_id += 1
                    old_num = len(objects)
                    im_rgb, old_objects, tracked_histograms,  lmkObsv = new_landmarks(objects, lmk_id, im0, im_rgb, online_data, names)
                    first_detection = False
                    codebook_match = {}
                    correct_hist = {}
                else:
                    tracker = track_detections(old_num,  im0, im_rgb, old_objects, objects, lmk_id, tracked_histograms, codebook_match, correct_hist, online_data, lmkObsv, names)
                    lmk_id = tracker.id
                    old_objects = tracker.old_objects
                    old_num = tracker.old_num
                    im_rgb = tracker.disp
                    tracked_histograms = tracker.tracked_histograms
                    codebook_match = tracker.codebook_match
                    correct_hist = tracker.keyframes_hist
                    lmkObsv = tracker.lmkObsv

                    if tracker.publish_flag:
                        lmk_copy = dict(lmkObsv)
                        keyframes_copy = dict(correct_hist)
                        tracked_hists = dict(tracked_histograms)
                        for key in lmkObsv:
                            if len(lmkObsv[key]) > min_samples:
                                #print(key)
                                lmk = lmkObsv[key]
                                gpose = np.array(list(map(operator.itemgetter(2), lmk)))
                                gpose = gpose[:,:,0].transpose()
                                gpose = np.true_divide(gpose.sum(1),(gpose!=0).sum(1)).reshape(3,1)
                                print('- - - - - -- - -- - - - - -- - - - -- - - ')
                                print('I saw ' + key + ' with a global pose of:')
                                print(gpose)
                                in_area, occupied = spatial_filter(lmk_obsv_poses, gpose, key)

                                if in_area and not occupied:
                                    print('Valid observation - in range - unoccupied space')
                                    for i in range(len(lmk)):
                                        lmk_gpose = lmk[i][2]
                                        lmk_list = landmark_pub(lmk[i], key)
                                        lmk_pub.publish(lmk_list)

                                    lmk_obsv_poses[key] = gpose
                                elif not in_area and not occupied:
                                    print('Mismatch - new observation - out of range - unoccupied space')
                                    new_id = key.split('_')
                                    lmk_id += 1
                                    new_id = new_id[0] + '_' + str(lmk_id)
                                    for i in range(len(lmk)):
                                        lmk_gpose = lmk[i][2]
                                        lmk_list = landmark_pub(lmk[i], new_id)
                                        lmk_pub.publish(lmk_list)

                                    lmk_obsv_poses[new_id] = gpose

                                    del lmk_copy[key]
                                    del keyframes_copy[key]
                                    del tracked_hists[key]
                                    lmk_copy[new_id] = lmkObsv[key]
                                    keyframes_copy[new_id] = correct_hist[key]
                                    tracked_hists[new_id] = tracked_histograms[key]

                                else:
                                    print('out of range - there is a landmark already there - IGNORE')
                                    del lmk_copy[key]
                                    del keyframes_copy[key]
                                    del tracked_histograms[key]
                                print('- - - - - -- - -- - - - - -- - - - -- - - ')
                        
                        lmkObsv = lmk_copy
                        correct_hist = keyframes_copy
                        tracked_histograms = tracked_hists

                        lmkObsv = {}

                    if hasattr(tracker, 'sampler_txt'): txt = tracker.sampler_txt

        img2 = cv2.putText(im_rgb, txt, (0, 60), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 2, cv2.LINE_AA)

        fps_disp = (fps_disp + (1. / (torch_utils.time_synchronized() - t1))) / 2
        img2 = cv2.putText(img2, "FPS: {:.2f}".format(fps_disp), (0, 30),
                          cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 2)
        # Stream results
        if view_img or save_img:
            if online_flag:
                #display = np.hstack((d4d, img2))
                cv2.imshow('output', img2)
                out_vid_write.write(img2)
            else:
                cv2.imshow('output', img2)
                out_vid_write.write(img2)

            if cv2.waitKey(1) == ord('q'):  # q to quit
                raise StopIteration

        # Save results (image with detections)
        if save_img:
            if dataset.mode == 'images':
                cv2.imwrite(save_path, display)
            else:
                if vid_path != save_path:  # new video
                    vid_path = save_path
                    if isinstance(vid_writer, cv2.VideoWriter):
                        vid_writer.release()  # release previous video writer

                    fps = vid_cap.get(cv2.CAP_PROP_FPS)
                    w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*opt.fourcc), fps, (w, h))

    if save_txt or save_img:
        print('Results saved to %s' % os.getcwd() + os.sep + out)
        if platform == 'darwin':  # MacOS
            os.system('open ' + save_path)

    print('Done. (%.3fs)' % (time.time() - t0))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='weights/best_yolov5x_custom_3.pt', help='model.pt path')
    parser.add_argument('--source', type=str, default='0', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--output', type=str, default='inference/output/', help='output folder')  # output folder
    parser.add_argument('--img-size', type=int, default=416, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.85, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='IOU threshold for NMS')
    parser.add_argument('--fourcc', type=str, default='mp4v', help='output video codec (verify ffmpeg support)')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    opt = parser.parse_args()
    opt.img_size = check_img_size(opt.img_size)
    print(opt)

    with torch.no_grad():
        detect()