#!/usr/bin/env python
## Resolve cv2 and ROS conflict of python versions
import sys
sys.path.remove('/opt/ros/melodic/lib/python2.7/dist-packages')
import cv2
import argparse

import torch.backends.cudnn as cudnn

from utils import google_utils
from utils.datasets import *
from utils.utils import *

import operator
import numpy as np

## Find best match
from V_SLAM_fcn.old_tracker import *

## Load codebook and re-weighted histograms
codebook =  np.load('./V_SLAM_fcn/codebook/Visual_Words100.npy') ## codebook
#re_hist = np.load('./PR_plot/kf_hists_329_surf_500.npy', allow_pickle=True).reshape(1, 1) ## dctionary histogram
#re_hist = re_hist[0,0]

width = 640
height = 480

online_flag = False ## Run online or from a video

def detect(save_img=False):
    ## Writing output video
    width = 640
    height = 480
    fps = 30
    codec = cv2.VideoWriter_fourcc(*'XVID')
    output_path = './pr_test.avi'
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
    #colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]

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
    frames_wo_det = 0

    ## Main loop
    for path, img, im0s, dmap, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0

        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        ## Online operation data
        online_data = {'flag': online_flag,
                       'timestamp': 0 if not online_flag else rospy.Time.now(),
                       'depth_map': 0 if not online_flag else dmap}

        # Inference
        t1 = torch_utils.time_synchronized()
        pred = model(img, augment=opt.augment)[0]

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        t2 = torch_utils.time_synchronized()

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

                    boxH = bbox[3] - bbox[1]
                    boxW = bbox[2] - bbox[0]
                    if boxH < 120:
                        objects = torch.cat([detections[0:i], detections[i+1:]])

                if not objects.cpu().tolist(): objects = None

            ## Track and sample landmarks
            if objects is not None:
                if first_detection:
                    frames_wo_det = 0
                    lmk_id += 1
                    old_num = len(objects)
                    im_rgb, old_objects, tracked_histograms = new_landmarks(objects, lmk_id, im0, im_rgb, online_data, names)
                    first_detection = False
                    codebook_match = {}
                    correct_hist = {}
                else:
                    tracker = track_detections(old_num, im0, im_rgb, old_objects, objects, lmk_id, tracked_histograms, codebook_match, correct_hist, online_data, names, frames_wo_det)
                    lmk_id = tracker.id
                    old_objects = tracker.old_objects
                    old_num = tracker.old_num
                    im_rgb = tracker.disp
                    tracked_histograms = tracker.tracked_histograms
                    codebook_match = tracker.codebook_match
                    correct_hist = tracker.keyframes_hist
                    frames_wo_det = 0

                    if hasattr(tracker, 'sampler_txt') and tracker.sampler_txt != " ": 
                        txt = tracker.sampler_txt

            else: ## no detections
                frames_wo_det +=1
        img2 = cv2.putText(im_rgb, txt, (0, 60), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 1, cv2.LINE_AA)

        fps_disp = (fps_disp + (1. / (torch_utils.time_synchronized() - t1))) / 2
        img2 = cv2.putText(img2, "FPS: {:.2f}".format(fps_disp), (0, 30),
                          cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 2)

        # Stream results
        if view_img or save_img:
            if online_flag:
                #display = np.hstack((d4d, img2))
                #cv2.imshow('output', img2)
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
                #out = display if online_flag else img2
                #vid_writer.write(display)

    if save_txt or save_img:
        #id_pct, id_tot, sampler_txt, correct_hist, correct_tracked = sample(codebook_match, correct_hist ,tracked_histograms, online_flag)

        print('Results saved to %s' % os.getcwd() + os.sep + out)
        if platform == 'darwin':  # MacOS
            os.system('open ' + save_path)
    print(correct_hist.keys())
    print('Done. (%.3fs)' % (time.time() - t0))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='weights/best_yolov5x_custom_3.pt', help='model.pt path')
    parser.add_argument('--source', type=str, default='inference/videos/329_two_classes_test_video_01.avi', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--output', type=str, default='inference/output/', help='output folder')  # output folder
    parser.add_argument('--img-size', type=int, default=416, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.7, help='object confidence threshold')
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