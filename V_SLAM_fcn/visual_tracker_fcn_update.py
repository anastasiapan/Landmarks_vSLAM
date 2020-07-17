#!/usr/bin/env python
import numpy as np
import cv2
import operator
from itertools import groupby

## Bag of Visual Words
from V_SLAM_fcn.bag_of_words_fcn import BoVW_comparison

## Point clouds
from V_SLAM_fcn.pcl_functions import cloud_object_center
from V_SLAM_fcn.robot_global_pose import *

import global_variables

## Variables
min_samples = 20 # Minimum number of samples to go through frame comparison 
bow_thres = 70  # BoVW matching percentage threshold to declare a match for each comparison
loop_cl_thres = 75 # Loop closure detection -- Sampler threshold -- number of good matches/total matches

codebook = global_variables.codebook
parameters = global_variables.parameters

def discard_tracks(finished_tracks):
    final_tracks = dict(finished_tracks)
    for key in finished_tracks:
        if finished_tracks[key].shape[0] < min_samples:
            del final_tracks[key]

    return final_tracks

def sample(codebook_match, histograms, tracked_histograms, online, lmkObsv):
    ## Correct the false ids
    id_pct = 0
    id_tot = 0
    txt = " "
    final_hist = dict(histograms)
    correct_tracked = dict(tracked_histograms)
    lmk_correct = dict(lmkObsv)
    for track_id in codebook_match:
        if len(codebook_match[track_id]) > min_samples:
            unique_ids = {i: codebook_match[track_id].count(i) for i in codebook_match[track_id]}
            unique_ids = sorted(unique_ids.items(), key=operator.itemgetter(1), reverse=True)

            best_match = unique_ids[0][0]
            best_pct = unique_ids[0][1]

            if best_match == 'bad_match' and len(unique_ids) > 1:
                best_match = unique_ids[1][0]
                best_pct = unique_ids[1][1]

            text = "I saw {} ".format(track_id) + " as {} {} ".format(best_match, best_pct) + "times!"

            #np.append(tracked_histograms[track_id],

            print(text)
            sum_det = 0
            sum_tot = 0
            for i in range(len(unique_ids)):
                if unique_ids[i][0] != 'bad_match':
                    sum_det = sum_det + unique_ids[i][1]
                sum_tot = sum_tot + unique_ids[i][1]

            id_pct = best_pct*100/sum_det if sum_det != 0 else 0
            id_tot = best_pct*100/sum_tot if sum_tot != 0 else 0
            #txt = "I am {} ".format(round(id_pct)) + "% sure that I saw {}.".format(best_match)

            if best_match != 'bad_match':
                txt = "I am {} ".format(round(id_tot)) + "% sure that I saw {}.".format(best_match)
            else:
                txt = ' '
            print(txt)
            print(unique_ids)
            print("-------------------------------------------------------------------------------")

            if id_tot > loop_cl_thres and best_match!='bad_match':
                del final_hist[track_id]
                final_hist[best_match] = np.append(final_hist[best_match],histograms[track_id], axis=0)

                del correct_tracked[track_id]
                correct_tracked[best_match] = np.append(tracked_histograms[best_match], histograms[track_id], axis=0)

                if online:
                    del lmk_correct[track_id]
                    lmk_correct[best_match] = lmkObsv[track_id]

    return id_pct, id_tot, txt, final_hist, correct_tracked, lmk_correct

def new_landmarks(detections, id, img, disp, online_data, names):
    old_objects = {}
    tracked_histograms = {}
    lmkObsv = {}

    hess_th = parameters['hess_th']
    exp_pct = parameters['exp_pct']
    
    for i in range(len(detections)):
        det = detections[i].data.cpu().tolist()
        obj_class = names[int(det[5])]
        label = obj_class + '_' + str(id)

        bbox = [int(j) for j in det[0:4]]  ## bbox = [x1 y1 x2 y2]

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        mask = np.zeros(img.shape[:2], dtype=np.uint8)
        ## patch [x1 y1 x2 y2]
        patch  = [int(bbox[0] - exp_pct * bbox[0]), int(bbox[1] - exp_pct * bbox[1]), int(bbox[2] + exp_pct * bbox[2]), int(bbox[3] + exp_pct * bbox[3])]
        cv2.rectangle(mask, (patch[0], patch[1]), (patch[2], bbox[1]), (255), thickness=-1)
        cv2.rectangle(mask, (patch[0], bbox[1]), (bbox[0], patch[3]), (255), thickness=-1)
        cv2.rectangle(mask, (bbox[0], bbox[3]), (patch[3], patch[3]), (255), thickness=-1)
        cv2.rectangle(mask, (bbox[2], bbox[1]), (patch[2], bbox[3]), (255), thickness=-1)

        surf = cv2.xfeatures2d.SURF_create(hess_th)
        kp, des = surf.detectAndCompute(gray, mask)

        hess = hess_th
        while des is None:
            hess_th = hess_th - 50
            surf = cv2.xfeatures2d.SURF_create(hess)
            kp = surf.detect(gray, mask)
            des = surf.compute(gray, kp)

        ## Calculate histogram
        k = codebook.shape[0]

        num_feat = des.shape[0]  # Number of extracted features for frame to be tested
        des = np.dstack(np.split(des, num_feat))

        words_stack = np.dstack([codebook] * num_feat)  ## stack words depthwise
        diff = words_stack - des
        dist = np.linalg.norm(diff, axis=1)
        idx = np.argmin(dist, axis=0)
        hist, n_bins = np.histogram(idx, bins=k)
        hist = hist.reshape(1, k)

        old_objects[label] = []
        old_objects[label].append(hist)

        tracked_histograms[label] = np.empty((0, hist.shape[1]))
        tracked_histograms[label] = np.append(tracked_histograms[label], hist, axis=0)

        ## Online operation
        if online_data['flag']:
            ## center pixel of the area the object is located
            obj_cent = [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2]
            pcl = cloud_object_center(online_data['depth_map'], obj_cent)
            valid_transformation = online_data['robot_global_pose'].size != 0
            lmk_global_pose = np.dot(online_data['robot_global_pose'],np.array(pcl).reshape(3,1)) if valid_transformation else np.zeros((3,1))
            lmkEntry = (online_data['timestamp'], pcl, lmk_global_pose)
            lmkObsv[label] = []
            lmkObsv[label].append(lmkEntry)

        ## Put text next to the bounding box
        org = (bbox[0] + 10, bbox[1] + 20)  # org - text starting point
        txt = '{}'.format(label)
        img = cv2.putText(disp, txt, org, cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 1, cv2.LINE_AA)

    return img, old_objects, tracked_histograms, lmkObsv

class track_detections:

    ## Draw text on frame
    def draw_text(self, x, y):
        ## Put text next to the bounding box
        org = (int(x + 10), int(y + 20))  # org - text starting point
        self.disp = cv2.putText(self.disp, self.txt, org, cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 1, cv2.LINE_AA)

        return self

    ## Find SURF features in the patch
    def features_histogram(self, bbox):
        ## Find SURF features in the patch
        gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        mask = np.zeros(self.img.shape[:2], dtype=np.uint8)
        ## patch [x1 y1 x2 y2]
        patch  = [int(bbox[0] - self.exp_pct * bbox[0]), int(bbox[1] - self.exp_pct * bbox[1]), int(bbox[2] + self.exp_pct * bbox[2]), int(bbox[3] + self.exp_pct * bbox[3])]
        cv2.rectangle(mask, (patch[0], patch[1]), (patch[2], bbox[1]), (255), thickness=-1)
        cv2.rectangle(mask, (patch[0], bbox[1]), (bbox[0], patch[3]), (255), thickness=-1)
        cv2.rectangle(mask, (bbox[0], bbox[3]), (patch[3], patch[3]), (255), thickness=-1)
        cv2.rectangle(mask, (bbox[2], bbox[1]), (patch[2], bbox[3]), (255), thickness=-1)

        surf = cv2.xfeatures2d.SURF_create(self.hessian)
        kp, des = surf.detectAndCompute(gray, mask)

        hess = self.hessian
        while des is None:
            hess = hess - 50
            surf = cv2.xfeatures2d.SURF_create(hess)
            kp = surf.detect(gray, mask)
            des = surf.compute(gray, kp)

        ## Calculate histogram
        k = codebook.shape[0]

        num_feat = des.shape[0]  # Number of extracted features for frame to be tested
        des = np.dstack(np.split(des, num_feat))

        words_stack = np.dstack([codebook] * num_feat)  ## stack words depthwise
        diff = words_stack - des
        dist = np.linalg.norm(diff, axis=1)
        idx = np.argmin(dist, axis=0)
        hist, n_bins = np.histogram(idx, bins=k)
        hist = hist.reshape(1, k)

        return kp, des, hist

        ## Find good matches - Brute force matching - Lowe ratio test

    ## Track objects
    def track_features(self, hist, obj_class):
        ## Best match
        for object in self.old_objects:
            old_id = object.split('_')
            old_id = old_id[0]
            if old_id == obj_class:
                # eucl_dist = np.linalg.norm(self.old_objects[object] - hist)
                dotP = np.sum(self.old_objects[object] * hist)
                norm = np.linalg.norm(hist)
                norm_codebook = np.linalg.norm(self.old_objects[object])
                sim_cos = dotP * 100 / (norm * norm_codebook)

                self.pcts[object] = sim_cos
            else:
                self.pcts[object] = 0

        self.tracked_objects = sorted(self.pcts.items(), key=operator.itemgetter(1), reverse=True)

        return self

    ## Search if the object is observed before
    def loop_closure_detection(self,des, box, obj_class, current_obj):
        if current_obj not in self.old_objects:
            self.keyframes_hist = discard_tracks(self.tracked_histograms)

            id_pct, id_tot, self.sampler_txt, correct_hist, correct_tracked, lmk_correct = sample(self.codebook_match, self.keyframes_hist,
                                                                        self.tracked_histograms, self.online_flag, self.lmkObsv)

            self.tracked_histograms = correct_tracked
            self.keyframes_hist = correct_hist

            self.lmkObsv = lmk_correct
            self.publish_flag = True
            self.codebook_match = {}

        if len(self.keyframes_hist) >= 1 and current_obj not in self.keyframes_hist:
            BoVW_match = BoVW_comparison(self.keyframes_hist, des, self.img, self.disp, box[0], box[1], obj_class)

            if current_obj not in self.codebook_match: self.codebook_match[current_obj] = []
            if self.online_flag and current_obj not in self.lmkObsv: self.lmkObsv[current_obj] = []

            if BoVW_match.cos_pct > bow_thres:
                ## Append found match
                self.codebook_match[current_obj].append(BoVW_match.object)
                if self.online_flag: self.lmkObsv[current_obj].append(track_detections.online_operation(self,box))
                    #if BoVW_match.object not in self.lmkObsv: self.lmkObsv[BoVW_match.object] = []
                    #self.lmkObsv[BoVW_match.object].append(track_detections.online_operation(self,box))
            else:
                self.codebook_match[current_obj].append('bad_match')
                if self.online_flag: self.lmkObsv[current_obj].append(track_detections.online_operation(self,box))

        elif len(self.keyframes_hist) == 0 and self.online_flag:
            if current_obj not in self.lmkObsv: self.lmkObsv[current_obj] = []
            ## Append found match
            self.lmkObsv[current_obj].append(track_detections.online_operation(self,box))

        return self

    def online_operation(self, box):
        obj_cent = [(box[0] + box[2])/2, (box[1] + box[3])/2]
        pcl = cloud_object_center(self.dmap, obj_cent)
        valid_transformation = self.robot_gtrans.size != 0
        x = pcl[2] + global_variables.T_cam_imu[0,0]
        y = pcl[0] + global_variables.T_cam_imu[2,0]
        pose = [x,y,1]
        lmk_global_pose = np.dot(self.robot_gtrans,np.array(pose).reshape(3,1)) if valid_transformation else np.zeros((3,1))
        lmkEntry = (self.obsv_time, pcl, lmk_global_pose)
        
        return lmkEntry

    ## Same number of objects in current and previous frame
    def less_or_equal_objects(self, detections, names):
        match_to_prev = []
        test = []
        for i in range(self.new_num):           
            det = detections[i].data.cpu().tolist()
            obj_class = names[int(det[5])]
            bbox = [int(j) for j in det[0:4]]  ## bbox = [x1 y1 x2 y2]

            kp, des, hist = track_detections.features_histogram(self, bbox)

            track_detections.track_features(self, hist, obj_class)
            best_match_obj = self.tracked_objects[0][0]
            best_match_pct = self.tracked_objects[0][1]

            match_to_prev.append((best_match_obj,best_match_pct, bbox, hist, des, obj_class))
            test.append((best_match_obj,best_match_pct, bbox, obj_class))
         
        sorted_matches = ([max(pct) for lmk_id, pct in groupby(match_to_prev, lambda y: y[0])])   

        tracks = []
        for i, det in enumerate(match_to_prev):
            (best_match_obj,best_match_pct, bbox, hist, des, obj_class) = det
            valid_track = best_match_pct > self.match_thres
            if det in sorted_matches and valid_track:
                tracks.append(det)
            else:
                self.id += 1
                label = obj_class + '_' + str(self.id)
                self.tracked_histograms[label] = np.empty((0, hist.shape[1]))
                tracks.append((label,best_match_pct, bbox, hist, des, obj_class))

        for det in tracks:
            (label,track_pct, bbox, hist, des, obj_class) = det 

            self.tracked_histograms[label] = np.append(self.tracked_histograms[label], hist, axis=0)   
            self.txt = '{} {}'.format(label, round(track_pct))   
            track_detections.draw_text(self, bbox[0], bbox[1])
            self.new_objects[label] = hist
            track_detections.loop_closure_detection(self, des, bbox, obj_class, label)           
            
        self.old_objects = self.new_objects
        self.old_num = self.new_num

        return self
   
    def __init__(self, old_num, frame, disp, old_landmarks, detections, lmk_id, tracked_histograms, codebook_match, final_tracks, online_data, lmkObsv, names):
        self.new_num = len(detections)
        self.old_num = old_num
        self.hessian = parameters['hess_th']
        self.match_thres = parameters['match_thres']
        self.exp_pct = parameters['exp_pct']
        self.img = frame
        self.disp = disp
        self.old_objects = old_landmarks
        self.new_objects = {}
        self.pcts = {}
        self.tracked_objects = []
        self.txt = " "
        self.id = lmk_id
        self.tracked_histograms = tracked_histograms
        self.nbins = codebook.shape[0]
        self.codebook_match = codebook_match
        self.keyframes_hist = final_tracks

        ## For running online
        self.online_flag = online_data['flag']
        self.obsv_time = online_data['timestamp']
        self.dmap = online_data['depth_map']
        self.robot_gtrans = online_data['robot_global_pose']
        self.lmkObsv = lmkObsv
        self.publish_flag = False

        track_detections.less_or_equal_objects(self, detections,names)