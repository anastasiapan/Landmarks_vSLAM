#!/usr/bin/env python
import numpy as np
import cv2
import operator
from V_SLAM_fcn.bag_of_words_fcn import BoVW_comparison

## Point clouds
from V_SLAM_fcn.pcl_functions import cloud_object_center

def sample(codebook_match, online):
    ## Correct the false ids
    unique_ids = {}
    corrected_poses = {}
    corrected_timestamps = {}
    for track_id in codebook_match:

        if len(codebook_match[track_id]) > 5:
            unique_ids = {i: codebook_match[track_id].count(i) for i in codebook_match[track_id]}
            unique_ids = sorted(unique_ids.items(), key=operator.itemgetter(1), reverse=True)
            text = "I saw {} ".format(track_id) + " as {} {} ".format(unique_ids[0][0], unique_ids[0][1]) + "times!"
            print(text)
            print(unique_ids)

            if online:
                corrected_timestamps[unique_ids[0][0]] = timestamps[track_id]
                corrected_poses[unique_ids[0][0]] = poses[track_id]

    return corrected_poses, corrected_timestamps

def serial_landmarks(corrected_poses, corrected_timestamps):
    ## Arange observations timewise
    aranged_timestamps = {}
    aranged_poses = {}

    for object in corrected_timestamps:
        for x in range(len(corrected_timestamps[object])):
            obj_ts = corrected_timestamps[object]
            obj_poses = corrected_poses[object]
            ts = obj_ts[x]
            ps = obj_poses[x]
            if str(ts) in arranged_timestamps.keys():
                arranged_timestamps[str(ts)].append(object)
                arranged_poses[str(ts)].append(ps)
            else:
                arranged_timestamps[str(ts)] = []
                arranged_timestamps[str(ts)].append(object)
                arranged_poses[str(ts)] = []
                arranged_poses[str(ts)].append(ps)

    return arranged_poses, arranged_timestamps

def new_landmarks(online_data, detections, id, img, disp, parameters, codebook, re_hist, names):
    old_objects = {}
    old_histograms = {}
    codebook_match = {}

    timestamps = {}
    poses = {}

    hess_th = parameters['hess_th']
    exp_pct = parameters['exp_pct']

    for i in range(len(detections)):
        det = detections[i].data.cpu().tolist()
        label = names[int(det[5])] + str(id)

        bbox = [int(j) for j in det[0:4]]  ## bbox = [x1 y1 x2 y2]

        ## Find SURF features in the patch
        patch = img[int(bbox[1] - exp_pct * bbox[1]) : int(bbox[3] + exp_pct * bbox[3]), int(bbox[0] - exp_pct * bbox[0]) : int(bbox[2] + exp_pct * bbox[2])]
        surf = cv2.xfeatures2d.SURF_create(hess_th)
        kp, des = surf.detectAndCompute(patch, None)
        old_objects[label] = des

        ## Bag of words frame comparison
        BoVW_match = BoVW_comparison(codebook, re_hist, des, img, disp, bbox[0], bbox[1])
        old_histograms[label] = BoVW_match.hist

        codebook_match[label] = []
        codebook_match[label].append(BoVW_match.object)

        ## Put text next to the bounding box
        org = (bbox[0] + 10, bbox[1] + 20)  # org - text starting point
        txt = '{}'.format(label)
        img = cv2.putText(disp, txt, org, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

        ## Online operation
        if online_data['flag']:
            ## center pixel of the area the object is located
            obj_cent = [(bbox[0] + bbox[2])/2, (bbox[1] + bbox[3])/2]
            timestamps[label] = []
            timestamps[label].append(online_data['timestamp'])
            poses[label] = []
            pcl = cloud_object_center(online_data['depth_map'], obj_cent)
            poses[label].append(pcl)

    return img, old_objects, old_histograms, codebook_match, timestamps, poses

class track_objects:

    ## Draw text on frame
    def draw_text(self, x, y):
        ## Put text next to the bounding box
        org = (int(x + 10), int(y + 20))  # org - text starting point
        self.disp = cv2.putText(self.disp, self.txt, org, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

        return self

    ## Find good matches - Brute force matching - Lowe ratio test
    def BF_match_finder(self, des):
        ## Best match
        for object in self.old_objects:
            good = []
            best_matches = []
            bf = cv2.BFMatcher()
            bf_matches = bf.knnMatch(des, self.old_objects[object], k=2)
            for m, n in bf_matches:
                if m.distance < self.lowe_ratio * n.distance:
                    good.append([m.distance])
            pct = len(good) * 100 / des.shape[0]
            self.pcts[object] = pct

        self.matches = sorted(self.pcts.items(), key=operator.itemgetter(1), reverse=True)

        return self

    ## Same number of objects in current and previous frame
    def less_or_equal_objects(self, detections, names, codebook, re_hist):
        self.text_scene = "less or same number of objects"

        for i in range(self.new_num):
            det = detections[i].data.cpu().tolist()
            bbox = [int(j) for j in det[0:4]]  ## bbox = [x1 y1 x2 y2]

            ## Find SURF features in the patch
            patch = self.img[int(bbox[1] - self.exp_pct * bbox[1]): int(bbox[3] + self.exp_pct * bbox[3]), int(bbox[0] - self.exp_pct * bbox[0]): int(bbox[2] + self.exp_pct * bbox[2])]
            surf = cv2.xfeatures2d.SURF_create(self.hessian)
            kp, des = surf.detectAndCompute(patch, None)

            ## Calculate match to object in previous frame
            track_objects.BF_match_finder(self, des)

            ## Same objects as before
            if self.matches[0][1] > self.match_thres:
                self.new_objects[self.matches[0][0]] = des

                ## Bag of words frame comparison
                BoVW_match = BoVW_comparison(codebook, re_hist, des, self.img, self.disp, bbox[0], bbox[1])
                self.new_histograms[self.matches[0][0]] = BoVW_match.hist

                ## Append found match
                self.codebook_match[self.matches[0][0]].append(BoVW_match.object)

                ## Online operation
                if self.online_flag:
                    ## center pixel of the area the object is located
                    obj_cent = [(bbox[0] + bbox[2])/2, (bbox[1] + bbox[3])/2]
                    self.timestamps[self.matches[0][0]].append(self.obsv_time)
                    pcl = cloud_object_center(self.dmap, obj_cent)
                    self.poses[self.matches[0][0]].append(pcl)

                ## Put text next to the bounding box
                self.txt = '{} {}'.format(self.matches[0][0], round(self.matches[0][1]))
                track_objects.draw_text(self, bbox[0], bbox[1])

            ## Different objects than in previous frame
            else:
                self.text_scene = "new object"
                self.id += 1
                label = names[int(det[5])] + str(self.id)

                ## Find SURF features in the patch
                patch = self.img[int(bbox[1] - self.exp_pct * bbox[1]): int(bbox[3] + self.exp_pct * bbox[3]), int(bbox[0] - self.exp_pct * bbox[0]): int(bbox[2] + self.exp_pct * bbox[2])]
                surf = cv2.xfeatures2d.SURF_create(self.hessian)
                kp, des = surf.detectAndCompute(patch, None)
                self.new_objects[label] = des

                ## Bag of words frame comparison
                BoVW_match = BoVW_comparison(codebook, re_hist, des, self.img, self.disp, bbox[0], bbox[1])
                self.new_histograms[label] = BoVW_match.hist

                ## Append found match
                self.codebook_match[label] = []
                self.codebook_match[label].append(BoVW_match.object)

                ## Online operation
                if self.online_flag:
                    ## center pixel of the area the object is located
                    obj_cent = [(bbox[0] + bbox[2])/2, (bbox[1] + bbox[3])/2]
                    self.timestamps[label] = []
                    self.timestamps[label].append(self.obsv_time)
                    self.poses[label] = []
                    pcl = cloud_object_center(self.dmap, obj_cent)
                    self.poses[label].append(pcl)

                ## Put text next to the bounding box
                self.txt = '{} new'.format(label)
                track_objects.draw_text(self, bbox[0], bbox[1])

            self.old_objects = self.new_objects
            self.old_num = self.new_num

            return self

    ## More objects in current frame
    def more_objects(self, detections, names, codebook, re_hist):
        self.text_scene = "more new objects"
        prev_match = 0

        ## Check in old objects and find which of the new ones is the best match
        found_matches = []
        found_match_box = {}
        found_match_des = {}
        found_match_box_end = {}

        for object in self.old_objects:

                prev_pct = self.match_thres

                for i in range(self.new_num):
                    det = detections[i].data.cpu().tolist()
                    bbox = [int(j) for j in det[0:4]]  ## bbox = [x1 y1 x2 y2]

                    ## Find SURF features in the patch
                    patch = self.img[int(bbox[1] - self.exp_pct * bbox[1]): int(bbox[3] + self.exp_pct * bbox[3]), int(bbox[0] - self.exp_pct * bbox[0]): int(bbox[2] + self.exp_pct * bbox[2])]
                    surf = cv2.xfeatures2d.SURF_create(self.hessian)
                    kp, des = surf.detectAndCompute(patch, None)

                    ## BF matcher - Find the best match in the current frame
                    good = []
                    bf = cv2.BFMatcher()
                    bf_matches = bf.knnMatch(des, self.old_objects[object], k=2)
                    for m, n in bf_matches:
                        if m.distance < self.lowe_ratio * n.distance:
                            good.append([m.distance])
                    pct = len(good) * 100 / len(kp)

                    ## Keep best match
                    if pct > prev_pct:
                        self.pcts[object] = pct
                        found_match_box[object] = [bbox[0], bbox[1]]
                        found_match_box_end[object] = [bbox[2], bbox[3]]
                        found_match_des[object] = des
                    prev_pct = pct

                self.new_objects[object] = found_match_des[object]

                x1y1 = found_match_box[object]
                x2y2 = found_match_box_end[object]
                track_objects.draw_text(self, x1y1[0], x1y1[1])

                ## Bag of words frame comparison
                BoVW_match = BoVW_comparison(codebook, re_hist, found_match_des[object], self.img, self.disp, x1y1[0], x1y1[1])
                self.new_histograms[object] = BoVW_match.hist

                ## Append found match
                self.codebook_match[object].append(BoVW_match.object)

                ## Online operation
                if self.online_flag:
                    ## center pixel of the area the object is located
                    obj_cent = [(x1y1[0] + x2y2[0]) / 2, (x1y1[1] + x2y2[1]) / 2]
                    self.timestamps[object].append(self.obsv_time)
                    pcl = cloud_object_center(self.dmap, obj_cent)
                    self.poses[object].append(pcl)

                found_matches.append(found_match_box[object])

                if object == list(self.old_objects.keys())[-1]: ## last loop - last object which has a match
                    for i in range(self.new_num): ## check which objects are left out
                        det = detections[i].data.cpu().tolist()
                        bbox = [int(j) for j in det[0:4]]  ## bbox = [x1 y1 x2 y2]

                        if [bbox[0], bbox[1]] not in found_matches:
                            self.id += 1
                            label = names[int(det[5])] + str(self.id)

                            ## Find SURF features in the patch
                            patch = self.img[int(bbox[1] - self.exp_pct * bbox[1]): int(bbox[3] + self.exp_pct * bbox[3]), int(bbox[0] - self.exp_pct * bbox[0]): int(bbox[2] + self.exp_pct * bbox[2])]
                            surf = cv2.xfeatures2d.SURF_create(self.hessian)
                            kp, des = surf.detectAndCompute(patch, None)

                            self.new_objects[label] = des

                            ## Bag of words frame comparison
                            BoVW_match = BoVW_comparison(codebook, re_hist, des, self.img, self.disp, bbox[0], bbox[1])
                            self.new_histograms[label] = BoVW_match.hist

                            ## Append found match
                            self.codebook_match[label] = []
                            self.codebook_match[label].append(BoVW_match.object)

                            ## Online operation
                            if self.online_flag:
                                ## center pixel of the area the object is located
                                obj_cent = [(bbox[0] + bbox[2])/2, (bbox[1] + bbox[3])/2]
                                self.timestamps[label] = []
                                self.timestamps[label].append(self.obsv_time)
                                self.poses[label] = []
                                pcl = cloud_object_center(self.dmap, obj_cent)
                                self.poses[label].append(pcl)

                            self.txt = '{} new'.format(label)
                            track_objects.draw_text(self, bbox[0], bbox[1])

        self.old_objects = self.new_objects
        self.old_num = self.new_num

        return self

    def __init__(self, online_data, old_num, parameters, frame, disp, old_landmarks, detections, id, codebook, re_hist, codebook_match,timestamps, poses, names):
        self.new_num = len(detections)
        self.old_num = old_num
        self.text_scene = " "
        self.hessian = parameters['hess_th']
        self.lowe_ratio = parameters['lowe_ratio']
        self.match_thres = parameters['match_thres']
        self.exp_pct = parameters['exp_pct']
        self.img = frame
        self.disp = disp
        self.old_objects = old_landmarks
        self.new_objects = {}
        self.pcts = {}
        self.matches = []
        self.txt = " "
        self.id = id
        self.new_histograms = {}
        self.codebook_match = codebook_match

        ## For running online
        self.online_flag = online_data['flag']
        self.obsv_time = online_data['timestamp']
        self.dmap = online_data['depth_map']
        self.timestamps = timestamps
        self.poses = poses

        if self.new_num  <= self.old_num: ## less or same number of objects
            track_objects.less_or_equal_objects(self, detections, names, codebook, re_hist)
        else: ## more objects
            track_objects.more_objects(self, detections, names, codebook, re_hist)