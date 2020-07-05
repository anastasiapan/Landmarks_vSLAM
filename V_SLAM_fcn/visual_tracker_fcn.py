#!/usr/bin/env python
import numpy as np
import cv2
import operator
from V_SLAM_fcn.bag_of_words_fcn import BoVW_comparison

## Point clouds
from V_SLAM_fcn.pcl_functions import cloud_object_center

bow_thres = 0.95

def sample(codebook_match, online, timestamps, poses):
    ## Correct the false ids
    id_pct = 0
    txt = " "
    unique_ids = {}
    corrected_poses = {}
    corrected_timestamps = {}
    for track_id in codebook_match:

        if len(codebook_match[track_id]) > 20:
            unique_ids = {i: codebook_match[track_id].count(i) for i in codebook_match[track_id]}
            unique_ids = sorted(unique_ids.items(), key=operator.itemgetter(1), reverse=True)

            best_match = unique_ids[0][0]
            best_pct = unique_ids[0][1]

            if best_match == 'bad_match' and len(unique_ids) > 1:
                best_match = unique_ids[1][0]
                best_pct = unique_ids[1][1]

            text = "I saw {} ".format(track_id) + " as {} {} ".format(best_match, best_pct) + "times!"
            print(text)
            sum_det = 0
            sum_tot = 0
            for i in range(len(unique_ids)):
                if unique_ids[i][0] != 'bad_match':
                    sum_det = sum_det + unique_ids[i][1]
                sum_tot = sum_tot + unique_ids[i][1]


            id_pct = best_pct*100/sum_det if sum_det != 0 else 0
            id_tot =best_pct*100/sum_tot if sum_tot != 0 else 0
            txt = "I am {} ".format(round(id_pct)) + "% sure that I saw {}.".format(best_match)
            print(txt)
            print("Tot: I am {} ".format(round(id_tot)) + "% sure that I saw {}.".format(best_match))
            print(unique_ids)
            print("-------------------------------------------------------------------------------")

            if online:
                corrected_timestamps[best_match] = timestamps[track_id]
                corrected_poses[best_match] = poses[track_id]

    return corrected_poses, corrected_timestamps, id_pct, txt

def serial_landmarks(corrected_poses, corrected_timestamps):
    ## Arange observations timewise
    arranged_timestamps = {}
    arranged_poses = {}

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
        obj_class = names[int(det[5])]
        label = names[int(det[5])] + '_' + str(id)

        bbox = [int(j) for j in det[0:4]]  ## bbox = [x1 y1 x2 y2]

        ## Find ORB features in the patch
        patch = img[int(bbox[1] - exp_pct * bbox[1]) : int(bbox[3] + exp_pct * bbox[3]), int(bbox[0] - exp_pct * bbox[0]) : int(bbox[2] + exp_pct * bbox[2])]
        surf = cv2.xfeatures2d.SURF_create(hess_th)
        kp, des = surf.detectAndCompute(patch, None)

        #old_objects[label] = hist

        ## Bag of words frame comparison
        BoVW_match = BoVW_comparison(codebook, re_hist, des, img, disp, bbox[0], bbox[1], obj_class)
        old_histograms[label] = BoVW_match.hist
        old_objects[label] = old_histograms[label]

        codebook_match[label] = []

        if BoVW_match.diff_match < bow_thres:
            ## Append found match
            codebook_match[label].append(BoVW_match.object)
        else:
            codebook_match[label].append('bad_match')

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
    def BF_match_finder(self, des, hist, obj_class):
        ## Best match
        for object in self.old_objects:
            old_id = object.split('_')
            old_id = old_id[0]
            #eucl_dist = np.linalg.norm(self.old_objects[object] - hist)
            if obj_class == old_id:
                dotP = np.sum(self.old_objects[object] * hist)
                norm = np.linalg.norm(hist)
                norm_codebook = np.linalg.norm(self.old_objects[object])
                sim_cos = dotP*100 / (norm * norm_codebook)
            else:
                sim_cos = 0

            self.pcts[object] = sim_cos

        self.matches = sorted(self.pcts.items(), key=operator.itemgetter(1), reverse=True)

        return self

    def extract_features(self, bbox, codebook):
        ## Find SURF features in the patch
        patch = self.img[int(bbox[1] - self.exp_pct * bbox[1]): int(bbox[3] + self.exp_pct * bbox[3]), int(bbox[0] - self.exp_pct * bbox[0]): int(bbox[2] + self.exp_pct * bbox[2])]
        surf = cv2.xfeatures2d.SURF_create(self.hessian)
        kp, des = surf.detectAndCompute(patch, None)

        hess = self.hessian
        while des is None:
            hess = hess - 50
            surf = cv2.xfeatures2d.SURF_create(hess)
            kp, des = surf.detectAndCompute(patch, None)

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

    ## Same number of objects in current and previous frame
    def less_or_equal_objects(self, detections, names, codebook, re_hist):
        self.text_scene = "less or same number of objects"

        for i in range(self.new_num):
            det = detections[i].data.cpu().tolist()
            obj_class = names[int(det[5])]
            bbox = [int(j) for j in det[0:4]]  ## bbox = [x1 y1 x2 y2]

            ## Find SURF features in the patch
            kp, des, hist = track_objects.extract_features(self, bbox, codebook)

            ## Calculate match to object in previous frame
            track_objects.BF_match_finder(self, des, hist, obj_class)

            ## Same objects as before
            if self.matches[0][1] > self.match_thres: ## check if the match is valid

                ## Current observation
                self.new_objects[self.matches[0][0]] = des

                ## Bag of words frame comparison
                BoVW_match = BoVW_comparison(codebook, re_hist, des, self.img, self.disp, bbox[0], bbox[1], obj_class)
                self.new_histograms[self.matches[0][0]] = BoVW_match.hist

                if BoVW_match.diff_match < bow_thres:
                    ## Append found match
                    self.codebook_match[self.matches[0][0]].append(BoVW_match.object)
                else:
                    self.codebook_match[self.matches[0][0]].append('bad_match')

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
                self.track_ended = True
                self.id += 1
                obj_class = names[int(det[5])]
                label = names[int(det[5])] + '_' + str(self.id)

                ## Find SURF features in the patch
                kp, des, hist = track_objects.extract_features(self, bbox, codebook)

                ## Current observation
                self.new_objects[label] = des

                ## Bag of words frame comparison
                BoVW_match = BoVW_comparison(codebook, re_hist, des, self.img, self.disp, bbox[0], bbox[1], obj_class)
                self.new_histograms[label] = BoVW_match.hist

                ## Append found match
                self.codebook_match[label] = []

                if BoVW_match.diff_match < bow_thres:
                    ## Append found match
                    self.codebook_match[label].append(BoVW_match.object)
                else:
                    self.codebook_match[label].append('bad_match')

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

            self.old_objects = self.new_histograms
            self.old_num = self.new_num

            return self

    ## More objects in current frame
    def more_objects(self, detections, names, codebook, re_hist):
        self.text_scene = "more new objects"

        ## Check in old objects and find which of the new ones is the best match
        found_matches = []
        found_match_box = {}
        found_match_des = {}
        found_match_box_end = {}

        for object in self.old_objects:

                prev_pct = self.match_thres
                old_id = object.split('_')
                old_id = old_id[0]

                for i in range(self.new_num):
                    det = detections[i].data.cpu().tolist()
                    obj_class = names[int(det[5])]
                    bbox = [int(j) for j in det[0:4]]  ## bbox = [x1 y1 x2 y2]

                    ## Find SURF features in the patch
                    kp, des, hist = track_objects.extract_features(self, bbox, codebook)

                    if obj_class == old_id:
                        ## BF matcher - Find the best match in the current frame
                        dotP = np.sum(self.old_objects[object] * hist)
                        norm = np.linalg.norm(hist)
                        norm_codebook = np.linalg.norm(self.old_objects[object])
                        pct = dotP*100 / (norm * norm_codebook)
                    else:
                        pct = 0

                    ## Keep best match
                    if pct > prev_pct:
                        self.pcts[object] = pct
                        found_match_box[object] = [bbox[0], bbox[1]]
                        found_match_box_end[object] = [bbox[2], bbox[3]]
                        found_match_des[object] = des
                    prev_pct = pct

                ## Current observation
                self.new_objects[object] = found_match_des[object]

                x1y1 = found_match_box[object]
                x2y2 = found_match_box_end[object]
                track_objects.draw_text(self, x1y1[0], x1y1[1])

                ## Bag of words frame comparison
                BoVW_match = BoVW_comparison(codebook, re_hist, found_match_des[object], self.img, self.disp, x1y1[0], x1y1[1], obj_class)
                self.new_histograms[object] = BoVW_match.hist

                ## Append found match
                if BoVW_match.diff_match < bow_thres:
                    ## Append found match
                    self.codebook_match[object].append(BoVW_match.object)
                else:
                    self.codebook_match[object].append('bad_match')

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
                        obj_class = names[int(det[5])]
                        bbox = [int(j) for j in det[0:4]]  ## bbox = [x1 y1 x2 y2]

                        if [bbox[0], bbox[1]] not in found_matches:
                            self.id += 1
                            label = names[int(det[5])] + '_' + str(self.id)

                            ## Find SURF features in the patch
                            kp, des, hist = track_objects.extract_features(self, bbox, codebook)

                            self.new_objects[label] = des
                            self.track_ended = True

                            ## Bag of words frame comparison
                            BoVW_match = BoVW_comparison(codebook, re_hist, des, self.img, self.disp, bbox[0], bbox[1], obj_class)
                            self.new_histograms[label] = BoVW_match.hist

                            ## Append found match
                            self.codebook_match[label] = []

                            if BoVW_match.diff_match < bow_thres:
                                ## Append found match
                                self.codebook_match[label].append(BoVW_match.object)
                            else:
                                self.codebook_match[label].append('bad_match')

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

        self.old_objects = self.new_histograms
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
        self.track_ended = False

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