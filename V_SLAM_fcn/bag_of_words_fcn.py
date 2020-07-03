#!/usr/bin/env python
import numpy as np
import cv2
import operator

class BoVW_comparison:

    def img_hist(self):
        ## Initializations
        k =self.codebook.shape[0]

        num_feat = self.current_des.shape[0]  # Number of extracted features for frame to be tested
        des = np.dstack(np.split(self.current_des, num_feat))

        words_stack = np.dstack([self.codebook] * num_feat)  ## stack words depthwise
        diff = words_stack - des
        dist = np.linalg.norm(diff, axis=1)
        idx = np.argmin(dist, axis=0)
        self.hist, n_bins = np.histogram(idx, bins=k)
        self.hist = self.hist.reshape(1, k)

        return self

    def find_match(self):
        ## codebook_hist is a dictionary
        #n_frames = len(self.codebook_hist)
        sim_cos = {}
        #eucl_dist = {}
        for key in self.codebook_hist:
            curr_id = key.split('_')
            curr_id = curr_id[0]
            if curr_id == self.obj_class:
                #eucl_dist[key] = np.linalg.norm(self.codebook_hist[key] - self.hist)
                dotP = np.sum(self.codebook_hist[key] * self.hist, axis=1)
                norm = np.linalg.norm(self.hist)
                norm_codebook = np.linalg.norm(self.codebook_hist[key], axis=1)
                sim_arr = dotP / (norm * norm_codebook)
                sim_arr[np.isinf(sim_arr)] = 0
                sim_cos[key] = max(sim_arr)


        ## Most similar frame
        sim_cos = sorted(sim_cos.items(), key=operator.itemgetter(1), reverse=True)

        #self.object = max(sim_cos.keys(), key=(lambda k: sim_cos[k]))
        self.object = sim_cos[0][0]

        ## Match percentage
        self.cos_pct = sim_cos[0][1]*100

        ## Second best match
        self.sbo = sim_cos[1][0]
        self.sbp = sim_cos[1][1]*100

        self.diff_match = self.sbp/self.cos_pct

        ## Split the strings
        #self.object = self.object.split('p')
        #self.object = self.object[0]

        return self

    def draw_ids(self, x_txt, y_txt):
        ## Put text next to the bounding box
        org = (int(x_txt + 10), int(y_txt + 50))  # org - text starting point
        disp_match = round(self.cos_pct)
        txt = '{} {}'.format(self.object, disp_match)
        self.disp = cv2.putText(self.disp, txt, org, cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255,230,0), 2, cv2.LINE_AA)

        return self

    def __init__(self, codebook, codebook_histograms, des, frame, disp, x_txt, y_txt, object_class):
        self.codebook = codebook
        self.codebook_hist = codebook_histograms
        self.current_des = des
        self.hist = np.empty((0,0))
        self.cos_pct = 0
        self.object = " "
        self.img = frame
        self.disp = disp
        self.obj_class = object_class
        self.diff_match = 0

        if des is not None:
            BoVW_comparison.img_hist(self) ## Create image histogram
            BoVW_comparison.find_match(self) ## Find best match
            BoVW_comparison.draw_ids(self, x_txt, y_txt) ## Draw ids on frame

            ## Put text next to the bounding box
            org = (int(x_txt + 10), int(y_txt + 80))  # org - text starting point
            disp_match = round(self.sbp)
            txt = '{} {}'.format(self.sbo, disp_match)
            self.disp = cv2.putText(self.disp, txt, org, cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 230, 0), 2, cv2.LINE_AA)

            ## Put text next to the bounding box
            org = (int(x_txt + 10), int(y_txt + 110))  # org - text starting point
            disp_match = round(self.diff_match,3)
            txt = '{}'.format(disp_match)
            col = (255, 150, 0) if disp_match < 0.8 else (0, 0, 255)
            self.disp = cv2.putText(self.disp, txt, org, cv2.FONT_HERSHEY_COMPLEX_SMALL, 1,col , 2, cv2.LINE_AA)
        else:
            self.hist = 0
            self.cos_pct = 0