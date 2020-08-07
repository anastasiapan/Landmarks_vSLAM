#!/usr/bin/env python
import numpy as np
import cv2
import operator

from parameters import codebook

## TF-IDF reweighting keyframes' histograms
def TF_IDF_reweight(kf_hist, hist):
    #tf_idf_kf_phrases = dict(kf_phrases)
    #for kf in kf_phrases:
    #hist_arr = kf_phrases[kf]
    hist_arr = np.append(kf_hist, hist, axis=0)
    n_frames = hist_arr.shape[0]
    nd = np.sum(hist_arr, axis=1).reshape(n_frames, 1)  # Total number of words in one frame
    norm_h = hist_arr / nd

    ## number of images that a word occurs
    ni = np.count_nonzero(hist_arr, axis=0).astype(float)
    div = np.divide(n_frames, ni, out=np.zeros_like(ni), where=ni != 0)
    log_div = np.log(div, out=np.zeros_like(div), where=div != 0)
    re_hist_arr = norm_h * log_div

    last_row = re_hist_arr.shape[0]-1
    re_hist = re_hist_arr[last_row, :]
    re_hist_arr = re_hist_arr[0:last_row, :]
        #tf_idf_kf_phrases[kf] = re_hist_arr

    return re_hist_arr, re_hist

class BoVW_comparison:

    def find_match(self):
        ## codebook_hist is a dictionary
        #n_frames = len(self.codebook_hist)
        sim_cos = {}
        #eucl_dist = {}
        #cb_hist = list(self.codebook_hist.items())
        #cb_hist = dict(cb_hist[0:-1])
        #for key in cb_hist:
        for key in self.codebook_hist:
            curr_id = key.split('_')
            curr_id = curr_id[0]
            #keyframe_hist = cb_hist[key]
            #print(keyframe_hist.shape)
            if curr_id == self.obj_class:# and keyframe_hist.shape[0] > 20:
                #eucl_dist[key] = np.linalg.norm(self.codebook_hist[key] - self.hist)
                #re_hist_arr, re_hist = TF_IDF_reweight(self.codebook_hist[key], self.hist)
                #dotP = np.sum(re_hist_arr * re_hist, axis=1)
                #norm = np.linalg.norm(re_hist)
                #norm_codebook = np.linalg.norm(re_hist_arr, axis=1)
                dotP = np.sum(self.codebook_hist[key] * self.hist, axis=1)
                norm = np.linalg.norm(self.hist)
                norm_codebook = np.linalg.norm(self.codebook_hist[key], axis=1)
                sim_arr = dotP / (norm * norm_codebook)
                sim_arr[np.isinf(sim_arr)] = 0
                sim_arr[np.isnan(sim_arr)] = 0
                sim_cos[key] = max(sim_arr)
            else:
                sim_cos[key] = 0
        ## Most similar frame
        sim_cos = sorted(sim_cos.items(), key=operator.itemgetter(1), reverse=True)

        #self.object = max(sim_cos.keys(), key=(lambda k: sim_cos[k]))
        self.object = sim_cos[0][0]

        ## Match percentage
        self.cos_pct = sim_cos[0][1]*100

        ## Second best match
        if len(sim_cos) > 1:
            self.sbo = sim_cos[1][0]
            self.sbp = sim_cos[1][1]*100

            self.diff_match = self.sbp/self.cos_pct if self.cos_pct != 0 else 1.0

        return self

    def draw_ids(self, x_txt, y_txt):
        ## Put text next to the bounding box
        org = (int(x_txt + 10), int(y_txt + 50))  # org - text starting point
        disp_match = round(self.cos_pct)
        obj_name = self.object.split('_')
        prnt_label = 'Fire_'+str(obj_name[1]) if obj_name[0]=='Fire Extinguisher' else self.object ## Fire Extinguisher is too large to display
        txt = '{} {}'.format(prnt_label, disp_match)
        self.disp = cv2.putText(self.disp, txt, org, cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (51, 255, 255), 1, cv2.LINE_AA)

        return self

    def __init__(self, codebook_histograms, hist, frame, disp, x_txt, y_txt, object_class):
        self.codebook_hist = codebook_histograms
        self.hist = hist
        self.cos_pct = 0
        self.object = " "
        self.img = frame
        self.disp = disp
        self.obj_class = object_class
        self.sbo = 'a_'
        self.sbp = 0
        self.diff_match = 0.0

        BoVW_comparison.find_match(self) ## Find best match

        curr_id = self.object.split('_')
        curr_id = curr_id[0]
        if self.obj_class == curr_id:
            BoVW_comparison.draw_ids(self, x_txt, y_txt) ## Draw ids on frame

        s_id = self.sbo.split('_')
        s_id = s_id[0]