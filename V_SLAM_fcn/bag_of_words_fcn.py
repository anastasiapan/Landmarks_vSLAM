#!/usr/bin/env python
import numpy as np
import cv2
import operator

from parameters import codebook, tf_idf_histograms_flag

'''
Class: BoVW_comparison
Comparison with cosine similarity score using the Bag-of-Visual-Words method

Inputs:
    - codebook_histograms : Histograms of keyframes in dictionary
    - hist : Histogram of current observation
    - frame : Input image
    - disp : Image to be outputted with printed useful information
    - x_txt, y_txt : Position to print the best match id
    - object_class : Class of the current observation

Returns/class members:
    - object : Most similar previously observed object
    - cos_pct : Matching percentage of the most similar object
    - disp : Output image with print-outs
'''
class BoVW_comparison:

    '''
    Class memeber function: find_match
    Aranges the current observation's similarities with the keyframes in descending order
    and returns the best match (id and percentage)
    '''
    def find_match(self):
        ## codebook_hist is a dictionary
        sim_cos = {}
        for key in self.codebook_hist: ## Run through each old object detected
            curr_id = key.split('_')
            curr_id = curr_id[0]
            if curr_id == self.obj_class: ## Similar only if they belong to the same class
                dotP = np.sum(self.codebook_hist[key] * self.hist, axis=1)
                norm = np.linalg.norm(self.hist)
                norm_codebook = np.linalg.norm(self.codebook_hist[key], axis=1)
                sim_arr = dotP / (norm * norm_codebook)
                sim_arr[np.isinf(sim_arr)] = 0
                sim_arr[np.isnan(sim_arr)] = 0
                sim_cos[key] = max(sim_arr)
            else: ## Cannot be similar
                sim_cos[key] = 0

        ## Sort results in descending order
        sim_cos = sorted(sim_cos.items(), key=operator.itemgetter(1), reverse=True)

        ## Most similar frame
        self.object = sim_cos[0][0]
        self.cos_pct = sim_cos[0][1]*100

        return self

    '''
    Class memeber function: draw_ids
    Draws the matching results
    '''
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

        BoVW_comparison.find_match(self) ## Find best match

        curr_id = self.object.split('_')
        curr_id = curr_id[0]
        if self.obj_class == curr_id:
            BoVW_comparison.draw_ids(self, x_txt, y_txt) ## Draw ids on frame