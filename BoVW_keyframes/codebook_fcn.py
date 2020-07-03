#!/usr/bin/env python
import numpy as np
import cv2
import os
import time
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

def extract_features(img_path, hess_th, des_size):
    features = np.empty((0, des_size))
    for imagepath in sorted(img_path):
        # Capture frame-by-frame
        frame = cv2.imread(str(imagepath))

        ## Extract SURF features
        surf = cv2.xfeatures2d.SURF_create(hess_th)
        kp, des = surf.detectAndCompute(frame, None)
        features = np.append(features, des, axis=0)
        print(features.shape)

        img2 = cv2.drawKeypoints(frame, kp, None)

        plt.imshow(img2), plt.show()

    return features

def cluster_vocabulary(voc, num_clusters):

    ## Kmeans clustering
    kmeans = KMeans(n_clusters=num_clusters)
    kmeans.fit(voc)
    print("Clustering done!")

    words = kmeans.cluster_centers_

    return words

class create_codebook:

    ## Create histogram for codebook images
    def create_codebook_histogram(self):
        ## Initializations
        num_w = self.words.shape[0]

        ## Main loop
        for imagepath in sorted(self.img_path):
            print(imagepath)
            self.n_frames += 1
            img = cv2.imread(str(imagepath))

            ## Extract SURF features
            surf = cv2.xfeatures2d.SURF_create(self.hess_th)
            kp, descr = surf.detectAndCompute(img, None)
            num_feat = descr.shape[0]  # Number of extracted features for frame to be tested
            print(num_feat)
            img2 = cv2.drawKeypoints(img, kp, None)
            plt.imshow(img2), plt.show()

            des = np.dstack(np.split(descr, num_feat))

            words_stack = np.dstack([self.words] * num_feat)  ## stack words depthwise
            diff = words_stack - des
            dist = np.linalg.norm(diff, axis=1)
            idx = np.argmin(dist, axis=0)
            hist, n_bins = np.histogram(idx, bins=num_w)
            hist = hist.reshape(1, num_w)

            head_tail = os.path.split(imagepath)
            label = head_tail[1].split('.')
            label = label[0]
            self.hist_cbook[label] = hist
            self.hist_arr = np.append(self.hist_arr, hist, axis=0)
        return self

    ## TF-IDF reweighting codebook's histogram
    def TF_IDF_reweight(self):
        ## normalized histogram
        nd = np.sum(self.hist_arr, axis=1).reshape(self.n_frames, 1)  # Total number of words in one frame
        norm_h = self.hist_arr / nd

        ## number of images that a word occurs
        ni = np.count_nonzero(self.hist_arr, axis=0).astype(float)
        div = np.divide(self.n_frames, ni, out=np.zeros_like(ni), where=ni != 0)
        log_div = np.log(div, out=np.zeros_like(div), where=div != 0)
        re_hist_arr = norm_h * log_div

        index = 0
        prev_id = 'start'
        for key, value in sorted(self.hist_cbook.items()):
            id_object = key.split('p')
            id_object = id_object[0]
            if id_object != prev_id:
                n_bins = re_hist_arr.shape[1]
                self.re_hist[id_object] = np.empty((1,n_bins))
                self.re_hist[id_object] = np.append(self.re_hist[id_object], re_hist_arr[index, :].reshape(1,n_bins), axis=0)
            else:
                self.re_hist[id_object] = np.append(self.re_hist[id_object], re_hist_arr[index, :].reshape(1,n_bins), axis=0)
            index+=1
            prev_id = id_object

        return self

    def __init__(self, path, descriptor_size, hessian, visual_words):
        self.n_frames = 0
        self.img_path = path
        self.des_size = descriptor_size
        self.hess_th = hessian
        self.words = visual_words
        self.re_hist = {}
        self.hist_cbook = {}
        self.hist_arr = np.empty((0, visual_words.shape[0]))
        create_codebook.create_codebook_histogram(self)
        create_codebook.TF_IDF_reweight(self)

## Cost matrix: Comparison of training dataset images with each other
def calculate_cost(hist):
    n_frames = hist.shape[0]
    cost_matrix_eucl = np.zeros((n_frames, n_frames))
    cost_matrix_cos = np.zeros((n_frames, n_frames))

    for row, hist_row in enumerate(hist):
        for col, hist_col in enumerate(hist):
            eucl_dist = np.linalg.norm(hist_row - hist_col)
            cost_matrix_eucl[row, col] = eucl_dist
            cos_sim = np.dot(hist_row, hist_col) / (np.linalg.norm(hist_row) * np.linalg.norm(hist_col))
            cost_matrix_cos[row, col] = cos_sim
    return cost_matrix_eucl, cost_matrix_cos


