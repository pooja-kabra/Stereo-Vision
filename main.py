# -*- coding: utf-8 -*-
"""
Created on Tue Apr  5 21:10:18 2022

@author: pooja
"""
import stereocam
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from imageProcess import normalize
import imutils


RESIZE_RATIO = 4.79

def main():
    img0 =cv.imread("im0.png")
    img1 =cv.imread("im1.png")
    h0, w0,_ = img0.shape
    H = int(h0/RESIZE_RATIO)
    W = int(w0/RESIZE_RATIO)
    img0 = imutils.resize(img0, width=W, height=H)
    img1 = imutils.resize(img1, width=W, height=H)

    
    K_l = np.array([[5299.313, 0, 1263.818], [0, 5299.313, 977.763], [0, 0, 1]])
    K_r = np.array([[5299.313, 0, 1438.004], [0, 5299.313, 977.763], [0, 0, 1]])
    B = 177.288
    f = 5299.313
    myStereo = stereocam.StereoCam(img0, img1, K_l, K_r, B, f, RESIZE_RATIO)
    
    disparity, depth = myStereo.getDepthMap(img0, img1)

    disparity_map= disparity #normalize(disparity)
    depth_map= depth #normalize(depth)

    
    fig = plt.figure()
    
    ax = fig.add_subplot(2, 2, 1)
    imgplot = plt.imshow(img0)
    ax.set_title('Left Image')
    
    ax = fig.add_subplot(2, 2, 2)
    imgplot = plt.imshow(img0)
    ax.set_title('Right Image')
    
    ax = fig.add_subplot(2, 2, 3)
    imgplot = plt.imshow(disparity_map, cmap="jet")
    ax.set_title('Disparity')
    
    
    ax = fig.add_subplot(2, 2, 4)
    imgplot = plt.imshow(depth_map, cmap="jet")
    ax.set_title('Depth')
    
    plt.subplots_adjust(left=0.1,
                    bottom=0.1, 
                    right=0.9, 
                    top=0.9, 
                    wspace=0.4, 
                    hspace=0.4)
    
    plt.savefig("Depth from Stereo")

if __name__ == "__main__":
    main()
    